"""
Marigold-style monocular depth fine-tuning on Modal.
Paper: "Repurposing Diffusion-Based Image Generators for Monocular Depth Estimation"
(Ke et al., CVPR 2024) -- arXiv:2312.02145

Protocol reproduced here:
  * backbone: Stable Diffusion 2 (768-v) -> v-objective, NOT epsilon
  * VAE + text encoder frozen; only the denoising UNet is trained
  * UNet conv_in expanded 4 -> 8 channels (RGB latent || noisy depth latent),
    duplicated weights scaled by 0.5 to preserve activation magnitude
  * depth: affine-invariant normalization to [-1, 1] using 2nd/98th
    percentiles, then replicated to 3 channels before VAE encoding
  * text conditioning disabled (single cached empty-prompt embedding)
  * train: DDPM 1000 steps, Adam 3e-5, effective batch 32, h-flip aug
  * infer: DDIM 50 steps, 10-member ensemble

Usage:
    modal run train.py::inspect      # ALWAYS run this first
    modal run train.py::train
    modal run train.py::evaluate
    modal run train.py::push
"""

import modal

# -------------------------------------------------------------------------
# Config -- edit here, nothing else.
# -------------------------------------------------------------------------
CFG = {
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",  # 768-v checkpoint (vprediction)
    "dataset": "ahd03/nyu-depth-controlnet",
    "dataset_split": "train",
    "rgb_col": "conditioning_image",
    "depth_col": "image",
    "proc_h": 480,
    "proc_w": 640,
    "val_frac": 0.10,
    "seed": 42,
    "micro_batch": 2,
    "grad_accum": 8,
    "lr": 3e-5,
    "warmup": 100,
    "max_steps": 2000,
    "val_every": 250,
    "ckpt_every": 500,
    "multires_noise": True,
    "use_8bit_adam": False,
    "ddim_steps": 50,
    "ensemble": 10,
    "eval_limit": 66,
    "hub_repo": "ahd03/nyu-depth-marigoldmodel",
}

GPU = "L40S"
app = modal.App("marigold-depth-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "datasets==2.21.0",
        "huggingface_hub==0.25.2",
        "safetensors==0.4.5",
        "bitsandbytes==0.44.1",
        "Pillow==10.4.0",
        "numpy==1.26.4",
        "scipy==1.14.1",
        "tqdm",
    )
    .env({"HF_HOME": "/cache/hf"})
)

out_vol = modal.Volume.from_name("marigoldoutput", create_if_missing=True)
cache_vol = modal.Volume.from_name("marigold-cache", create_if_missing=True)
VOLUMES = {"/output": out_vol, "/cache": cache_vol}
SECRETS = [modal.Secret.from_name("huggingface-secret2")]


# =========================================================================
# Shared helpers (defined at module level, imported inside functions)
# =========================================================================
def _build_helpers():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    def to_gray_array(pil_img):
        return np.asarray(pil_img.convert("L"), dtype=np.float32) / 255.0

    def affine_invariant_norm(depth, lo=2.0, hi=98.0):
        valid = depth > 0
        if valid.sum() < 64:
            valid = np.ones_like(depth, dtype=bool)
        d2, d98 = np.percentile(depth[valid], [lo, hi])
        scale = max(float(d98 - d2), 1e-6)
        out = (depth - d2) / scale * 2.0 - 1.0
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    def rgb_to_tensor(pil_img, h, w):
        img = pil_img.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def depth_to_tensor(pil_img, h, w):
        d = pil_img.convert("L").resize((w, h), Image.NEAREST)
        d = np.asarray(d, dtype=np.float32) / 255.0
        d = affine_invariant_norm(d)
        return torch.from_numpy(d).unsqueeze(0).repeat(3, 1, 1)

    def multires_noise(latents, timesteps, num_train_ts, strength=0.9):
        b, c, h, w = latents.shape
        noise = torch.randn_like(latents)
        anneal = (timesteps.float() / num_train_ts).view(-1, 1, 1, 1)
        for i in range(1, 8):
            r = 2**i
            h2, w2 = max(1, h // r), max(1, w // r)
            up = F.interpolate(
                torch.randn(b, c, h2, w2, device=latents.device, dtype=latents.dtype),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            noise = noise + up * (strength**i) * anneal
            if h2 == 1 and w2 == 1:
                break
        return noise / noise.std(dim=(1, 2, 3), keepdim=True)

    def align_least_squares(pred, gt, mask):
        p, g = pred[mask], gt[mask]
        A = np.stack([p, np.ones_like(p)], axis=1)
        s, t = np.linalg.lstsq(A, g, rcond=None)[0]
        return pred * s + t

    def depth_metrics(pred, gt, mask):
        pred = np.clip(pred, 1e-6, None)
        gt_v, pr_v = gt[mask], pred[mask]
        absrel = float(np.mean(np.abs(pr_v - gt_v) / np.clip(gt_v, 1e-6, None)))
        ratio = np.maximum(
            pr_v / np.clip(gt_v, 1e-6, None), gt_v / np.clip(pr_v, 1e-6, None)
        )
        return {
            "absrel": absrel,
            "delta1": float(np.mean(ratio < 1.25)),
            "rmse": float(np.sqrt(np.mean((pr_v - gt_v) ** 2))),
        }

    return dict(
        to_gray_array=to_gray_array,
        affine_invariant_norm=affine_invariant_norm,
        rgb_to_tensor=rgb_to_tensor,
        depth_to_tensor=depth_to_tensor,
        multires_noise=multires_noise,
        align_least_squares=align_least_squares,
        depth_metrics=depth_metrics,
    )


def _detect_columns(ds):
    import numpy as np

    if CFG["rgb_col"] and CFG["depth_col"]:
        return CFG["rgb_col"], CFG["depth_col"]

    img_cols = [
        k for k, v in ds.features.items() if v.__class__.__name__ == "Image"
    ]
    if len(img_cols) < 2:
        raise ValueError(
            f"Need 2 image columns, found {img_cols}. Set rgb_col/depth_col in CFG."
        )

    scores = {}
    for col in img_cols:
        chan_var = []
        for i in range(min(8, len(ds))):
            a = np.asarray(ds[i][col].convert("RGB"), dtype=np.float32)
            chan_var.append(float(a.std(axis=2).mean()))
        scores[col] = float(np.mean(chan_var))

    depth_col = min(scores, key=scores.get)
    rgb_col = max(scores, key=scores.get)

    if scores[depth_col] > 3.0:
        print(f"!! channel-variance heuristic inconclusive: {scores}")
        print("!! falling back to ControlNet convention.")
        rgb_col, depth_col = "conditioning_image", "image"

    print(f"columns -> rgb={rgb_col}  depth={depth_col}  (variance {scores})")
    return rgb_col, depth_col


def _load_models(device, train_mode=True):
    import torch
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    mid = CFG["model_id"]
    vae = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(device).eval()
    unet = UNet2DConditionModel.from_pretrained(mid, subfolder="unet").to(device)
    sched = DDPMScheduler.from_pretrained(mid, subfolder="scheduler")
    print(f"prediction_type = {sched.config.prediction_type}")

    vae.requires_grad_(False)

    old = unet.conv_in
    if old.in_channels == 4:
        new = torch.nn.Conv2d(
            8,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            dilation=old.dilation,
            groups=old.groups,
            bias=old.bias is not None,
        ).to(device=device, dtype=old.weight.dtype)
        with torch.no_grad():
            new.weight[:, :4].copy_(old.weight * 0.5)
            new.weight[:, 4:].copy_(old.weight * 0.5)
            if old.bias is not None:
                new.bias.copy_(old.bias)
        unet.conv_in = new
        unet.register_to_config(in_channels=8)
        print("conv_in expanded 4 -> 8 channels")

    tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").to(device)
    txt.requires_grad_(False)

    with torch.no_grad():
        ids = tok(
            [""],
            padding="max_length",
            max_length=tok.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        empty_embed = txt(ids)[0].detach()

    del txt, tok
    torch.cuda.empty_cache()

    if train_mode:
        unet.enable_gradient_checkpointing()
        unet.train()
    else:
        unet.eval()

    return vae, unet, sched, empty_embed


def _make_datasets(H):
    import torch
    from datasets import load_dataset
    from torch.utils.data import Dataset

    hf = load_dataset(CFG["dataset"], split=CFG["dataset_split"])
    rgb_col, depth_col = _detect_columns(hf)

    hf = hf.shuffle(seed=CFG["seed"])
    n_val = max(1, int(len(hf) * CFG["val_frac"]))
    train_hf = hf.select(range(n_val, len(hf)))
    val_hf = hf.select(range(n_val))
    print(f"{len(hf)} pairs -> train {len(train_hf)} / val {len(val_hf)}")

    class DepthPairs(Dataset):
        def __init__(self, ds, augment):
            self.ds, self.augment = ds, augment

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            ex = self.ds[i]
            h, w = CFG["proc_h"], CFG["proc_w"]
            img = H["rgb_to_tensor"](ex[rgb_col], h, w)
            dep = H["depth_to_tensor"](ex[depth_col], h, w)
            if self.augment and torch.rand(1).item() < 0.5:
                img, dep = torch.flip(img, [2]), torch.flip(dep, [2])
            return img, dep

    return DepthPairs(train_hf, True), DepthPairs(val_hf, False)


def _encode(vae, x):
    return vae.encode(x).latent_dist.mode() * vae.config.scaling_factor


# =========================================================================
# 1. Inspect the dataset -- run this BEFORE training
# =========================================================================
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=1800)
def inspect():
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(CFG["dataset"], split=CFG["dataset_split"])
    print(f"n = {len(ds)}")
    print(f"features = {ds.features}")
    rgb_col, depth_col = _detect_columns(ds)

    for col in (rgb_col, depth_col):
        im = ds[0][col]
        a = np.asarray(im.convert("L"), dtype=np.float32)
        print(
            f"{col}: mode={im.mode} size={im.size} "
            f"min={a.min():.0f} max={a.max():.0f} "
            f"zeros={(a == 0).mean():.3f}"
        )

    print("\nIf rgb/depth look swapped, set CFG['rgb_col'] / CFG['depth_col'].")
    cache_vol.commit()


# =========================================================================
# 2. Train
# =========================================================================
@app.function(
    image=image,
    gpu=GPU,
    timeout=8 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
    retries=modal.Retries(max_retries=2),
)
def train():
    import os
    import time
    import json
    import torch
    from torch.utils.data import DataLoader
    from diffusers.optimization import get_scheduler
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    device = "cuda"
    H = _build_helpers()

    vae, unet, sched, empty_embed = _load_models(device, train_mode=True)
    n_train_ts = sched.config.num_train_timesteps
    pred_type = sched.config.prediction_type

    train_ds, val_ds = _make_datasets(H)
    dl = DataLoader(train_ds, batch_size=CFG["micro_batch"], shuffle=True, num_workers=2)
    vdl = DataLoader(val_ds, batch_size=CFG["micro_batch"], shuffle=False, num_workers=2)

    if CFG["use_8bit_adam"]:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(unet.parameters(), lr=CFG["lr"])
    else:
        opt = torch.optim.AdamW(unet.parameters(), lr=CFG["lr"])

    lr_sched = get_scheduler(
        "cosine",
        optimizer=opt,
        num_warmup_steps=CFG["warmup"],
        num_training_steps=CFG["max_steps"],
    )

    ck = "/output/state.pt"
    step = 0
    if os.path.exists(ck):
        st = torch.load(ck, map_location="cpu")
        unet.load_state_dict(st["unet"])
        opt.load_state_dict(st["opt"])
        lr_sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed at step {step}")

    def loss_for(img_t, dep_t, gen=None):
        img_t, dep_t = img_t.to(device), dep_t.to(device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            z_img = _encode(vae, img_t).float()
            z_dep = _encode(vae, dep_t).float()

        b = z_dep.shape[0]
        ts = torch.randint(0, n_train_ts, (b,), device=device, generator=gen).long()

        if CFG["multires_noise"]:
            noise = H["multires_noise"](z_dep, ts, n_train_ts)
        else:
            noise = torch.randn(z_dep.shape, device=device, generator=gen)

        noisy = sched.add_noise(z_dep, noise, ts)
        target = (
            sched.get_velocity(z_dep, noise, ts)
            if pred_type == "v_prediction"
            else noise
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = unet(
                torch.cat([z_img, noisy], dim=1),
                ts,
                encoder_hidden_states=empty_embed.repeat(b, 1, 1),
            ).sample

        return torch.nn.functional.mse_loss(out.float(), target.float())

    @torch.no_grad()
    def validate():
        unet.eval()
        g = torch.Generator(device=device).manual_seed(0)
        tot, n = 0.0, 0
        for img_t, dep_t in vdl:
            tot += loss_for(img_t, dep_t, gen=g).item()
            n += 1
        unet.train()
        return tot / max(n, 1)

    def save(tag):
        unet.save_pretrained(f"/output/{tag}")
        torch.save(
            {"unet": unet.state_dict(), "opt": opt.state_dict(),
             "sched": lr_sched.state_dict(), "step": step},
            ck,
        )
        out_vol.commit()

    print(f"training: {CFG['max_steps']} optimizer steps "
          f"(micro {CFG['micro_batch']} x accum {CFG['grad_accum']})")
    best, t0, accum = float("inf"), time.time(), 0
    opt.zero_grad(set_to_none=True)
    done = False

    while not done:
        for img_t, dep_t in dl:
            loss = loss_for(img_t, dep_t) / CFG["grad_accum"]
            loss.backward()
            accum += 1

            if accum < CFG["grad_accum"]:
                continue

            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            opt.step()
            lr_sched.step()
            opt.zero_grad(set_to_none=True)
            accum, step = 0, step + 1

            if step % 20 == 0:
                el = time.time() - t0
                print(f"step {step}/{CFG['max_steps']} "
                      f"loss {loss.item() * CFG['grad_accum']:.4f} "
                      f"lr {lr_sched.get_last_lr()[0]:.2e} "
                      f"{el / max(step, 1):.1f}s/step", flush=True)

            if step % CFG["val_every"] == 0:
                vl = validate()
                print(f"  val_loss {vl:.4f}", flush=True)
                if vl < best:
                    best = vl
                    save("unet_best")
                    with open("/output/best.json", "w") as f:
                        json.dump({"step": step, "val_loss": vl}, f)

            if step % CFG["ckpt_every"] == 0:
                save(f"unet_step{step}")

            if step >= CFG["max_steps"]:
                done = True
                break

    save("unet_final")
    print(f"done. best val_loss {best:.4f}")


# =========================================================================
# 3. Evaluate -- DDIM sampling + ensembling + affine-invariant metrics
# =========================================================================
@app.function(
    image=image,
    gpu=GPU,
    timeout=4 * 60 * 60,
    volumes=VOLUMES,
    secrets=SECRETS,
)
def evaluate(ckpt: str = "unet_best"):
    import os

    import numpy as np
    import torch
    from diffusers import DDIMScheduler, UNet2DConditionModel
    from PIL import Image

    device = "cuda"
    H = _build_helpers()
    vae, unet, _, empty_embed = _load_models(device, train_mode=False)

    path = f"/output/{ckpt}"
    if os.path.exists(path):
        unet = UNet2DConditionModel.from_pretrained(path).to(device).eval()
        print(f"loaded {path}")
    else:
        print(f"!! {path} not found -- evaluating the un-finetuned baseline")

    ddim = DDIMScheduler.from_pretrained(CFG["model_id"], subfolder="scheduler")
    ddim.set_timesteps(CFG["ddim_steps"], device=device)

    _, val_ds = _make_datasets(H)
    os.makedirs("/output/preds", exist_ok=True)

    @torch.no_grad()
    def predict(img_t):
        z_img = _encode(vae, img_t.unsqueeze(0).to(device))
        members = []
        for k in range(CFG["ensemble"]):
            g = torch.Generator(device=device).manual_seed(k)
            z = torch.randn(z_img.shape, device=device, generator=g)
            for t in ddim.timesteps:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = unet(
                        torch.cat([z_img, z], dim=1),
                        t,
                        encoder_hidden_states=empty_embed,
                    ).sample
                z = ddim.step(out.float(), t, z).prev_sample
            d = vae.decode(z / vae.config.scaling_factor).sample
            d = d.mean(dim=1).squeeze(0)
            members.append(((d + 1.0) / 2.0).clamp(0, 1).cpu().numpy())

        ref = members[0]
        m = np.ones_like(ref, dtype=bool)
        aligned = [ref] + [
            H["align_least_squares"](x, ref, m) for x in members[1:]
        ]
        return np.clip(np.mean(aligned, axis=0), 0, 1)

    scores = []
    for i in range(min(CFG["eval_limit"], len(val_ds))):
        img_t, dep_t = val_ds[i]
        pred = predict(img_t)
        gt = (dep_t[0].numpy() + 1.0) / 2.0
        mask = gt > 1e-3
        pred_a = H["align_least_squares"](pred, gt, mask)
        scores.append(H["depth_metrics"](pred_a, gt, mask))
        if i < 8:
            Image.fromarray((pred * 255).astype(np.uint8)).save(
                f"/output/preds/{i:03d}_pred.png"
            )
            Image.fromarray((gt * 255).astype(np.uint8)).save(
                f"/output/preds/{i:03d}_gt.png"
            )

    agg = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    print(f"\nn={len(scores)}  " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    print("NOTE: metrics are vs the 8-bit depth PNGs, not metric NYU depth.")
    out_vol.commit()
    return agg


# =========================================================================
# 4. Push to the Hub (separate step -- never inside the training loop)
# =========================================================================
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def push(ckpt: str = "unet_best"):
    import os
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF token in the 'huggingface-secret' Modal secret. "
            "Add a key named HF_TOKEN."
        )

    api = HfApi(token=token)
    api.create_repo(CFG["hub_repo"], exist_ok=True)
    api.upload_folder(folder_path=f"/output/{ckpt}", repo_id=CFG["hub_repo"])
    print(f"pushed /output/{ckpt} -> {CFG['hub_repo']}")


@app.local_entrypoint()
def main():
    print("Run one of:")
    print("  modal run train.py::inspect")
    print("  modal run train.py::train")
    print("  modal run train.py::evaluate")
    print("  modal run train.py::push")