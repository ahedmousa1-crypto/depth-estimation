"""
Monocular metric depth fine-tuning on Modal.

Backbone protocol follows Ke et al., CVPR 2024 ("Repurposing Diffusion-Based
Image Generators for Monocular Depth Estimation", arXiv:2312.02145):
  * backbone: Stable Diffusion 1.5
  * VAE + text encoder frozen; only the denoising UNet is trained
  * UNet conv_in expanded 4 -> 8 channels (RGB latent || depth-RGB latent),
    duplicated weights scaled by 0.5 to preserve activation magnitude
  * text conditioning disabled (single cached empty-prompt embedding)
  * train: DDPM 1000 steps, epsilon-prediction, Adam 3e-5, effective batch 16, h-flip aug
  * infer: DDIM 50 steps, 10-member ensemble

Depth representation follows "Image Generators are Generalist Vision
Learners" (Vision Banana, Gabeur et al. 2026), Sec. 3.2: instead of an
affine-invariant grayscale target, ground-truth metric depth (meters) is
encoded as an RGB image via a bijection (see depth_codec.py):
  1. Barron 2025 power transform, lambda=-3, c=10/3, warps d in [0, inf)
     to a curved distance t in [0, 1).
  2. t selects a position along a 7-edge path across the RGB cube's 8
     corners, black -> white.
This makes the training target and the model's output an ordinary RGB
image (so it VAE-encodes/decodes like any other picture) while remaining
exactly invertible back to metric depth -- no affine-invariant rescaling,
no per-image normalization, no post-hoc alignment at eval time.

Dataset assumption (see CFG["depth_png_scale"]): the `depth` column holds
a 16-bit single-channel PNG in centimeters (uint16, so 0-655.35 m range).
If the DIODE-derived HF dataset stores depth differently (e.g. raw float32
meters), change _depth_png_to_meters() and depth_png_scale accordingly.

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
    # Model
    "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",

    # Dataset -- DIODE 3K on HuggingFace
    "dataset": "ahd03/diode-depth-3k",
    "dataset_split": "train",
    "rgb_col": "image",    # column name in your HF dataset
    "depth_col": "depth",  # column name in your HF dataset

    # Metric depth is stored as a 16-bit PNG scaled by this factor
    # (default: centimeters, so meters = pixel_value / 100.0).
    "depth_png_scale": 100.0,

    # Resolution -- 640x640 is above SD1.5's native 512 but fits on L40S.
    # If you get OOM errors, drop to 512 and set micro_batch to 4.
    "proc_h": 640,
    "proc_w": 640,

    # Train/val split -- 10% held out for validation (200 images)
    "val_frac": 0.10,
    "seed": 42,

    # Batch:
    #   micro_batch = images processed on GPU at once (limited by VRAM)
    #   grad_accum  = how many micro batches before a weight update
    #   effective batch = micro_batch x grad_accum = 2 x 8 = 16
    "micro_batch": 2,
    "grad_accum": 8,

    # Optimizer
    "lr": 3e-5,
    "warmup": 200,
    "max_steps": 10000,

    # Logging / checkpointing
    "val_every": 500,
    "ckpt_every": 1000,

    # Misc training options
    "multires_noise": False,
    "use_8bit_adam": False,

    # Inference / evaluation
    "ddim_steps": 50,
    "ensemble": 10,
    "eval_limit": 100,

    # Where to push the final model
    "hub_repo": "ahd03/diode-depth-model-v2",
}

GPU = "L40S"
app = modal.App("depth-estimation-training")

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
    # Ships depth_codec.py into the container so the bijective encode/decode
    # is defined in exactly one place, shared with local testing.
    .add_local_python_source("depth_codec", copy=True)
)

out_vol = modal.Volume.from_name("depth-estimation-output", create_if_missing=True)
cache_vol = modal.Volume.from_name("depth-estimation-cache", create_if_missing=True)
VOLUMES = {"/output": out_vol, "/cache": cache_vol}
SECRETS = [modal.Secret.from_name("huggingface-secret2")]


# =========================================================================
# Shared helpers
# =========================================================================
def _build_helpers():
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    import depth_codec

    def rgb_to_tensor(pil_img, h, w):
        img = pil_img.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def depth_png_to_meters(pil_img, h, w, scale=CFG["depth_png_scale"]):
        """16-bit single-channel PNG -> metric depth array (H, W), meters."""
        d = pil_img.resize((w, h), Image.NEAREST)
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim == 3:  # some encoders write 16-bit depth as I;16 -> 'L' fallback
            arr = arr[..., 0]
        return arr / scale

    def depth_to_tensor(pil_img, h, w):
        """Ground-truth metric depth (meters) -> bijective RGB target in
        [-1, 1], ready for the VAE, via depth_codec.depth_to_rgb."""
        meters = depth_png_to_meters(pil_img, h, w)
        rgb01 = depth_codec.depth_to_rgb(meters)  # (H, W, 3) in [0, 1]
        rgb = rgb01.astype(np.float32) * 2.0 - 1.0
        return torch.from_numpy(rgb).permute(2, 0, 1)

    def depth_tensor_to_meters(t):
        """Inverse of depth_to_tensor. t: (3, H, W) tensor/array in [-1, 1]
        (or a numpy array shaped (H, W, 3) in [0, 1]) -> meters (H, W)."""
        if torch.is_tensor(t):
            arr = t.permute(1, 2, 0).cpu().numpy()
            rgb01 = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
        else:
            rgb01 = np.clip(t, 0.0, 1.0)
        return depth_codec.rgb_to_depth(rgb01)

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

    def depth_metrics(pred_m, gt_m, mask):
        """Direct metric-space comparison, no scale/shift alignment --
        the codec is trained to predict absolute depth, so alignment would
        hide exactly the error we care about."""
        pred_v = np.clip(pred_m[mask], 1e-6, None)
        gt_v = np.clip(gt_m[mask], 1e-6, None)
        absrel = float(np.mean(np.abs(pred_v - gt_v) / gt_v))
        ratio = np.maximum(pred_v / gt_v, gt_v / pred_v)
        return {
            "absrel": absrel,
            "delta1": float(np.mean(ratio < 1.25)),
            "rmse": float(np.sqrt(np.mean((pred_v - gt_v) ** 2))),
        }

    return dict(
        rgb_to_tensor=rgb_to_tensor,
        depth_to_tensor=depth_to_tensor,
        depth_tensor_to_meters=depth_tensor_to_meters,
        multires_noise=multires_noise,
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

    # Expand conv_in from 4 to 8 channels to accept
    # concatenated [RGB latent | noisy depth-RGB latent]
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
# 1. Inspect the dataset -- run this BEFORE training to verify columns
#    and depth value ranges look correct
# =========================================================================
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=1800)
def inspect():
    import numpy as np
    from datasets import load_dataset

    ds = load_dataset(CFG["dataset"], split=CFG["dataset_split"])
    print(f"n = {len(ds)}")
    print(f"features = {ds.features}")
    rgb_col, depth_col = _detect_columns(ds)

    im = ds[0][rgb_col]
    print(f"{rgb_col}: mode={im.mode} size={im.size}")

    dep = ds[0][depth_col]
    arr = np.asarray(dep, dtype=np.float32)
    meters = arr / CFG["depth_png_scale"]
    print(
        f"{depth_col}: mode={dep.mode} size={dep.size} "
        f"raw min={arr.min():.0f} raw max={arr.max():.0f} "
        f"-> meters min={meters.min():.3f} max={meters.max():.3f} "
        f"zeros={(arr == 0).mean():.3f}"
    )
    print(
        "\nSanity check: does 'meters max' look like a plausible DIODE depth "
        "(indoor: tens of meters, outdoor: up to a few hundred)?"
    )
    print(
        "If min/max look wrong, CFG['depth_png_scale'] does not match how "
        "the Colab dataset-builder encoded depth -- fix it before training."
    )
    print("If rgb/depth look swapped, fix CFG['rgb_col'] / CFG['depth_col'].")
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

    train_ds, val_ds = _make_datasets(H)
    dl = DataLoader(
        train_ds,
        batch_size=CFG["micro_batch"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    vdl = DataLoader(
        val_ds,
        batch_size=CFG["micro_batch"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

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
        target = noise  # SD1.5 uses epsilon-prediction

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
            {
                "unet": unet.state_dict(),
                "opt": opt.state_dict(),
                "sched": lr_sched.state_dict(),
                "step": step,
            },
            ck,
        )
        out_vol.commit()

    print(
        f"training: {CFG['max_steps']} steps | "
        f"micro_batch {CFG['micro_batch']} x grad_accum {CFG['grad_accum']} "
        f"= effective batch {CFG['micro_batch'] * CFG['grad_accum']} | "
        f"resolution {CFG['proc_h']}x{CFG['proc_w']}"
    )

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
                print(
                    f"step {step}/{CFG['max_steps']} "
                    f"loss {loss.item() * CFG['grad_accum']:.4f} "
                    f"lr {lr_sched.get_last_lr()[0]:.2e} "
                    f"{el / max(step, 1):.1f}s/step",
                    flush=True,
                )

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
# 3. Evaluate -- DDIM sampling + ensemble, decoded to metric depth via the
#    bijective codec. No least-squares alignment: this is a true zero-shot
#    metric-depth evaluation, matching the Vision Banana protocol.
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
        print(f"!! {path} not found -- evaluating un-finetuned baseline")

    ddim = DDIMScheduler.from_pretrained(CFG["model_id"], subfolder="scheduler")
    ddim.set_timesteps(CFG["ddim_steps"], device=device)

    _, val_ds = _make_datasets(H)
    os.makedirs("/output/preds", exist_ok=True)

    @torch.no_grad()
    def predict_rgb01(img_t):
        """Returns the ensemble-averaged predicted depth-RGB image in
        [0, 1], shape (H, W, 3) -- averaging happens in RGB space, then a
        single codec decode recovers metric depth."""
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
            d = vae.decode(z / vae.config.scaling_factor).sample.squeeze(0)  # (3,H,W)
            rgb01 = ((d + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            members.append(rgb01)
        return np.clip(np.mean(members, axis=0), 0, 1)

    scores = []
    for i in range(min(CFG["eval_limit"], len(val_ds))):
        img_t, dep_t = val_ds[i]
        pred_rgb01 = predict_rgb01(img_t)
        pred_m = H["depth_tensor_to_meters"](pred_rgb01)
        gt_m = H["depth_tensor_to_meters"](dep_t)

        mask = gt_m > 1e-3
        scores.append(H["depth_metrics"](pred_m, gt_m, mask))

        if i < 8:
            Image.fromarray((pred_rgb01 * 255).astype(np.uint8)).save(
                f"/output/preds/{i:03d}_pred_rgb.png"
            )
            gt_rgb01 = ((dep_t.permute(1, 2, 0).numpy() + 1.0) / 2.0).clip(0, 1)
            Image.fromarray((gt_rgb01 * 255).astype(np.uint8)).save(
                f"/output/preds/{i:03d}_gt_rgb.png"
            )
            rgb_np = (img_t.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
            Image.fromarray(rgb_np.astype(np.uint8)).save(
                f"/output/preds/{i:03d}_rgb.png"
            )

    agg = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    print(f"\nn={len(scores)}  " + "  ".join(f"{k}={v:.4f}" for k, v in agg.items()))
    print("Metrics are direct metric-depth comparisons (meters), no alignment applied.")
    out_vol.commit()
    return agg


# =========================================================================
# 4. Push to HuggingFace Hub
# =========================================================================
@app.function(image=image, volumes=VOLUMES, secrets=SECRETS, timeout=3600)
def push(ckpt: str = "unet_best"):
    import os
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError(
            "No HF token found in Modal secret. "
            "Add a key named HF_TOKEN to your huggingface-secret2 secret."
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