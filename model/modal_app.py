"""
Modal backend for the depth estimation demo.

Serves the DIODE-trained metric depth model (see train.py / depth_codec.py):
an SD1.5-based diffusion UNet that denoises a depth map encoded as an RGB
image via the bijective codec, ensembled over several DDIM samples and
decoded back to metric depth in meters.

Falls back to Depth Anything V2 Small (relative depth) if the fine-tuned
checkpoint isn't on the Hub yet, so the demo works end-to-end before
training finishes and upgrades itself automatically once you push a
checkpoint with `modal run train.py::push` -- no changes needed here.

Deploy:      modal deploy modal_app.py
Test first:  modal serve modal_app.py
"""

import base64
import contextlib
import io
import os
import time

import modal

app = modal.App("depth-demo")

# ---------------------------------------------------------------------------
# Model + inference config -- tune here, nothing else needs to change.
# ---------------------------------------------------------------------------
CFG = {
    "base_model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "hub_repo": "ahd03/diode-depth-model-v2",  # must match train.py CFG["hub_repo"]
    "proc_h": 640,
    "proc_w": 640,
    "ddim_steps": 25,  # lower than train.py::evaluate's 50 -- interactive latency
    "ensemble": 4,      # lower than train.py::evaluate's 10 -- interactive latency
}

SECRETS = [modal.Secret.from_name("huggingface-secret2")]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate==0.34.2",
        "huggingface_hub==0.25.2",
        "safetensors==0.4.5",
        "pillow",
        "numpy",
        "fastapi[standard]",
    )
    # Ships the bijective codec so encode/decode logic lives in exactly one
    # place, shared with train.py and dataset_prep.py.
    .add_local_python_source("depth_codec", copy=True)
)

# ---------------------------------------------------------------------------
# Settings you will want to look at
# ---------------------------------------------------------------------------

# Origins allowed to call this endpoint. Add your GitHub Pages URL.
ALLOWED_ORIGINS = [
    "https://ahedmousa1-crypto.github.io",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# Optional shared token. Set the same string in app.js CONFIG.TOKEN.
# Leave empty to disable. This is a speed bump, not real authentication.
SHARED_TOKEN = os.environ.get("DEMO_TOKEN", "")

# Reject anything larger than this, so nobody can burn your credits with a
# 200 megapixel upload.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 4096 * 4096


@app.cls(
    image=image,
    gpu="L4",  # diffusion ensemble inference needs more than a T4 gives comfortably
    secrets=SECRETS,
    scaledown_window=300,  # keep the container alive 5 min after the last call
    max_containers=2,      # ceiling on concurrent GPU containers
    timeout=300,
)
class DepthService:

    @modal.enter()
    def load(self):
        """Runs once per container, before the first request. Tries the
        fine-tuned metric model first; falls back to Depth Anything V2 if
        it isn't on the Hub yet."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = False

        try:
            self._load_metric_model()
            self.metric = True
            print(f"loaded fine-tuned metric model from {CFG['hub_repo']}")
        except Exception as e:
            print(f"!! could not load {CFG['hub_repo']} ({e}) -- falling back to Depth Anything V2")
            self._load_fallback()

    def _load_metric_model(self):
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        mid = CFG["base_model_id"]
        self.vae = AutoencoderKL.from_pretrained(mid, subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(CFG["hub_repo"]).to(self.device).eval()
        self.ddim = DDIMScheduler.from_pretrained(mid, subfolder="scheduler")
        self.ddim.set_timesteps(CFG["ddim_steps"], device=self.device)

        tok = CLIPTokenizer.from_pretrained(mid, subfolder="tokenizer")
        txt = CLIPTextModel.from_pretrained(mid, subfolder="text_encoder").to(self.device)
        with torch.no_grad():
            ids = tok(
                [""],
                padding="max_length",
                max_length=tok.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            self.empty_embed = txt(ids)[0].detach()
        del txt, tok
        torch.cuda.empty_cache()

    def _load_fallback(self):
        from transformers import pipeline

        self.pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1,
        )

    def _infer_metric(self, pil_image):
        """DDIM-ensemble denoise, decode the bijective RGB codec back to
        metric depth. Returns a 2D float32 array of meters, same size as
        the input image."""
        import numpy as np
        import torch
        from PIL import Image
        import depth_codec

        h, w = CFG["proc_h"], CFG["proc_w"]
        img = pil_image.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)

        autocast_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else contextlib.nullcontext()
        )

        with torch.no_grad():
            z_img = self.vae.encode(img_t).latent_dist.mode() * self.vae.config.scaling_factor

            members = []
            for k in range(CFG["ensemble"]):
                g = torch.Generator(device=self.device).manual_seed(k)
                z = torch.randn(z_img.shape, device=self.device, generator=g)
                for t in self.ddim.timesteps:
                    with autocast_ctx:
                        out = self.unet(
                            torch.cat([z_img, z], dim=1),
                            t,
                            encoder_hidden_states=self.empty_embed,
                        ).sample
                    z = self.ddim.step(out.float(), t, z).prev_sample
                d = self.vae.decode(z / self.vae.config.scaling_factor).sample.squeeze(0)
                rgb01 = ((d + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                members.append(rgb01)

        rgb01 = np.clip(np.mean(members, axis=0), 0, 1)
        meters = depth_codec.rgb_to_depth(rgb01).astype(np.float32)

        meters_img = Image.fromarray(meters, mode="F")
        meters_full = np.asarray(meters_img.resize(pil_image.size, Image.BILINEAR), dtype=np.float32)
        return meters_full

    def _infer_relative(self, pil_image):
        import numpy as np

        result = self.pipe(pil_image)
        return np.array(result["predicted_depth"].squeeze().cpu())

    @modal.asgi_app()
    def web(self):
        import numpy as np
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.middleware.cors import CORSMiddleware
        from PIL import Image

        api = FastAPI(title="depth-demo")

        # Without this the browser blocks every call from GitHub Pages.
        api.add_middleware(
            CORSMiddleware,
            allow_origins=ALLOWED_ORIGINS,
            allow_methods=["POST", "OPTIONS"],
            allow_headers=["Content-Type", "X-Demo-Token"],
        )

        @api.get("/health")
        def health():
            return {"ok": True, "device": self.device, "metric": self.metric}

        @api.post("/predict")
        async def predict(request: Request):
            if SHARED_TOKEN and request.headers.get("X-Demo-Token") != SHARED_TOKEN:
                raise HTTPException(status_code=401, detail="Bad or missing token.")

            payload = await request.json()
            b64 = payload.get("image")
            if not b64:
                raise HTTPException(status_code=400, detail="No image in request body.")

            try:
                raw = base64.b64decode(b64)
            except Exception:
                raise HTTPException(status_code=400, detail="Image was not valid base64.")

            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image too large.")

            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                raise HTTPException(status_code=400, detail="Could not decode that image.")

            if pil.width * pil.height > MAX_PIXELS:
                raise HTTPException(status_code=413, detail="Too many pixels.")

            t0 = time.perf_counter()
            depth_min_m = depth_max_m = None

            if self.metric:
                depth_m = self._infer_metric(pil)
                elapsed = (time.perf_counter() - t0) * 1000
                depth_min_m, depth_max_m = float(depth_m.min()), float(depth_m.max())
                # Nearer must be brighter for the client's colour ramp; metric
                # depth has nearer = smaller, so invert before normalising.
                span = depth_max_m - depth_min_m
                norm = (
                    np.zeros_like(depth_m)
                    if span < 1e-8
                    else (depth_max_m - depth_m) / span
                )
            else:
                depth = self._infer_relative(pil)
                elapsed = (time.perf_counter() - t0) * 1000
                depth = np.asarray(depth, dtype=np.float32)
                lo, hi = float(depth.min()), float(depth.max())
                span = hi - lo
                norm = np.zeros_like(depth) if span < 1e-8 else (depth - lo) / span

            gray = (norm * 255).astype(np.uint8)
            out = Image.fromarray(gray, mode="L").resize(pil.size, Image.BILINEAR)
            buf = io.BytesIO()
            out.save(buf, format="PNG", optimize=True)

            return {
                "depth": base64.b64encode(buf.getvalue()).decode(),
                "width": pil.width,
                "height": pil.height,
                "metric": self.metric,
                "depth_min_m": depth_min_m,
                "depth_max_m": depth_max_m,
                "inference_ms": elapsed,
                "device": self.device,
            }

        return api
