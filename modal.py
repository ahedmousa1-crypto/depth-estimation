"""
Modal backend for the depth estimation demo.

This file is a wrapper. It does not replace your model code and it does not ask
you to rewrite it. There is exactly one place below where you point it at your
own inference function, marked YOUR CODE GOES HERE.

Deploy:      modal deploy modal_app.py
Test first:  modal serve modal_app.py
"""

import base64
import io
import os
import time

import modal

app = modal.App("depth-demo")

# Your trained weights live here, uploaded once, mounted into the container.
# Create and fill it before deploying:
#     modal volume create depth-weights
#     modal volume put depth-weights ./checkpoints/best.pt /best.pt
weights = modal.Volume.from_name("depth-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# ---------------------------------------------------------------------------
# Container image
#
# Add whatever your model needs. If you already have a requirements.txt next to
# this file, swap the pip_install call for:
#     .pip_install_from_requirements("requirements.txt")
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "pillow",
        "numpy",
        "fastapi[standard]",
    )
    # Ships your training script into the container so its model class and
    # preprocessing can be imported. Importing it runs every top-level line,
    # so the training call must sit under `if __name__ == "__main__":`.
    .add_local_python_source("train", copy=True)
)

# ---------------------------------------------------------------------------
# Settings you will want to look at
# ---------------------------------------------------------------------------

# Origins allowed to call this endpoint. Add your GitHub Pages URL.
ALLOWED_ORIGINS = [
    "https://YOUR-USERNAME.github.io",
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
    gpu="T4",              # bump to "A10G" or "L4" if your model needs it
    volumes={WEIGHTS_DIR: weights},
    scaledown_window=300,  # keep the container alive 5 min after the last call
    max_containers=2,      # ceiling on concurrent GPU containers
)
class DepthService:

    @modal.enter()
    def load(self):
        """Runs once per container, before the first request. Load weights here,
        never inside the request handler."""
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # ===================================================================
        # YOUR CODE GOES HERE
        #
        # Import the class and preprocessing out of your training script and
        # load the checkpoint it produced. Nothing in train.py changes except
        # putting the training call under `if __name__ == "__main__":`, which
        # is what makes it importable without training on import.
        #
        #     from train import DepthNet          # your model class
        #
        #     self.model = DepthNet()
        #     ckpt = torch.load(f"{WEIGHTS_DIR}/best.pt", map_location=self.device)
        #     # Training checkpoints are usually a dict holding more than weights.
        #     # Print ckpt.keys() once to see which key yours uses.
        #     self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        #     self.model.to(self.device).eval()
        #
        # If you saved with nn.DataParallel, every key is prefixed "module.":
        #     sd = {k.replace("module.", ""): v for k, v in sd.items()}
        # ===================================================================

        # --- fallback so the demo runs before you wire yours in -------------
        from transformers import pipeline

        self.pipe = pipeline(
            "depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            device=0 if self.device == "cuda" else -1,
        )
        # -------------------------------------------------------------------

    def _infer(self, pil_image):
        """Take a PIL image, return a 2D float array of predicted depth.
        Larger values must mean nearer. Any range is fine, it gets normalised."""
        import numpy as np

        # ===================================================================
        # YOUR CODE GOES HERE
        #
        # Reuse the exact transform your validation loop uses. If train.py has
        # separate train and val transforms, take the val one, and do not
        # include any augmentation.
        #
        #     import torch
        #     from train import val_transform
        #
        #     x = val_transform(pil_image).unsqueeze(0).to(self.device)
        #     with torch.no_grad():
        #         out = self.model(x)
        #     return out.squeeze().cpu().numpy()
        #
        # If the numbers come out inverted (far looks bright), return -array.
        # ===================================================================

        # --- fallback -------------------------------------------------------
        result = self.pipe(pil_image)
        return np.array(result["predicted_depth"].squeeze().cpu())
        # --------------------------------------------------------------------

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
            return {"ok": True, "device": self.device}

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
            depth = self._infer(pil)
            elapsed = (time.perf_counter() - t0) * 1000

            depth = np.asarray(depth, dtype=np.float32)
            lo, hi = float(depth.min()), float(depth.max())

            # Normalise to 0..255 for transport. The browser applies the colour
            # ramp, so the user can switch ramps without another GPU call.
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
                "raw_min": lo,
                "raw_max": hi,
                "inference_ms": elapsed,
                "device": self.device,
            }

        return api