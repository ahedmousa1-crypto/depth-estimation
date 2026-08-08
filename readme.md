# Monocular depth estimation — live demo

Upload or shoot a photo, get a predicted depth map back. A static page on GitHub
Pages calls a GPU container on Modal. No accounts, no sign-up, no database.

```
browser  ──POST base64 jpeg──▶  Modal GPU container
   ▲                                   │
   └──── base64 grayscale png ─────────┘
   applies the colour ramp locally
```

The server returns a normalised grayscale map. The browser applies the colour ramp,
so switching between Inferno, Turbo and Grey costs nothing and never re-runs the model.

---

## 1. Make train.py importable

`import train` runs the whole file. If the training call sits at module level,
importing it on Modal starts a training run on the GPU and the request never returns.

Guard it:

```python
if __name__ == "__main__":
    train()          # or whatever your last lines are
```

Nothing above that line changes, and `python train.py` still trains exactly as before.
Your model class, transforms and helpers are now importable.

## 2. Upload your checkpoint

```bash
modal volume create depth-weights
modal volume put depth-weights ./checkpoints/best.pt /best.pt
```

It mounts at `/weights/best.pt` inside the container.

## 3. Wire in your model

Open `modal_app.py`. Two blocks are marked `YOUR CODE GOES HERE`.

**In `load()`** — build the model and load the checkpoint:

```python
from train import DepthNet

self.model = DepthNet()
ckpt = torch.load(f"{WEIGHTS_DIR}/best.pt", map_location=self.device)
self.model.load_state_dict(ckpt.get("model_state_dict", ckpt))
self.model.to(self.device).eval()
```

Training checkpoints usually hold the optimiser state and epoch number alongside the
weights, so print `ckpt.keys()` once to find the right key. If you trained with
`nn.DataParallel`, strip the `module.` prefix from every key.

**In `_infer()`** — use the same transform your validation loop uses, not the training
one, and no augmentation:

```python
x = val_transform(pil_image).unsqueeze(0).to(self.device)
with torch.no_grad():
    out = self.model(x)
return out.squeeze().cpu().numpy()
```

Return any 2D array where **larger means nearer**. Range does not matter, it gets
normalised. If yours is inverted, return `-array`.

Delete the two fallback blocks once yours works. Until then the demo runs on
Depth Anything V2 Small, so you can prove the plumbing before touching anything.

List your training dependencies in `pip_install` — whatever `train.py` imports at the
top, the container needs too.

## 4. Deploy the backend

```bash
pip install modal
modal setup
modal serve modal_app.py     # temporary URL, live-reloads, use this while testing
modal deploy modal_app.py    # permanent URL
```

Modal prints a URL. Check it:

```bash
curl https://YOUR-WORKSPACE--depth-demo-web.modal.run/health
```

## 5. Point the front end at it

In `app.js`, set `CONFIG.ENDPOINT` to that URL with `/predict` on the end.

In `modal_app.py`, add your Pages URL to `ALLOWED_ORIGINS`. **Skipping this is the
single most common failure.** The endpoint answers `curl` fine and the browser
refuses it, because the two are on different origins and the server has not said the
page may call it.

## 6. Publish

Push, then Settings → Pages → Deploy from a branch → `main` → `/ (root)`.
Live in about a minute at `https://YOUR-USERNAME.github.io/REPO-NAME/`.

To test locally first: `python3 -m http.server 8000`, then open
`http://localhost:8000`. That origin is already allowed.

---

## Bundled examples

`Load an example` reads `examples/manifest.json` and needs no GPU call. Worth doing
before you show anyone — if the container is asleep or out of credits, there is still
something on screen.

```json
[
  { "photo": "corridor.jpg", "depth": "corridor_depth.png", "label": "Corridor" }
]
```

Put both files in `examples/`. The depth file is a grayscale PNG, white nearest,
the same one the endpoint returns. Easiest way to make them: run the demo, hit
`Save depth map` with the **Grey** ramp selected, and drop both files in.

## Cost and abuse

The endpoint is public. Anyone with the URL can spend your credits.

- `MAX_UPLOAD_BYTES` and `MAX_PIXELS` in `modal_app.py` reject oversized inputs.
- `max_containers=2` caps how many GPUs can run at once.
- `SHARED_TOKEN` turns away crawlers. It sits in `app.js`, so anyone reading devtools
  can see it. It is a speed bump, not authentication.
- Set a spending limit in your Modal dashboard. Do this before you share the link.

## Cold starts

`scaledown_window=300` keeps a container alive five minutes after the last request.
Past that it sleeps and the next call takes roughly ten to thirty seconds. The status
line tells the user this is happening rather than leaving them staring at nothing.

Before a demo, set `min_containers=1` in the `@app.cls` decorator and redeploy. Put it
back afterwards, since a warm GPU bills continuously.

Modal renames these arguments occasionally. If deploy rejects one, check their current
docs — `scaledown_window` was `container_idle_timeout` and `min_containers` was
`keep_warm` in older versions.

## What the output is

Relative inverse depth, normalised per image. Bright is nearer. There is no metric
scale, and two images are not comparable to each other. Metric depth in real units
needs a metric-trained model, which is a different checkpoint.

## Files

| | |
|---|---|
| `index.html` | markup |
| `style.css` | all styling |
| `app.js` | capture, upload, colour ramps, wipe compare |
| `modal_app.py` | GPU backend, the only file that touches your model |
| `examples/` | precomputed pairs for the offline fallback |