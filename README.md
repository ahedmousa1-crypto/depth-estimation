# Monocular Depth Estimation

Metric depth from a single photograph. A Stable Diffusion 1.5 UNet is fine-tuned on
[DIODE](https://diode-dataset.org) to denoise a depth map, which is encoded as an
ordinary RGB image and decoded back to distance in metres. A static web demo lets you
upload or shoot a photo and compare it against the predicted depth map.

- **Live demo:** `https://ahedmousa1-crypto.github.io/depth-estimation/` (once GitHub Pages is enabled, see below)
- **Model:** [`ahd03/diode-depth-model-v2`](https://huggingface.co/ahd03/diode-depth-model-v2)
- **Dataset:** [`ahd03/diode-depth-3k`](https://huggingface.co/datasets/ahd03/diode-depth-3k)

## How it works

1. **Encode the photo.** The frozen SD 1.5 VAE turns the photo into a latent. It's
   concatenated to the UNet input (`conv_in` widened from 4 to 8 channels), so the
   network is always conditioned on the scene.
2. **Denoise a depth image.** The UNet, the only part that is trained, runs DDIM
   sampling from noise. Several seeds are averaged into one ensemble prediction.
3. **Decode back to metres.** The output is an RGB image, so the VAE decodes it. The
   bijective codec in `model/depth_codec.py` then turns each colour back into exact
   metric depth, with no per-image scale or shift alignment.

The codec composes two invertible steps: a power transform (Barron 2025, λ = −3) that
warps depth in `[0, ∞)` into `[0, 1)` and gives near ranges more resolution, then a
path along the seven edges of the RGB cube from black to white. The training recipe
follows Marigold (Ke et al., CVPR 2024). The depth-as-RGB representation follows
*Image Generators are Generalist Vision Learners* (Gabeur et al., 2026).

## Repository layout

```
docs/                    Static website, served by GitHub Pages
  index.html
  style.css
  app.js                 Upload, wipe viewer, colour ramps; calls the Modal endpoint
model/                   Everything that runs on Modal
  depth_codec.py         Bijective metric depth <-> RGB codec (shared by all scripts)
  dataset_prep.py        Downloads DIODE, samples 3k frames, pushes to the HF Hub
  train.py               Fine-tune, evaluate, push the UNet
  modal_app.py           GPU inference API behind the website
  legacy/                Early smoke tests from the Marigold-style grayscale version
```

## Running it

All model commands run from the `model/` folder, because the scripts import
`depth_codec.py` from beside them.

### Prerequisites

```bash
pip install modal
modal token new
```

Create a Modal secret named `huggingface-secret2` with a key `HF_TOKEN` (a Hugging Face
write token), either in the Modal dashboard or with:

```bash
modal secret create huggingface-secret2 HF_TOKEN=hf_...
```

### 1. Build the dataset

```bash
cd model
modal run dataset_prep.py::download --which val
modal run dataset_prep.py::extract  --which val
modal run dataset_prep.py::inspect_raw            # check the file naming first
modal run dataset_prep.py::download --which train # ~81 GB, slow
modal run dataset_prep.py::extract  --which train
modal run dataset_prep.py::build_and_push         # dry run, 20 samples
modal run dataset_prep.py::build_and_push --no-dry-run
```

### 2. Train, evaluate, publish

```bash
modal run train.py::inspect    # verify columns and depth ranges before training
modal run train.py::train      # L40S, resumes from /output/state.pt if interrupted
modal run train.py::evaluate   # AbsRel, δ1, RMSE in metres, no alignment
modal run train.py::push       # uploads unet_best to the HF Hub
```

Hyperparameters are in the `CFG` dict at the top of `train.py`.

### 3. Serve the API

```bash
modal serve modal_app.py    # temporary dev URL, hot reloads
modal deploy modal_app.py   # permanent URL
```

The API loads the fine-tuned model from the Hub. If that fails, it falls back to Depth
Anything V2 Small (relative depth only), so the demo works before training finishes.

| Route           | Body                         | Returns |
|-----------------|------------------------------|---------|
| `GET /health`   | –                            | `{ ok, device, metric }` |
| `POST /predict` | `{ "image": "<base64 JPEG>" }` | 8-bit depth PNG (base64, near = bright), `depth_min_m`, `depth_max_m`, `inference_ms`, `device` |

### 4. Run the website

Set `CONFIG.ENDPOINT` in `docs/app.js` to the URL that `modal deploy` prints, with
`/predict` on the end. Then preview it locally:

```bash
python -m http.server 8000 -d docs
```

Open http://localhost:8000. `localhost:8000` is already in `ALLOWED_ORIGINS` in
`modal_app.py`, so CORS works.

To publish it, go to **Settings → Pages** on GitHub and choose **Deploy from a branch**,
branch `main`, folder `/docs`.

### Optional: bundled examples

The **Load an example** button only appears if `docs/examples/manifest.json` exists. It
shows precomputed results without calling the GPU:

```json
[{ "photo": "kitchen.jpg", "depth": "kitchen_depth.png", "label": "Kitchen" }]
```

The depth file is a grayscale PNG with near = bright, the same as the API returns.

## Limits

Mirrors and glass, large textureless walls, heavy motion blur, and framing far from
DIODE's indoor/outdoor scenes (aerial, macro) all degrade the prediction. Thin
structures like hair and fences tend to smear.
