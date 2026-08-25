"""
Builds a ~3000-sample DIODE subset and pushes it to HuggingFace, entirely on
Modal so Colab's disk quota never enters the picture.

DIODE is distributed as two monolithic tarballs (no per-scene random access):
    val.tar.gz    2.6 GB   -- 325 indoor + 446 outdoor frames (771 total)
    train.tar.gz  81 GB    -- 8574 indoor + 16884 outdoor frames
Source: https://diode-dataset.org (S3-hosted, http://diode-dataset.s3.amazonaws.com/)

Both get downloaded once into a persistent Modal Volume (resumable via HTTP
Range requests -- a dropped connection resumes instead of restarting), then
extracted, indexed, sampled, converted, and pushed.

Depth conversion matches train.py's assumption exactly:
    CFG["depth_png_scale"] = 100.0  ->  16-bit PNG, meters = pixel / 100
    (covers 0-655.35 m, comfortably above DIODE's ~350 m sensor range)

Pipeline (run in this order):
    modal run dataset_prep.py::download --which val
    modal run dataset_prep.py::extract  --which val
    modal run dataset_prep.py::inspect_raw          # ALWAYS run before train
    modal run dataset_prep.py::download --which train    # slow, ~81GB
    modal run dataset_prep.py::extract  --which train     # slow
    modal run dataset_prep.py::build_and_push               # dry_run defaults True: sanity check
    modal run dataset_prep.py::build_and_push --no-dry-run  # real push
"""

import modal

app = modal.App("diode-dataset-prep")

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "requests",
    "datasets==2.21.0",
    "huggingface_hub==0.25.2",
    "numpy==1.26.4",
    "Pillow==10.4.0",
    "tqdm",
)

RAW_DIR = "/data"
raw_vol = modal.Volume.from_name("diode-raw", create_if_missing=True)
SECRETS = [modal.Secret.from_name("huggingface-secret2")]

DIODE_URLS = {
    "val": "http://diode-dataset.s3.amazonaws.com/val.tar.gz",
    "train": "http://diode-dataset.s3.amazonaws.com/train.tar.gz",
}

CFG = {
    "target_n": 3000,
    "indoor_frac": 0.33,       # DIODE's own train ratio is ~8574:16884 (indoor:outdoor)
    "max_invalid_frac": 0.15,  # skip frames where >15% of pixels are masked invalid
    "depth_cm_scale": 100.0,   # must match train.py CFG["depth_png_scale"]
    "seed": 42,
    "hub_repo": "ahd03/diode-depth-3k",
}


# =========================================================================
# 1. Download -- resumable, so a dropped connection never means starting over
# =========================================================================
@app.function(image=image, volumes={RAW_DIR: raw_vol}, timeout=6 * 60 * 60)
def download(which: str = "val"):
    import os
    import requests

    url = DIODE_URLS[which]
    dest = f"{RAW_DIR}/{which}.tar.gz"
    tmp = dest + ".part"

    if os.path.exists(dest):
        print(f"{dest} already present ({os.path.getsize(dest)/1e9:.2f} GB), skipping")
        return

    headers, mode, downloaded = {}, "wb", 0
    if os.path.exists(tmp):
        downloaded = os.path.getsize(tmp)
        headers["Range"] = f"bytes={downloaded}-"
        mode = "ab"
        print(f"resuming {which} download at byte {downloaded}")

    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, mode) as f:
            last_log = downloaded
            for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded - last_log > 500 * 1024 * 1024:
                    print(f"{which}: {downloaded/1e9:.2f} GB", flush=True)
                    last_log = downloaded
                    raw_vol.commit()  # checkpoint the volume periodically

    os.rename(tmp, dest)
    raw_vol.commit()
    print(f"{which} download complete -> {dest} ({downloaded/1e9:.2f} GB)")


# =========================================================================
# 2. Extract
# =========================================================================
@app.function(image=image, volumes={RAW_DIR: raw_vol}, timeout=3 * 60 * 60)
def extract(which: str = "val"):
    import os
    import tarfile

    src = f"{RAW_DIR}/{which}.tar.gz"
    dest_dir = f"{RAW_DIR}/{which}_extracted"
    if os.path.exists(dest_dir) and os.listdir(dest_dir):
        print(f"{dest_dir} already populated, skipping")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"extracting {src} -> {dest_dir} ...")
    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(dest_dir)
    raw_vol.commit()
    print(f"done extracting {which}")


# =========================================================================
# 3. Inspect -- ALWAYS run this before touching the train tarball. Confirms
#    the real file-naming convention instead of trusting an assumption.
# =========================================================================
@app.function(image=image, volumes={RAW_DIR: raw_vol}, timeout=600)
def inspect_raw(which: str = "val"):
    import os

    root = f"{RAW_DIR}/{which}_extracted"
    if not os.path.exists(root):
        print(f"!! {root} does not exist -- run download+extract for '{which}' first")
        return

    print(f"top-level entries under {root}:")
    for entry in sorted(os.listdir(root))[:20]:
        print(f"  {entry}")

    sample_files = []
    for dirpath, _, filenames in os.walk(root):
        if filenames:
            sample_files = [(dirpath, sorted(filenames)[:6])]
            break
    print(f"\nfirst populated directory: {sample_files[0][0] if sample_files else 'NONE FOUND'}")
    for fn in (sample_files[0][1] if sample_files else []):
        print(f"  {fn}")

    n_rgb = n_depth = n_mask = 0
    for _, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith("_depth_mask.npy"):
                n_mask += 1
            elif fn.endswith("_depth.npy"):
                n_depth += 1
            elif fn.endswith(".png"):
                n_rgb += 1
    print(f"\ncounts matching assumed convention: rgb={n_rgb} depth={n_depth} mask={n_mask}")
    print(
        "If any of these are 0, the naming convention differs from what "
        "_index_split() in this file assumes -- fix the suffixes there "
        "before running build_and_push on the full train split."
    )


# =========================================================================
# Shared indexing / conversion logic
# =========================================================================
def _index_split(root, split_name):
    import os

    frames = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith("_depth.npy"):
                continue
            base = os.path.join(dirpath, fn[: -len("_depth.npy")])
            rgb_path = base + ".png"
            mask_path = base + "_depth_mask.npy"
            if os.path.exists(rgb_path) and os.path.exists(mask_path):
                indoor = "indoors" in dirpath.lower()
                frames.append(
                    {
                        "rgb": rgb_path,
                        "depth": base + "_depth.npy",
                        "mask": mask_path,
                        "indoor": indoor,
                        "split": split_name,
                    }
                )
    return frames


def _convert_frame(frame, depth_cm_scale, max_invalid_frac):
    """Returns (rgb_png_bytes, depth_png_bytes, invalid_frac), or None if the
    frame exceeds max_invalid_frac, or if its files are corrupt/truncated
    (e.g. an interrupted tarball extraction) -- caller counts and skips."""
    import io

    import numpy as np
    from PIL import Image

    try:
        depth_m = np.load(frame["depth"]).astype(np.float32)
        mask = np.load(frame["mask"]).astype(bool)
    except (ValueError, EOFError, OSError) as e:
        print(f"!! corrupt/truncated .npy, skipping {frame['depth']}: {e}")
        return None

    if depth_m.ndim == 3:
        depth_m = depth_m[..., 0]
    if mask.ndim == 3:
        mask = mask[..., 0]

    if depth_m.shape != mask.shape:
        print(f"!! depth/mask shape mismatch, skipping {frame['depth']}: {depth_m.shape} vs {mask.shape}")
        return None

    invalid_frac = 1.0 - float(mask.mean())
    if invalid_frac > max_invalid_frac:
        return None

    depth_m = np.where(mask, depth_m, 0.0)  # invalid -> 0m, matches codec's black sentinel
    depth_cm = np.clip(np.round(depth_m * depth_cm_scale), 0, 65535).astype(np.uint16)

    try:
        depth_img = Image.fromarray(depth_cm, mode="I;16")
        depth_buf = io.BytesIO()
        depth_img.save(depth_buf, format="PNG")

        rgb_img = Image.open(frame["rgb"]).convert("RGB")
        rgb_buf = io.BytesIO()
        rgb_img.save(rgb_buf, format="PNG")
    except Exception as e:
        print(f"!! image decode/encode failed, skipping {frame['rgb']}: {e}")
        return None

    return rgb_buf.getvalue(), depth_buf.getvalue(), invalid_frac


def _verify_16bit_roundtrip(depth_cm_array):
    """HF's Image() feature round-trips through PIL -- 16-bit PNGs have a
    known history of being silently mangled by some encoders. Verify before
    trusting a full push."""
    import io

    import numpy as np
    from datasets import Dataset, Features
    from datasets import Image as HFImage
    from PIL import Image

    img = Image.fromarray(depth_cm_array, mode="I;16")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    ds = Dataset.from_dict({"depth": [buf.getvalue()]}, features=Features({"depth": HFImage()}))
    loaded = np.asarray(ds[0]["depth"])
    if loaded.ndim == 3:
        loaded = loaded[..., 0]
    return bool(np.array_equal(loaded.astype(np.uint16), depth_cm_array))


# =========================================================================
# 4. Build the sample + push to HuggingFace
# =========================================================================
@app.function(image=image, volumes={RAW_DIR: raw_vol}, secrets=SECRETS, timeout=4 * 60 * 60)
def build_and_push(dry_run: bool = True):
    import random

    import numpy as np
    from datasets import Dataset, Features, Value
    from datasets import Image as HFImage
    from huggingface_hub import login
    import os

    n_target = 20 if dry_run else CFG["target_n"]

    val_root = f"{RAW_DIR}/val_extracted"
    if not os.path.exists(val_root) or not os.listdir(val_root):
        raise RuntimeError(
            f"{val_root} doesn't exist or is empty -- you need to download "
            "and extract the val split before build_and_push can index it:\n"
            "  modal run dataset_prep.py::download --which val\n"
            "  modal run dataset_prep.py::extract --which val\n"
            "  modal run dataset_prep.py::inspect_raw"
        )

    print("indexing val split...")
    frames = _index_split(val_root, "val")

    train_root = f"{RAW_DIR}/train_extracted"
    if os.path.exists(train_root) and os.listdir(train_root):
        print("indexing train split...")
        frames += _index_split(train_root, "train")
    else:
        print(f"{train_root} not present yet -- indexing val split only")

    print(f"indexed {len(frames)} candidate frames total")

    if not frames:
        raise RuntimeError(
            f"{val_root} exists but 0 frames matched the expected "
            "*_rgb.png / *_depth.npy / *_depth_mask.npy naming -- run "
            "`modal run dataset_prep.py::inspect_raw` and check its output "
            "against _index_split() in this file; the convention likely "
            "doesn't match your tarball."
        )

    indoor = [f for f in frames if f["indoor"]]
    outdoor = [f for f in frames if not f["indoor"]]
    rng = random.Random(CFG["seed"])
    rng.shuffle(indoor)
    rng.shuffle(outdoor)

    n_indoor = int(n_target * CFG["indoor_frac"])
    n_outdoor = n_target - n_indoor
    candidates = indoor[: n_indoor * 3] + outdoor[: n_outdoor * 3]  # oversample, some get filtered
    rng.shuffle(candidates)

    # Self-test the 16-bit PNG round trip once before doing real work.
    probe = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    if not _verify_16bit_roundtrip(probe):
        raise RuntimeError(
            "16-bit depth PNG did not round-trip through datasets.Image() "
            "unchanged -- HF is quantizing/mangling it. Stop and fix the "
            "encoding before converting the full sample."
        )
    print("16-bit PNG round-trip check: OK")

    rgb_bytes, depth_bytes, indoor_flags, meta = [], [], [], []
    n_indoor_kept = n_outdoor_kept = 0
    skipped_invalid = 0

    for frame in candidates:
        if len(rgb_bytes) >= n_target:
            break
        want_indoor = n_indoor_kept < n_indoor
        want_outdoor = n_outdoor_kept < n_outdoor
        if frame["indoor"] and not want_indoor:
            continue
        if not frame["indoor"] and not want_outdoor:
            continue

        result = _convert_frame(frame, CFG["depth_cm_scale"], CFG["max_invalid_frac"])
        if result is None:
            skipped_invalid += 1
            continue

        rgb_b, depth_b, invalid_frac = result
        rgb_bytes.append(rgb_b)
        depth_bytes.append(depth_b)
        indoor_flags.append(frame["indoor"])
        meta.append({"split_origin": frame["split"], "invalid_frac": invalid_frac})
        if frame["indoor"]:
            n_indoor_kept += 1
        else:
            n_outdoor_kept += 1

    print(
        f"kept {len(rgb_bytes)} frames "
        f"(indoor={n_indoor_kept} outdoor={n_outdoor_kept}), "
        f"skipped {skipped_invalid} for >{CFG['max_invalid_frac']*100:.0f}% invalid pixels"
    )

    ds = Dataset.from_dict(
        {
            "image": rgb_bytes,
            "depth": depth_bytes,
            "indoor": indoor_flags,
            "split_origin": [m["split_origin"] for m in meta],
            "invalid_frac": [m["invalid_frac"] for m in meta],
        },
        features=Features(
            {
                "image": HFImage(),
                "depth": HFImage(),
                "indoor": Value("bool"),
                "split_origin": Value("string"),
                "invalid_frac": Value("float32"),
            }
        ),
    )

    if dry_run:
        print(f"DRY RUN -- built {len(ds)} examples, not pushing. Inspect ds[0] before the real run.")
        example = ds[0]
        print(f"image: mode={example['image'].mode} size={example['image'].size}")
        print(f"depth: mode={example['depth'].mode} size={example['depth'].size}")
        return

    login(token=os.environ["HF_TOKEN"])
    ds.push_to_hub(CFG["hub_repo"])
    print(f"pushed {len(ds)} examples -> {CFG['hub_repo']}")


@app.local_entrypoint()
def main():
    print("Run in order:")
    print("  modal run dataset_prep.py::download --which val")
    print("  modal run dataset_prep.py::extract --which val")
    print("  modal run dataset_prep.py::inspect_raw")
    print("  modal run dataset_prep.py::download --which train")
    print("  modal run dataset_prep.py::extract --which train")
    print("  modal run dataset_prep.py::build_and_push               # dry_run defaults True: sanity check")
    print("  modal run dataset_prep.py::build_and_push --no-dry-run  # real push")
