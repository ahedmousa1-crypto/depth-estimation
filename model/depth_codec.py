"""
Bijective metric-depth <-> RGB codec, from "Image Generators are Generalist
Vision Learners" (Vision Banana, Gabeur et al. 2026), Section 3.2.

Two composed bijections:
  1. Power transform (Barron 2025, arXiv:2502.10647), shape lambda < -1:
       f(d, lam, c) = 1 - (1 - d/(lam*c))**(lam+1)
     maps metric depth d in [0, inf) to a curved distance t in [0, 1),
     giving near depth more resolution than far depth.
  2. A piecewise-linear path along the 7 edges of the RGB unit cube,
     black -> white, visiting all 8 corners once (the first iteration of a
     3D Hilbert curve). t in [0, 1) selects a position along that path.

Encode: depth (meters) -> t -> RGB in [0, 1].
Decode: RGB -> nearest point on the path -> t -> depth (meters), by
projecting onto the nearest of the 7 segments and inverting the transform.
"""

import numpy as np

LAM = -3.0
C = 10.0 / 3.0

# Hamiltonian path across the 8 cube corners, black -> white, each step
# flipping exactly one bit (one RGB channel), 7 edges total.
CUBE_PATH = np.array(
    [
        [0, 0, 0],  # black
        [0, 0, 1],  # blue
        [0, 1, 1],  # cyan
        [0, 1, 0],  # green
        [1, 1, 0],  # yellow
        [1, 0, 0],  # red
        [1, 0, 1],  # magenta
        [1, 1, 1],  # white
    ],
    dtype=np.float64,
)
N_EDGES = len(CUBE_PATH) - 1


def _power_forward(d, lam=LAM, c=C):
    lam_c = lam * c
    return 1.0 - (1.0 - d / lam_c) ** (lam + 1.0)


def _power_inverse(t, lam=LAM, c=C):
    lam_c = lam * c
    return lam_c * (1.0 - (1.0 - t) ** (1.0 / (lam + 1.0)))


def _t_to_rgb(t):
    """t: (...,) in [0, 1) -> rgb: (..., 3) in [0, 1]."""
    t = np.clip(t, 0.0, 1.0 - 1e-9)
    pos = t * N_EDGES
    seg = np.floor(pos).astype(np.int64)
    seg = np.clip(seg, 0, N_EDGES - 1)
    local = (pos - seg)[..., None]
    a = CUBE_PATH[seg]
    b = CUBE_PATH[seg + 1]
    return a * (1.0 - local) + b * local


def _rgb_to_t(rgb):
    """rgb: (..., 3) in [0, 1] -> t: (...,) in [0, 1], by nearest-segment
    projection onto the 7-edge polyline."""
    rgb = rgb[..., None, :]  # (..., 1, 3)
    a = CUBE_PATH[:-1]  # (7, 3)
    b = CUBE_PATH[1:]  # (7, 3)
    ab = b - a  # (7, 3)
    denom = np.sum(ab * ab, axis=-1)  # (7,)

    u = np.sum((rgb - a) * ab, axis=-1) / denom  # (..., 7)
    u = np.clip(u, 0.0, 1.0)

    proj = a + u[..., None] * ab  # (..., 7, 3)
    dist2 = np.sum((rgb - proj) ** 2, axis=-1)  # (..., 7)

    best = np.argmin(dist2, axis=-1)  # (...,)
    best_u = np.take_along_axis(u, best[..., None], axis=-1)[..., 0]
    return (best.astype(np.float64) + best_u) / N_EDGES


def depth_to_rgb(depth_m, lam=LAM, c=C):
    """depth_m: array of metric depth in meters, d >= 0. Returns RGB in
    [0, 1] with the same leading shape plus a trailing channel dim of 3."""
    depth_m = np.asarray(depth_m, dtype=np.float64)
    t = _power_forward(np.maximum(depth_m, 0.0), lam, c)
    return _t_to_rgb(t)


def rgb_to_depth(rgb, lam=LAM, c=C):
    """rgb: array in [0, 1] with a trailing channel dim of 3. Returns
    metric depth in meters, same leading shape."""
    rgb = np.asarray(rgb, dtype=np.float64)
    t = _rgb_to_t(rgb)
    return _power_inverse(t, lam, c)


if __name__ == "__main__":
    depths = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 100.0, 1000.0])
    rgb = depth_to_rgb(depths)
    recovered = rgb_to_depth(rgb)

    print(f"{'depth (m)':>10} {'RGB':>22} {'recovered (m)':>14} {'abs err':>10}")
    for d, c_, r in zip(depths, rgb, recovered):
        print(f"{d:10.3f} {np.array2string(c_, precision=3):>22} {r:14.4f} {abs(d - r):10.2e}")

    depth_map = np.random.exponential(scale=3.0, size=(64, 64)).astype(np.float64)
    rgb_map = depth_to_rgb(depth_map)
    round_trip = rgb_to_depth(rgb_map)
    max_err = np.max(np.abs(depth_map - round_trip))
    rel_err = np.max(np.abs(depth_map - round_trip) / np.maximum(depth_map, 1e-6))
    print(f"\n64x64 random depth map round-trip: max_abs_err={max_err:.3e} max_rel_err={rel_err:.3e}")
    assert max_err < 1e-4, "codec is not bijective within tolerance"
    print("OK: codec round-trips within tolerance.")