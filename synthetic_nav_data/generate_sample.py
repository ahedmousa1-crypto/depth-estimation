"""End-to-end: one seed -> one full training sample, in both target formats.

- NavSpace-style: instruction + discrete action_sequence + reference_path (3D)
- NaviTrace-style: one image + 2D pixel trace + embodiment tag
Both come from the exact same underlying scene, so they're guaranteed consistent
with each other (same episode, two label views) -- useful if you end up training
one model against both benchmark formats.
"""
import json
import os

from scene_gen import generate_scene
from render import render_scene
from instruction_gen import generate_instruction, CATEGORIES


def build_sample(seed: int, category: str, out_root: str, use_claude: bool = False):
    scene = generate_scene(seed=seed)
    out_dir = os.path.join(out_root, f"sample_{seed:04d}_{category}")
    render = render_scene(scene, out_dir, walkthrough_every=1)
    instruction = generate_instruction(scene, category, use_claude=use_claude)

    goal = next(f for f in scene["facts"] if f["is_goal"])

    sample = {
        "sample_id": f"synthnav_{seed:04d}_{category}",
        "category": category,
        "instruction_text": instruction,
        "start_position": scene["start_position"],
        "start_heading_deg": scene["start_heading_deg"],
        "goal": {"id": goal["id"], "shape": goal["shape"], "color": goal["color"]},
        "action_sequence": scene["actions"],
        "reference_path_3d": scene["waypoints"],
        "trace_pixels_2d": render["trace_pixels"],
        "start_frame_image": os.path.relpath(render["start_frame"], out_dir),
        "walkthrough_frames": [os.path.relpath(p, out_dir) for p in render["walkthrough_frames"]],
        "objects": scene["facts"],
    }
    with open(os.path.join(out_dir, "sample.json"), "w") as f:
        json.dump(sample, f, indent=2)
    return sample, out_dir


if __name__ == "__main__":
    import sys
    out_root = sys.argv[1] if len(sys.argv) > 1 else "generated_samples"
    seeds_categories = [(101, "precise_movement"), (202, "spatial_relationship"),
                         (303, "viewpoint_shifting"), (404, "spatial_relationship")]
    for seed, cat in seeds_categories:
        sample, out_dir = build_sample(seed, cat, out_root)
        print(f"[{cat}] seed={seed} -> {sample['instruction_text']!r}  (saved to {out_dir})")
