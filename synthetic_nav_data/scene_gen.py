"""
Procedural scene + camera-path generator for synthetic navigation data.

Everything here is exact and known by construction -- object positions, camera
path, discrete actions, and the 2D trace are all derived from the same numbers,
so there's nothing to "verify" after the fact (contrast with the NavSpace
mismatch we found: instruction text describing a path that didn't match the
actual reference_path).

Coordinate system: right-handed, x=right, z=forward-is-negative (three.js/
Habitat convention), y=up. Angles in degrees, clockwise-positive for "turn right".
"""
import random
import math

SHAPES = ["cone", "cube", "sphere", "cylinder"]
COLORS = {
    "red": 0xe53935, "blue": 0x1e88e5, "green": 0x43a047,
    "purple": 0x8e24aa, "orange": 0xfb8c00, "yellow": 0xfdd835,
    "pink": 0xec407a, "teal": 0x00897b,
}

STEP_FORWARD_M = 1.5   # meters per discrete 'forward' action
STEP_TURN_DEG = 30      # degrees per discrete 'left'/'right' action


def _rot_y(vec, deg):
    """Rotate an (x, z) vector by deg around the vertical axis."""
    rad = math.radians(deg)
    x, z = vec
    return (x * math.cos(rad) + z * math.sin(rad), -x * math.sin(rad) + z * math.cos(rad))


def generate_scene(seed: int, n_objects: int = 5, n_actions: int = 6):
    rng = random.Random(seed)

    # 1. Place objects randomly in a walkable area in front of the start pose.
    objects = []
    for i in range(n_objects):
        shape = rng.choice(SHAPES)
        color_name = rng.choice(list(COLORS.keys()))
        x = rng.uniform(-4, 4)
        z = rng.uniform(-8, -2)  # in front of camera (negative z = forward)
        objects.append({
            "id": f"obj_{i}",
            "shape": shape,
            "color_name": color_name,
            "color_hex": COLORS[color_name],
            "position": [x, 0.5, z],
        })

    # 2. Start pose: origin, facing -z (three.js/Habitat "forward").
    start_pos = [0.0, 1.6, 0.0]
    start_heading = 0.0  # degrees, 0 = facing -z

    # 3. Random discrete action sequence -> exact waypoint path.
    actions = []
    heading = start_heading
    pos = list(start_pos)
    waypoints = [list(pos)]
    headings = [heading]
    for _ in range(n_actions):
        choice = rng.choices(["forward", "left", "right"], weights=[0.55, 0.225, 0.225])[0]
        if choice == "forward":
            dx, dz = _rot_y((0, -1), heading)
            pos = [pos[0] + dx * STEP_FORWARD_M, pos[1], pos[2] + dz * STEP_FORWARD_M]
        elif choice == "left":
            heading -= STEP_TURN_DEG
        else:
            heading += STEP_TURN_DEG
        actions.append(choice)
        waypoints.append(list(pos))
        headings.append(heading)
    actions.append("stop")

    # 4. Pick the goal object = the one closest to the final waypoint.
    end_pos = waypoints[-1]
    def dist(o):
        return math.hypot(o["position"][0] - end_pos[0], o["position"][2] - end_pos[2])
    goal = min(objects, key=dist)

    # 5. Geometric facts relative to the START pose (for instruction generation).
    facts = []
    for o in objects:
        rel_x = o["position"][0] - start_pos[0]
        rel_z = o["position"][2] - start_pos[2]
        # rotate into start-heading-relative frame
        lx, lz = _rot_y((rel_x, rel_z), -start_heading)
        distance_m = math.hypot(lx, lz)
        side = "left" if lx < -0.15 else ("right" if lx > 0.15 else "ahead")
        facts.append({
            "id": o["id"], "shape": o["shape"], "color": o["color_name"],
            "distance_m": round(distance_m, 2), "side_from_start": side,
            "is_goal": o["id"] == goal["id"],
        })
    # ordinal position left-to-right (for "the Nth object from the left" phrasing)
    ordered = sorted(facts, key=lambda f: f["distance_m"] * (-1 if f["side_from_start"] == "left" else 1))
    left_to_right = sorted(facts, key=lambda f: next(o["position"][0] for o in objects if o["id"] == f["id"]))
    for i, f in enumerate(left_to_right):
        f["ordinal_left_to_right"] = i + 1

    # Ground-level copies of the path, for plotting the 2D trace (a point at
    # exactly eye-height always projects onto the horizon row regardless of
    # distance, which is why the trace must be plotted at foot-height, not
    # camera-height).
    waypoints_ground = [[w[0], 0.02, w[2]] for w in waypoints]

    return {
        "seed": seed,
        "objects": objects,
        "start_position": start_pos,
        "start_heading_deg": start_heading,
        "actions": actions,
        "waypoints": waypoints,
        "waypoints_ground": waypoints_ground,
        "headings": headings,
        "goal_id": goal["id"],
        "facts": facts,
        "step_forward_m": STEP_FORWARD_M,
        "step_turn_deg": STEP_TURN_DEG,
    }


if __name__ == "__main__":
    import json
    s = generate_scene(seed=42)
    print(json.dumps(s, indent=2))
