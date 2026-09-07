"""
Turn a scene's exact geometric facts into a natural-language instruction.

Because the scene is procedurally generated, the facts passed in here are
already ground truth (positions, distances, left/right, the discrete action
sequence) -- the LLM's only job is to phrase them naturally, targeting one of
the weakness categories found in step 2 (mirrors NavSpace's own categories so
eval numbers line up category-for-category). It cannot introduce a fact that
doesn't match the geometry, because it's only ever paraphrasing facts we
already computed -- this sidesteps the verification problem we found in the
real NavSpace data (instruction describing a path that didn't match the
actual reference_path).

Two backends:
  - call_claude(): real generation via the Anthropic API (what the at-scale
    pipeline should use -- needs ANTHROPIC_API_KEY, e.g. as a Modal secret).
  - template_fallback(): deterministic, no API key needed. Used here so the
    pipeline is runnable/testable without credentials; swap for call_claude
    in production for phrasing variety.
"""
import os
import json

CATEGORIES = ["precise_movement", "spatial_relationship", "viewpoint_shifting"]


def _describe_actions(scene):
    """Turn the discrete action sequence into exact human-readable segments."""
    segs = []
    fwd_run = 0
    for a in scene["actions"]:
        if a == "forward":
            fwd_run += 1
        else:
            if fwd_run:
                segs.append(f"go straight for {fwd_run * scene['step_forward_m']:.1f} meters")
                fwd_run = 0
            if a == "left":
                segs.append(f"turn left {scene['step_turn_deg']} degrees")
            elif a == "right":
                segs.append(f"turn right {scene['step_turn_deg']} degrees")
    if fwd_run:
        segs.append(f"go straight for {fwd_run * scene['step_forward_m']:.1f} meters")
    return segs


def template_fallback(scene: dict, category: str) -> str:
    goal = next(f for f in scene["facts"] if f["is_goal"])
    if category == "precise_movement":
        segs = _describe_actions(scene)
        return "From your current position, " + ", then ".join(segs) + f", and stop at the {goal['color']} {goal['shape']}."
    if category == "spatial_relationship":
        others_same_side = [f for f in scene["facts"] if f["side_from_start"] == goal["side_from_start"] and not f["is_goal"]]
        nearer = [f for f in others_same_side if f["distance_m"] < goal["distance_m"]]
        ordinal = len(nearer) + 1
        ordword = {1: "first", 2: "second", 3: "third", 4: "fourth"}.get(ordinal, f"{ordinal}th")
        return f"Go to the {ordword} {goal['shape']} on your {goal['side_from_start']}, about {goal['distance_m']:.1f} meters away."
    if category == "viewpoint_shifting":
        ref = next(f for f in scene["facts"] if not f["is_goal"])
        rel_side = "left" if goal["side_from_start"] != ref["side_from_start"] else "right"
        return (f"Imagine you are the {ref['color']} {ref['shape']}, facing the way you were facing. "
                f"Move toward the {goal['color']} {goal['shape']} on your {rel_side}.")
    raise ValueError(category)


def call_claude(scene: dict, category: str, model: str = "claude-sonnet-5") -> str:
    """Real generation path for the at-scale pipeline. Requires the `anthropic`
    package and ANTHROPIC_API_KEY in the environment (wire this up as a Modal
    secret, same pattern as the huggingface-secret2 secret already used
    elsewhere in this repo)."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    facts_json = json.dumps({
        "objects": scene["facts"],
        "goal_id": scene["goal_id"],
        "action_sequence_exact": _describe_actions(scene),
    }, indent=2)

    prompt = f"""You are generating one training example for a navigation-instruction
dataset targeting the "{category}" spatial-reasoning skill.

Here are the EXACT geometric facts for this scene (distances in meters, left/right
relative to the walker's starting orientation, and the exact discrete action
sequence the walker actually takes). Do not invent any fact not listed here --
only phrase what's given, naturally, the way a person giving directions would.

{facts_json}

Write ONE instruction sentence (or two short sentences) a person would say to
send someone to the goal object, in the style of "{category}". Return only the
instruction text, nothing else."""

    msg = client.messages.create(
        model=model, max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def generate_instruction(scene: dict, category: str, use_claude: bool = False) -> str:
    if use_claude and os.environ.get("ANTHROPIC_API_KEY"):
        return call_claude(scene, category)
    return template_fallback(scene, category)
