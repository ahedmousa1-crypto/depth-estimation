"""Render a generated scene: start-frame image, first-person walkthrough frames,
and the 2D pixel-space trace (waypoints projected into the start frame) --
all computed by the same camera/projection three.js uses, so it's pixel-exact.
"""
import base64
import json
import os

from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_chromium() -> str:
    """Locate a Chromium binary. Override with CHROMIUM_PATH if auto-detect
    fails (e.g. running outside this dev environment)."""
    override = os.environ.get("CHROMIUM_PATH")
    if override:
        return override
    browsers_dir = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    if os.path.isdir(browsers_dir):
        for name in sorted(os.listdir(browsers_dir), reverse=True):
            candidate = os.path.join(browsers_dir, name, "chrome-linux", "chrome")
            if os.path.isfile(candidate):
                return candidate
    # fall back to whatever `playwright install` put in the default location
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        return p.chromium.executable_path


CHROMIUM = _find_chromium()


def _save_data_url(data_url: str, path: str):
    header, b64 = data_url.split(",", 1)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))


def render_scene(scene: dict, out_dir: str, walkthrough_every: int = 1):
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(HERE, "scene_template.html")) as f:
        template = f.read()
    three_js_path = os.path.join(HERE, "three.min.js")
    html = template.replace("__SCENE_JSON__", json.dumps(scene))
    html = html.replace('src="three.min.js"', f'src="file://{three_js_path}"')
    html_path = os.path.join(out_dir, "_scene.html")
    with open(html_path, "w") as f:
        f.write(html)

    result = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROMIUM,
            args=["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
        )
        page = browser.new_page(viewport={"width": 640, "height": 480})
        page.goto(f"file://{html_path}")
        page.wait_for_function("window.__SCENE_READY__ === true", timeout=15000)

        # 1. start frame (this is the NaviTrace-style single image)
        data_url = page.evaluate("() => window.__renderFrame(0)")
        start_frame_path = os.path.join(out_dir, "start_frame.png")
        _save_data_url(data_url, start_frame_path)
        result["start_frame"] = start_frame_path

        # 2. project the whole waypoint path into the start frame's pixel space
        trace_px = page.evaluate("() => window.__projectWaypointsFrom(0)")
        result["trace_pixels"] = trace_px  # list of [x, y, depth] per waypoint

        # 2b. object pixel positions from the start frame (for sanity/debug overlays)
        obj_px = page.evaluate("() => window.__projectObjectsFrom(0)")
        result["object_pixels"] = obj_px

        # 3. first-person walkthrough frames (NavSpace-style multi-frame)
        n_wp = len(scene["waypoints"])
        frame_paths = []
        for i in range(0, n_wp, walkthrough_every):
            data_url = page.evaluate(f"() => window.__renderFrame({i})")
            fp = os.path.join(out_dir, f"walk_frame_{i:02d}.png")
            _save_data_url(data_url, fp)
            frame_paths.append(fp)
        result["walkthrough_frames"] = frame_paths

        browser.close()

    return result
