#!/usr/bin/env bash
# One-time setup for the synthetic nav data pipeline.
set -euo pipefail
cd "$(dirname "$0")"

pip install -q playwright anthropic
curl -sS -L "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js" -o three.min.js

echo "Done. Chromium is expected to already be installed at:"
echo "  /opt/pw-browsers/chromium-1194/chrome-linux/chrome"
echo "If running elsewhere, run 'playwright install chromium' and update"
echo "CHROMIUM in render.py to match."
