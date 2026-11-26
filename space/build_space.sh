#!/usr/bin/env bash
# Assemble a deployable Space folder (app + the two pure-python modules it needs) and push it.
# Usage: space/build_space.sh <hf-user>/<space-name>
set -euo pipefail
SPACE_ID="${1:?usage: build_space.sh user/space}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/runs/space_build"
rm -rf "$OUT" && mkdir -p "$OUT/telltale"
cp "$ROOT/space/app.py" "$ROOT/space/requirements.txt" "$ROOT/space/README.md" "$OUT/"
cp "$ROOT/telltale/__init__.py" "$ROOT/telltale/text.py" "$ROOT/telltale/inference.py" "$OUT/telltale/"
huggingface-cli repo create "$SPACE_ID" --repo-type space --space_sdk gradio -y 2>/dev/null || true
huggingface-cli upload "$SPACE_ID" "$OUT" . --repo-type space
echo "Deployed https://huggingface.co/spaces/$SPACE_ID (set TELLTALE_MODEL_REPO / HF_TOKEN secrets if the model is private)"
