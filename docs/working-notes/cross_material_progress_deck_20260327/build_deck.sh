#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
OUT_DIR="$REPO_ROOT/paper/out"
OUT_PPTX="$OUT_DIR/cross_material_progress_20260327.pptx"

mkdir -p "$OUT_DIR"

python3 "$SCRIPT_DIR/generate_assets.py"
rm -f "$OUT_PPTX"

pandoc \
  --from=markdown+link_attributes \
  --to=pptx \
  --slide-level=1 \
  --resource-path="$SCRIPT_DIR:$REPO_ROOT" \
  "$SCRIPT_DIR/cross_material_progress_20260327.md" \
  -o "$OUT_PPTX"

printf 'Built %s\n' "$OUT_PPTX"
