#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/record_all_gifs.sh <model.zip>
# Records the 6 TRAINING levels only. Holdout levels (1-4, 5-1) are recorded
# separately as zeroshot-*.gif so they are never mislabeled as trained.
MODEL="${1:?model path required}"
mkdir -p assets/gifs
for LVL in 1-1 1-2 1-3 2-1 3-1 4-1; do
  python -m marioai.record_gif --model "$MODEL" --level "$LVL" \
    --out "assets/gifs/${LVL}.gif" --rollouts 5
done
