#!/usr/bin/env bash
set -euo pipefail
# Usage: scripts/train.sh <run-name> [extra args...]
RUN="${1:?run-name required}"; shift || true
python -m marioai.train --config configs/default.yaml --run-name "$RUN" "$@"
