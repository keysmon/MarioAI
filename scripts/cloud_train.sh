#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: cloud_train.sh PHASE MAX_SECONDS REPO_DIR S3_PREFIX [TRAIN_ARGS...]" >&2
  exit 64
fi

PHASE=$1
MAX_SECONDS=$2
REPO_DIR=$3
S3_PREFIX=$4
shift 4

if [[ ! "$PHASE" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "phase must contain only letters, numbers, underscores, and hyphens" >&2
  exit 64
fi
if [[ ! "$MAX_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "maximum seconds must be a positive integer" >&2
  exit 64
fi
if [[ ! -d "$REPO_DIR" ]]; then
  echo "repository directory does not exist: $REPO_DIR" >&2
  exit 64
fi
if [[ ! "$S3_PREFIX" =~ ^s3://[^/]+/.+/$ ]]; then
  echo "S3 prefix must name a bucket prefix ending in /" >&2
  exit 64
fi

finish() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  aws s3 sync "$REPO_DIR/models/" "${S3_PREFIX}models/"
  aws s3 sync "$REPO_DIR/reports/" "${S3_PREFIX}reports/"
  sudo shutdown -h now
  exit "$status"
}
trap finish EXIT INT TERM

cd "$REPO_DIR" || exit 1
timeout --signal=TERM --kill-after=300 "$MAX_SECONDS" \
  .venv/bin/python -m marioai.train --phase "$PHASE" "$@"
