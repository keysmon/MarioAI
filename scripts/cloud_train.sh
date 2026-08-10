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
SYNC_INTERVAL_SECONDS=${MARIOAI_SYNC_INTERVAL_SECONDS:-900}
if [[ ! "$SYNC_INTERVAL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "sync interval must be a positive integer" >&2
  exit 64
fi
DURABLE_STOP_RESERVE_SECONDS=${MARIOAI_DURABLE_STOP_RESERVE_SECONDS:-60}
if [[ ! "$DURABLE_STOP_RESERVE_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
  (( DURABLE_STOP_RESERVE_SECONDS >= MAX_SECONDS )); then
  echo "durable-stop reserve must be positive and below maximum seconds" >&2
  exit 64
fi

sync_checkpoints() {
  local snapshot_dir
  local manifest
  local relative
  local status=0
  snapshot_dir=$(mktemp -d)
  while IFS= read -r -d '' manifest; do
    relative=${manifest#"$REPO_DIR/models/"}
    mkdir -p "$snapshot_dir/$(dirname "$relative")"
    cp -- "$manifest" "$snapshot_dir/$relative"
  done < <(find "$REPO_DIR/models" -type f -name latest.json -print0)

  if aws s3 sync "$REPO_DIR/models/" "${S3_PREFIX}models/" \
    --exclude "*/latest.json"; then
    while IFS= read -r -d '' manifest; do
      relative=${manifest#"$snapshot_dir/"}
      aws s3 cp "$manifest" "${S3_PREFIX}models/${relative}" \
        --only-show-errors || status=$?
    done < <(find "$snapshot_dir" -type f -name latest.json -print0)
  else
    status=$?
  fi
  rm -rf "$snapshot_dir"
  return "$status"
}

sync_all() {
  local status=0
  sync_checkpoints || status=$?
  aws s3 sync "$REPO_DIR/reports/" "${S3_PREFIX}reports/" || status=$?
  return "$status"
}

periodic_sync() {
  local timer_pid=
  trap '
    if [[ -n "$timer_pid" ]]; then
      kill "$timer_pid" 2>/dev/null
      wait "$timer_pid" 2>/dev/null
    fi
    exit 0
  ' INT TERM
  while true; do
    sleep "$SYNC_INTERVAL_SECONDS" &
    timer_pid=$!
    wait "$timer_pid" || exit 0
    timer_pid=
    sync_all || true
  done
}

finish() {
  local status=$?
  trap - EXIT INT TERM
  set +e
  if [[ -n "${SYNC_PID:-}" ]]; then
    kill "$SYNC_PID" 2>/dev/null
    wait "$SYNC_PID" 2>/dev/null
  fi
  sync_all
  sudo shutdown -h now
  exit "$status"
}
trap finish EXIT INT TERM

cd "$REPO_DIR" || exit 1
periodic_sync &
SYNC_PID=$!
DEADLINE_EPOCH=$((
  $(date +%s) + MAX_SECONDS - DURABLE_STOP_RESERVE_SECONDS
))
timeout --signal=TERM --kill-after=300 "$MAX_SECONDS" \
  .venv/bin/python scripts/train_phase.py phase \
  --phase "$PHASE" \
  --deadline-epoch "$DEADLINE_EPOCH" \
  "$@"
