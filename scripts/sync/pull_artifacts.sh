#!/bin/bash
# On the other computer, inside a clone of the repository:
#   scripts/sync/pull_artifacts.sh light          # cloud tier: training runs, splits, evaluations (no traces), dataset metadata
#   scripts/sync/pull_artifacts.sh heavy          # LAN tier: episode recordings + traces, rsync over SSH from the lab PC
#   scripts/sync/pull_artifacts.sh verify         # check every episode .npz against the sha256 in its .json
# Both pulls are incremental and resumable; re-running only transfers what changed.
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$REPO_ROOT"
case "${1:-light}" in
  light)
    need_hf
    hf download "$HF_REPO" --repo-type dataset --local-dir . --exclude ".cache/*"
    rm -rf .cache 2>/dev/null || true
    python3 -I scripts/sync/pack.py --unpack
    echo "== light tier pulled; manifest: runs/sync/manifest.json" ;;
  heavy)
    for d in datasets runs/evaluation; do
      echo "== $HEAVY_HOST:$HEAVY_PATH/$d -> $d"
      rsync -aP --partial "${HEAVY_EXCLUDES[@]}" "$HEAVY_HOST:$HEAVY_PATH/$d/" "$d/"
    done ;;
  verify) python3 -I scripts/sync/manifest.py --verify ;;
  *) echo "usage: $0 light|heavy|verify" >&2; exit 1 ;;
esac
