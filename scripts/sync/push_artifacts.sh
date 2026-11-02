#!/bin/bash
# Push the light tier to the private Hugging Face dataset repo. Incremental: hf upload only sends files whose content
# changed. Run at the end of a chain or from the nightly timer (scripts/sync/install_timer.sh).
#   scripts/sync/push_artifacts.sh            # upload
#   scripts/sync/push_artifacts.sh --list     # only list what would be uploaded, with sizes
set -euo pipefail
source "$(dirname "$0")/common.sh"; cd "$REPO_ROOT"
python3 -I scripts/sync/manifest.py > /dev/null   # refresh runs/sync/manifest.json (sizes, hashes, episode counts)
python3 -I scripts/sync/pack.py                    # archives of the evaluations (no traces) and the dataset metadata
if [ "${1:-}" = "--list" ]; then python3 -I scripts/sync/manifest.py --light; exit 0; fi
need_hf
hf repos create "$HF_REPO" --repo-type dataset --private >/dev/null 2>&1 || true
for spec in "${LIGHT_UPLOADS[@]}"; do
  IFS='|' read -r local remote excludes <<< "$spec"
  [ -d "$local" ] || continue
  args=(); IFS=',' read -ra ex <<< "$excludes"; for e in "${ex[@]}"; do [ -n "$e" ] && args+=(--exclude "$e"); done
  echo "== $local -> $HF_REPO/$remote"
  hf upload "$HF_REPO" "$local" "$remote" --repo-type dataset "${args[@]}" --commit-message "sync $(hostname) $(date -u +%FT%TZ): $local" | tail -1
done
hf upload "$HF_REPO" runs/sync/manifest.json runs/sync/manifest.json --repo-type dataset --commit-message "manifest $(date -u +%FT%TZ)" | tail -1
echo "== pushed $(date +%T)"
