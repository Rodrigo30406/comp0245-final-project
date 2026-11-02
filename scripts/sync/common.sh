#!/bin/bash
# Shared settings of the artifact sync (scripts/sync/). Not part of the provenance hash (only scripts/*.py is).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_REPO="${COMP0245_HF_REPO:-Rodrigo30406/comp0245-artifacts}"   # private Hugging Face *dataset* repo
HF_BIN="${HF_BIN:-$HOME/.local/share/hf-tools/bin/hf}"
export PYTHONPATH="$HOME/.local/share/hf-tools${PYTHONPATH:+:$PYTHONPATH}"
HEAVY_HOST="${COMP0245_HEAVY_HOST:-labia10@192.168.27.91}"       # the lab PC that holds the datasets and traces
HEAVY_PATH="${COMP0245_HEAVY_PATH:-repos/comp0245-final-project}"
# Light tier (cloud, ~2 GB): everything needed to read, re-evaluate and re-train, except the episode recordings and traces.
# Evaluations (minus traces) and dataset metadata travel as one archive each (scripts/sync/pack.py).
LIGHT_UPLOADS=(
  "runs/training|runs/training|"
  "runs/splits|runs/splits|"
  "runs/diagnostics|runs/diagnostics|"
  "runs/sync/pack|runs/sync/pack|"
  "docs/figures|docs/figures|"
)
# Heavy tier (LAN / USB): the recordings and the evaluation traces. Aborted or interrupted collections are not synced.
HEAVY_EXCLUDES=(--exclude '*aborted*' --exclude '*interrupted*' --exclude '*smoke_check*' --exclude '__pycache__')
hf() { "$HF_BIN" "$@"; }
need_hf() { [ -x "$HF_BIN" ] || { echo "hf CLI missing: pip3 install --target ~/.local/share/hf-tools 'huggingface_hub[cli,hf_xet]'" >&2; exit 2; }
            hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run  $HF_BIN auth login  (write token from https://huggingface.co/settings/tokens)" >&2; exit 2; }; }
