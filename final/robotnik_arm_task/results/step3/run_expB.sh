#!/bin/bash
# Step 3, experiment B (pre-registered in report.txt): base/arm 2x2 split on DEV with the adopted arm of experiment A.
# Usage: MODEL=ego ./results/step3/run_expB.sh   (EE = results/B-EE_dev, MM = results/<MODEL>_s<seed>_dev, already run)
set -e
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
W=${WORKERS:-20}
MODEL=${MODEL:?set MODEL to the adopted arm of experiment A (ego or ego-novw)}
SEEDS=${SEEDS:-"0 1 2 3 4"}
cell() {  # cell <name> <policy arg> <out dir>
  local name=$1 arg=$2 out=$3
  if [ -f "$out/summary.txt" ]; then echo "skip $out"; return; fi
  echo "== $(date +%T) $out ($arg)"
  $PY evaluate_policy.py --policy policies/hybrid.py:make_policy --policy-arg "$arg" --no-early-stop \
      --eval-set data/dev --workers $W --out-dir "$out" > "$out.log" 2>&1
  grep -E "^Policy|^Stop reasons" "$out/summary.txt"
}
for s in $SEEDS; do
  cell EM "base=expert;arm=trained/${MODEL}_s$s.npz" results/B-EM_${MODEL}_s${s}_dev
  cell ME "base=trained/${MODEL}_s$s.npz;arm=expert" results/B-ME_${MODEL}_s${s}_dev
done
echo "== $(date +%T) done"
