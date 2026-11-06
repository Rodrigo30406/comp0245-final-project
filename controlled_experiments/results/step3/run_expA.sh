#!/bin/bash
# Step 3, experiment A (pre-registered in report.txt, frozen 2026-10-29 19:05): ego vs ego-novw, seeds 0-4, protocol v1.1.
# 1) wait for the trained models; check that the retrained ego seeds 0-2 equal the pilot models bit for bit
# 2) DEV runs of both arms, 3) the early-stop check (stalled episodes re-run with --no-early-stop),
# 4) HARD runs of both arms, 5) descriptive DEV runs (raw seed 0, raw-ols, ego-ols).
set -e
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
W=${WORKERS:-20}
SEEDS="0 1 2 3 4"
MODELS="trained/ego_s0.npz trained/ego_s1.npz trained/ego_s2.npz trained/ego_s3.npz trained/ego_s4.npz
        trained/ego-novw_s0.npz trained/ego-novw_s1.npz trained/ego-novw_s2.npz trained/ego-novw_s3.npz trained/ego-novw_s4.npz
        trained/raw_s0.npz trained/raw-ols.npz trained/ego-ols.npz"
for m in $MODELS; do until [ -f "$m" ]; do sleep 20; done; done
until ! pgrep -f "train_policy.py train" > /dev/null; do sleep 20; done
echo "== $(date +%T) models ready"
$PY - <<'EOF'
import os
import numpy as np
for s in (0, 1, 2):
    if not os.path.exists(f"trained/pilot_models/ego_s{s}.npz"):  # pilot models are not in git
        print(f"ego seed {s}: pilot model not in this copy, bit-identity check skipped")
        continue
    a, b = np.load(f"trained/ego_s{s}.npz"), np.load(f"trained/pilot_models/ego_s{s}.npz")
    keys = [k for k in a.files if k[0] in "Wb" or k.startswith(("x_", "y_"))]
    same = all(np.array_equal(a[k], b[k]) for k in keys)
    print(f"ego seed {s}: retrained model bit-identical to the pilot model: {same} ({len(keys)} arrays)")
EOF

run() {  # run <model file> <eval set> <out dir> [extra args]
  local m=$1 set=$2 out=$3; shift 3
  if [ -f "$out/summary.txt" ]; then echo "skip $out"; return; fi
  echo "== $(date +%T) $out"
  $PY evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg "$m" --eval-set "data/$set" \
      --workers $W --out-dir "$out" "$@" > "$out.log" 2>&1
  grep -E "^Policy|^Stop reasons" "$out/summary.txt"
}
stall_check() {  # re-run the stalled episodes of a run without the early stop
  local m=$1 run_dir=$2
  local eps
  eps=$($PY -c "import csv; print(','.join(r['episode'] for r in csv.DictReader(open('$run_dir/rollouts.csv')) if r['stop_reason'] == 'stalled'))")
  if [ -z "$eps" ]; then echo "$run_dir: no stalled episode"; return; fi
  mkdir -p results/step3/nostall
  run "$m" dev "results/step3/nostall/$(basename $run_dir)" --no-early-stop --episodes "$eps"
}

for s in $SEEDS; do run trained/ego_s$s.npz dev results/ego_s${s}_dev; done
for s in $SEEDS; do run trained/ego-novw_s$s.npz dev results/ego-novw_s${s}_dev; done
for s in $SEEDS; do stall_check trained/ego_s$s.npz results/ego_s${s}_dev; done
for s in $SEEDS; do stall_check trained/ego-novw_s$s.npz results/ego-novw_s${s}_dev; done
for s in $SEEDS; do run trained/ego_s$s.npz hard results/ego_s${s}_hard; done
for s in $SEEDS; do run trained/ego-novw_s$s.npz hard results/ego-novw_s${s}_hard; done
run trained/raw_s0.npz dev results/raw_s0_dev
run trained/raw-ols.npz dev results/raw-ols_dev
run trained/ego-ols.npz dev results/ego-ols_dev
stall_check trained/raw_s0.npz results/raw_s0_dev
stall_check trained/raw-ols.npz results/raw-ols_dev
stall_check trained/ego-ols.npz results/ego-ols_dev
echo "== $(date +%T) done"
