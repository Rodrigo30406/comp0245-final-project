#!/usr/bin/env bash
# Step 9 sealed test2: opened ONCE, after the H-MM confirmation on confirm2 (2026-11-05, results/step9/step9_confirm2.txt),
# for the final frozen model L2 (dev seeds 0-4 = the models listed in prereg_step9_frozen.txt) and the expert.
# Same harness and arguments as run_step9_eval.sh (protocol v2.0), plus --allow-test2. Resumable like that script.
#   bash results/step9/run_step9_test2.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
WORKERS=${WORKERS:-22}
SET=test2

run() {  # out_dir, then the policy arguments
    local out=$1; shift
    if [[ -f "$out/summary.txt" ]]; then echo "skip $out"; return; fi
    rm -rf "$out"
    echo "$(date -Is) start $out"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY -m obstacle_task.evaluate "$@" --eval-set "data/obstacle/$SET" --allow-test2 --workers "$WORKERS" --out-dir "$out" \
        > "$out.log" 2>&1
    echo "$(date -Is) done $out"
}

run "results/step9/expert_$SET" --policy expert --compare-recorded
for s in 0 1 2 3 4; do
    run "results/step9/L2_s${s}_$SET" --policy obstacle_task/policies.py:make_config \
        --policy-arg "L2,seed=$s" --no-early-stop --save-traj --traj-every 100
done
echo "$(date -Is) chain complete for $SET"
