#!/usr/bin/env bash
# Step 9 closed-loop evaluation chain (protocol v2.0 harness obstacle_task/evaluate.py), resumable:
# a run folder with summary.txt is skipped; a folder without it (interrupted run) is deleted and rerun.
#   bash results/step9/run_step9_eval.sh dev2 [configs] [seeds]      (default: expert, then L0 L1 L1-K0 L2 x seeds 0-4)
#   bash results/step9/run_step9_eval.sh confirm2 "L1 L2" "100 101 102 103 104"
# Learned policies run with --no-early-stop (as every new model family since step 4); the expert keeps the
# early stop and is compared with its recordings. Learner trajectories are kept at 10 Hz (--traj-every 100)
# for the top-down figures. test2 is never run from here (one manual run at the very end).
set -euo pipefail
cd "$(dirname "$0")/../.."
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
SET=${1:?usage: run_step9_eval.sh dev2|confirm2 [configs] [seeds]}
CONFIGS=${2:-"L0 L1 L1-K0 L2"}
SEEDS=${3:-"0 1 2 3 4"}
WORKERS=${WORKERS:-22}
case "$SET" in dev2|confirm2) ;; *) echo "set must be dev2 or confirm2" >&2; exit 2 ;; esac

run() {  # out_dir, then the policy arguments
    local out=$1; shift
    if [[ -f "$out/summary.txt" ]]; then echo "skip $out"; return; fi
    rm -rf "$out"
    echo "$(date -Is) start $out"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY -m obstacle_task.evaluate "$@" --eval-set "data/obstacle/$SET" --workers "$WORKERS" --out-dir "$out" \
        > "$out.log" 2>&1
    echo "$(date -Is) done $out"
}

run "results/step9/expert_$SET" --policy expert --compare-recorded
for c in $CONFIGS; do
    for s in $SEEDS; do
        run "results/step9/${c}_s${s}_$SET" --policy obstacle_task/policies.py:make_config \
            --policy-arg "$c,seed=$s" --no-early-stop --save-traj --traj-every 100
    done
done
echo "$(date -Is) chain complete for $SET"
