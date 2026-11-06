#!/usr/bin/env bash
# Cross-evaluation chain (2026-11-05 night): Rodrigo's v2 policies (branch rodrigo, weights/v2) under this task's
# protocol v1.1 harness on DEV and HARD, so that both lines of the group are scored on the same episodes with the
# same success definition. Descriptive: no adoption decision, CONFIRM and test are not opened.
# Resumable: a run folder with summary.txt is skipped; a folder without it is deleted and rerun.
# Gates: (1) the pilot smoke run of K144 seed 0 must have finished without policy / harness errors;
#        (2) waits until the step-9 confirm2 chain (run_step9_eval.sh) has exited, then runs its analysis first.
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
WORKERS=${WORKERS:-22}
OUT=results/step10_cross
LOG=$OUT/chain_cross.log
log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

# ---- gate 1: smoke ----------------------------------------------------------------------------------------------
SMOKE=$OUT/smoke/rodrigo-k144_s0_pilot
while [[ ! -f $SMOKE/summary.txt ]]; do sleep 30; done
if $PY - "$SMOKE/rollouts.csv" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r for r in rows if r["stop_reason"] in ("policy_error", "harness_error", "invalid_action")]
succ = sum(r["success"] == "True" for r in rows)
print(f"smoke: {len(rows)} episodes, success {succ}, bad stops {len(bad)}")
sys.exit(1 if bad or not rows else 0)
PYEOF
then log "gate 1 (smoke) passed"; else log "gate 1 (smoke) FAILED: policy/harness errors in $SMOKE; chain aborted"; exit 1; fi

# ---- gate 2: wait for the step-9 confirm2 chain, then its analysis -------------------------------------------------
while pgrep -f "run_step9_eval.sh confirm2" > /dev/null; do sleep 60; done
log "step-9 confirm2 chain has exited"
if [[ ! -f results/step9/step9_confirm2.txt ]]; then
    log "running step9_analysis.py --set confirm2"
    $PY results/step9/step9_analysis.py --set confirm2 --configs L1 L2 --seeds 100-104 > $OUT/step9_confirm2_analysis.log 2>&1 \
        && log "step9 confirm2 analysis done (results/step9/step9_confirm2.txt)" \
        || log "step9 confirm2 analysis FAILED (see $OUT/step9_confirm2_analysis.log); continuing"
fi

# ---- the cross-evaluation runs ------------------------------------------------------------------------------------
run() {  # out_dir set model_npz
    local out=$1 set=$2 npz=$3
    if [[ -f "$out/summary.txt" ]]; then log "skip $out"; return; fi
    rm -rf "$out"
    log "start $out"
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        $PY evaluate_policy.py --policy policies/rodrigo_policy.py:make_policy --policy-arg "$npz" \
        --name "$(basename "$out")" --eval-set "data/$set" --workers "$WORKERS" --no-early-stop \
        --save-traj --traj-every 100 --out-dir "$out" > "$out.log" 2>&1
    log "done $out ($(grep -m1 -i 'success' "$out/summary.txt" 2>/dev/null | head -c 120))"
}
for set in dev hard; do
    for s in 0 1 2 3 4; do run "$OUT/rodrigo-k144_s${s}_$set" $set "trained/rodrigo_v2/v2_mlp28_trig144_seed${s}/model.npz"; done
    for s in 0 1 2 3 4; do run "$OUT/rodrigo-gru40_s${s}_$set" $set "trained/rodrigo_v2/v2_gru28_patience40_seed${s}/model.npz"; done
    if [[ $set == dev ]]; then run "$OUT/rodrigo-ctrl_s0_$set" $set "trained/rodrigo_v2/v2_mlp28_control_seed0/model.npz"; fi
    $PY $OUT/cross_analysis.py --set $set > $OUT/cross_analysis_$set.log 2>&1 && log "analysis $set done" || log "analysis $set FAILED"
done
log "chain complete"
