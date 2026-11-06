#!/bin/bash
# Step 3 continued, experiment C3 (trigger-frame copies; pre-registered in report.txt, frozen in
# results/step3/prereg_C3_frozen.txt). One stage per call, in this order; each stage checks the decision table of the
# pre-registration (item 7) on the previous stage's output and refuses otherwise:
#   calib    train K = 16, 64, 256 (seed 0) in parallel -> trained/ego-novw-trig-k<K>_s0.npz; K probe (validation only)
#   train    seeds 0-4 and 100-104 with the chosen K -> trained/ego-novw-trig_s<k>.npz; seed 0 must equal the
#            calibration model bit for bit; every model's trigger settings and code hash are checked
#   em       DEV C3-EM (expert base + C3 arm, --no-early-stop) -> results/B-EM_ego-novw-trig_s<k>_dev; verdict
#   mm       only if C3-EM triggered: DEV C3-MM (early stop, trajectories every 10 steps) -> results/ego-novw-trig_s<k>_dev;
#            stall re-runs
#   hard     only if DEV (1)-(3) and the no-worse rule hold and H-trajectory does not: HARD C3-MM
#   confirm  only if, in addition, HARD (4) holds: CONFIRM C3-MM (seeds 100-104), opened once
# Usage: results/step3/run_expC3.sh <stage>   (from final/robotnik_arm_task; WORKERS=20 by default)
set -eo pipefail
STAGE=${1:?stage: calib, train, em, mm, hard or confirm}
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
W=${WORKERS:-20}
M=ego-novw-trig
SEEDS="0 1 2 3 4"
FROZEN=results/step3/prereg_C3_frozen.txt
[ -f $FROZEN ] || { echo "the C3 pre-registration is not frozen ($FROZEN missing)"; exit 1; }

nopb() { grep -v "^pybullet build time" || true; }   # drop pybullet's banner; keep the producer's exit status

evalrun() {  # evalrun <policy> <policy arg> <eval set> <out dir> [extra args]
  local pol=$1 arg=$2 set=$3 out=$4; shift 4
  if [ -f "$out/summary.txt" ]; then echo "skip $out"; return; fi
  echo "== $(date +%T) $out"
  $PY evaluate_policy.py --policy "$pol" --policy-arg "$arg" --eval-set "data/$set" --workers $W --out-dir "$out" "$@" \
      > "$out.log" 2>&1
  grep -E "^Policy|^Stop reasons" "$out/summary.txt"
}

stall_check() {  # re-run the stalled episodes of a run without the early stop (protocol)
  local m=$1 run_dir=$2 eps
  eps=$($PY -c "import csv; print(','.join(r['episode'] for r in csv.DictReader(open('$run_dir/rollouts.csv')) if r['stop_reason'] == 'stalled'))" 2>/dev/null)
  if [ -z "$eps" ]; then echo "$run_dir: no stalled episode"; return; fi
  mkdir -p results/step3/nostall
  evalrun policies/bc_policy.py:make_policy "$m" dev "results/step3/nostall/$(basename $run_dir)" --no-early-stop --episodes "$eps"
}

gate() {  # gate <python expression on j = results/step3/C3_mechanisms.json> <message>
  $PY - "$1" <<'EOF' 2>&1 | nopb || { echo "refused: $2"; exit 1; }
import json, sys
j = json.load(open("results/step3/C3_mechanisms.json"))
ok = eval(sys.argv[1], {"j": j})
print(f"gate {sys.argv[1]}: {bool(ok)}; verdict so far: {(j.get('adoption') or {}).get('verdict')}")
sys.exit(0 if ok else 1)
EOF
}

mechanisms() {  # mechanisms [--allow-confirm]
  $PY results/step3/C3_mechanisms.py "$@" 2>&1 | nopb
}

case $STAGE in
calib)
  [ -f results/step3/cache/trigger/manifest.json ] || $PY train_policy.py build-trigger-cache --workers 12
  for K in 16 64 256; do
    out=trained/$M-k${K}_s0.npz
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "== $(date +%T) train $out"
    $PY train_policy.py train --model $M --trigger-copies $K --seed 0 --out $out > ${out%.npz}.log 2>&1 &
  done
  wait
  for K in 16 64 256; do [ -f trained/$M-k${K}_s0.npz ] || { echo "missing trained/$M-k${K}_s0.npz"; exit 1; }; done
  echo "== $(date +%T) K probe"
  rm -f results/step3/C3_probe.json
  $PY results/step3/C3_probe.py 2>&1 | nopb
  ;;
train)
  # amended rule 1 (report.txt, prereg_C3_amend1_frozen.txt): K from results/step3/C3_K_choice.json (C3_K_choice.py)
  K=$($PY -c "import json; k = json.load(open('results/step3/C3_K_choice.json'))['chosen_k']; print('' if k is None else k)" 2>/dev/null || true)
  [ -n "$K" ] || { echo "no K chosen (results/step3/C3_K_choice.txt): stop and propose"; exit 1; }
  for s in 0 1 2 3 4 100 101 102 103 104; do
    [ ! -f trained/${M}_s$s.npz ] || { echo "trained/${M}_s$s.npz exists: refusing to reuse models of an earlier attempt"; exit 1; }
  done
  echo "== $(date +%T) train seeds 0-4, 100-104 with K = $K"
  $PY train_policy.py train-many --model $M --seeds 0-4,100-104 --jobs 10 --trigger-copies $K \
      > results/step3/train_${M}_s0-4_100-104.log 2>&1
  $PY - "$K" "$FROZEN" <<'EOF' 2>&1 | nopb
import json, sys
import numpy as np
k, frozen = int(sys.argv[1]), sys.argv[2]
code_sha = next(line.split()[0] for line in open(frozen) if line.strip().endswith(" train_policy.py"))
a, b = np.load("trained/ego-novw-trig_s0.npz"), np.load(f"trained/ego-novw-trig-k{k}_s0.npz")
keys = [x for x in a.files if x[0] in "Wb" or x.startswith(("x_", "y_"))]
same = all(np.array_equal(a[x], b[x]) for x in keys)
print(f"seed 0 with K = {k}: bit-identical to the calibration model: {same} ({len(keys)} arrays)")
ok = same
for s in (0, 1, 2, 3, 4, 100, 101, 102, 103, 104):
    m = json.loads(str(np.load(f"trained/ego-novw-trig_s{s}.npz")["meta"]))
    t = m["trigger"]
    good = (m["seed"] == s and t["copies"] == k and t["qd_jitter"] == "auto" and t["jitter_seed"] == 0
            and t["n_rows_added"] == 1613 * k and m["code_sha256"]["train_policy.py"] == code_sha)
    ok &= good
    print(f"  seed {s}: K {t['copies']}, rows added {t['n_rows_added']}, jitter {t['qd_jitter']}, code = frozen "
          f"{m['code_sha256']['train_policy.py'] == code_sha}, trigger cache {t['trigger_cache_manifest_sha256'][:12]}, "
          f"epochs {m['epochs_run']} -> {'ok' if good else 'MISMATCH'}")
sys.exit(0 if ok else 1)
EOF
  ;;
em)
  for s in $SEEDS; do
    evalrun policies/hybrid.py:make_policy "base=expert;arm=trained/${M}_s$s.npz" dev results/B-EM_${M}_s${s}_dev --no-early-stop
  done
  $PY analyze_rollouts.py compare --a-runs $(for s in $SEEDS; do printf "results/B-EM_ego-novw_s%s_dev " $s; done) \
      --b-runs $(for s in $SEEDS; do printf "results/B-EM_${M}_s%s_dev " $s; done) --set dev \
      --predicted-classes parked_no_arm,circling > results/step3/C3_EM_compare_dev.stdout 2>&1
  for ext in txt json; do mv results/step3/A_A_vs_B_dev.$ext results/step3/C3_EM_dev.$ext; done
  mechanisms
  ;;
mm)
  gate "j['em'] is not None and j['em']['verdict'] == 'trigger'" "C3-MM runs only when C3-EM triggered"
  for s in $SEEDS; do
    evalrun policies/bc_policy.py:make_policy trained/${M}_s$s.npz dev results/${M}_s${s}_dev --save-traj --traj-every 10
  done
  for s in $SEEDS; do stall_check trained/${M}_s$s.npz results/${M}_s${s}_dev; done
  $PY analyze_rollouts.py compare --a ego-novw --b $M --set dev --seeds 0-4 --predicted-classes parked_no_arm,circling \
      > results/step3/C3_MM_compare_dev.stdout 2>&1
  mechanisms
  ;;
hard)
  gate "j['em']['verdict'] == 'trigger' and not (j['em'].get('h_trajectory') or {}).get('holds') and j['mm_dev'] is not None and j['mm_dev']['cond1'] and j['mm_dev']['cond2'] and j['mm_dev']['cond3'] and j['mm_dev']['no_worse'] is not False" \
       "HARD runs only after DEV (1)-(3) and the no-worse rule, without H-trajectory"
  for s in $SEEDS; do evalrun policies/bc_policy.py:make_policy trained/${M}_s$s.npz hard results/${M}_s${s}_hard; done
  $PY analyze_rollouts.py compare --a ego-novw --b $M --set hard --seeds 0-4 --predicted-classes parked_no_arm,circling \
      > results/step3/C3_MM_compare_hard.stdout 2>&1
  mechanisms
  ;;
confirm)
  gate "j['em']['verdict'] == 'trigger' and not (j['em'].get('h_trajectory') or {}).get('holds') and j['mm_dev']['cond1'] and j['mm_dev']['cond2'] and j['mm_dev']['cond3'] and j['mm_dev']['no_worse'] is not False and j['mm_hard'] is not None and j['mm_hard']['direction']" \
       "CONFIRM is opened only after DEV (1)-(3), the no-worse rule and HARD (4)"
  for s in 100 101 102 103 104; do
    evalrun policies/bc_policy.py:make_policy trained/${M}_s$s.npz confirm results/${M}_s${s}_confirm
  done
  $PY analyze_rollouts.py --allow-confirm compare --a ego-novw --b $M --set confirm --seeds 100-104 \
      --predicted-classes parked_no_arm,circling > results/step3/C3_MM_compare_confirm.stdout 2>&1
  mechanisms --allow-confirm
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "== $(date +%T) stage $STAGE done"
