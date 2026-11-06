#!/bin/bash
# Step 4, experiment D (near-field end-effector error features; pre-registered in report.txt, frozen in
# results/step3/prereg_D_frozen.txt). One stage per call; each stage checks the decision table (STEP4_PLAN_PROMPT.md 6)
# on results/step3/D_mechanisms.json and refuses otherwise. Every new run is made with --no-early-stop (the C3 family
# failed the stall check); the older early-stop C3 runs get their stalled episodes re-run without the early stop.
#   train    ego-novw-trig-ee seeds 0-9 and 100-104, and the C3 controls ego-novw-trig seeds 5-9 (K 16)
#   em       DEV D-EM (expert base + D arm) seeds 0-4 -> results/B-EM_ego-novw-trig-ee_s<k>_dev; trigger check
#   mm       only if the trigger check passed: DEV D-MM seeds 0-9 and C3 seeds 5-9
#   hard     only if DEV (0)-(3) hold (and H-shortcut does not): C3 HARD s0-4 stall re-runs, D HARD s0-9, C3 HARD s5-9
#   confirm  only if HARD (4) holds too: C3 CONFIRM stall re-runs, D CONFIRM s100-104 (opened once)
# Usage: results/step3/run_expD.sh <stage>   (from final/robotnik_arm_task; WORKERS=20 by default)
set -eo pipefail
STAGE=${1:?stage: train, em, mm, hard or confirm}
PY="pixi run --manifest-path ${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml python"
W=${WORKERS:-20}
D=ego-novw-trig-ee
C=ego-novw-trig
FROZEN=results/step3/prereg_D_frozen.txt
[ -f $FROZEN ] || { echo "the D pre-registration is not frozen ($FROZEN missing)"; exit 1; }

nopb() { grep -v -E "^pybullet build time|b3Warning|No inertial data|^robot_" || true; }

evalrun() {  # evalrun <policy> <policy arg> <eval set> <out dir> [extra args]
  local pol=$1 arg=$2 set=$3 out=$4; shift 4
  if [ -f "$out/summary.txt" ]; then echo "skip $out"; return; fi
  echo "== $(date +%T) $out"
  $PY evaluate_policy.py --policy "$pol" --policy-arg "$arg" --eval-set "data/$set" --workers $W --out-dir "$out" "$@" \
      > "$out.log" 2>&1
  grep -E "^Policy|^Stop reasons" "$out/summary.txt"
}

stall_rerun() {  # stall_rerun <model file> <run dir> <eval set>: the stalled episodes of an early-stop run, without it
  local m=$1 run_dir=$2 set=$3 eps
  [ -f "$run_dir/rollouts.csv" ] || { echo "missing $run_dir/rollouts.csv"; exit 1; }
  eps=$($PY -c "import csv; print(','.join(r['episode'] for r in csv.DictReader(open('$run_dir/rollouts.csv')) if r['stop_reason'] == 'stalled'))" 2>/dev/null)
  if [ -z "$eps" ]; then echo "$run_dir: no stalled episode"; return; fi
  mkdir -p results/step3/nostall
  evalrun policies/bc_policy.py:make_policy "$m" "$set" "results/step3/nostall/$(basename $run_dir)" --no-early-stop --episodes "$eps"
}

gate() {  # gate <python expression on j = results/step3/D_mechanisms.json> <message>
  $PY - "$1" <<'EOF' 2>&1 | nopb || { echo "refused: $2"; exit 1; }
import json, sys
j = json.load(open("results/step3/D_mechanisms.json"))
ok = eval(sys.argv[1], {"j": j})
print(f"gate {sys.argv[1]}: {bool(ok)}; verdict so far: {(j.get('adoption') or {}).get('verdict')}")
sys.exit(0 if ok else 1)
EOF
}

mechanisms() { $PY results/step3/D_mechanisms.py "$@" 2>&1 | nopb; }

check_models() {  # every D model and the C3 s5-9 controls against the D freeze record (code and cache hashes)
  $PY - "$FROZEN" <<'EOF' 2>&1 | nopb || { echo "refused: model check failed"; exit 1; }
import json, sys
import numpy as np
code = {}
for line in open(sys.argv[1]):
    p = line.split()
    if len(p) == 2 and len(p[0]) == 64:
        code[p[1]] = p[0]
ok = True
for model, seeds, fs in (("ego-novw-trig-ee", list(range(10)) + [100, 101, 102, 103, 104], "ego_novw_ee"),
                         ("ego-novw-trig", [5, 6, 7, 8, 9], "ego_novw")):
    for s in seeds:
        m = json.loads(str(np.load(f"trained/{model}_s{s}.npz")["meta"]))
        t = m["trigger"]
        same_code = all(m["code_sha256"][f] == code[f] for f in ("train_policy.py", "bc_features.py"))
        same_cache = (t["trigger_cache_manifest_sha256"] == code["results/step3/cache/trigger/manifest.json"]
                      and (fs != "ego_novw_ee"
                           or m["ee_cache_manifest_sha256"] == code["results/step3/cache/ee/manifest.json"]))
        good = (m["seed"] == s and m["feature_set"] == fs and t["copies"] == 16 and t["qd_jitter"] == "auto"
                and t["jitter_seed"] == 0 and t["n_rows_added"] == 16 * 1613 and same_code and same_cache)
        ok &= good
        print(f"  {model}_s{s}: features {m['feature_set']}, K {t['copies']}, code = frozen {same_code}, caches = "
              f"frozen {same_cache}, epochs {m['epochs_run']} (best {m['best_epoch']}) -> {'ok' if good else 'MISMATCH'}")
sys.exit(0 if ok else 1)
EOF
}

case $STAGE in
train)
  [ -f results/step3/cache/ee/manifest.json ] || { echo "no ee side cache: train_policy.py build-ee-cache"; exit 1; }
  for s in 0 1 2 3 4 5 6 7 8 9 100 101 102 103 104; do
    [ ! -f trained/${D}_s$s.npz ] || { echo "trained/${D}_s$s.npz exists: refusing to reuse models of an earlier attempt"; exit 1; }
  done
  for s in 5 6 7 8 9; do
    [ ! -f trained/${C}_s$s.npz ] || { echo "trained/${C}_s$s.npz exists: refusing"; exit 1; }
  done
  DJ=${DJOBS:-7}; CJ=${CJOBS:-3}
  if pgrep -f "python train_step4e.py|python train_policy.py train" > /dev/null; then
    echo "other trainings are running: refusing"; exit 1
  fi
  avail=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
  need=$(( (DJ + CJ) * 2400 ))
  [ "$avail" -ge "$need" ] || { echo "MemAvailable ${avail} MiB < ${need} MiB for $((DJ + CJ)) trainings: lower DJOBS / CJOBS"; exit 1; }
  echo "== $(date +%T) train $D seeds 0-9, 100-104 ($DJ parallel) and $C seeds 5-9 ($CJ parallel); MemAvailable ${avail} MiB"
  $PY train_policy.py train-many --model $D --seeds 0-9,100-104 --jobs $DJ --trigger-copies 16 \
      > results/step3/train_${D}.log 2>&1 &
  p1=$!
  $PY train_policy.py train-many --model $C --seeds 5-9 --jobs $CJ --trigger-copies 16 \
      > results/step3/train_${C}_s5-9.log 2>&1 &
  p2=$!
  wait $p1 || { echo "train-many $D failed (results/step3/train_${D}.log)"; exit 1; }
  wait $p2 || { echo "train-many $C failed (results/step3/train_${C}_s5-9.log)"; exit 1; }
  check_models
  ;;
em)
  check_models
  for s in 0 1 2 3 4; do
    evalrun policies/hybrid.py:make_policy "base=expert;arm=trained/${D}_s$s.npz" dev results/B-EM_${D}_s${s}_dev --no-early-stop
  done
  mechanisms
  ;;
mm)
  check_models
  gate "j['em'] is not None and j['em']['trigger_ok'] and j['em']['runs_b'] == [f'B-EM_ego-novw-trig-ee_s{k}_dev' for k in range(5)]" \
       "D-MM runs only after the trigger check passed on the D-EM runs"
  for s in 0 1 2 3 4 5 6 7 8 9; do
    evalrun policies/bc_policy.py:make_policy trained/${D}_s$s.npz dev results/${D}_s${s}_dev --no-early-stop --save-traj --traj-every 10
  done
  for s in 5 6 7 8 9; do
    evalrun policies/bc_policy.py:make_policy trained/${C}_s$s.npz dev results/${C}_s${s}_dev --no-early-stop --save-traj --traj-every 10
  done
  mechanisms
  ;;
hard)
  check_models
  gate "(j.get('adoption') or {}).get('verdict', '').startswith('NEXT: DEV (0)-(3) met: run HARD')" \
       "HARD runs only after DEV (0)-(3) (without H-shortcut)"
  for s in 0 1 2 3 4; do stall_rerun trained/${C}_s$s.npz results/${C}_s${s}_hard hard; done
  for s in 0 1 2 3 4 5 6 7 8 9; do
    evalrun policies/bc_policy.py:make_policy trained/${D}_s$s.npz hard results/${D}_s${s}_hard --no-early-stop
  done
  for s in 5 6 7 8 9; do
    evalrun policies/bc_policy.py:make_policy trained/${C}_s$s.npz hard results/${C}_s${s}_hard --no-early-stop
  done
  mechanisms
  ;;
confirm)
  check_models
  gate "(j.get('adoption') or {}).get('verdict', '').startswith('NEXT: DEV (0)-(3) and HARD (4) met')" \
       "CONFIRM is opened only after DEV (0)-(3) and HARD (4)"
  for s in 100 101 102 103 104; do stall_rerun trained/${C}_s$s.npz results/${C}_s${s}_confirm confirm; done
  for s in 100 101 102 103 104; do
    evalrun policies/bc_policy.py:make_policy trained/${D}_s$s.npz confirm results/${D}_s${s}_confirm --no-early-stop
  done
  mechanisms --allow-confirm
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "== $(date +%T) stage $STAGE done"
