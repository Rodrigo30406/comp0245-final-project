#!/bin/bash
# Step 4 E (descriptive; STEP4_PLAN_PROMPT.md 3): learning curve and stride on the C3 recipe, seeds 0-2.
#   train    train_step4e.py prepare + train-many (15 models; JOBS parallel, default 4 so it can share the machine)
#   offline  offline metrics of the 15 models (validation cache of their own split variant)
#   eval     DEV, --no-early-stop, one run at a time (WORKERS, default 20)
#   analyze  results/step3/step4E_analysis.py -> step4E.txt / .json and the two figures
# Usage: results/step3/run_step4E.sh <stage>   (from final/robotnik_arm_task)
set -eo pipefail
STAGE=${1:?stage: train, offline, eval or analyze}
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
W=${WORKERS:-20}
J=${JOBS:-4}
MODELS="ego-novw-trig-f0125 ego-novw-trig-f025 ego-novw-trig-f05 ego-novw-trig-s27 ego-novw-trig-s81"
SEEDS="0 1 2"
nopb() { grep -v -E "^pybullet build time|b3Warning|No inertial data|^robot_" || true; }
split_of() { case $1 in *-s27) echo trained/data_split_s27.json;; *-s81) echo trained/data_split_s81.json;; *) echo trained/data_split.json;; esac; }
cache_of() { case $1 in *-s27) echo results/step3/cache_s27;; *-s81) echo results/step3/cache_s81;; *) echo results/step3/cache;; esac; }

case $STAGE in
train)
  $PY train_step4e.py prepare 2>&1 | nopb
  $PY train_step4e.py train-many --models all --seeds 0-2 --jobs $J 2>&1 | nopb
  ;;
offline)
  for m in $MODELS; do for s in $SEEDS; do
    f=trained/${m}_s$s.npz
    [ -f "${f%.npz}.offline.json" ] && { echo "skip $f"; continue; }
    $PY train_policy.py offline --model $f --split $(split_of $m) --cache $(cache_of $m) > results/step3/offline_${m}_s$s.txt 2>&1
    grep "val loss" results/step3/offline_${m}_s$s.txt | head -1
  done; done
  ;;
eval)
  for m in $MODELS; do for s in $SEEDS; do
    out=results/${m}_s${s}_dev
    if [ -f "$out/summary.txt" ]; then echo "skip $out"; continue; fi
    echo "== $(date +%T) $out"
    $PY evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg trained/${m}_s$s.npz --eval-set data/dev \
        --workers $W --out-dir $out --no-early-stop > $out.log 2>&1
    grep -E "^Policy|^Stop reasons" $out/summary.txt
  done; done
  ;;
analyze)
  $PY results/step3/step4E_analysis.py 2>&1 | nopb
  ;;
*)
  echo "unknown stage $STAGE"; exit 1 ;;
esac
echo "== $(date +%T) stage $STAGE done"
