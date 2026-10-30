#!/bin/bash
# Step 3, experiment A, condition (5) (pre-registered): ego and ego-novw with NEW training seeds 100-104 on CONFIRM.
# Run after DEV conditions (1)-(3) and HARD condition (4) were met on the reach_5cm decision metric (floor rule).
set -e
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
W=${WORKERS:-20}
SEEDS="100 101 102 103 104"
echo "== $(date +%T) training"
$PY train_policy.py train-many --model ego --seeds 100-104 --jobs 5 > results/step3/train_ego_s100-104.log 2>&1 &
$PY train_policy.py train-many --model ego-novw --seeds 100-104 --jobs 5 > results/step3/train_ego-novw_s100-104.log 2>&1 &
wait
for a in ego ego-novw; do for s in $SEEDS; do [ -f trained/${a}_s$s.npz ] || { echo "missing trained/${a}_s$s.npz"; exit 1; }; done; done
for a in ego ego-novw; do for s in $SEEDS; do
  out=results/${a}_s${s}_confirm
  if [ -f "$out/summary.txt" ]; then echo "skip $out"; continue; fi
  echo "== $(date +%T) $out"
  $PY evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg trained/${a}_s$s.npz --eval-set data/confirm \
      --workers $W --out-dir $out > $out.log 2>&1
  grep -E "^Policy|^Stop reasons" $out/summary.txt
done; done
echo "== $(date +%T) done"
