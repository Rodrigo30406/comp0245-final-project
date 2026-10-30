#!/bin/bash
# Protocol v1.1 baselines (step 3.0/3.3): expert on DEV/HARD/CONFIRM with --compare-recorded, the zero policy on DEV,
# and the hybrid with the expert on both halves on DEV (must reproduce the recorded expert step by step).
set -e
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
W=${WORKERS:-16}
for s in dev hard confirm; do
  $PY evaluate_policy.py --policy expert --eval-set data/$s --workers $W --compare-recorded --out-dir results/expert_v11_$s
  $PY tests/check_v11_against_v1.py results/expert_$s results/expert_v11_$s
done
$PY evaluate_policy.py --policy zero --eval-set data/dev --workers $W --out-dir results/zero_v11_dev
$PY evaluate_policy.py --policy policies/hybrid.py:make_policy --policy-arg "base=expert;arm=expert" --compare-recorded \
    --no-early-stop --eval-set data/dev --workers $W --out-dir results/B-EE_dev
