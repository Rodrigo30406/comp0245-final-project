#!/bin/bash
# Step 11, local machine: train R and S (stage train), then the DEV runs of R and S (stage eval; only after the freeze).
# Usage (from final/robotnik_arm_task): bash results/step11/run_step11_local.sh train|eval
set -uo pipefail
PY="pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python"
L=results/step11/logs
case "$1" in
train)
  {
    for s in 0 1 2 3 4; do
      echo "[ -f trained/step11/R_s$s.npz ] || $PY results/step11/step11.py train-r --seed $s > $L/train_R_s$s.log 2>&1"
      for n in base arm-K16 arm-K0; do
        echo "[ -f trained/step11/S-${n}_s$s.npz ] || $PY results/step11/step11.py train-s --net $n --seed $s > $L/train_S-${n}_s$s.log 2>&1"
      done
    done
  } | xargs -P ${JOBS:-10} -I{} bash -c "{}"
  ls trained/step11/*.npz | wc -l ;;
eval)
  [ -f results/step11/prereg_step11_frozen.txt ] || { echo "not frozen"; exit 1; }
  W=${WORKERS:-22}
  for s in 0 1 2 3 4; do
    for run in "R_s${s}_dev|policies/bc_policy.py:make_policy|trained/step11/R_s$s.npz" \
               "S-K0_s${s}_dev|policies/split_policy.py:make_policy|base=trained/step11/S-base_s$s.npz;arm=trained/step11/S-arm-K0_s$s.npz" \
               "S-K16_s${s}_dev|policies/split_policy.py:make_policy|base=trained/step11/S-base_s$s.npz;arm=trained/step11/S-arm-K16_s$s.npz"; do
      IFS='|' read -r name pol arg <<< "$run"
      out=results/step11/$name
      [ -f $out/summary.txt ] && { echo "skip $name"; continue; }
      rm -rf $out
      echo "[$(date +%T)] $name"
      $PY evaluate_policy.py --policy $pol --policy-arg "$arg" --eval-set data/dev --workers $W --save-traj --out-dir $out > $L/eval_$name.log 2>&1
      grep -E "^Policy .* success" $out/summary.txt
    done
  done ;;
esac
