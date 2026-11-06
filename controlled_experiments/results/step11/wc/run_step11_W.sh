#!/bin/bash
# Step 11 W on windcharger (/scratch0/yudiwang/task): train W-K0 / W-K16 seeds 0-4 while re-running the comparators
# (C3 = ego-novw-trig s0-4, ego-novw s0-4) on this machine's simulator; W's DEV runs only after the freeze file exists.
set -uo pipefail
cd /scratch0/yudiwang/task
export PIXI_CACHE_DIR=/scratch0/yudiwang/pixi-cache
PY="/scratch0/yudiwang/bin/pixi run --manifest-path /scratch0/yudiwang/RoboEnv/pixi.toml python"
L=results/step11/logs; mkdir -p $L trained/step11
evalrun () {  # name policy-arg workers
  out=results/step11/$1
  [ -f $out/summary.txt ] && { echo "skip $1"; return; }
  rm -rf $out
  echo "[$(date +%T)] eval $1"
  $PY evaluate_policy.py --policy policies/bc_policy.py:make_policy --policy-arg "$2" --eval-set data/dev \
      --workers $3 --save-traj --out-dir $out > $L/eval_$1.log 2>&1
  grep -E "^Policy .* success" $out/summary.txt
}
{
  for s in 0 1 2 3 4; do for v in K0 K16; do
    echo "[ -f trained/step11/W-${v}_s$s.npz ] || $PY results/step11/step11.py train-w --variant $v --seed $s > $L/train_W-${v}_s$s.log 2>&1"
  done; done
} | xargs -P 10 -I{} bash -c "{}" &
TRAIN=$!
for s in 0 1 2 3 4; do
  evalrun wc_C3_s${s}_dev trained/ego-novw-trig_s$s.npz 20
  evalrun wc_ego-novw_s${s}_dev trained/ego-novw_s$s.npz 20
done
wait $TRAIN
echo "[$(date +%T)] W training done: $(ls trained/step11/W-*.npz | wc -l) models"
while [ ! -f results/step11/prereg_step11_frozen.txt ]; do sleep 30; done
for s in 0 1 2 3 4; do for v in K0 K16; do
  evalrun W-${v}_s${s}_dev trained/step11/W-${v}_s$s.npz 30
done; done
echo "[$(date +%T)] all done"
