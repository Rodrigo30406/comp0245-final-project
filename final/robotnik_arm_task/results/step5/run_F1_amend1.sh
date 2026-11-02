#!/bin/bash
# Amendment 1 verification; preserve all evidence from the first verification.
set -euo pipefail
cd /home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
export MPLCONFIGDIR=/tmp/mplcfg
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=(pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python)
out=results/step5/F1_expert_pilot
echo "F1 amendment 1 start $(date -u --iso-8601=seconds)"
if [ ! -f "$out/summary.txt" ]; then
  if [ -d "$out" ]; then rm -rf -- "$out"; fi
  "${PY[@]}" evaluate_policy.py --policy expert --eval-set data/pilot --episodes 0-4 \
    --save-traj --traj-every 1 --workers 5 --out-dir "$out" \
    > results/step5/F1_expert_pilot_amend1.log 2>&1
fi
export DAGGER_F1_DIAGNOSTICS=${DAGGER_F1_DIAGNOSTICS:-results/step5/F1_amend1_recorded.json}
export DAGGER_F1_PILOT_DIAGNOSTICS=${DAGGER_F1_PILOT_DIAGNOSTICS:-results/step5/F1_amend1_exact.json}
"${PY[@]}" -m unittest discover -s tests -p test_dagger_relabel.py -v \
  > "${DAGGER_F1_LOG:-results/step5/F1_amend1_verification.log}" 2>&1
echo "F1 amendment 1 PASS $(date -u --iso-8601=seconds)"
