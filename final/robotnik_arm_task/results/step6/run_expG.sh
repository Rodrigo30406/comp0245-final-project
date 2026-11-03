#!/bin/bash
# Step 6 G: cache -> pilot/selection -> train -> DEV -> HARD -> CONFIRM.
set -euo pipefail
cd /home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
export MPLCONFIGDIR=/tmp/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
  results/step6/step6_pipeline.py --experiment G --stage "${1:-all}" "${@:2}"
