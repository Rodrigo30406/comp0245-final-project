#!/bin/bash
set -euo pipefail
cd /home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
  results/step7/pc_pipeline.py --stage "${1:-all}" "${@:2}"
