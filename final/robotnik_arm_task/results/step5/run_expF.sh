#!/bin/bash
# Frozen experiment F, resumable stages and sequential DEV/HARD/CONFIRM gates.
# Launch with nohup and wait for the child to retain the execution session.
set -euo pipefail
cd /home/w/comp0245/LAB_SESSION_COMP_0245_2026_PUBLIC/.claude/worktrees/eval-protocol/final/robotnik_arm_task
export MPLCONFIGDIR=/tmp/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec pixi run --manifest-path /home/w/comp0245/RoboEnv/pixi.toml python \
  results/step5/step5_pipeline.py --stage "${1:-all}"
