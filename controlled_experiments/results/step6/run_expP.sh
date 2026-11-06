#!/bin/bash
# Step 6 P: cache -> pilot program check -> train -> DEV -> HARD -> CONFIRM.
set -euo pipefail
cd "$(dirname "$0")/../.."
export MPLCONFIGDIR=/tmp/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
exec pixi run --manifest-path "${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml" python \
  results/step6/step6_pipeline.py --experiment P --stage "${1:-all}" "${@:2}"
