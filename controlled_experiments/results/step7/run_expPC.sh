#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/mplcfg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec pixi run --manifest-path "${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml" python \
  results/step7/pc_pipeline.py --stage "${1:-all}" "${@:2}"
