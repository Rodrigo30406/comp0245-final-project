#!/usr/bin/env bash
# Builds the protocol v2.0 sets (plan section 4) into data/obstacle/<set>/ with the expert v2: resumable chain of
# collector runs of at most --batch episodes each (default 100: about 4-6 min with 20 workers), every run capped at
# 570 s. An interrupted chain is resumed by running the same command again (saved episodes are kept).
#   results/step9/run_step9_sets.sh                       # train2 pilot2 pilot2n dev2 confirm2
#   results/step9/run_step9_sets.sh --set dev2 --set confirm2
#   results/step9/run_step9_sets.sh --set test2 --allow-test2    # sealed: only for the final frozen model
#   results/step9/run_step9_sets.sh --set pilot2 --root results/step9/smoke/sets_check --batch 50   # script check
# Sets (seed, episodes, noise_frac): train2 (3, 2000, 0.5), pilot2 (3000, 100, 0), pilot2n (3000, 100, 1.0),
# dev2 (300, 300, 0), confirm2 (302, 300, 0), test2 (301, 300, 0) = obstacle_task/collect.py SET_DEFAULTS.
set -euo pipefail
cd "$(dirname "$0")/../.."
allow_test2=0
batch=100
root=data/obstacle
sets=()
while (($#)); do
    case "$1" in
        --allow-test2) allow_test2=1 ;;
        --set) shift; sets+=("${1:?missing set name}") ;;
        --batch) shift; batch="${1:?missing batch size}" ;;
        --root) shift; root="${1:?missing root folder}" ;;
        *) echo "Usage: $0 [--set NAME ...] [--batch N] [--root DIR] [--allow-test2]" >&2; exit 2 ;;
    esac
    shift
done
if ((${#sets[@]} == 0)); then sets=(train2 pilot2 pilot2n dev2 confirm2); fi
for name in "${sets[@]}"; do
    extra=()
    case "$name" in
        train2) count=2000 ;;
        pilot2|pilot2n) count=100 ;;
        dev2|confirm2) count=300 ;;
        test2)
            if ((allow_test2 == 0)); then echo "test2 is sealed: it requires --allow-test2" >&2; exit 2; fi
            count=300; extra=(--allow-test2) ;;
        *) echo "Unknown set: $name" >&2; exit 2 ;;
    esac
    for ((first=0; first<count; first+=batch)); do
        last=$((first+batch-1)); if ((last>=count)); then last=$((count-1)); fi
        echo "== $name episodes $first-$last"
        PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        timeout --signal=TERM --kill-after=10s 570s \
        pixi run --manifest-path "${ROBOENV_DIR:?set ROBOENV_DIR to your RoboEnv checkout}/pixi.toml" python -m obstacle_task.collect \
            --set-name "$name" --workers 20 --episodes "$first-$last" --resume --out-dir "$root/$name" \
            "${extra[@]}"
    done
done
