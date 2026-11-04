"""E13 efficiency table from the training run directories (REPORT_PLAN_20261103.md §4).

Usage: efficiency_table.py [RUN_DIR ...]   (default: runs/training/v2_*)

Reads model_info.json only: parameters, single-observation inference latency (NumPy, median / p95 over 2 000 calls, measured
by the trainer), training wall time, epochs, offline fit. No evaluation is run.

The latency medians were measured by each trainer on a machine that was usually running collectors or evaluators at the
same time, so they range 35–150 µs for the same architecture; the evaluator's per-episode `mean_online_prediction_us`
is the in-loop figure. Both are far below the 1 000 µs control period.
"""
import glob
import json
import sys
from pathlib import Path

KIND = {
    "egocentric_mlp_regressor": "MLP",
    "gru_regressor": "GRU",
    "transformer_regressor": "transformer",
    "gru_gated_regressor": "gated GRU",
}


def row(run: Path) -> str:
    info = json.loads((run / "model_info.json").read_text())
    us = info.get("inference_single_observation_us") or {}
    secs = info.get("fit_and_transform_seconds")
    minutes = f"{secs / 60:.0f}" if secs else ""
    epochs = f"{info.get('epochs_completed', '')} (best {info.get('best_epoch', '')})"
    nrmse = info.get("best_validation_episode_macro_balanced_nrmse")
    extra = []
    if info.get("trigger_copies"):
        extra.append(f"K {info['trigger_copies']['copies_per_episode']}")
    if info.get("trigger_weight"):
        w = info["trigger_weight"]
        extra.append(f"trigger weight {w['weight'] if isinstance(w, dict) else w:g}")
    if info.get("wheel_output") and info["wheel_output"] != "regression":
        extra.append(info["wheel_output"])
    if info.get("memory_stride_steps"):
        extra.append(f"memory every {info['memory_stride_steps']} ms")
    return (
        f"| `{run.name}` | {KIND.get(info.get('kind'), info.get('kind'))}, {info.get('input_features')} inputs"
        f"{(', ' + ', '.join(extra)) if extra else ''} | {info.get('regression_parameters', 0):,} | "
        f"{us.get('median', float('nan')):.0f} / {us.get('p95', float('nan')):.0f} | {minutes} | {epochs} | "
        f"{info.get('training_episodes', '')} | {nrmse:.3f} |" if nrmse is not None else
        f"| `{run.name}` | {KIND.get(info.get('kind'), info.get('kind'))} | {info.get('regression_parameters', 0):,} | "
        f"{us.get('median', float('nan')):.0f} / {us.get('p95', float('nan')):.0f} | {minutes} | {epochs} | "
        f"{info.get('training_episodes', '')} | |"
    )


def main(argv):
    runs = [Path(a) for a in argv] or sorted(Path(p) for p in glob.glob("runs/training/v2_*"))
    runs = [r for r in runs if (r / "model_info.json").exists()]
    print("| run | model | parameters | inference µs median / p95 | training min | epochs | episodes | val NRMSE |")
    print("|---|---|---:|---:|---:|---|---:|---:|")
    for r in runs:
        print(row(r))


if __name__ == "__main__":
    main(sys.argv[1:])
