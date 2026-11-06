"""Experiment PC: frozen HARD + CONFIRM statistics, P minus C3.

Only explicit result paths and existing episode-reference caches are read.  No
evaluation recording is opened, and no reference cache is built implicitly.
The single-set path calls step 6's exact ``gp_common.difference`` function.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
sys.path.insert(0, str(TASK / "results" / "step6"))
sys.path.insert(0, str(TASK))
sys.path.insert(0, str(HERE))
import gp_common as gc  # noqa: E402

ar = gc.ar
OLD = "ego-novw-trig"
NEW = "ego-novw-trig-pd"
SEEDS = {"hard": list(range(5)), "confirm": list(range(100, 105))}
COUNTS = {"hard": 200, "confirm": 300}
BINARY_METRICS = ("success", "reach_2cm", "reach_5cm", "held_10s", "arm_extended_toward")
METRICS = tuple(dict.fromkeys((*BINARY_METRICS, *ar.FAILURE_CLASSES, "risk", "policy_ms")))
LOW_METRICS = ("success", "circling", "reach_2cm", "extended_not_reached")
RNG_RULE = ("10000 paired two-level draws; base seed 0; independent HARD/CONFIRM "
            "streams np.random.SeedSequence(0).spawn(2), spawn_key (0,) and (1,); "
            "one episode resample shared between arms within each set; arms' seeds independent")


def _matrices(A, B):
    A, B = np.atleast_2d(np.asarray(A, dtype=float)), np.atleast_2d(np.asarray(B, dtype=float))
    if not A.size or not B.size or A.shape[1] != B.shape[1]:
        raise ValueError("Both nonempty arms must have the same episodes")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("Metric matrices must contain only finite values")
    return A, B


def matrix_difference(A, B, level=.95, boot=None):
    """Step-6 calculation plus the unfiltered paired draws and Welch metadata.

    With the default bootstrap this is bit-for-bit the step-6 single-set path;
    none of the original numerical fields returned by gc.difference is changed.
    """
    A, B = _matrices(A, B)
    if boot is None:
        boot = ar.Bootstrap([len(A), len(B)], A.shape[1], ar.N_BOOT, ar.BOOT_SEED)
    out = gc.difference(A, B, level=level, boot=boot)
    paired = ar.paired_diff(A, B, boot=boot)
    welch = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B), level=level)
    out.update(draws=paired["draws"], welch_se=welch["se"], welch_df=welch["df"],
               n_episodes=A.shape[1])
    return out


def metric_matrix(runs, episodes, metric):
    if metric in BINARY_METRICS:
        result = ar.metric_matrix(runs, episodes, metric)
        if result is None:
            raise ValueError(f"Required metric column missing: {metric}")
        return result
    if metric in ar.FAILURE_CLASSES:
        return ar.matrix(runs, episodes, lambda row: float(row["fclass"] == metric))
    if metric == "risk":
        return ar.matrix(runs, episodes, lambda row: float(row["fclass"] in ("crash", "left_or_overshot")))
    if metric == "policy_ms":
        return ar.matrix(runs, episodes, lambda row: float(row["policy_ms_mean"]))
    raise ValueError(f"Unknown PC metric: {metric}")


def single_set(ra, rb, metric="success", episodes=None, level=.95, boot=None):
    """Compare prepared Run lists on one set, through the original step-6 path."""
    if episodes is None:
        ar.check_integrity(ra + rb, "PC single set", allow_incomplete=False)
        episodes, excluded = ar.paired_episodes(ra + rb)
        if excluded:
            raise ValueError("PC excludes no episodes; repair harness errors first")
    return matrix_difference(metric_matrix(ra, episodes, metric), metric_matrix(rb, episodes, metric), level, boot)


def combine_statistics(hard, confirm, n_hard=200, n_confirm=300, level=.95):
    """Episode-weighted independent-set bootstrap and four-term Welch formula.

    Inputs are matrix_difference dictionaries.  The four variance terms are
    w_H^2 s_H,A^2/n_H,A, w_H^2 s_H,B^2/n_H,B, then the corresponding CONFIRM
    terms; their Welch-Satterthwaite denominator uses each term's own n-1.
    """
    if n_hard < 0 or n_confirm < 0 or n_hard + n_confirm <= 0:
        raise ValueError("Episode weights must be nonnegative with a positive total")
    total = n_hard + n_confirm
    weights = [n_hard / total, n_confirm / total]
    draws_h, draws_c = np.asarray(hard["draws"], dtype=float), np.asarray(confirm["draws"], dtype=float)
    if draws_h.ndim != 1 or draws_h.shape != draws_c.shape or not len(draws_h):
        raise ValueError("Independent-set draws must be nonempty vectors of equal length")
    if not (np.isfinite(draws_h).all() and np.isfinite(draws_c).all()):
        raise ValueError("Bootstrap draws must be finite")
    draws = (n_hard * draws_h + n_confirm * draws_c) / total
    a = (n_hard * hard["a"] + n_confirm * confirm["a"]) / total
    b = (n_hard * hard["b"] + n_confirm * confirm["b"]) / total
    diff = (n_hard * hard["diff"] + n_confirm * confirm["diff"]) / total
    terms, denominator = [], 0.0
    for weight, result in zip(weights, (hard, confirm)):
        for arm in ("a", "b"):
            rates = np.asarray(result[f"per_seed_{arm}"], dtype=float).ravel()
            if len(rates) < 2 or not np.isfinite(rates).all():
                raise ValueError("Combined Welch requires at least two finite seed rates per arm and set")
            variance = float(rates.var(ddof=1) / len(rates))
            term = weight ** 2 * variance
            terms.append(term)
            denominator += term ** 2 / (len(rates) - 1)
    variance = float(sum(terms))
    se = float(np.sqrt(variance))
    df = float(variance ** 2 / denominator) if denominator > 0 else np.nan
    if se == 0:
        welch_lo, welch_hi, p_welch = diff, diff, 1.0 if diff == 0 else 0.0
    else:
        q = float(stats.t.ppf(.5 + level / 2, df))
        welch_lo, welch_hi = diff - q * se, diff + q * se
        p_welch = float(2 * stats.t.sf(abs(diff) / se, df))
    lo, hi = ar.percentile_ci(draws, level=level)
    return {"a": float(a), "b": float(b), "diff": float(diff), "lo": lo, "hi": hi,
            "level": level, "p_boot": ar.boot_p(draws), "draws": draws,
            "welch_lo": float(welch_lo), "welch_hi": float(welch_hi), "p_welch": p_welch,
            "welch_se": se, "welch_df": df, "welch_variance_terms": terms,
            "welch_denominator": float(denominator), "n_episodes": total,
            "episode_counts": {"hard": n_hard, "confirm": n_confirm},
            "weights": {"hard": weights[0], "confirm": weights[1]}}


def conditions(hard, confirm, combined):
    """Only STEP7 §3's three conditions and mirror rule determine the verdict."""
    success = combined["success"]
    c1 = bool(success["lo"] > 0 and success["welch_lo"] > 0)
    c2 = bool(hard["success"]["diff"] > 0 and confirm["success"]["diff"] > 0)
    c3 = bool(combined["risk"]["diff"] <= .03)
    confirmed = c1 and c2 and c3
    mirror = bool(success["hi"] < 0 and success["welch_hi"] < 0)
    verdict = "P 在留出集上得到确认" if confirmed else "没有得到确认"
    return {"condition_1": c1, "condition_2": c2, "condition_3": c3,
            "confirmed": confirmed, "mirror_worse": mirror, "verdict": verdict,
            "mirror_verdict": "P 在留出集上更差" if mirror else None,
            "keep_c3": not confirmed,
            "main_model": ("P 仅在助教确认固定 PD 输出层符合作业要求后；否则 C3" if confirmed else "C3")}


def _raw_runs(task_root, model, set_name, seeds):
    runs = [ar.load_run(str(task_root / "results" / f"{model}_s{seed}_{set_name}")) for seed in seeds]
    if any(run.meta.get("warnings") for run in runs):
        raise RuntimeError("Run name/configuration mismatch")
    return runs


def merged_controls(runs, task_root=TASK):
    """D/F/step-6 stalled-row replacement, with step7's new output as fallback."""
    task_root = Path(task_root)
    result, notes = [], []
    for run in runs:
        stalled = sorted(row["episode"] for row in run.rows if row.get("stop_reason") == "stalled")
        if run.meta.get("no_early_stop") is True or not stalled:
            result.append(run)
            notes.append(f"{run.name}: original rows; no stalled replacement needed")
            continue
        candidates = [task_root / "results" / step / "nostall" / run.name
                      for step in ("step3", "step5", "step6", "step7")]
        path = next((path for path in candidates if (path / "summary.txt").exists()), None)
        if path is None:
            raise RuntimeError(f"Missing stalled rerun: {run.name}")
        from step6_pipeline import validate_run
        model_arg = f"trained/{run.meta['model']}_s{run.meta['seed']}.npz"
        rerun, _ = validate_run(path, stalled, model_arg)
        if (rerun.meta.get("no_early_stop") is not True or sorted(rerun.by_episode) != stalled
                or len(rerun.rows) != len(stalled) or rerun.meta.get("warnings")
                or any(_row_error(row) for row in rerun.rows)):
            raise RuntimeError(f"Invalid stalled rerun: {path}")
        rows = [rerun.by_episode.get(row["episode"], row) for row in run.rows]
        result.append(ar.Run(run.path, rows, run.header, run.config, dict(run.meta)))
        notes.append(f"{run.name}: replaced {len(stalled)} stalled rows from {path.relative_to(task_root)}")
    return result, notes


def _row_error(row):
    return (row.get("stop_reason") in ("harness_error", "policy_error")
            or bool(row.get("error_message")) or row.get("hold10_stop") == "policy_error")


def _prepare(runs, task_root):
    for run in runs:
        cache_dir = task_root / "results" / "step3"
        if not (cache_dir / f"cache_ref_{run.meta['set']}.csv").is_file():
            raise RuntimeError("Missing existing episode reference; implicit rebuilding is forbidden")
        ar.prepare(run, out_dir=str(cache_dir), data_dir=str(task_root / "data"))
    return runs


def _line(name, result):
    multiplier, units = (1, "ms") if name == "policy_ms" else (100, "pp")
    a_units = "ms" if name == "policy_ms" else "%"
    return (f"{name}: C3 {multiplier * result['a']:.6f}{a_units}, P {multiplier * result['b']:.6f}{a_units}; "
            f"Δ {multiplier * result['diff']:+.6f} {units}; "
            f"bootstrap {100 * result['level']:g}% [{multiplier * result['lo']:+.6f}, "
            f"{multiplier * result['hi']:+.6f}], Welch {100 * result['level']:g}% "
            f"[{multiplier * result['welch_lo']:+.6f}, {multiplier * result['welch_hi']:+.6f}] {units}")


def _compare_set(task_root, out_dir, set_name, bootstrap_seed):
    from step6_pipeline import validate_run

    A, notes = merged_controls(_raw_runs(task_root, OLD, set_name, SEEDS[set_name]), task_root)
    A = _prepare(A, task_root)
    B = [validate_run(task_root / "results" / f"{NEW}_s{seed}_{set_name}",
                      range(COUNTS[set_name]), f"trained/{NEW}_s{seed}.npz")[0]
         for seed in SEEDS[set_name]]
    B = _prepare(B, task_root)
    if any(run.meta.get("no_early_stop") is not True for run in B):
        raise RuntimeError("Every P run must have --no-early-stop")
    if any(_row_error(row) for run in A + B for row in run.rows):
        raise RuntimeError("Repair harness_error / policy_error before PC analysis")
    ar.check_integrity(A + B, f"PC {set_name}", allow_incomplete=False)
    episodes, excluded = ar.paired_episodes(A + B)
    if excluded or len(episodes) != COUNTS[set_name]:
        raise RuntimeError(f"Expected all {COUNTS[set_name]} episodes in {set_name}")
    boot = ar.Bootstrap([len(A), len(B)], len(episodes), ar.N_BOOT, bootstrap_seed)
    compare_boot_seed = int(bootstrap_seed.generate_state(1, dtype=np.uint32)[0])
    out = {"n_episodes": len(episodes), "seeds": SEEDS[set_name], "replacement_notes": notes,
           "runs_a": [run.path for run in A], "runs_b": [run.path for run in B],
           "bootstrap_spawn_key": list(bootstrap_seed.spawn_key),
           "compare_boot_seed": compare_boot_seed}
    lines = ["", set_name.upper(), *notes]
    for metric in METRICS:
        out[metric] = single_set(A, B, metric, episodes, boot=boot)
        lines.append(_line(metric, out[metric]))
    low = [episode for episode in episodes if float(A[0].by_episode[episode]["target_z"]) < .45]
    if not low:
        raise RuntimeError(f"{set_name}: empty low-z subgroup")
    # Independent subgroup draws from a child of this set's stream; the primary
    # draws and all main quantities above stay unchanged.
    low_seed = bootstrap_seed.spawn(1)[0]
    low_boot = ar.Bootstrap([len(A), len(B)], len(low), ar.N_BOOT, low_seed)
    out["low_z"] = {"n_episodes": len(low), "threshold_m": .45,
                    "bootstrap_spawn_key": list(low_seed.spawn_key)}
    for metric in LOW_METRICS:
        out["low_z"][metric] = single_set(A, B, metric, low, boot=low_boot)
        lines.append(_line("low_z " + metric, out["low_z"][metric]))
    expert_path = task_root / "results" / f"expert_v11_{set_name}"
    expert = _prepare([ar.load_run(str(expert_path))], task_root)[0] if (expert_path / "rollouts.csv").exists() else None
    full_lines, full = ar.compare_arms(A, B, "C3", "P", (), expert_run=expert,
                                     allow_incomplete=False, seed=compare_boot_seed)
    # Complete descriptive comparison, with all Step-3 adoption/floor rules
    # removed.  PC's sole decision is reported separately by conditions().
    cut = next((i for i, value in enumerate(full_lines) if value.startswith("Adoption-rule inputs")), len(full_lines))
    full_lines = [value.replace("descriptive unless the floor rule is active", "descriptive only")
                  for value in full_lines[:cut] if not value.startswith("Floor rule")]
    full_lines[0] = full_lines[0].replace("Experiment A:", "Step 7 descriptive comparison:")
    full.pop("adoption_inputs", None)
    full.pop("floor_rule_active", None)
    full["PC_statistics"] = out
    ar.write_outputs(str(out_dir), f"PC_compare_{set_name}",
                     ["Complete descriptive comparison; the verdict is given only by the STEP7 §3 conditions "
                      "in PC_mechanisms.txt.",
                      f"The generic complete comparison uses the integer seed {compare_boot_seed} derived from this "
                      "set's SeedSequence, for compatibility with strata_analysis.",
                      "The generic complete comparison CI and the PC custom statistics appended below use different "
                      "bootstrap draws; the primary verdict uses the PC custom statistics.",
                      *full_lines, "", "PC custom statistics (source of the primary verdict):", *lines], full)
    return out, lines


def analyze(task_root=TASK, out_dir=None, allow_confirm=False, smoke=False):
    """Run the complete analysis after checking the freeze and opening token."""
    from pc_freeze import check_freeze

    check_freeze()
    task_root = Path(task_root).resolve()
    out_dir = Path(out_dir).resolve() if out_dir is not None else task_root / "results" / "step7"
    if smoke:
        boundary = HERE / "smoke"
        if task_root == boundary or not task_root.is_relative_to(boundary):
            raise ValueError("Synthetic task root must be a child of results/step7/smoke")
        if out_dir != task_root / "results" / "step7":
            raise ValueError("Synthetic output must remain at <smoke-root>/results/step7")
    elif task_root != TASK or out_dir != HERE:
        raise ValueError("Formal analysis uses only the task's frozen result paths")
    if not allow_confirm:
        raise RuntimeError("PC analysis requires --allow-confirm")
    if not smoke:
        from pc_pipeline import confirm_token
        confirm_token(HERE, allow_confirm)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_h, seed_c = np.random.SeedSequence(ar.BOOT_SEED).spawn(2)
    hard, hard_lines = _compare_set(task_root, out_dir, "hard", seed_h)
    confirm, confirm_lines = _compare_set(task_root, out_dir, "confirm", seed_c)
    combined = {"n_episodes": 500, "episode_weights": {"hard": .4, "confirm": .6}}
    for metric in METRICS:
        combined[metric] = combine_statistics(hard[metric], confirm[metric])
    low_h, low_c = hard["low_z"], confirm["low_z"]
    combined["low_z"] = {"n_episodes": low_h["n_episodes"] + low_c["n_episodes"], "threshold_m": .45,
                         "weighting": "within low-z subgroup, weights equal each set's low-z episode count"}
    for metric in LOW_METRICS:
        combined["low_z"][metric] = combine_statistics(low_h[metric], low_c[metric],
                                                        low_h["n_episodes"], low_c["n_episodes"])
    result = {"experiment": "PC", "software_smoke": bool(smoke), "delta_direction": "P minus C3",
              "rule": "STEP7_PLAN_PROMPT.md §3; 95%; one comparison; no multiplicity correction",
              "bootstrap_rule": RNG_RULE, "hard": hard, "confirm": confirm, "combined": combined,
              "decision": conditions(hard, confirm, combined),
              "step6_DEV_verdict_unchanged": "KEEP C3",
              "design_context": "This experiment was designed after seeing P's DEV results; the hypothesis comes "
                                "from DEV; DEV is presented together with this experiment.",
              "limitations": "10 vs 10 seeds after pooling; the HARD distribution differs from DEV; the effect is "
                             "the pooled average over the two sets; second test of the same change."}
    result["descriptive_predictions"] = {
        "descriptive_only": True,
        "low_z_success_increase": bool(combined["low_z"]["success"]["diff"] > 0),
        "circling_decrease": bool(combined["circling"]["diff"] < 0),
        "reach_2cm_increase": bool(combined["reach_2cm"]["diff"] > 0),
        "low_z_circling_decrease": bool(combined["low_z"]["circling"]["diff"] < 0),
        "low_z_reach_2cm_increase": bool(combined["low_z"]["reach_2cm"]["diff"] > 0),
        "extended_not_reached": "no direction was pre-registered"}
    lines = ["Experiment PC: frozen pre-registered numbers and verdict; differences are P minus C3; 95% CI.",
             "SYNTHETIC SOFTWARE SMOKE ONLY" if smoke else "Formal HARD + CONFIRM analysis",
             RNG_RULE, result["design_context"], *hard_lines, *confirm_lines, "", "COMBINED (200 HARD + 300 CONFIRM)"]
    for metric in METRICS:
        lines.append(_line(metric, combined[metric]))
    lines.append(f"low_z: n={combined['low_z']['n_episodes']}; weighted by each set's number of low-z episodes.")
    for metric in LOW_METRICS:
        lines.append(_line("low_z " + metric, combined["low_z"][metric]))
    success, risk, decision = combined["success"], combined["risk"], result["decision"]
    lines += ["", "§3 item-by-item verdict:",
              f"(1) pooled Δsuccess bootstrap lower bound {100 * success['lo']:+.6f} pp, Welch lower bound "
              f"{100 * success['welch_lo']:+.6f} pp, both > 0: {decision['condition_1']}",
              f"(2) HARD Δsuccess {100 * hard['success']['diff']:+.6f} pp, CONFIRM "
              f"{100 * confirm['success']['diff']:+.6f} pp, both > 0: {decision['condition_2']}",
              f"(3) pooled crash + left_or_overshot Δ {100 * risk['diff']:+.6f} pp ≤ +3 pp: {decision['condition_3']}",
              "Verdict: " + decision["verdict"] + (", KEEP C3" if decision["keep_c3"] else ""),
              f"Mirror: bootstrap upper bound {100 * success['hi']:+.6f} pp, Welch upper bound "
              f"{100 * success['welch_hi']:+.6f} pp, both < 0: {decision['mirror_worse']}",
              *([decision["mirror_verdict"]] if decision["mirror_verdict"] else []),
              "Main model after confirmation: " + decision["main_model"],
              "Descriptive predictions (not part of the verdict): " + json.dumps(result["descriptive_predictions"], ensure_ascii=False),
              "Step 6 DEV verdict: KEEP C3 (unchanged); in the report DEV is presented together with this experiment.",
              "Pre-registered power and limitations: " + result["limitations"]]
    ar.write_outputs(str(out_dir), "PC_mechanisms", lines, result)
    print("\n".join(lines), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-confirm", action="store_true")
    parser.add_argument("--smoke-root", help="synthetic task root below results/step7/smoke")
    args = parser.parse_args(argv)
    analyze(task_root=args.smoke_root or TASK, allow_confirm=args.allow_confirm, smoke=bool(args.smoke_root))


if __name__ == "__main__":
    main()
