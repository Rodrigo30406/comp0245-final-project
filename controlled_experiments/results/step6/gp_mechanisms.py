"""Frozen G/P statistics and sequential gates, following experiment F."""
import argparse
import json

import numpy as np

from gp_common import (HERE, TASK, OLD, MODELS, SEEDS, ar, check_freeze, difference,
                       dev_pass, hard_pass, verdict, merged_old, prepare, raw_runs, write_json)


def line(name, d, experiment):
    return (f"{name}: C3 {100*d['a']:.6f}%, {experiment} {100*d['b']:.6f}%; "
            f"Δ {100*d['diff']:+.6f} pp; bootstrap {100*d['level']:g}% "
            f"[{100*d['lo']:+.6f}, {100*d['hi']:+.6f}], Welch {100*d['level']:g}% "
            f"[{100*d['welch_lo']:+.6f}, {100*d['welch_hi']:+.6f}] pp")


def apply_conditions(out, set_name, experiment):
    d = out["success"]
    if set_name == "dev":
        out.update(condition_1=bool(d["lo"] > 0 and d["welch_lo"] > 0),
                   condition_2=bool(out["risk"]["diff"] <= .03 + 1e-12),
                   mirror_worse=bool(d["hi"] < 0 and d["welch_hi"] < 0))
        ext, circ = out["extended_not_reached"], out["circling"]
        within = lambda x: bool(x["lo"] >= -.07 and x["hi"] <= .07
                                and x["welch_lo"] >= -.07 and x["welch_hi"] <= .07)
        if experiment == "G":
            out["mechanism_predictions"] = {
                "MG1": bool(circ["diff"] < 0), "MG2": bool(out["held_10s"]["diff"] > 0),
                "MG3": within(ext), "MG4": "see G_smoothness.json; descriptive"}
        else:
            out["mechanism_predictions"] = {
                "MP1": bool(ext["hi"] < 0 and ext["welch_hi"] < 0),
                "MP2": bool(out["crash"]["diff"] < 0), "MP3": within(circ),
                "MP4": "see P_offline_table.json; descriptive"}
    elif set_name == "hard":
        out["condition_3"] = bool(d["diff"] > 0)
    else:
        out["condition_4"] = bool(d["diff"] > 0 and d["lo"] > 0)
    return out


def compare(experiment, set_name, lines):
    A_raw = raw_runs(OLD, set_name, SEEDS[set_name])
    A, notes = merged_old(A_raw)
    A = prepare(A)
    B = prepare(raw_runs(MODELS[experiment], set_name, SEEDS[set_name]))
    if any(r.meta.get("no_early_stop") is not True for r in B):
        raise RuntimeError("All new G/P runs must use --no-early-stop")
    if any(row.get("stop_reason") in ("harness_error", "policy_error") or row.get("error_message")
           or row.get("hold10_stop") == "policy_error" for r in A + B for row in r.rows):
        raise RuntimeError("harness_error / policy_error present; verdict not allowed")
    ar.check_integrity(A + B, f"experiment {experiment} {set_name}")
    eps, _ = ar.paired_episodes(A + B)
    boot = ar.Bootstrap([len(A), len(B)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    out = {"n_episodes": len(eps), "seeds": SEEDS[set_name], "replacement_notes": notes,
           "runs_a": [r.path for r in A], "runs_b": [r.path for r in B]}
    lines += ["", set_name.upper(), *notes]
    for metric in ("success", "reach_2cm", "reach_5cm", "held_10s", "arm_extended_toward"):
        level = .975 if metric == "success" and set_name == "dev" else .95
        out[metric] = difference(ar.metric_matrix(A, eps, metric), ar.metric_matrix(B, eps, metric), level, boot)
        lines.append(line(metric, out[metric], experiment))
    for cls in ar.FAILURE_CLASSES:
        fn = lambda row, cls=cls: float(row["fclass"] == cls)
        out[cls] = difference(ar.matrix(A, eps, fn), ar.matrix(B, eps, fn), boot=boot)
        lines.append(line(cls, out[cls], experiment))
    fn = lambda row: float(row["fclass"] in ("crash", "left_or_overshot"))
    out["risk"] = difference(ar.matrix(A, eps, fn), ar.matrix(B, eps, fn), boot=boot)
    lines.append(line("crash + left_or_overshot", out["risk"], experiment))
    if set_name == "dev":
        low = [e for e in eps if float(A[0].by_episode[e]["target_z"]) < .45]
        if not low:
            raise RuntimeError("DEV low-z subgroup empty")
        out["low_z_success"] = difference(ar.metric_matrix(A, low, "success"), ar.metric_matrix(B, low, "success"))
        out["low_z_success"]["n_episodes"] = len(low)
        lines.append(line("target z < 0.45 m success", out["low_z_success"], experiment))
        # policy_ms is continuous milliseconds, not a success percentage.
        timing = lambda r: float(r["policy_ms_mean"])
        out["policy_ms"] = difference(ar.matrix(A, eps, timing), ar.matrix(B, eps, timing), boot=boot)
        lines.append("policy_ms (ms): " + json.dumps(ar._clean(out["policy_ms"]), ensure_ascii=False))
        names = ["reach_2cm", "reach_5cm", "held_10s", "arm_extended_toward", "low_z_success", "policy_ms"]
        ps = [ar.combined_p(out[n]["p_boot"], out[n]["p_welch"]) for n in names]
        out["secondary_holm"] = {"names": names, "combined_p": ps, "holm": ar.holm(ps), "descriptive_only": True}
        lines.append("Secondary Holm (6 metrics, descriptive): " + json.dumps(ar._clean(out["secondary_holm"]), ensure_ascii=False))
        out["early_stop_sensitivity"] = difference(ar.metric_matrix(prepare(A_raw), eps, "success"),
                                                    ar.metric_matrix(B, eps, "success"), .975, boot)
        lines.append(line("C3 original early-stop rows sensitivity (not used for verdict)", out["early_stop_sensitivity"], experiment))
        if experiment == "G":
            losses = []
            for model in (OLD, MODELS[experiment]):
                vals = []
                for seed in SEEDS["dev"]:
                    with np.load(TASK / "trained" / f"{model}_s{seed}.npz") as z:
                        vals.append(json.loads(str(z["meta"]))["val_loss"])
                losses.append(np.asarray(vals))
            out["j0_validation_loss"] = difference(losses[0][:, None], losses[1][:, None])
            lines.append("j=0 standardized validation MSE (descriptive): " + json.dumps(ar._clean(out["j0_validation_loss"]), ensure_ascii=False))
    apply_conditions(out, set_name, experiment)
    for name in ("condition_1", "condition_2", "condition_3", "condition_4", "mirror_worse", "mechanism_predictions"):
        if name in out:
            lines.append(f"{name}: {out[name]}")
    if out.get("mirror_worse"):
        lines.append("Mirror verdict: this change makes success worse.")
    expert = prepare([ar.load_run(str(TASK / "results" / f"expert_v11_{set_name}"))])[0]
    full_lines, full = ar.compare_arms(A, B, "C3", experiment, (), expert_run=expert)
    # Reuse the descriptive tables, never Step 3's adoption/floor rules.
    cut = next((i for i, value in enumerate(full_lines) if value.startswith("Adoption-rule inputs")), len(full_lines))
    full_lines = [value.replace("descriptive unless the floor rule is active", "descriptive only")
                  for value in full_lines[:cut] if not value.startswith("Floor rule")]
    full_lines[0] = full_lines[0].replace("Experiment A:", "Step 6 descriptive comparison:")
    full.pop("adoption_inputs", None)
    full.pop("floor_rule_active", None)
    full["step6_conditions"] = {k: v for k, v in out.items() if k.startswith("condition_") or k == "mirror_worse"}
    full["step6_secondary_holm"] = out.get("secondary_holm")
    full_lines = ["The main DEV success and the mirror verdict use the 97.5% CI below; all other CIs in the generic full comparison are 95%.",
                  line("success primary verdict", out["success"], experiment), *full_lines]
    full["preregistered_success"] = out["success"]
    ar.write_outputs(str(HERE), f"{experiment}_MM_compare_{set_name}", full_lines, full)
    return out


def main(experiment, argv=None):
    global TASK, HERE, SEEDS
    p = argparse.ArgumentParser()
    p.add_argument("--through", choices=("dev", "hard", "confirm"), default="dev")
    p.add_argument("--allow-confirm", action="store_true")
    p.add_argument("--smoke-root", help="synthetic fixtures below results/step6/smoke only")
    a = p.parse_args(argv)
    check_freeze()
    if a.smoke_root:
        from gp_common import configure_smoke_root
        TASK, HERE = configure_smoke_root(a.smoke_root)
        SEEDS = {"dev": [0, 1], "hard": [0, 1], "confirm": [100, 101]}
    j = {"experiment": experiment, "dev": None, "hard": None, "confirm": None,
         "rule": "STEP6 §6; DEV success/mirror 97.5% bootstrap and Welch; all other CI 95%; new minus C3"}
    lines = [f"Experiment {experiment}: frozen pre-registered numbers and verdicts; differences are {experiment} minus C3.",
             "Two-level bootstrap with 10000 draws, seed 0; Welch by seed; DEV primary verdict/mirror 97.5%, all others 95%."]
    j["dev"] = compare(experiment, "dev", lines)
    if a.through in ("hard", "confirm"):
        if not dev_pass(j):
            raise RuntimeError("HARD gate not passed; refusing to read HARD")
        j["hard"] = compare(experiment, "hard", lines)
    if a.through == "confirm":
        if not a.allow_confirm or not hard_pass(j):
            raise RuntimeError("CONFIRM gate or --allow-confirm missing; refusing to read CONFIRM")
        opened = HERE / f"{experiment}_confirm_opened.json"
        if not opened.exists():
            raise RuntimeError("CONFIRM was not opened by the one-time opening step of the chain script")
        ar.ALLOW_CONFIRM = True
        j["confirm"] = compare(experiment, "confirm", lines)
    j["verdict"] = verdict(j, experiment)
    lines += ["", "Final/current verdict: " + j["verdict"], "The G + P combination is not tested (pre-registration limitation)."]
    write_json(HERE / f"{experiment}_mechanisms.json", j)
    (HERE / f"{experiment}_mechanisms.txt").write_text("\n".join(lines) + "\n")
    joint_report()
    print("\n".join(lines), flush=True)


def joint_report():
    """Apply the preregistered final G/P choice once both chains have verdicts."""
    from gp_common import joint_choice
    paths = [HERE / f"{e}_mechanisms.json" for e in ("G", "P")]
    if not all(p.exists() for p in paths):
        return
    g, p = [json.loads(path.read_text()) for path in paths]
    if any(j["verdict"].startswith(("NEXT", "PENDING")) for j in (g, p)):
        return
    ga, pa = g["verdict"] == "ADOPT G", p["verdict"] == "ADOPT P"
    d = None
    if ga and pa:
        G = prepare(raw_runs(MODELS["G"], "dev", SEEDS["dev"]))
        P = prepare(raw_runs(MODELS["P"], "dev", SEEDS["dev"]))
        eps, _ = ar.paired_episodes(G + P)
        # difference(A,B) is B-A: pass P first for G minus P.
        d = difference(ar.metric_matrix(P, eps, "success"), ar.metric_matrix(G, eps, "success"))
    result = {"G": g["verdict"], "P": p["verdict"], "G_minus_P_DEV": d,
              "main_model": joint_choice(ga, pa, d), "G_plus_P_tested": False,
              "rule": "both adopted: DEV paired bootstrap 95% excludes zero -> higher, else G"}
    write_json(HERE / "GP_main_model_selection.json", result)
    (HERE / "GP_main_model_selection.txt").write_text(json.dumps(ar._clean(result), indent=2, ensure_ascii=False) + "\n")
