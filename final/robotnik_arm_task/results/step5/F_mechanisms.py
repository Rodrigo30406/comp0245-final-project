"""Pre-registered experiment F comparisons and sequential gates; no post-hoc rules."""
import argparse
import json

import numpy as np

from f_common import HERE, TASK, OLD, NEW, SEEDS, ar, check_freeze, clean, merged_old, prepare, raw_runs, write_json


def difference(A, B, boot=None):
    d = ar.paired_diff(A, B, boot=boot)
    t = ar.seed_t_test(ar.seed_rates(A), ar.seed_rates(B))
    return {"a": d["mean_a"], "b": d["mean_b"], "diff": d["diff"],
            "lo": d["lo"], "hi": d["hi"], "p_boot": d["p"],
            "welch_lo": t["lo"], "welch_hi": t["hi"], "p_welch": t["p"],
            "per_seed_a": ar.seed_rates(A), "per_seed_b": ar.seed_rates(B)}


def line(name, d):
    return (f"{name}: C3 {100*d['a']:.6f}%, dag1 {100*d['b']:.6f}%; Δ {100*d['diff']:+.6f} pp; "
            f"bootstrap 95% [{100*d['lo']:+.6f}, {100*d['hi']:+.6f}], "
            f"Welch 95% [{100*d['welch_lo']:+.6f}, {100*d['welch_hi']:+.6f}] pp")


def dev_pass(j):
    d = j.get("dev") or {}
    return bool(d.get("condition_1") and d.get("condition_2"))


def hard_pass(j):
    return dev_pass(j) and bool((j.get("hard") or {}).get("condition_3"))


def verdict(j):
    d, h, c = (j.get(k) for k in ("dev", "hard", "confirm"))
    if d is None:
        return "PENDING DEV"
    if not dev_pass(j):
        return "KEEP C3"
    if h is None:
        return "NEXT HARD"
    if not h["condition_3"]:
        return "KEEP C3"
    if c is None:
        return "NEXT CONFIRM"
    return "ADOPT dag1" if c["condition_4"] else "KEEP C3"


def compare(set_name, lines):
    A_raw = raw_runs(OLD, set_name, SEEDS[set_name])
    A, notes = merged_old(A_raw)
    A = prepare(A)
    B = prepare(raw_runs(NEW, set_name, SEEDS[set_name]))
    if any(r.meta.get("no_early_stop") is not True for r in B):
        raise RuntimeError("all new F evaluations require --no-early-stop")
    if any(r["stop_reason"] in ("harness_error", "policy_error")
           for run in A + B for r in run.rows):
        raise RuntimeError("harness_error/policy_error prevents a decision")
    ar.check_integrity(A + B, f"experiment F {set_name}")
    eps, _ = ar.paired_episodes(A + B)
    boot = ar.Bootstrap([len(A), len(B)], len(eps), ar.N_BOOT, ar.BOOT_SEED)
    out = {"n_episodes": len(eps), "seeds": SEEDS[set_name], "replacement_notes": notes,
           "runs_a": [r.path for r in A], "runs_b": [r.path for r in B]}
    lines += ["", set_name.upper(), *notes]
    for metric in ("success", "reach_2cm", "reach_5cm", "held_10s", "arm_extended_toward"):
        out[metric] = difference(ar.metric_matrix(A, eps, metric), ar.metric_matrix(B, eps, metric), boot)
        lines.append(line(metric, out[metric]))
    for name, classes in (("M1_extended_not_reached", ("extended_not_reached",)),
                          ("M2_circling", ("circling",)),
                          ("M4_risk", ("crash", "left_or_overshot"))):
        fn = lambda r: float(r["fclass"] in classes)
        out[name] = difference(ar.matrix(A, eps, fn), ar.matrix(B, eps, fn), boot)
        lines.append(line(name, out[name]))
    d = out["success"]
    if set_name == "dev":
        out["condition_1"] = bool(d["lo"] > 0 and d["welch_lo"] > 0)
        out["condition_2"] = bool(out["M4_risk"]["diff"] <= .03)
        m1 = out["M1_extended_not_reached"]
        out["M1_pass"] = bool(m1["hi"] < 0 and m1["welch_hi"] < 0)
        out["M2_pass_descriptive"] = bool(out["M2_circling"]["diff"] < 0)
        out["M3_pass_descriptive"] = bool(out["held_10s"]["diff"] > 0)
        out["M4_pass"] = out["condition_2"]
        out["mirror_worse"] = bool(d["hi"] < 0 and d["welch_hi"] < 0)
        low = [e for e in eps if float(A[0].by_episode[e]["target_z"]) < .45]
        out["low_z_success"] = difference(ar.metric_matrix(A, low, "success"), ar.metric_matrix(B, low, "success"))
        out["low_z_success"]["n_episodes"] = len(low)
        lines.append(line("target z < 0.45 m success（描述性）", out["low_z_success"]))
        losses = []
        for model in (OLD, NEW):
            a = []
            for s in SEEDS["dev"]:
                with np.load(TASK / "trained" / f"{model}_s{s}.npz") as z:
                    a.append(json.loads(str(z["meta"]))["val_loss"])
            losses.append(np.array(a))
        lt = ar.seed_t_test(*losses)
        lb = ar.paired_diff(losses[0][:, None], losses[1][:, None])
        out["expert_validation_loss"] = {"per_seed_a": losses[0], "per_seed_b": losses[1],
            "a": losses[0].mean(), "b": losses[1].mean(), "diff": lt["diff"],
            "welch_lo": lt["lo"], "welch_hi": lt["hi"], "p_welch": lt["p"],
            "lo": lb["lo"], "hi": lb["hi"], "p_boot": lb["p"],
            "bootstrap_unit": "seed; one aggregate expert-validation loss per model"}
        names = ["reach_2cm", "reach_5cm", "held_10s", "arm_extended_toward", "low_z_success", "expert_validation_loss"]
        pvals = [ar.combined_p(out[n]["p_boot"], out[n]["p_welch"]) for n in names]
        out["secondary_holm"] = {"names": names, "combined_p": pvals, "holm": ar.holm(pvals),
                                  "descriptive_only": True}
        lines.append("专家验证集损失（标准化 MSE；C2 检查）：" + json.dumps(clean(out["expert_validation_loss"]), ensure_ascii=False))
        lines.append("次要量 Holm（6 项，描述性）：" + json.dumps(clean(out["secondary_holm"]), ensure_ascii=False))
        for k in ("condition_1", "condition_2", "M1_pass", "M2_pass_descriptive", "M3_pass_descriptive", "M4_pass", "mirror_worse"):
            lines.append(f"{k}: {out[k]}")
        if out["mirror_worse"]:
            lines.append("镜像判定：DAgger 一轮使 success 变差。")
        # Report the unchanged original early-stop controls as sensitivity, as
        # in D; the primary decision always uses the replaced rows above.
        out["early_stop_sensitivity"] = difference(ar.metric_matrix(prepare(A_raw), eps, "success"),
                                                    ar.metric_matrix(B, eps, "success"), boot)
        lines.append(line("C3 原早停行敏感性（不参与判定）", out["early_stop_sensitivity"]))
    elif set_name == "hard":
        out["condition_3"] = bool(d["diff"] > 0)
        lines.append(f"condition_3: {out['condition_3']}")
    else:
        out["condition_4"] = bool(d["diff"] > 0 and d["lo"] > 0)
        lines.append(f"condition_4: {out['condition_4']}")
    expert = ar.load_prepared(str(TASK / "results" / f"expert_v11_{set_name}"))
    full_lines, full = ar.compare_arms(A, B, "C3", "dag1", ("extended_not_reached",), expert_run=expert)
    ar.write_outputs(str(HERE), f"F_MM_compare_{set_name}", full_lines, full)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--through", choices=("dev", "hard", "confirm"), default="dev")
    p.add_argument("--allow-confirm", action="store_true")
    a = p.parse_args(argv)
    check_freeze()
    j = {"dev": None, "hard": None, "confirm": None,
         "rule": "STEP5_DAGGER_PLAN_PROMPT.md section 3; success only; differences dag1 minus C3"}
    lines = ["实验 F：冻结预注册的数字与判定；差值均为 dag1 减 C3。",
             "两层 bootstrap 10000 次、seed 0；Welch 按 seed；95% CI。"]
    j["dev"] = compare("dev", lines)
    if a.through in ("hard", "confirm"):
        if not dev_pass(j):
            raise RuntimeError("HARD analysis refused: DEV gate failed")
        j["hard"] = compare("hard", lines)
    if a.through == "confirm":
        if not a.allow_confirm or not hard_pass(j):
            raise RuntimeError("CONFIRM analysis refused: HARD gate or --allow-confirm missing")
        ar.ALLOW_CONFIRM = True
        j["confirm"] = compare("confirm", lines)
    j["verdict"] = verdict(j)
    lines += ["", "最终／当前判定：" + j["verdict"],
              "局限（预注册）：数据行数、Adam 步数和早停时机同时变化；本实验不能分离 C3（只是数据更多）。"]
    write_json(HERE / "F_mechanisms.json", j)
    (HERE / "F_mechanisms.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
