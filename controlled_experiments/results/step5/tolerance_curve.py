"""F0: descriptive tolerance curves using existing rollout rows only."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from f_common import HERE, TASK, ar, check_freeze, merged_old, raw_runs, write_json


def main():
    check_freeze()
    # F0 is explicitly allowed to read existing CONFIRM results. No evaluation
    # data or recorded trajectory is opened and no reference is regenerated.
    ar.ALLOW_CONFIRM = True
    table = np.array([5, 7.5, 10, 15, 20, 30, 50, 100], dtype=float)
    grid = np.unique(np.r_[np.geomspace(2, 100, 121), table])
    out = {"amendment": 2,
           "row_rule": "Main curves: original rows for all sets. DEV only: additional stalled-replaced "
                       "C3 sensitivity curve, as in D. No HARD/CONFIRM replacements or new simulations in F0.",
           "definition": "P(min_error < tau); min_error is frozen at the protocol v1 stop point "
                         "(one second after first reach), not the subsequent extended hold. "
                         "Unreached rows cover their whole run. Existing CONFIRM results only.",
           "bootstrap": {"draws": ar.N_BOOT, "seed": ar.BOOT_SEED, "method": "analyze_rollouts.boot2"},
           "sets": {}}
    lines = [out["definition"], out["row_rule"],
             "Amendment 2: published original DEV C3 60.5%; stalled-replaced sensitivity 60.8% "
             "is already recorded in report.txt C3 result item 2. This explains the prior F0 mismatch.",
             "Each entry is seed mean [two-level bootstrap 95% CI], percent."]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, constrained_layout=True)
    expected = {"dev": (60.5, 0.3), "hard": (56.3, 0.4), "confirm": (64.3, 0.3)}
    for ax, set_name in zip(axes, ("dev", "hard", "confirm")):
        seeds = list(range(100, 105)) if set_name == "confirm" else list(range(5))
        c3 = raw_runs("ego-novw-trig", set_name, seeds)
        notes = ["Main curves use original rollout rows, without stalled replacement."]
        expert = ar.load_run(str(TASK / "results" / f"expert_v11_{set_name}"))
        arms = {"expert": [expert], "ego-novw": raw_runs("ego-novw", set_name, seeds), "ego-novw-trig": c3}
        if set_name == "dev":
            sensitivity, replacement_notes = merged_old(c3)
            arms["ego-novw-trig (DEV stalled sensitivity)"] = sensitivity
            notes += replacement_notes
        out["sets"][set_name] = {"replacement_notes": notes, "models": {}}
        lines += ["", set_name.upper(), *notes, "model / tau_mm: " + " ".join(f"{v:g}" for v in table)]
        for model, runs in arms.items():
            ar.check_integrity(runs, f"F0 {set_name} {model}")
            eps = sorted(runs[0].by_episode)
            errors = ar.matrix(runs, eps, lambda r: ar._f(r, "min_error"))
            points = []
            for tau in grid:
                Y = (errors < tau / 1000).astype(float)
                b = ar.boot2(Y)
                points.append({"tau_mm": tau, **{k: b[k] for k in ("mean", "lo", "hi")}})
            checks = {}
            for tau, metric in ((10, "success"), (20, "reach_2cm"), (50, "reach_5cm")):
                checks[metric] = bool(np.array_equal((errors < tau / 1000).astype(float),
                                                    ar.metric_matrix(runs, eps, metric)))
            rate10 = float((errors < .01).mean())
            if model != "expert":
                sensitive = model.endswith("sensitivity)")
                want = 60.8 if sensitive else expected[set_name][0 if model == "ego-novw-trig" else 1]
                key = "report_sensitivity_10mm_rounded" if sensitive else "published_10mm_rounded"
                checks[key] = round(100 * rate10, 1) == want
            if not all(checks.values()):
                raise RuntimeError(f"F0 consistency mismatch: {set_name} {model}: {checks}")
            out["sets"][set_name]["models"][model] = {"n_seeds": len(runs), "n_episodes": len(eps),
                                                       "checks": checks, "curve": points,
                                                       "runs": [r.path for r in runs]}
            tab = [p for p in points if p["tau_mm"] in table]
            lines.append(model + ": " + "; ".join(f"{p['tau_mm']:g}: {100*p['mean']:.3f} "
                                                   f"[{100*p['lo']:.3f}, {100*p['hi']:.3f}]" for p in tab))
            lines.append(f"  consistency: {checks}")
            x = [p["tau_mm"] for p in points]
            ax.plot(x, [100*p["mean"] for p in points], label=model,
                    linestyle="--" if model.endswith("sensitivity)") else "-")
            ax.fill_between(x, [100*p["lo"] for p in points], [100*p["hi"] for p in points], alpha=.12)
        ax.set(xscale="log", xlabel="Tolerance (mm)", title=set_name.upper(), ylim=(0, 101))
        ax.axvline(10, color="grey", linestyle=":", linewidth=.8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("P(min_error < tolerance) (%)")
    axes[0].legend(fontsize=6, loc="lower right")
    axes[-1].legend(fontsize=8, loc="lower right")
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"fig_tolerance.{ext}", dpi=180)
    plt.close(fig)
    write_json(HERE / "tolerance_curve.json", out)
    (HERE / "tolerance_curve.txt").write_text("\n".join(lines) + "\n")
    print("F0 consistency: PASS", flush=True)


if __name__ == "__main__":
    main()
