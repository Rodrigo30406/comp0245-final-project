"""fig_step3_C3_arm.{png,pdf}: DEV success and arm_extended_toward of B-EM, C3-EM, MM, C3-MM (seed mean, two-level
bootstrap 95% CI) and the expert (single run). Same cells and computations as C3_mechanisms.figure() (frozen); only the
value labels are placed above the CIs so they do not overlap the error bars."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import C3_mechanisms as cm  # noqa: E402

ar = cm.ar


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cells = [("B-EM\nego-novw", cm.run_dirs(cm.OLD, "dev", "EM")), ("C3-EM\nego-novw-trig", cm.run_dirs(cm.NEW, "dev", "EM")),
             ("MM\nego-novw", cm.run_dirs(cm.OLD, "dev")), ("C3-MM\nego-novw-trig", cm.run_dirs(cm.NEW, "dev")),
             ("expert", [ar.run_dir_for(cm.RES, "expert", None, "dev")])]
    runs = {n: cm.load(p) for n, p in cells if cm.have(p)}
    eps, _ = ar.paired_episodes([r for rs in runs.values() for r in rs])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric in zip(axes, ("success", "arm_extended_toward")):
        for i, (n, rs) in enumerate(runs.items()):
            Y = ar.metric_matrix(rs, eps, metric)
            top = 100 * Y.mean()
            if len(rs) > 1:
                b = ar.boot2(Y)
                ax.bar(i, 100 * b["mean"], color="#4C78A8")
                ax.errorbar(i, 100 * b["mean"], yerr=[[100 * (b["mean"] - b["lo"])], [100 * (b["hi"] - b["mean"])]],
                            color="k", capsize=3)
                top = 100 * b["hi"]
            else:
                ax.bar(i, 100 * Y.mean(), color="#9D9D9D")
            ax.text(i, top + 2, f"{100 * Y.mean():.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(list(runs), fontsize=7)
        ax.set_ylim(0, 110)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_title(f"DEV {metric} (seed mean, 2-level bootstrap 95% CI)", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ar.STEP3_DIR, f"fig_step3_C3_arm.{ext}"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
