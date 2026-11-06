"""Synthetic rollouts for exercising the real G/P analysis and sequential gates.

These are software fixtures, never experiment outcomes. All writes are confined
to a child of results/step6/smoke. No evaluation recordings or models are opened,
and no trained model is created or replaced. The chain supplies actual smoke
models separately and calls G/P_mechanisms.py against these fixture paths.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

import numpy as np

SOURCE_TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_TASK))
import analyze_rollouts as ar  # noqa: E402
from evaluate_policy import PROTOCOL, ROW_COLUMNS  # noqa: E402

OLD = "ego-novw-trig"
MODELS = {"G": "ego-novw-trig-chunk", "P": "ego-novw-trig-pd"}
CASES = ("pass", "dev_fail", "dev_risk_fail", "hard_fail", "confirm_fail")
SMOKE_SEEDS = {"dev": [0, 1], "hard": [0, 1], "confirm": [100, 101]}


def smoke_root(root):
    root = Path(root).resolve()
    boundary = SOURCE_TASK / "results" / "step6" / "smoke"
    if root == boundary or not root.is_relative_to(boundary):
        raise ValueError("Synthetic fixtures require a child of results/step6/smoke")
    return root


def write_once(path, content):
    """Permit identical resumes; reject replacing a previous fixture."""
    path = Path(path)
    payload = content.encode() if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Synthetic fixture exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def write_csv(path, columns, rows):
    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_once(path, content.getvalue())


def write_json(path, obj):
    write_once(path, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _episode_geometry(episode):
    d0 = (-.3, 0., .1, .3, .75, 1.3)[episode % 6]
    return {"d0": d0, "r": d0 + .7, "theta_deg": 20., "target_z": .3 if episode % 2 == 0 else .7,
            "standoff": .7, "target_x": d0 + .7, "target_y": 0., "init_x": 0., "init_y": 0.,
            "init_yaw_deg": 0.}


def reference_rows(n_episodes):
    result = []
    for episode in range(n_episodes):
        geometry = _episode_geometry(episode)
        row = {key: 0. for key in ar.REF_COLUMNS}
        row.update(geometry)
        row.update(episode=episode, noisy=False, expert_status="reached", z=geometry["target_z"],
                   mode="M1" if geometry["d0"] < -.08 else "M0" if abs(geometry["d0"]) < .08 else "M2",
                   align_req_deg=20., orient_req_deg=0., navdone_step=20, navdone_time=.2,
                   navdone_time_meta=.2, navdone_ee_err=.3, cov_pct_world=.5, cov_pct_rel=.5,
                   cov_nn_world=.1, cov_nn_rel=.1)
        result.append(row)
    return result


def _outcome(case, set_name, arm, episode, n_episodes):
    if arm == "expert":
        return "success"
    fraction = .3 if arm == "C3" else .9
    if arm != "C3" and case == f"{set_name}_fail":
        fraction = .1
    successful = episode < round(n_episodes * fraction)
    if successful:
        return "success"
    if arm != "C3" and case == "dev_risk_fail" and set_name == "dev":
        return "crash"
    return "extended_not_reached"


def rollout_row(episode, outcome, arm, template=None):
    """Use the harness's complete schema, replacing every decision input."""
    row = {key: 0. for key in ROW_COLUMNS}
    if template:
        row.update({key: value for key, value in template.items() if key in ROW_COLUMNS})
    for key in ar.STRING_COLUMNS:
        row[key] = ""
    geometry = _episode_geometry(episode)
    row.update(geometry)
    success = outcome == "success"
    row.update(episode=episode, success=success, settled=success,
               t_reach=.8 if success else np.nan,
               stop_reason="reached" if success else "tipped" if outcome == "crash" else "timeout",
               min_error=.005 if success else .03, final_error=.005 if success else .03,
               hold_error=.005 if success else np.nan, hold_error_max=.008 if success else np.nan,
               duration=1., n_steps=1000, base_goal_dist_start=abs(geometry["d0"]),
               base_goal_dist_min=.05, base_goal_dist_final=.08,
               base_path_length=abs(geometry["d0"]) + .4, base_turn_total_deg=45.,
               arm_move_time=.25, base_goal_dist_at_arm_move=.08, torque_saturated_frac=0.,
               wheel_cmd_max=1., policy_ms_mean=.1 if arm == "C3" else .12,
               policy_ms_p99=.15, policy_ms_max=.2, expert_status_recorded="reached",
               expert_nav_timed_out_recorded=False, expert_noisy_recorded=False,
               wall_time=0., error_message="", held_10s=success,
               hold10_stop="completed" if success else "", hold10_error_end=.005 if success else np.nan,
               hold10_error_mean=.005 if success else np.nan, hold10_error_max=.008 if success else np.nan,
               hold10_leave_time=np.nan, hold10_base_disp_max=0., hold10_yaw_change_max_deg=0.,
               hold10_wheel_cmd_max=0., stable_1cm_time=.7 if success else np.nan,
               stable_2cm_time=.6 if success else np.nan, stable_5cm_time=.5,
               stable_5cm_leave_time=np.nan, run_end_time=11. if success else 1.,
               arm_extend_time=.3, ee_base_disp_max=.3, ring_dist_start=abs(geometry["d0"]),
               ring_dist_min=.05, ring_dist_final=.08, ring_dist_at_arm_extend=.08,
               target_bearing_err_at_arm_extend_deg=0., target_bearing_err_start_deg=20.,
               target_bearing_err_final_deg=0., target_dist_start=geometry["r"], target_dist_final=.7,
               parked_time=.2, parked=True, target_bearing_err_at_park_deg=0., wheel_over_envelope_frac=0.)
    # These columns describe a comparison with a recording, which was not run.
    for key in ROW_COLUMNS:
        if key.startswith("cmp_"):
            row[key] = ""
    return row


def write_trajectory(path, row, arm):
    t = np.arange(101, dtype=float) / 100.
    amplitude = .2 if arm == "C3" else .1
    phase = np.arange(7, dtype=float)[None, :]
    target = np.array([row["target_x"], row["target_y"], row["target_z"]])
    arrays = {"t": t, "nav_done": t >= .2,
              "torque_arm": amplitude * np.sin(t[:, None] * 15 + phase),
              "base_pos": np.c_[np.minimum(t, .2), np.zeros((len(t), 2))],
              "base_ori": np.tile([0., 0., 0., 1.], (len(t), 1)),
              "q_mes": np.zeros((len(t), 17)), "qd_mes": np.zeros((len(t), 17)),
              "ee_pos": target + np.c_[np.linspace(.3, row["min_error"], len(t)), np.zeros((len(t), 2))],
              "target": target,
              "meta": np.array(json.dumps({"synthetic": True, "episode": row["episode"],
                                             "purpose": "gate/plot software smoke only"}))}
    path = Path(path)
    if path.exists():
        with np.load(path) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError(f"Synthetic trajectory exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def create_fixture(root, experiment, case="pass", set_name=None, n_episodes=100,
                   seeds=None, template_run=None, with_trajectories=True):
    """Create full synthetic sets for real gp_mechanisms.compare/apply_conditions.

    ``set_name=None`` creates all three synthetic sets; it never reads their real
    data. Reuse a root only for identical fixtures (e.g. G and P's common C3).
    ``template_run`` may name an actual two-episode pilot run; its first CSV row
    supplies extra descriptive values, while all outcome inputs are synthetic.
    Models in root/trained are supplied by the caller and remain untouched.
    """
    root = smoke_root(root)
    if experiment not in MODELS or case not in CASES:
        raise ValueError("Unknown experiment or synthetic case")
    if n_episodes != 100:
        raise ValueError("Gate smoke fixtures use complete synthetic sets of exactly 100 episodes")
    sets = (set_name,) if set_name else ("dev", "hard", "confirm")
    if any(name not in SMOKE_SEEDS for name in sets):
        raise ValueError("Synthetic set must be dev, hard or confirm")
    template = None
    if template_run:
        template_path = smoke_root(template_run) / "rollouts.csv"
        with template_path.open(newline="") as stream:
            template = next(csv.DictReader(stream))
    records = []
    for name in sets:
        selected_seeds = list(seeds) if seeds is not None else SMOKE_SEEDS[name]
        write_csv(root / "results" / "step3" / f"cache_ref_{name}.csv", ar.REF_COLUMNS,
                  reference_rows(n_episodes))
        specs = [(OLD, "C3", seed) for seed in selected_seeds]
        specs += [(MODELS[experiment], experiment, seed) for seed in selected_seeds]
        specs += [("expert", "expert", None)]
        for model, arm, seed in specs:
            stem = f"expert_v11_{name}" if seed is None else f"{model}_s{seed}_{name}"
            folder = root / "results" / stem
            rows = [rollout_row(episode, _outcome(case, name, arm, episode, n_episodes), arm, template)
                    for episode in range(n_episodes)]
            write_csv(folder / "rollouts.csv", ROW_COLUMNS, rows)
            config = {"args": {"policy": "expert" if seed is None else "policies/bc_policy.py:make_policy",
                                "policy_arg": None if seed is None else str(
                                    (root / "trained" / f"{model}_s{seed}.npz").relative_to(SOURCE_TASK)),
                                "eval_set": f"synthetic/{name}", "out_dir": str(folder.relative_to(root)),
                                "workers": 1, "no_early_stop": True, "save_traj": with_trajectories,
                                "traj_every": 10, "episodes": None, "limit": None, "name": None,
                                "compare_recorded": False, "gui": False},
                      "protocol": PROTOCOL, "eval_set_config": {"n_episodes": n_episodes, "synthetic": True},
                      "episodes": list(range(n_episodes)), "synthetic": True,
                      "purpose": "Software smoke fixture; excluded from all experiment conclusions"}
            write_json(folder / "config.json", config)
            write_once(folder / "summary.txt",
                       f"SYNTHETIC ONLY: {stem}; {sum(r['success'] for r in rows)}/{n_episodes} success.\n"
                       "Software smoke fixture; excluded from all experiment conclusions.\n")
            if with_trajectories:
                for row in rows:
                    write_trajectory(folder / "traj" / f"ep_{row['episode']:04d}.npz", row, arm)
            records.append({"set": name, "model": model, "seed": seed,
                            "folder": str(folder.relative_to(root)), "episodes": n_episodes,
                            "success": sum(row["success"] for row in rows),
                            "tipped": sum(row["stop_reason"] == "tipped" for row in rows)})
    result = {"synthetic": True, "experiment": experiment, "case": case, "runs": records,
              "real_evaluation_data_opened": False, "models_written": False,
              "expected": {"dev_condition_1": case != "dev_fail", "dev_condition_2": case != "dev_risk_fail",
                           "hard_condition_3": case != "hard_fail", "confirm_condition_4": case != "confirm_fail"}}
    suffix = f"_{set_name}" if set_name else ""
    write_json(root / "results" / "step6" / f"synthetic_fixture_{experiment}_{case}{suffix}.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--experiment", choices=tuple(MODELS), required=True)
    parser.add_argument("--case", choices=CASES, default="pass")
    parser.add_argument("--set", choices=tuple(SMOKE_SEEDS), dest="set_name")
    parser.add_argument("--template-run")
    parser.add_argument("--no-trajectories", action="store_true")
    args = parser.parse_args(argv)
    result = create_fixture(args.smoke_root, args.experiment, args.case, args.set_name,
                            template_run=args.template_run, with_trajectories=not args.no_trajectories)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
