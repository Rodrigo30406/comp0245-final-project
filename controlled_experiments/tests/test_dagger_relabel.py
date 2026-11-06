"""Required F1 verification on five clean and five noisy training episodes.

Run from the task directory:
  MPLCONFIGDIR=/tmp/mplcfg pixi run --manifest-path $ROBOENV_DIR/pixi.toml \
      python -m unittest discover -s tests -p test_dagger_relabel.py -v

Amendment 1 uses 1e-9 for float64 harness trajectories and, for the
float32 recordings, 1e-4 rad/s for wheels and 1e-3 N*m for torques.
Optional DAGGER_F1_DIAGNOSTICS names a new JSON file for the per-episode errors.
"""
import datetime
import csv
import io
import json
import os
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TASK_DIR)

import numpy as np  # noqa: E402

import dagger_relabel as dr  # noqa: E402
import train_policy as tp  # noqa: E402
from evaluate_policy import ARM_MOVE_ANGLE, load_eval_set  # noqa: E402


TOLERANCES = {"wheel_cmds": 1e-4, "torque_arm": 1e-3}


class TestRecordedExpertReplay(unittest.TestCase):
    def test_ten_training_episodes(self):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        split = tp.load_split()
        eps = split["train_episodes"]
        clean = [e for e in eps if not split["episodes"][str(e)]["noisy"]][:5]
        noisy = [e for e in eps if split["episodes"][str(e)]["noisy"]][:5]
        self.assertEqual((len(clean), len(noisy)), (5, 5))
        self.assertFalse(set(clean + noisy) & set(split["val_episodes"]))
        with redirect_stdout(io.StringIO()):
            episodes, cfg = load_eval_set(tp.TRAIN_DIR)
        self.assertEqual(cfg["log_every"], 1)
        by = {ep["episode"]: ep for ep in episodes}
        rows = []
        for e in clean + noisy:
            episode = by[e]
            path = os.path.join(tp.TRAIN_DIR, split["episodes"][str(e)]["file"])
            self.assertEqual(os.path.realpath(path), os.path.realpath(episode["recorded_file"]))
            fields = (*dr.OBS_FIELDS, "target", "meta", "wheel_cmds", "torque_arm", "nav_done", "q_des_arm",
                      "wheel_cmds_exec", "torque_arm_exec")
            with np.load(path) as z:
                traj = {name: z[name] for name in fields}
            meta = json.loads(str(traj["meta"]))
            self.assertEqual(meta["log_every"], 1)
            self.assertEqual(len(traj["t"]), split["episodes"][str(e)]["n_steps"])
            np.testing.assert_array_equal(episode["target"], traj["target"])
            setup = dr.setup_simulation
            init = {}

            def capture_setup(*args, **kwargs):
                result = setup(*args, **kwargs)
                init["q_mes"] = result[3]["init_q_mes"].copy()
                return result

            with redirect_stdout(io.StringIO()), mock.patch.object(dr, "setup_simulation", side_effect=capture_setup):
                shadow = dr.replay_expert(traj, episode)
            row = {"episode": e, "noisy": bool(split["episodes"][str(e)]["noisy"]),
                   "n_steps": len(traj["t"]), "trajectory_sha256": tp.sha256_file(path),
                   "dtypes": {name: str(traj[name].dtype) for name in dr.OBS_FIELDS + ("wheel_cmds", "torque_arm")},
                   "init_q_mes_max_abs_error": float(np.max(np.abs(init["q_mes"] - traj["q_mes"][0]))),
                   "nav_done_mismatched_steps": int(np.count_nonzero(shadow["nav_done"] != traj["nav_done"])),
                   "q_des_arm_max_abs_error": float(np.max(np.abs(shadow["q_des_arm"] - traj["q_des_arm"])))}
            for name in ("wheel_cmds", "torque_arm"):
                error = np.abs(shadow[name] - traj[name].astype(float))
                step, joint = np.unravel_index(np.argmax(error), error.shape)
                row[name] = {"max_abs_error": float(error[step, joint]), "step": int(step), "joint": int(joint),
                             "shadow": float(shadow[name][step, joint]), "recorded": float(traj[name][step, joint]),
                             "step0_max_abs_error": float(error[0].max()),
                             "shadow_float32_rounding_max": float(np.max(np.abs(
                                 shadow[name] - shadow[name].astype(np.float32).astype(float)))),
                             "exec_minus_label_max_abs": float(np.max(np.abs(
                                 traj[name + "_exec"].astype(float) - traj[name].astype(float))))}
            row["passed"] = (all(row[name]["max_abs_error"] <= TOLERANCES[name] for name in TOLERANCES)
                             and row["nav_done_mismatched_steps"] == 0)
            rows.append(row)
            print(f"F1 ep {e:04d} noisy={row['noisy']} steps={row['n_steps']}: "
                  f"wheels {row['wheel_cmds']['max_abs_error']:.17g}, "
                  f"torques {row['torque_arm']['max_abs_error']:.17g}, "
                  f"nav_done mismatches {row['nav_done_mismatched_steps']}", flush=True)
        report = {"started": started, "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "absolute_tolerances": TOLERANCES, "relative_tolerance": 0, "amendment": 1,
                  "selected_clean": clean, "selected_noisy": noisy, "episodes": rows,
                  "passed": all(row["passed"] for row in rows),
                  "split_sha256": tp.sha256_file(tp.SPLIT_PATH),
                  "code_sha256": {name: tp.sha256_file(os.path.join(TASK_DIR, name))
                                  for name in ("dagger_relabel.py", "tests/test_dagger_relabel.py",
                                               "evaluate_policy.py", "robotnik_arm_control.py", "collect_data.py")}}
        output = os.environ.get("DAGGER_F1_DIAGNOSTICS")
        if output:
            with open(output, "x") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
        for row in rows:
            with self.subTest(episode=row["episode"], noisy=row["noisy"]):
                for name in ("wheel_cmds", "torque_arm"):
                    self.assertLessEqual(row[name]["max_abs_error"], TOLERANCES[name], f"{name}: {row[name]}")
                self.assertEqual(row["nav_done_mismatched_steps"], 0)


class TestHarnessExpertReplay(unittest.TestCase):
    def test_five_float64_pilot_episodes(self):
        folder = os.path.join(TASK_DIR, "results", "step5", "F1_expert_pilot")
        if not os.path.isfile(os.path.join(folder, "summary.txt")):
            self.skipTest("amendment 1 pilot trajectories have not been generated")
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        episodes, _ = load_eval_set(os.path.join(TASK_DIR, "data", "pilot"))
        with open(os.path.join(folder, "rollouts.csv")) as f:
            rollouts = {int(r["episode"]): r for r in csv.DictReader(f)}
        self.assertEqual(sorted(rollouts), [e["episode"] for e in episodes[:5]])
        rows = []
        for episode in episodes[:5]:
            e = episode["episode"]
            path = os.path.join(folder, "traj", f"ep_{e:04d}.npz")
            with np.load(path) as z:
                traj = {name: z[name] for name in z.files}
            for name in (*dr.OBS_FIELDS, "wheel_cmds", "torque_arm"):
                self.assertEqual(traj[name].dtype, np.dtype("float64"), name)
            with redirect_stdout(io.StringIO()):
                shadow = dr.replay_expert(traj, episode)
            done = np.flatnonzero(shadow["nav_done"])
            first_done = float(traj["t"][done[0]]) if len(done) else None
            arm_move = float(rollouts[e]["arm_move_time"])
            # arm_move_time is measured displacement, not the reach trigger.
            # Gravity and settling can cross its threshold during navigation.
            # Recompute that diagnostic and verify that the shadow reference
            # still holds the initial posture before its first nav_done step.
            moved = np.flatnonzero(np.max(np.abs(traj["q_mes"][:, 4:11] - traj["q_mes"][0, 4:11]), axis=1)
                                   > ARM_MOVE_ANGLE)
            measured_move = float(traj["t"][moved[0]]) if len(moved) else np.nan
            same_move = (np.isnan(measured_move) and np.isnan(arm_move)) or measured_move == arm_move
            pre = shadow["q_des_arm"][:done[0]] if len(done) else shadow["q_des_arm"]
            hold_error = float(np.max(np.abs(pre - traj["q_mes"][0, 4:11]))) if len(pre) else 0.0
            consistent = bool(same_move and hold_error <= 1e-9
                              and not np.any(np.diff(shadow["nav_done"].astype(int)) < 0))
            row = {"episode": e, "n_steps": len(traj["t"]),
                   "trajectory_sha256": tp.sha256_file(path),
                   "dtypes": {n: str(traj[n].dtype) for n in traj},
                   "first_nav_done": first_done, "arm_move_time": arm_move,
                   "recomputed_arm_move_time": measured_move, "pre_nav_reference_max_abs_error": hold_error,
                   "nav_done_arm_move_consistent": consistent,
                   "stop_reason": rollouts[e]["stop_reason"]}
            for name in ("wheel_cmds", "torque_arm"):
                row[name] = {"max_abs_error": float(np.max(np.abs(shadow[name] - traj[name])))}
            row["passed"] = consistent and all(row[n]["max_abs_error"] <= 1e-9
                                               for n in ("wheel_cmds", "torque_arm"))
            rows.append(row)
            print(f"F1(a) ep {e:04d}: {row}", flush=True)
        report = {"amendment": 1, "started": started,
                  "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  "absolute_tolerance": 1e-9, "relative_tolerance": 0,
                  "nav_done_check": "nav_done is not saved. Recompute measured arm_move_time exactly; verify "
                                    "the initial reference is held before first shadow nav_done. Measured "
                                    "settling during navigation is not a reach trigger.",
                  "episodes": rows, "passed": all(r["passed"] for r in rows)}
        output = os.environ.get("DAGGER_F1_PILOT_DIAGNOSTICS")
        if output:
            with open(output, "x") as f:
                json.dump(report, f, indent=2)
                f.write("\n")
        for row in rows:
            with self.subTest(episode=row["episode"]):
                self.assertTrue(row["nav_done_arm_move_consistent"], row)
                self.assertNotIn(row["stop_reason"], ("policy_error", "harness_error"))
                for name in ("wheel_cmds", "torque_arm"):
                    self.assertLessEqual(row[name]["max_abs_error"], 1e-9)


if __name__ == "__main__":
    unittest.main()
