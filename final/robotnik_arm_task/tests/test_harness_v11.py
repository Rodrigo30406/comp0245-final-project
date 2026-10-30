"""Unit tests of the evaluation harness, protocol v1.1 (evaluate_policy.py) and of tests/check_v11_against_v1.py.

  - metric helpers on synthetic sequences (no simulator): dwell detection (an interval of exactly 0.5 s, one step
    short, broken by one step, ending after the 40 s horizon), stable arrival and leaving, base-frame EE position
    and displacement (invariant to a planar rotation + translation of the whole robot, full quaternion), parking
    ring distance and target bearing, parking detection, extended-hold bookkeeping;
  - the protocol constants, the columns (v1 columns first and unchanged), summarize();
  - the check script on synthetic CSV files (complete and partial runs, duplicated rows, protocol versions,
    error_message, --require-compare);
  - a few short closed-loop episodes in this process (about 20 s): the expert on DEV episode 11 with an error
    injected in the extended hold (v1 columns identical to results/expert_dev, hold10_stop policy_error /
    invalid_action), and the zero policy on DEV episode 0;
  - RUN_SLOW=1: the expert on DEV episodes 67 (reached, not settled) and 169 (stalled) in fresh processes through
    run_policy_episode, v1 columns checked against results/expert_dev/rollouts.csv (about 45 s).

Run from the task directory:
  python -m unittest tests/test_harness_v11.py -v
  RUN_SLOW=1 python -m unittest tests/test_harness_v11.py -v
"""
import csv
import importlib.util
import io
import json
import multiprocessing as mp
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import numpy as np  # noqa: E402

import evaluate_policy as ev  # noqa: E402

_spec = importlib.util.spec_from_file_location("check_v11_against_v1", os.path.join(TESTS_DIR, "check_v11_against_v1.py"))
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)

DT = 0.001
DEV_DIR = os.path.join(TASK_DIR, "data", "dev")
EXPERT_DEV_CSV = os.path.join(TASK_DIR, "results", "expert_dev", "rollouts.csv")
RUN_SLOW = os.environ.get("RUN_SLOW", "") == "1"

# Columns of rollouts.csv in protocol v1 (results/expert_dev), in their order
V1_COLUMNS = [
    "episode", "success", "settled", "t_reach", "stop_reason", "min_error", "hold_error", "hold_error_max",
    "final_error", "duration", "n_steps", "r", "theta_deg", "target_z", "d0", "standoff", "base_goal_dist_start",
    "base_goal_dist_min", "base_goal_dist_final", "base_path_length", "base_turn_total_deg", "arm_move_time",
    "base_goal_dist_at_arm_move", "torque_saturated_frac", "wheel_cmd_max", "policy_ms_mean", "policy_ms_p99",
    "policy_ms_max", "target_x", "target_y", "init_x", "init_y", "init_yaw_deg", "expert_status_recorded",
    "expert_nav_timed_out_recorded", "expert_noisy_recorded", "cmp_steps", "cmp_identical", "cmp_first_diff_step",
    "cmp_max_base_pos", "cmp_max_q_mes", "cmp_max_qd_mes", "cmp_max_wheel_cmds", "cmp_max_torque_arm", "wall_time",
    "error_message",
]
# New columns of protocol v1.1, as specified (shared naming conventions of step 3)
SPEC_V11_COLUMNS = [
    "hold10_stop", "held_10s", "hold10_error_end", "hold10_error_mean", "hold10_error_max", "hold10_leave_time",
    "hold10_base_disp_max", "hold10_yaw_change_max_deg", "hold10_wheel_cmd_max", "stable_1cm_time",
    "stable_2cm_time", "stable_5cm_time", "stable_5cm_leave_time", "run_end_time", "arm_extend_time",
    "ee_base_disp_max", "ring_dist_start", "ring_dist_min", "ring_dist_final", "ring_dist_at_arm_extend",
    "target_bearing_err_at_arm_extend_deg", "target_bearing_err_start_deg", "target_bearing_err_final_deg",
    "target_dist_start", "target_dist_final", "parked_time", "parked", "target_bearing_err_at_park_deg",
    "wheel_over_envelope_frac",
]


def quat_xyzw(roll, pitch, yaw):
    """Quaternion (x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll) (the simulator's convention)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy])


def quat_mul_xyzw(a, b):
    """Hamilton product a * b of quaternions (x, y, z, w)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def planar_motion(phi, tx, ty):
    """Rotation about the world z axis by phi followed by a translation (tx, ty): (point map, quaternion map)."""
    c, s = np.cos(phi), np.sin(phi)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    q_rot = np.array([0.0, 0.0, np.sin(phi / 2), np.cos(phi / 2)])
    return (lambda x: rot @ np.asarray(x, dtype=float) + np.array([tx, ty, 0.0]),
            lambda q: quat_mul_xyzw(q_rot, q))


def run_dwell(conditions, min_steps):
    detector = ev.DwellDetector(min_steps)
    fired = [k for k, c in enumerate(conditions) if detector.update(k, c)]
    return detector.end_step, fired


def as_csv_strings(row):
    """A result row as the strings evaluate_policy.py writes into rollouts.csv (same csv.DictWriter call)."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=ev.ROW_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerow(row)
    buffer.seek(0)
    return next(csv.DictReader(buffer))


def expert_dev_rows():
    with open(EXPERT_DEV_CSV, newline="") as f:
        return {int(row["episode"]): row for row in csv.DictReader(f)}


def v1_differences(row_strings, reference, skip=()):
    """[(column, reference, value)] of the v1 columns (except the excluded ones) that differ, by the check script's
    rule (error_message may change from empty to 'extended hold: ...')."""
    return [(c, reference[c], row_strings[c]) for c in V1_COLUMNS
            if c not in check.EXCLUDED and c not in skip and not check.values_equal(c, reference[c], row_strings[c])]


# ----------------------------------------------------------------------------------------------------------------
class TestDwellDetector(unittest.TestCase):
    MIN = round(ev.STABLE_TIME / DT)  # 500 steps = 0.5 s

    def test_interval_of_exactly_min_duration(self):
        # steps 100..600 true: (600 - 100) steps = 0.5 s exactly -> found, END at step 600
        cond = [False] * 100 + [True] * (self.MIN + 1) + [False] * 50
        end, fired = run_dwell(cond, self.MIN)
        self.assertEqual(end, 100 + self.MIN)
        self.assertEqual(fired, [600])
        self.assertAlmostEqual(end * DT, 0.6, places=12)

    def test_one_step_short(self):
        cond = [False] * 100 + [True] * self.MIN + [False] * 50  # 499 steps = 0.499 s
        self.assertIsNone(run_dwell(cond, self.MIN)[0])

    def test_true_from_the_first_step(self):
        self.assertEqual(run_dwell([True] * 2000, self.MIN)[0], self.MIN)

    def test_broken_by_one_step(self):
        cond = [True] * 400 + [False] + [True] * 400  # 0.4 s + one step + 0.4 s: no interval of 0.5 s
        self.assertIsNone(run_dwell(cond, self.MIN)[0])
        cond = [True] * 400 + [False] + [True] * (self.MIN + 1)  # restarts at step 401
        self.assertEqual(run_dwell(cond, self.MIN)[0], 401 + self.MIN)

    def test_first_interval_is_reported_once(self):
        cond = [True] * 600 + [False] * 10 + [True] * 2000
        end, fired = run_dwell(cond, self.MIN)
        self.assertEqual(end, self.MIN)
        self.assertEqual(fired, [self.MIN])

    def test_never_true(self):
        self.assertIsNone(run_dwell([False] * 1000, self.MIN)[0])


class TestStableArrival(unittest.TestCase):
    def run_errors(self, errors, tol):
        s = ev.StableArrival(tol, DT)
        for k, e in enumerate(errors):
            s.update(k, e)
        return s.times()

    def test_tolerances_and_leaving(self):
        # error 0.10 m for 1 s, 0.03 m for 1 s, 0.005 m for 1 s, then 0.08 m for 0.5 s
        errors = [0.10] * 1000 + [0.03] * 1000 + [0.005] * 1000 + [0.08] * 500
        np.testing.assert_allclose(self.run_errors(errors, 0.05), (1.5, 3.0), atol=1e-12)  # leaves at 3.0 s
        self.assertAlmostEqual(self.run_errors(errors, 0.01)[0], 2.5, places=12)            # within 1 cm from 2 s
        self.assertTrue(np.isnan(self.run_errors(errors[:2400], 0.01)[1]))  # never leaves within the run

    def test_boundary_counts_as_within(self):
        # <= tol is within, the leave test is > tol
        errors = [0.05] * 600 + [0.05] * 100 + [0.0500001]
        stable, leave = self.run_errors(errors, 0.05)
        self.assertAlmostEqual(stable, 0.5, places=12)
        self.assertAlmostEqual(leave, 0.7, places=12)

    def test_horizon(self):
        h = round(ev.STABLE_HORIZON / DT)  # step 40000
        start = h - round(ev.STABLE_TIME / DT)
        # within tolerance from start: the interval ends exactly at t = 40 s -> counted
        errors = np.full(h + 2000, 0.2)
        errors[start:] = 0.0
        self.assertAlmostEqual(self.run_errors(errors, 0.01)[0], 40.0, places=9)
        # starting one step later, it ends at 40.001 s -> not counted, and nothing after it either
        errors = np.full(h + 2000, 0.2)
        errors[start + 1:] = 0.0
        stable, leave = self.run_errors(errors, 0.01)
        self.assertTrue(np.isnan(stable) and np.isnan(leave))

    def test_stable_columns(self):
        stable = [ev.StableArrival(tol, DT) for tol in ev.STABLE_TOLS]
        for k, e in enumerate([0.04] * 700 + [0.06] * 10):
            for s in stable:
                s.update(k, e)
        cols = ev.stable_columns(stable)
        self.assertEqual(set(cols), {"stable_1cm_time", "stable_2cm_time", "stable_5cm_time", "stable_5cm_leave_time"})
        self.assertAlmostEqual(cols["stable_5cm_time"], 0.5, places=12)
        self.assertAlmostEqual(cols["stable_5cm_leave_time"], 0.7, places=12)
        self.assertTrue(np.isnan(cols["stable_1cm_time"]) and np.isnan(cols["stable_2cm_time"]))


class TestFrames(unittest.TestCase):
    def test_known_pose(self):
        # base at (1, 2, 0.1) facing +y: a point 1 m ahead and 0.5 m up is (1, 0, 0.5) in the base frame
        p = ev.ee_in_base_frame([1.0, 3.0, 0.6], [1.0, 2.0, 0.1], quat_xyzw(0.0, 0.0, np.pi / 2))
        np.testing.assert_allclose(p, [1.0, 0.0, 0.5], atol=1e-12)

    def test_quaternion_order_matches_scipy(self):
        from scipy.spatial.transform import Rotation  # scalar-last (x, y, z, w) like the simulator
        rng = np.random.default_rng(0)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            d = rng.normal(size=3)
            expected = Rotation.from_quat(q).inv().apply(d)
            np.testing.assert_allclose(ev.ee_in_base_frame(d, np.zeros(3), q), expected, atol=1e-12)

    def test_full_quaternion_is_used(self):
        # a pitched base sees the same world point at a different base-frame position than a level one
        level = ev.ee_in_base_frame([0.5, 0.0, 1.0], np.zeros(3), quat_xyzw(0.0, 0.0, 0.3))
        pitched = ev.ee_in_base_frame([0.5, 0.0, 1.0], np.zeros(3), quat_xyzw(0.0, 0.1, 0.3))
        self.assertGreater(np.linalg.norm(level - pitched), 0.05)

    def test_invariant_to_planar_rigid_motion(self):
        rng = np.random.default_rng(1)
        for _ in range(50):
            base = np.array([*rng.uniform(-3, 3, 2), 0.1])
            q = quat_xyzw(*rng.uniform(-0.05, 0.05, 2), rng.uniform(-np.pi, np.pi))
            ee = base + rng.uniform(-1, 1, 3)
            move_point, move_quat = planar_motion(rng.uniform(-np.pi, np.pi), *rng.uniform(-5, 5, 2))
            np.testing.assert_allclose(ev.ee_in_base_frame(move_point(ee), move_point(base), move_quat(q)),
                                       ev.ee_in_base_frame(ee, base, q), atol=1e-12)


def synthetic_episode(n=6000, seed=0):
    """A base that turns towards the target, drives onto the parking ring, stops, then the arm extends.

    Returns (target, standoff, list of per-step states (ee_pos, base_pos, base_ori, yaw, lin_vel, yaw_rate),
    wheel commands); the EE is fixed in the base frame until step 4000, then moves 0.2 m forward over 1 s."""
    rng = np.random.default_rng(seed)
    target = np.array([2.0, 1.0, 0.5])
    standoff = float(np.interp(target[2], ev.STANDOFF_Z_POINTS, ev.STANDOFF_POINTS))
    to_target = np.arctan2(target[1], target[0])
    goal = target[:2] - standoff * np.array([np.cos(to_target), np.sin(to_target)])
    states, commands = [], []
    for k in range(n):
        t = k * DT
        if k < 1000:    # turn in place from yaw 0 to the target bearing
            xy, yaw = np.zeros(2), to_target * k / 1000
            vel, rate = np.zeros(2), to_target
        elif k < 2500:  # drive to the goal on the ring
            xy, yaw = goal * (k - 1000) / 1500, to_target
            vel, rate = goal / 1.5, 0.0
        else:           # parked
            xy, yaw = goal.copy(), to_target
            vel, rate = np.zeros(2), 0.0
        reach = 0.2 * min(1.0, max(0.0, (t - 4.0) / 1.0))
        ee_local = np.array([0.3 + reach, 0.0, 0.4]) + rng.normal(0, 1e-4, 3)
        c, s = np.cos(yaw), np.sin(yaw)
        base_pos = np.array([xy[0], xy[1], 0.1])
        ee = base_pos + np.array([c * ee_local[0] - s * ee_local[1], s * ee_local[0] + c * ee_local[1], ee_local[2]])
        states.append((ee, base_pos, quat_xyzw(0.0, 0.0, yaw), yaw, np.array([vel[0], vel[1], 0.0]), rate))
        commands.append(np.full(4, 4.0 if k < 100 else 1.0))  # 100 steps above the 3.6 rad/s envelope
    return target, standoff, states, commands


def run_approach(target, standoff, states, commands):
    diag = ev.ApproachDiagnostics(target, standoff, DT)
    for k, ((ee, base, q, yaw, vel, rate), cmd) in enumerate(zip(states, commands)):
        diag.update(k, ee, base, q, yaw, vel, rate)
        diag.command(cmd)
    return diag.columns()


class TestRingAndBearing(unittest.TestCase):
    def test_ring_distance(self):
        target = np.array([3.0, 4.0, 0.5])
        self.assertAlmostEqual(ev.ring_distance([0.0, 0.0], target, 0.65), 4.35, places=12)
        self.assertAlmostEqual(ev.ring_distance([3.0, 4.0 - 0.65], target, 0.65), 0.0, places=12)
        self.assertAlmostEqual(ev.ring_distance([3.0, 4.0], target, 0.65), 0.65, places=12)  # inside: absolute
        # anywhere on the ring, not only at the expert's parking point
        for phi in np.linspace(-np.pi, np.pi, 9):
            on_ring = target[:2] + 0.65 * np.array([np.cos(phi), np.sin(phi)])
            self.assertAlmostEqual(ev.ring_distance(on_ring, target, 0.65), 0.0, places=12)

    def test_expert_goal_is_on_the_ring(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            target = np.array([*rng.uniform(-3, 3, 2), rng.uniform(0.2, 1.2)])
            goal, _, standoff = ev.expert_base_goal(target, np.array([*rng.uniform(-2, 2, 2), 0.1]), 0.0)
            self.assertAlmostEqual(standoff, np.interp(target[2], ev.STANDOFF_Z_POINTS, ev.STANDOFF_POINTS))
            self.assertAlmostEqual(ev.ring_distance(goal, target, standoff), 0.0, places=12)

    def test_target_bearing_error(self):
        target = np.array([1.0, 1.0, 0.5])
        self.assertAlmostEqual(ev.target_bearing_error([0.0, 0.0], np.pi / 4, target), 0.0, places=12)
        self.assertAlmostEqual(ev.target_bearing_error([0.0, 0.0], np.pi / 4 + np.pi, target), np.pi, places=12)
        self.assertAlmostEqual(ev.target_bearing_error([0.0, 0.0], np.pi / 4 - 0.3, target), 0.3, places=12)
        # wraps: target direction -3.1 rad, yaw 3.1 rad -> 2 pi - 6.2 rad apart
        behind = np.array([np.cos(-3.1), np.sin(-3.1), 0.5])
        self.assertAlmostEqual(ev.target_bearing_error([0.0, 0.0], 3.1, behind), 2 * np.pi - 6.2, places=9)


class TestApproachDiagnostics(unittest.TestCase):
    def test_synthetic_episode(self):
        target, standoff, states, commands = synthetic_episode()
        cols = run_approach(target, standoff, states, commands)
        # parked: on the ring and still from step 2500 -> END of the 1 s interval at 3.5 s
        self.assertTrue(cols["parked"])
        self.assertAlmostEqual(cols["parked_time"], 3.5, places=9)
        self.assertLess(cols["target_bearing_err_at_park_deg"], 1e-6)
        # the arm extends 0.10 m at t = 4.5 s (half of the 0.2 m reach; noise 1e-4 m)
        self.assertAlmostEqual(cols["arm_extend_time"], 4.5, delta=0.005)
        self.assertAlmostEqual(cols["ee_base_disp_max"], 0.2, delta=0.001)
        self.assertLess(cols["ring_dist_at_arm_extend"], 1e-9)
        self.assertLess(cols["target_bearing_err_at_arm_extend_deg"], 1e-6)
        # ring, distance and bearing at the start and at the end
        dist0 = np.hypot(*target[:2])
        self.assertAlmostEqual(cols["target_dist_start"], dist0, places=12)
        self.assertAlmostEqual(cols["ring_dist_start"], dist0 - standoff, places=12)
        self.assertAlmostEqual(cols["target_bearing_err_start_deg"], np.rad2deg(np.arctan2(1.0, 2.0)), places=9)
        self.assertAlmostEqual(cols["ring_dist_final"], 0.0, places=12)
        self.assertAlmostEqual(cols["ring_dist_min"], 0.0, places=12)
        self.assertAlmostEqual(cols["target_dist_final"], standoff, places=12)
        self.assertLess(cols["target_bearing_err_final_deg"], 1e-6)
        self.assertAlmostEqual(cols["wheel_over_envelope_frac"], 100 / 6000, places=12)

    def test_invariant_to_planar_rigid_motion(self):
        target, standoff, states, commands = synthetic_episode(n=5000, seed=3)
        reference = run_approach(target, standoff, states, commands)
        move_point, move_quat = planar_motion(2.0, -1.5, 3.0)
        rotation = 2.0
        moved = [(move_point(ee), move_point(base), move_quat(q), float(ev.wrap_angle(yaw + rotation)),
                  move_point(vel) - move_point(np.zeros(3)), rate) for ee, base, q, yaw, vel, rate in states]
        cols = run_approach(move_point(target), standoff, moved, commands)
        for name, value in reference.items():
            if isinstance(value, float) and np.isnan(value):
                self.assertTrue(np.isnan(cols[name]), name)
            else:
                self.assertAlmostEqual(cols[name], value, places=9, msg=name)

    def test_moving_on_the_ring_is_not_parked(self):
        target, standoff, states, commands = synthetic_episode(n=5000)
        # same positions, but the base reports a yaw rate of 0.03 rad/s (above STILL_ANG) after step 2500
        states = [(ee, base, q, yaw, vel, 0.03 if k >= 2500 else rate)
                  for k, (ee, base, q, yaw, vel, rate) in enumerate(states)]
        cols = run_approach(target, standoff, states, commands)
        self.assertFalse(cols["parked"])
        self.assertTrue(np.isnan(cols["parked_time"]) and np.isnan(cols["target_bearing_err_at_park_deg"]))

    def test_no_arm_extension(self):
        target, standoff, states, commands = synthetic_episode(n=3000)  # ends before the arm moves
        cols = run_approach(target, standoff, states, commands)
        for name in ("arm_extend_time", "ring_dist_at_arm_extend", "target_bearing_err_at_arm_extend_deg"):
            self.assertTrue(np.isnan(cols[name]), name)
        self.assertLess(cols["ee_base_disp_max"], 0.002)


class TestExtendedHold(unittest.TestCase):
    N = round(ev.HOLD_EXT_TIME / DT)  # steps of the window after the reach step

    def run_hold(self, errors, stop, base_xy=None, yaws=None, commands=()):
        hold = ev.ExtendedHold([1.0, 2.0], 3.1)
        for k, e in enumerate(errors):
            hold.update(e, base_xy[k] if base_xy is not None else [1.0, 2.0], yaws[k] if yaws is not None else 3.1)
        for c in commands:
            hold.command(c)
        hold.stop = stop
        return hold.columns(DT)

    def test_completed_and_held(self):
        errors = np.full(self.N + 1, 0.004)
        errors[-1] = 0.006
        cols = self.run_hold(errors, "completed", commands=[np.zeros(4)] * self.N)
        self.assertEqual(cols["hold10_stop"], "completed")
        self.assertTrue(cols["held_10s"])
        self.assertEqual(cols["hold10_error_end"], 0.006)
        self.assertAlmostEqual(cols["hold10_error_max"], 0.006)
        self.assertAlmostEqual(cols["hold10_error_mean"], (0.004 * self.N + 0.006) / (self.N + 1), places=12)
        self.assertTrue(np.isnan(cols["hold10_leave_time"]))
        self.assertEqual(cols["hold10_wheel_cmd_max"], 0.0)
        self.assertEqual(cols["hold10_base_disp_max"], 0.0)

    def test_leaving_the_15mm_band(self):
        errors = np.full(self.N + 1, 0.004)
        errors[7000:] = 0.015  # >= 15 mm counts as left (held needs < 15 mm)
        cols = self.run_hold(errors, "completed")
        self.assertFalse(cols["held_10s"])
        self.assertAlmostEqual(cols["hold10_leave_time"], 7.0, places=12)
        self.assertEqual(cols["hold10_error_end"], 0.015)

    def test_window_ended_early(self):
        errors = np.full(3001, 0.004)
        for stop in ("policy_error", "invalid_action", "tipped"):
            cols = self.run_hold(errors, stop)
            self.assertEqual(cols["hold10_stop"], stop)
            self.assertFalse(cols["held_10s"])
            self.assertTrue(np.isnan(cols["hold10_error_end"]))
            self.assertAlmostEqual(cols["hold10_error_max"], 0.004)  # over the part of the window that ran

    def test_base_motion_and_yaw_wrap(self):
        n = 101
        base_xy = [[1.0 + 0.003 * k, 2.0 + 0.004 * k] for k in range(n)]  # 5 mm per step
        yaws = [3.1] * 50 + [-3.1] * 51                                    # crosses +-pi: 0.0832 rad, not 6.2
        cols = self.run_hold([0.004] * n, "completed", base_xy=base_xy, yaws=yaws,
                             commands=[np.array([0.1, -0.5, 0.2, 0.0])] * 5)
        self.assertAlmostEqual(cols["hold10_base_disp_max"], 0.5, places=12)
        self.assertAlmostEqual(cols["hold10_yaw_change_max_deg"], np.rad2deg(2 * np.pi - 6.2), places=9)
        self.assertEqual(cols["hold10_wheel_cmd_max"], 0.5)

    def test_empty_columns(self):
        cols = ev.ExtendedHold.empty_columns()
        self.assertEqual(set(cols), set(self.run_hold([0.004], "completed")))
        self.assertEqual((cols["hold10_stop"], cols["held_10s"]), ("", False))


class TestProtocolAndColumns(unittest.TestCase):
    def test_protocol_constants(self):
        expected = {"PROTOCOL_VERSION": "v1.1", "HOLD_EXT_TIME": 10.0, "ARM_EXTEND_DIST": 0.10, "PARK_RING_TOL": 0.20,
                    "PARK_TIME": 1.0, "STABLE_TOLS": (0.01, 0.02, 0.05), "STABLE_TIME": 0.5, "STABLE_HORIZON": 40.0,
                    "WHEEL_ENVELOPE": 3.6}
        for name, value in expected.items():
            self.assertEqual(ev.PROTOCOL[name], value, name)
        # the v1 values are unchanged
        v1 = {"T_MAX": 40.0, "REACH_TOL": 0.01, "HOLD_TIME": 1.0, "SETTLE_TOL": 0.015, "STALL_TIME": 5.0,
              "STALL_IMPROVEMENT": 0.001, "STALL_MIN_ERROR": 0.03, "STILL_LIN": 0.01, "STILL_ANG": 0.02,
              "STILL_ARM": 0.05, "TIP_ANGLE": 0.5, "ARM_MOVE_ANGLE": 0.05, "TORQUE_LIMIT": 30.0}
        self.assertEqual({name: ev.PROTOCOL[name] for name in v1}, v1)
        self.assertEqual(len(ev.PROTOCOL), len(expected) + len(v1))
        json.dumps(ev.PROTOCOL)  # written into config.json

    def test_columns(self):
        self.assertEqual(ev.ROW_COLUMNS[:len(V1_COLUMNS)], V1_COLUMNS)
        self.assertEqual(ev.ROW_COLUMNS[len(V1_COLUMNS):], SPEC_V11_COLUMNS)
        self.assertEqual(ev.V11_COLUMNS, SPEC_V11_COLUMNS)
        self.assertEqual(set(ev.empty_v11_columns()), set(SPEC_V11_COLUMNS))
        if os.path.exists(EXPERT_DEV_CSV):  # the v1 baseline run
            with open(EXPERT_DEV_CSV, newline="") as f:
                self.assertEqual(next(csv.reader(f)), V1_COLUMNS)

    def test_summarize(self):
        base = {"success": True, "settled": True, "stop_reason": "reached", "policy_ms_mean": 0.2,
                "policy_ms_p99": 0.4, "expert_noisy_recorded": False, "expert_status_recorded": "reached"}
        rows = [
            {**base, "held_10s": True, "hold10_stop": "completed", "stable_5cm_time": 9.0, "stable_2cm_time": 9.1,
             "stable_1cm_time": 9.5, "parked": True, "arm_extend_time": 8.0},
            {**base, "held_10s": False, "hold10_stop": "policy_error", "stable_5cm_time": 9.0,
             "stable_2cm_time": np.nan, "stable_1cm_time": np.nan, "parked": True, "arm_extend_time": np.nan},
            {**base, "success": False, "settled": False, "stop_reason": "stalled", **ev.ExtendedHold.empty_columns(),
             "stable_5cm_time": np.nan, "stable_2cm_time": np.nan, "stable_1cm_time": np.nan, "parked": False,
             "arm_extend_time": np.nan},
            {"success": False, "stop_reason": "harness_error"},  # a worker error row without the new columns
        ]
        lines = ev.summarize(rows, "test", "data/dev").splitlines()
        self.assertTrue(lines[0].startswith("Policy test on data/dev: success 2/4 = 50.0% ["))
        self.assertTrue(lines[1].startswith("Stop reasons: "))
        text = "\n".join(lines)
        self.assertIn("held_10s (EE < 15 mm for 10 s after the first reach): 1/4 = 25.0% [", text)
        self.assertIn("extended hold ended: completed 1, not reached 2, policy_error 1", text)
        self.assertIn("5cm 2/4 = 50.0% [", text)
        self.assertIn("2cm 1, 1cm 1", text)
        self.assertIn("parked (ring distance < 0.2 m and base still for >= 1.0 s): 2/4 = 50.0% [", text)
        self.assertIn("in the base frame): 1/4 = 25.0% [", text)


def write_run(directory, header, rows, version):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "rollouts.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump({"protocol": {"PROTOCOL_VERSION": version}}, f)


class TestCheckScript(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.v1_dir, self.v11_dir = os.path.join(self.tmp.name, "v1"), os.path.join(self.tmp.name, "v11")
        self.v1_header = ["episode", "success", "t_reach", "cmp_steps", "cmp_identical", "wall_time", "error_message"]
        self.v1_rows = [{"episode": str(k), "success": "True", "t_reach": repr(8.0 + k / 7), "cmp_steps": "9000",
                         "cmp_identical": "True", "wall_time": "10.5", "error_message": ""} for k in range(5)]
        self.v11_header = self.v1_header + ["held_10s"]
        self.v11_rows = [{**row, "held_10s": "True"} for row in self.v1_rows]

    def compare(self, *flags, v1_version="v1", v11_version="v1.1"):
        write_run(self.v1_dir, self.v1_header, self.v1_rows, v1_version)
        write_run(self.v11_dir, self.v11_header, self.v11_rows, v11_version)
        with redirect_stdout(io.StringIO()) as out:
            status = check.main([self.v1_dir, self.v11_dir, *flags])
        return status, out.getvalue()

    def test_pass(self):
        # excluded columns may differ, and an empty v1 error_message may become an extended-hold error
        self.v11_rows[0].update(wall_time="20.1", cmp_steps="9500", error_message="extended hold: RuntimeError()")
        for flags in ((), ("--require-compare",)):
            status, text = self.compare(*flags)
            self.assertEqual(status, 0, text)
            self.assertIn("PASS: all 5 v1 columns identical, 5 of 5 v1 episodes compared (exact)", text)

    def test_subset_needs_allow_subset(self):
        self.v11_rows = self.v11_rows[1:4]  # an incomplete run: episodes 0 and 4 are missing
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("3 of 5 v1 episodes compared", text)
        self.assertIn("2 episodes of the v1 run are missing from the v1.1 run", text)
        self.assertIn("[0, 4]", text)
        status, text = self.compare("--allow-subset")
        self.assertEqual(status, 0, text)
        self.assertIn("3 of 5 v1 episodes compared (2 not in the v1.1 run, --allow-subset)", text)
        self.assertIn("PASS", text)

    def test_value_differs(self):
        self.v11_rows[2]["t_reach"] = repr(float(self.v11_rows[2]["t_reach"]) + 1e-15)
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("FAIL", text)
        self.assertIn("ep    2  t_reach", text)
        self.assertIn("5 of 5 v1 episodes compared", text)
        # accepted with a tolerance
        self.assertEqual(self.compare("--float-tol", "1e-12")[0], 0)

    def test_error_message(self):
        cases = [("", "RuntimeError('x')", 1),                         # a new error in the v1 part
                 ("", "extended hold:RuntimeError('x')", 1),           # not the exact prefix
                 ("ValueError('a')", "extended hold: ValueError('a')", 1),  # a v1 error is never replaced
                 ("ValueError('a')", "ValueError('a')", 0),
                 ("", "extended hold: RuntimeError('x')", 0)]
        for v1_value, v11_value, expected in cases:
            with self.subTest(v1=v1_value, v11=v11_value):
                self.v1_rows[3]["error_message"], self.v11_rows[3]["error_message"] = v1_value, v11_value
                status, text = self.compare()
                self.assertEqual(status, expected, text)
                if expected:
                    self.assertIn("ep    3  error_message", text)

    def test_recording_not_identical(self):
        for row in (self.v1_rows[2], self.v11_rows[2]):
            row["cmp_identical"] = "False"
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("cmp_identical is not True in the v1.1 run for episodes [2]", text)

    def test_require_compare(self):
        # one episode without a recorded comparison (cmp_identical empty in both runs): fine unless required
        for row in (self.v1_rows[1], self.v11_rows[1]):
            row["cmp_identical"] = ""
        status, text = self.compare()
        self.assertEqual(status, 0, text)
        self.assertIn("4/4 episodes identical, 1 episodes without a comparison", text)
        status, text = self.compare("--require-compare")
        self.assertEqual(status, 1)
        self.assertIn("--require-compare: no recorded comparison (cmp_identical empty) in the v1.1 run for 1 "
                      "episodes: [1]", text)
        # a run without --compare-recorded: every value empty
        for row in self.v1_rows + self.v11_rows:
            row["cmp_identical"] = ""
        self.assertEqual(self.compare()[0], 0)
        status, text = self.compare("--require-compare")
        self.assertEqual(status, 1)
        self.assertIn("0/0 episodes identical, 5 episodes without a comparison", text)
        # no cmp_identical column at all
        self.v1_header.remove("cmp_identical")
        self.v11_header.remove("cmp_identical")
        for row in self.v1_rows + self.v11_rows:
            del row["cmp_identical"]
        self.assertEqual(self.compare()[0], 0)
        status, text = self.compare("--require-compare")
        self.assertEqual(status, 1)
        self.assertIn("no cmp_identical column", text)

    def test_protocol_versions(self):
        for v1_version, v11_version in (("v1.1", "v1.1"), ("v1.1", "v1"), ("v1", "v1"), ("v1", "v1.2")):
            with self.subTest(v1=v1_version, v11=v11_version):
                status, text = self.compare(v1_version=v1_version, v11_version=v11_version)
                self.assertEqual(status, 1)
                self.assertIn(f"must be protocol v1 and the checked run protocol v1.1 (got {v1_version} and "
                              f"{v11_version})", text)
        # a v1.1 run compared with itself
        write_run(self.v11_dir, self.v11_header, self.v11_rows, "v1.1")
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check.main([self.v11_dir, self.v11_dir]), 1)
        self.assertIn("(got v1.1 and v1.1)", out.getvalue())
        # no config.json: version unknown
        self.compare()
        os.remove(os.path.join(self.v1_dir, "config.json"))
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(check.main([self.v1_dir, self.v11_dir]), 1)
        self.assertIn("(got ? and v1.1)", out.getvalue())

    def test_duplicate_episodes(self):
        # a duplicated episode in the v1.1 run, the good copy last (it would win in a dict keyed by episode)
        bad = {**self.v11_rows[2], "t_reach": "99.0"}
        self.v11_rows = self.v11_rows[:2] + [bad] + self.v11_rows[2:]
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("1 episodes appear more than once in the v1.1 run: [2]", text)
        # a duplicated episode in the v1 run
        self.v11_rows.pop(2)
        self.v1_rows.append(dict(self.v1_rows[4]))
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("1 episodes appear more than once in the v1 run: [4]", text)

    def test_unknown_episode_and_header(self):
        self.v11_rows.append({**self.v1_rows[0], "episode": "7", "held_10s": ""})
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("not in the v1 run: [7]", text)
        self.v11_rows.pop()
        self.v11_header = ["success", "episode", *self.v1_header[2:], "held_10s"]
        status, text = self.compare()
        self.assertEqual(status, 1)
        self.assertIn("order differs", text)

    def tearDown(self):
        self.tmp.cleanup()


# ----------------------------------------------------------------------------------------------------------------
class ExpertFailingAfter:
    """The expert, with an error injected from time t_fail on: an exception or a NaN wheel command."""

    def __init__(self, t_fail, mode):
        self.expert, self.t_fail, self.mode = ev.ExpertPolicy(), t_fail, mode

    def reset(self, target, info):
        self.expert.reset(target, info)

    def act(self, obs):
        wheel_cmds, torque_arm = self.expert.act(obs)
        if obs["t"] >= self.t_fail:
            if self.mode == "raise":
                raise RuntimeError("injected error")
            return np.full(4, np.nan), torque_arm
        return wheel_cmds, torque_arm


@unittest.skipUnless(os.path.isdir(DEV_DIR) and os.path.exists(EXPERT_DEV_CSV), "needs data/dev and results/expert_dev")
class TestEpisodes(unittest.TestCase):
    """Short closed-loop runs in this process: the expert on DEV episode 11 (reaches at 1.574 s), with an error
    injected at t = 4 s in the extended hold, and the zero policy on DEV episode 0."""

    @classmethod
    def setUpClass(cls):
        cls.episodes, _ = ev.load_eval_set(DEV_DIR)
        cls.reference = expert_dev_rows()

    def run_episode(self, policy, index):
        episode = self.episodes[index]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):  # the injected errors' tracebacks
            row, traj = ev.run_policy_episode(policy, episode, keep_traj=True, verbose=False)
        row.update(ev.compare_with_recorded(traj, episode["recorded_file"], episode["recorded_every"]))
        return {**ev.episode_info(episode), **row}, traj

    def check_v1_columns(self, row, index):
        self.assertEqual(v1_differences(as_csv_strings(row), self.reference[index]), [])
        self.assertIs(row["cmp_identical"], True)

    def test_error_in_extended_hold(self):
        for mode, stop in (("raise", "policy_error"), ("nan", "invalid_action")):
            with self.subTest(mode=mode):
                row, traj = self.run_episode(ExpertFailingAfter(4.0, mode), 11)
                self.check_v1_columns(row, 11)  # the v1 row was frozen at t_reach + 1 s = 2.574 s
                self.assertTrue(row["success"] and row["settled"])
                self.assertEqual(row["stop_reason"], "reached")
                self.assertEqual(row["hold10_stop"], stop)
                self.assertFalse(row["held_10s"])
                self.assertTrue(np.isnan(row["hold10_error_end"]))
                self.assertAlmostEqual(row["run_end_time"], 4.0, delta=0.0015)
                self.assertEqual(len(traj["t"]), round(row["run_end_time"] / 0.001))  # the whole run
                self.assertEqual(row["error_message"], "extended hold: RuntimeError('injected error')"
                                 if mode == "raise" else "")
                # metrics of the run: the expert's base is done (zero wheel commands) during the hold
                self.assertEqual(row["hold10_wheel_cmd_max"], 0.0)
                self.assertGreaterEqual(row["stable_1cm_time"], row["t_reach"] + ev.STABLE_TIME - 1e-9)
                self.assertTrue(row["parked"])

    def test_zero_policy(self):
        row, _ = self.run_episode(ev.ZeroPolicy(), 0)
        self.assertEqual(row["stop_reason"], "stalled")
        self.assertEqual((row["hold10_stop"], row["held_10s"]), ("", False))
        # Known limitation of the specified arm_extend_time (EE displacement in the base frame > 0.10 m): with zero
        # torques the arm collapses under gravity (the EE drops about 0.7 m), which counts as "extended" at 0.14 s
        self.assertAlmostEqual(row["arm_extend_time"], 0.142, delta=0.005)
        self.assertGreater(row["ee_base_disp_max"], 0.7)
        self.assertLess(row["base_path_length"], 0.03)         # only jolted by the falling arm
        self.assertLess(row["base_turn_total_deg"], 1.0)
        self.assertFalse(row["parked"])                        # starts 1.31 m from the ring
        self.assertAlmostEqual(row["ring_dist_final"], row["ring_dist_start"], delta=0.01)
        self.assertEqual(row["wheel_over_envelope_frac"], 0.0)
        for tol in ev.STABLE_TOLS:
            self.assertTrue(np.isnan(row[f"stable_{ev._cm(tol)}_time"]))
        self.assertEqual(row["run_end_time"], row["duration"])


@unittest.skipUnless(RUN_SLOW, "slow: set RUN_SLOW=1")
class TestExpertAgainstV1Slow(unittest.TestCase):
    """The expert on DEV episodes 67 (reached, not settled) and 169 (stalled), each in a fresh process through
    _worker / run_policy_episode with --compare-recorded: every v1 column as in results/expert_dev."""

    def test_two_dev_episodes(self):
        episodes, _ = ev.load_eval_set(DEV_DIR)
        reference = expert_dev_rows()
        with tempfile.TemporaryDirectory() as tmp:
            jobs = [{"episode": episodes[i], "policy": "expert", "policy_arg": None, "compare": True,
                     "traj_path": None, "traj_every": 10, "early_stop": True,
                     "log_path": os.path.join(tmp, f"ep_{i:04d}.log")} for i in (67, 169)]
            with mp.get_context("spawn").Pool(2, maxtasksperchild=1) as pool:
                rows = pool.map(ev._worker, jobs)
        for row in rows:
            ep = row["episode"]
            with self.subTest(episode=ep):
                self.assertEqual(v1_differences(as_csv_strings(row), reference[ep]), [])
                self.assertIs(row["cmp_identical"], True)
        reached, stalled = rows
        self.assertEqual((reached["stop_reason"], reached["settled"]), ("reached", False))
        self.assertEqual(reached["hold10_stop"], "completed")
        self.assertAlmostEqual(reached["run_end_time"], reached["t_reach"] + ev.HOLD_EXT_TIME, delta=0.0015)
        self.assertFalse(reached["held_10s"])  # not settled in the first second
        self.assertEqual(stalled["stop_reason"], "stalled")
        self.assertEqual((stalled["hold10_stop"], stalled["run_end_time"]), ("", stalled["duration"]))


if __name__ == "__main__":
    unittest.main()
