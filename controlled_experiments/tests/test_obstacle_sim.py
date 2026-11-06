"""Tests of the step-9 simulation side (Track A): obstacle bodies and contacts, the two-segment expert navigation,
the protocol v2.0 harness (observation, info, contact windows, passage, docking), the collector's planning and
guards, the policy loader and the failure classes.

Pybullet is only used for short static setups (no recorded set is opened, data/test and test2 never). The closed-loop
gates (A2-A6) are scripts under results/step9/. Run:
  python -m unittest discover -s tests -p 'test_obstacle*' -v
"""
import csv
from dataclasses import replace
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

import collect_data  # noqa: E402
from evaluate_policy import setup_simulation  # noqa: E402
from robotnik_arm_control import BaseControllerParams, BaseNavState, base_controller  # noqa: E402
from obstacle_task import classify, collect, lidar, scenario  # noqa: E402
from obstacle_task import evaluate as ev  # noqa: E402
from obstacle_task.expert import NAV_TIMEOUT, TwoSegmentNavigation  # noqa: E402
from obstacle_task.sim import link_name, min_clearance, robot_obstacle_contact, spawn_obstacles  # noqa: E402


def synthetic_scenario():
    """Start at the origin facing +x, waypoint (1, 1), dock (2, 0) facing +x; obstacle of a real draw."""
    return replace(scenario.sample_scenario(7, 0), init_xy=[0.0, 0.0], init_yaw=0.0,
                   waypoint_xy=[1.0, 1.0], dock_xy=[2.0, 0.0], dock_bearing=0.0)


# ==============================================================================
# Obstacle bodies, contact and clearance (pybullet, no physics step except where stated)
# ==============================================================================
class TestObstacleBodies(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim, cls.client, cls.robot, cls.info = setup_simulation([0.0, 0.0], 0.0)

    @classmethod
    def tearDownClass(cls):
        cls.client.disconnect()

    def setUp(self):
        self.bodies = []

    def tearDown(self):
        for body in self.bodies:
            self.client.removeBody(body)

    def spawn(self, obstacles):
        bodies = spawn_obstacles(self.client, obstacles)
        self.bodies.extend(bodies)
        return bodies

    def clearance(self, body):
        return min_clearance(self.client, self.robot, [body])

    def place_at_clearance(self, body, gap, direction):
        """Move a cylinder along `direction` (unit, planar) until its clearance from the robot is `gap` (m)."""
        pos = np.array(self.client.getBasePositionAndOrientation(body)[0])
        for _ in range(6):
            pos[:2] -= direction * (self.clearance(body) - gap)
            self.client.resetBasePositionAndOrientation(body, pos.tolist(), [0.0, 0.0, 0.0, 1.0])
        self.client.performCollisionDetection()   # refresh the contact manifolds without a physics step
        return self.clearance(body)

    def test_fixed_cylinder_geometry(self):
        obs = [{"x": 10.0, "y": 4.0, "r": 0.3, "h": 1.5}, {"x": -10.0, "y": -4.0, "r": 0.25, "h": 1.0}]
        bodies = self.spawn(obs)
        self.assertEqual(len(set(bodies)), 2)
        for body, ob in zip(bodies, obs):
            position, _ = self.client.getBasePositionAndOrientation(body)
            np.testing.assert_array_equal(position, [ob["x"], ob["y"], ob["h"] / 2])   # standing on the floor
            self.assertEqual(self.client.getDynamicsInfo(body, -1)[0], 0.0)              # fixed (mass 0)
            self.assertEqual(self.client.getCollisionShapeData(body, -1)[0][2], p.GEOM_CYLINDER)
            lower, upper = map(np.asarray, self.client.getAABB(body))
            np.testing.assert_allclose(upper - lower, [2 * ob["r"], 2 * ob["r"], ob["h"]], atol=0.01)

    def test_contact_with_self_collision_disabled_config(self):
        with open(os.path.join(TASK_DIR, "configs", "robotnik_arm.json")) as f:
            config = json.load(f)
        self.assertEqual(config["robot_pybullet"]["collision_enable"], [0])   # the setting the contacts must survive

    def test_body_against_the_arm_tip(self):
        # A cylinder in front of the robot: the arm's fingers stick out ~0.73 m and touch it first
        body = self.spawn([{"x": 3.0, "y": 0.0, "r": 0.3, "h": 1.5}])[0]
        self.client.performCollisionDetection()
        self.assertEqual(robot_obstacle_contact(self.client, self.robot, [body]), (False, None))
        gap = self.place_at_clearance(body, 0.002, np.array([1.0, 0.0]))
        self.assertAlmostEqual(gap, 0.002, delta=2e-4)
        self.assertEqual(robot_obstacle_contact(self.client, self.robot, [body]), (False, None))   # 2 mm apart
        # (the fingers' manifold starts at 0.8 mm: obstacle_task/sim.py)
        gap = self.place_at_clearance(body, -0.001, np.array([1.0, 0.0]))
        self.assertLess(gap, 0.0)
        contact, link = robot_obstacle_contact(self.client, self.robot, [body])                    # touching
        self.assertTrue(contact)
        self.assertEqual(link, self.client.getContactPoints(bodyA=self.robot, bodyB=body)[0][3])
        self.assertTrue(link_name(self.client, self.robot, link).startswith("robot_j2s7s300_link_finger"))

    def test_body_against_the_chassis_side_after_a_step(self):
        # Beside the robot (+y): the wheels touch it; the contact shows after a physics step, as in the loop. The
        # wheels' manifold starts at a separation of 3.5 mm (collision margin, see obstacle_task/sim.py)
        body = self.spawn([{"x": 0.0, "y": 3.0, "r": 0.3, "h": 1.5}])[0]
        self.place_at_clearance(body, 0.01, np.array([0.0, 1.0]))
        self.client.stepSimulation()
        self.assertEqual(robot_obstacle_contact(self.client, self.robot, [body]), (False, None))
        self.place_at_clearance(body, -0.001, np.array([0.0, 1.0]))
        self.client.stepSimulation()
        contact, link = robot_obstacle_contact(self.client, self.robot, [body])
        self.assertTrue(contact)
        name = link_name(self.client, self.robot, link)
        self.assertFalse(name.startswith("robot_j2s7s300"), name)   # not an arm link

    def test_body_inside_the_robot_and_base_link_name(self):
        bodies = self.spawn([{"x": 5.0, "y": 0.0, "r": 0.3, "h": 1.5}])
        self.assertGreater(self.clearance(bodies[0]), 3.0)
        self.client.resetBasePositionAndOrientation(bodies[0], [0.0, 0.0, 0.75], [0.0, 0.0, 0.0, 1.0])
        self.client.performCollisionDetection()
        contact, link = robot_obstacle_contact(self.client, self.robot, bodies)
        self.assertTrue(contact)
        self.assertTrue(-1 <= link < self.client.getNumJoints(self.robot))
        self.assertLess(self.clearance(bodies[0]), 0.0)
        self.assertEqual(link_name(self.client, self.robot, -1), "robot_base_footprint")

    def test_empty_obstacles(self):
        self.assertEqual(spawn_obstacles(self.client, []), [])
        self.assertEqual(robot_obstacle_contact(self.client, self.robot, []), (False, None))
        self.assertTrue(np.isinf(min_clearance(self.client, self.robot, [])))


# ==============================================================================
# Two-segment navigation on synthetic poses
# ==============================================================================
class TestTwoSegments(unittest.TestCase):
    def test_waypoint_orientation_command_is_discarded(self):
        sc = synthetic_scenario()
        nav = TwoSegmentNavigation(sc)
        nav.state0.phase = "drive"
        pos, yaw, t = np.array([1.0, 1.0, 0.2]), -np.pi / 4, 10.0      # on the waypoint, facing the dock
        expected_state = BaseNavState()
        expected = base_controller(expected_state, pos, yaw, np.asarray(sc.dock_xy), sc.dock_bearing, t,
                                   BaseControllerParams(nav_timeout=NAV_TIMEOUT))
        actual = nav.command(pos, yaw, t)
        np.testing.assert_array_equal(actual, expected)   # the fresh segment-1 controller, same step
        self.assertTrue(np.all(actual > 0.0))             # drives straight on: no orient turn at the waypoint
        self.assertEqual((nav.segment, nav.state1.phase, nav.done), (1, expected_state.phase, False))
        self.assertEqual(nav.segment_switch_time, t)

    def test_passed_waypoint_switches_and_final_dock_finishes(self):
        sc = synthetic_scenario()
        nav = TwoSegmentNavigation(sc)
        nav.state0.phase = "drive"
        nav.command(np.array([1.1, 1.0, 0.2]), 0.0, 10.0)    # 0.1 m beyond the waypoint: passed_goal rule
        self.assertEqual((nav.segment, nav.done), (1, False))
        nav.command(np.array([2.0, 0.0, 0.2]), 0.0, 20.0)    # on the dock with its bearing
        self.assertTrue(nav.done)
        self.assertEqual(nav.nav_done_time, 20.0)
        np.testing.assert_array_equal(nav.command(np.array([2.0, 0.0, 0.2]), 0.0, 21.0), np.zeros(4))
        self.assertEqual(nav.segment_switch_time, 10.0)

    def test_segment_zero_timeout_is_absolute_and_done(self):
        nav = TwoSegmentNavigation(synthetic_scenario())
        nav.command(np.array([0.0, 0.0, 0.2]), 0.0, NAV_TIMEOUT)
        self.assertEqual((nav.segment, nav.done), (0, False))
        command = nav.command(np.array([0.0, 0.0, 0.2]), 0.0, NAV_TIMEOUT + 0.001)
        self.assertEqual(nav.segment, 1)
        self.assertTrue(nav.state1.done and nav.done)
        np.testing.assert_array_equal(command, np.zeros(4))
        self.assertEqual(nav.nav_done_time, NAV_TIMEOUT + 0.001)

    def test_segment_one_timeout_counts_the_whole_navigation(self):
        nav = TwoSegmentNavigation(synthetic_scenario())
        nav.state0.phase = "drive"
        nav.command(np.array([1.0, 1.0, 0.2]), 0.0, 30.0)                      # switch at 30 s
        self.assertEqual((nav.segment, nav.done), (1, False))
        nav.command(np.array([1.2, 0.9, 0.2]), -0.5, NAV_TIMEOUT + 0.001)     # far from the dock, after 45 s
        self.assertTrue(nav.done)

    def test_no_obstacle_uses_original_controller_and_timeout(self):
        sc = replace(synthetic_scenario(), obstacles=[])
        nav, state = TwoSegmentNavigation(sc), BaseNavState()
        params = BaseControllerParams(nav_timeout=25.0)
        self.assertEqual((nav.segment, nav.params.nav_timeout), (1, 25.0))
        for x, y, yaw, t in [(0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 2.0), (1.0, 0.0, 0.0, 8.0), (1.95, 0.0, 0.1, 12.0),
                             (2.0, 0.0, 0.0, 13.0)]:
            pos = np.array([x, y, 0.2])
            expected = base_controller(state, pos, yaw, np.asarray(sc.dock_xy), sc.dock_bearing, t, params)
            np.testing.assert_array_equal(nav.command(pos, yaw, t), expected)
            self.assertEqual(nav.state1, state)
        self.assertIsNone(nav.segment_switch_time)

    def test_kinematic_run_through_both_segments(self):
        """An ideal unicycle driven by the wheel commands: align, drive to the waypoint, switch without an orient
        turn there, align, drive, orient to the dock bearing, done."""
        sc = synthetic_scenario()
        nav, params = TwoSegmentNavigation(sc), BaseControllerParams(nav_timeout=NAV_TIMEOUT)
        x, y, yaw, dt = 0.0, 0.0, 0.0, 0.002
        phases, switch_pose = [], None
        for k in range(int(40.0 / dt)):
            segment = nav.segment
            wheels = nav.command(np.array([x, y, 0.2]), yaw, k * dt)
            if segment == 0 and nav.segment == 1:
                switch_pose = (x, y, yaw)
            phases.append((nav.segment, nav.state.phase))
            if nav.done:
                break
            rw, lw = wheels[0], wheels[1]
            v = params.wheel_radius * (rw + lw) / 2
            w = params.wheel_radius * (rw - lw) / params.wheel_base_width
            x, y, yaw = x + v * np.cos(yaw) * dt, y + v * np.sin(yaw) * dt, yaw + w * dt
        self.assertTrue(nav.done)
        self.assertLess(np.hypot(switch_pose[0] - 1.0, switch_pose[1] - 1.0), 0.15)
        self.assertNotIn((0, "orient"), phases)                       # segment 0 never orients
        seq = [ph for i, ph in enumerate(phases) if i == 0 or ph != phases[i - 1]]
        self.assertEqual(seq, [(0, "align"), (0, "drive"), (1, "align"), (1, "drive"), (1, "orient")])
        self.assertLess(np.hypot(x - 2.0, y), params.pos_tol + 0.07)
        self.assertLess(abs(yaw), params.ang_tol)


# ==============================================================================
# Harness: observation, info, contact windows, passage, docking (scripted states, no simulator)
# ==============================================================================
class TestHarnessObservation(unittest.TestCase):
    def test_lidar_matches_offline_batch(self):
        rng = np.random.default_rng(12)
        xy, yaw = rng.uniform(-3, 3, (32, 2)), rng.uniform(-np.pi, np.pi, 32)
        obstacles = scenario.sample_scenario(3000, 0).obstacles
        scans = []
        for i in range(len(xy)):
            obs = ev.build_observation(i * 0.001, np.r_[xy[i], 0.2], np.array([0.0, 0.0, 0.0, 1.0]), yaw[i],
                                       np.zeros(17), np.zeros(17), np.array([3.0, 1.0, 0.6]), obstacles)
            self.assertEqual(set(obs), {"t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "target",
                                        "lidar"})
            self.assertEqual(obs["lidar"].shape, (lidar.N_BEAMS,))
            scans.append(obs["lidar"])
        np.testing.assert_allclose(scans, lidar.scan_batch(xy, yaw, obstacles), atol=1e-12, rtol=0)
        self.assertTrue(np.any(np.array(scans) < lidar.MAX_RANGE))    # the obstacle is seen

    def test_policy_inputs_are_copied_and_info_is_sanitized(self):
        pos, ori, q, qd, target = np.ones(3), np.ones(4), np.ones(17), np.ones(17), np.ones(3)
        obs = ev.build_observation(0.0, pos, ori, 0.0, q, qd, target, [])
        for key, original in [("base_pos", pos), ("base_ori", ori), ("q_mes", q), ("qd_mes", qd), ("target", target)]:
            obs[key][:] = -100.0
            np.testing.assert_array_equal(original, np.ones_like(original))
        info = {"dyn_model": object(), "ee_frame_id": 1, "shoulder_frame_id": 2, "joint_lower": q.copy(),
                "joint_upper": q.copy(), "time_step": 0.001, "init_base_pos": pos.copy(), "init_base_bearing": 0.0,
                "init_q_mes": q.copy()}
        expected = set(info)
        info.update(scenario=synthetic_scenario(), obstacles=[{}], waypoint_xy=[1, 2], side=-1, dock_xy=[2, 3])
        sanitized = ev.sanitize_policy_info(info)
        self.assertEqual(set(sanitized), expected)
        self.assertEqual(set(ev.INFO_KEYS), expected)
        sanitized["init_q_mes"][:] = -100.0
        np.testing.assert_array_equal(info["init_q_mes"], np.ones(17))


class FakeSim:
    """Scripted measured states: base position path(step), base still or moving at 0.1 m/s; Step() counts steps."""

    def __init__(self, path, moving=True):
        self.step, self.path, self.moving = 0, path, moving

    def GetBasePosition(self):
        return list(self.path(self.step))

    def GetBaseOrientation(self):
        return [0.0, 0.0, 0.0, 1.0]

    def GetMotorAngles(self, index):
        return np.zeros(17)

    def GetMotorVelocities(self, index):
        return np.zeros(17)

    def GetBaseLinVelocity(self):
        return [0.1 if self.moving else 0.0, 0.0, 0.0]

    def GetBaseAngVelocity(self):
        return [0.0, 0.0, 0.0]

    def Step(self, command, mode):
        self.step += 1


class CapturingPolicy:
    """Zero commands; records its info and the number of act() calls; can scribble on its observation."""

    def __init__(self, mutate=False):
        self.calls, self.mutate, self.info = 0, mutate, None

    def reset(self, target, info):
        self.info = info

    def act(self, obs):
        self.calls += 1
        if self.mutate:
            obs["lidar"][:] = -100.0
            obs["base_pos"][:] = -100.0
        return np.zeros(4), np.zeros(7)


def run_harness(collision_step=None, path=None, moving=True, ee=None, policy=None, segment_policy=None):
    """run_policy_episode on scripted states with time step 0.1 s (1 s window = 10 steps, T_MAX = 600 steps).
    The EE sits at `ee` (default: on the target, i.e. reached at step 0). Returns (row, traj, policy, scenario)."""
    sc = replace(synthetic_scenario(), target=[1.0, 0.0, 0.6], obstacles=[{"x": 1.0, "y": 0.5, "r": 0.3, "h": 1.5}])
    sim = FakeSim(path or (lambda k: [0.01 * k, 0.02 * np.sin(k), 0.2]), moving)
    client = SimpleNamespace(getEulerFromQuaternion=lambda _: [0.0, 0.0, 0.0], disconnect=lambda: None)
    ee_pos = np.asarray(sc.target if ee is None else ee, dtype=float)
    model = SimpleNamespace(ReoderJoints2PinVec=lambda value, _: value, pin_model=object(),
                            pin_data=SimpleNamespace(oMf=[SimpleNamespace(translation=ee_pos)]))
    info = {"dyn_model": model, "ee_frame_id": 0, "shoulder_frame_id": 0, "joint_lower": -np.ones(17),
            "joint_upper": np.ones(17), "time_step": 0.1, "init_base_pos": np.asarray(sim.GetBasePosition()),
            "init_base_bearing": 0.0, "init_q_mes": np.zeros(17)}
    policy = policy or CapturingPolicy()
    contact = lambda *args: (sim.step == collision_step, 4 if sim.step == collision_step else None)  # noqa: E731
    patches = [patch.object(ev, "setup_simulation", return_value=(sim, client, 0, info, [1])),
               patch.object(ev, "robot_obstacle_contact", side_effect=contact),
               patch.object(ev, "min_clearance", return_value=0.2),
               patch.object(ev, "link_name", side_effect=lambda client, body, link: f"link{link}"),
               patch.object(ev.pin, "forwardKinematics"), patch.object(ev.pin, "updateFramePlacements")]
    if segment_policy is not None:
        patches.append(patch.object(ev, "ExpertPolicy", segment_policy))
    for item in patches:
        item.start()
    try:
        row, traj = ev.run_policy_episode(policy, {"target": sc.target, "init_xy": sc.init_xy,
                                                   "init_yaw": sc.init_yaw, "scenario": sc},
                                          keep_traj=True, early_stop=False, verbose=False)
    finally:
        for item in patches:
            item.stop()
    return row, traj, policy, sc


class TestHarnessContactWindows(unittest.TestCase):
    def test_contact_simultaneous_with_first_reach_is_failure(self):
        row, _, policy, _ = run_harness(collision_step=0)
        self.assertFalse(row["success"])
        self.assertTrue(row["collision"])
        self.assertEqual((row["collision_link"], row["stop_reason"]), ("link4", "collision"))
        self.assertTrue(np.isnan(row["t_reach"]))
        self.assertEqual(policy.calls, 0)
        self.assertNotIn("scenario", policy.info)

    def test_contact_before_any_reach_is_a_collision_failure(self):
        row, _, policy, _ = run_harness(collision_step=7, ee=[5.0, 5.0, 0.6])
        self.assertEqual((row["success"], row["stop_reason"], row["hold10_stop"]), (False, "collision", ""))
        self.assertAlmostEqual(row["collision_time"], 0.7)
        self.assertEqual(policy.calls, 7)
        self.assertEqual(classify.classify_row(row), "collision")

    def test_contact_inside_one_second_window_preserves_arrival(self):
        row, _, policy, _ = run_harness(collision_step=5)
        self.assertTrue(row["success"])
        self.assertFalse(row["settled"])
        self.assertFalse(row["held_10s"])
        self.assertEqual(row["hold10_stop"], "collision")
        self.assertAlmostEqual(row["collision_time"], 0.5)
        self.assertEqual(policy.calls, 5)

    def test_contact_in_extended_hold_preserves_primary_metrics(self):
        row, _, policy, _ = run_harness(collision_step=20)
        self.assertTrue(row["success"] and row["settled"])
        self.assertFalse(row["held_10s"])
        self.assertEqual(row["hold10_stop"], "collision")
        self.assertAlmostEqual(row["duration"], 1.0)
        self.assertAlmostEqual(row["run_end_time"], 2.0)
        self.assertEqual(policy.calls, 20)

    def test_timeout_at_t_max_without_reach(self):
        row, _, policy, _ = run_harness(ee=[5.0, 5.0, 0.6])
        self.assertEqual((row["success"], row["stop_reason"], row["collision"]), (False, "timeout", False))
        self.assertAlmostEqual(row["duration"], ev.T_MAX)
        self.assertEqual(policy.calls, round(ev.T_MAX / 0.1))

    def test_logged_lidar_matches_poses_despite_policy_mutation(self):
        row, traj, _, sc = run_harness(collision_step=20, policy=CapturingPolicy(mutate=True))
        self.assertTrue(row["success"])
        expected = lidar.scan_batch(traj["base_pos"][:, :2], traj["base_bearing"], sc.obstacles)
        np.testing.assert_allclose(traj["lidar"], expected, atol=1e-12, rtol=0)
        self.assertGreater(np.ptp(traj["base_pos"][:, 0]), 0.1)
        np.testing.assert_array_equal(traj["segment"], -1)    # no segment for a learned policy

    def test_only_the_expert_class_gets_the_scenario(self):
        class FakeExpert(CapturingPolicy):
            segment = 1

        class Subclass(FakeExpert):
            pass

        _, traj, expert, _ = run_harness(collision_step=3, policy=FakeExpert(), segment_policy=FakeExpert)
        self.assertEqual(expert.info["scenario"].to_dict()["waypoint_xy"], [1.0, 1.0])
        np.testing.assert_array_equal(traj["segment"], 1)
        _, _, other, _ = run_harness(collision_step=3, policy=Subclass(), segment_policy=FakeExpert)
        self.assertNotIn("scenario", other.info)


class TestPassageAndDocking(unittest.TestCase):
    def test_passage_side_on_synthetic_paths(self):
        sc = replace(scenario.sample_scenario(3000, 0))
        t, n, length = scenario.frame_of(sc.init_xy, sc.dock_xy)
        ob = np.array([sc.obstacles[0]["x"], sc.obstacles[0]["y"]])
        along = float((ob - np.asarray(sc.init_xy)) @ t)
        for side in (1, -1):
            diag = ev.ObstacleDiagnostics(sc)
            for s in np.linspace(0.0, length, 200):    # a path offset 1.2 m to one side of the start->dock line
                diag.update(np.asarray(sc.init_xy) + s * t + side * 1.2 * n)
                if s <= along:
                    self.assertFalse(diag.passed)
            self.assertEqual((diag.passed, diag.side), (True, side))
        stopped = ev.ObstacleDiagnostics(sc)
        for s in np.linspace(0.0, 0.9 * along, 50):
            stopped.update(np.asarray(sc.init_xy) + s * t)
        self.assertEqual((stopped.passed, stopped.side), (False, 0))
        empty = ev.ObstacleDiagnostics(replace(sc, obstacles=[]))
        empty.update(sc.dock_xy)
        self.assertEqual((empty.passed, empty.side), (False, 0))

    def park_path(self):
        """A still base on the parking ring of the run_harness target (standoff(0.6) = 0.65 m)."""
        return lambda k: [1.0 - 0.65, 0.0, 0.2]

    def test_docked_then_contact_is_docked_clean(self):
        row, _, _, _ = run_harness(collision_step=15, path=self.park_path(), moving=False)
        self.assertTrue(row["docked"] and row["docked_clean"] and row["collision"])
        self.assertAlmostEqual(row["time_to_dock"], 1.0)   # end of the 1 s still interval (10 steps)
        self.assertLess(row["time_to_dock"], row["collision_time"])

    def test_contact_before_docking_is_not_docked(self):
        row, _, _, _ = run_harness(collision_step=5, path=self.park_path(), moving=False)
        self.assertFalse(row["docked"] or row["docked_clean"])
        moving, _, _, _ = run_harness(path=self.park_path(), moving=True)
        self.assertFalse(moving["docked"])


# ==============================================================================
# Policy loader
# ==============================================================================
class TestPolicyLoader(unittest.TestCase):
    def test_builtins(self):
        self.assertIs(type(ev.load_policy("expert")), ev.ExpertPolicy)
        self.assertEqual(type(ev.load_policy("zero")).__name__, "ZeroPolicy")

    def test_package_file_and_module_specs(self):
        sys.modules.pop("classify", None)
        path = os.path.join(TASK_DIR, "obstacle_task", "classify.py")
        self.assertTrue(ev.load_policy(f"{path}:_true", "True"))     # factory(arg) is called with --policy-arg
        self.assertNotIn("classify", sys.modules)                     # imported as obstacle_task.classify
        self.assertFalse(ev.load_policy("obstacle_task.classify:_true", "False"))
        with self.assertRaises(ValueError):
            ev.load_policy("no_factory_given")

    def test_other_files_use_the_v1_loader(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "my_policy_v2test.py")
            with open(path, "w") as f:
                f.write("def make(arg):\n    return ('made', arg)\n")
            self.assertEqual(ev.load_policy(f"{path}:make", "m.npz"), ("made", "m.npz"))
        sys.modules.pop("my_policy_v2test", None)


# ==============================================================================
# Collector: planning, resume, guards, no-obstacle sampling
# ==============================================================================
class TestCollectorPlanning(unittest.TestCase):
    def args(self, out_dir, **kw):
        return collect.parse_args(["--set-name", "pilot2n", "--n-episodes", "6", "--out-dir", out_dir,
                                   *sum(([f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
                                         for k, v in kw.items()), [])])

    def test_set_defaults_and_paths(self):
        args = collect.parse_args(["--set-name", "dev2"])
        self.assertEqual((args.seed, args.n_episodes, args.noise_frac, args.max_steps), (300, 300, 0.0, 60000))
        self.assertEqual(args.out_dir, os.path.join("data", "obstacle", "dev2"))
        a2 = collect.parse_args(["--no-obstacle", "--seed", "909"])
        self.assertEqual(a2.max_steps, 40000)

    def test_sealed_guards(self):
        with patch("sys.stderr"):
            for argv in (["--set-name", "test2"], ["--seed", "301"]):
                with self.assertRaises(SystemExit):
                    collect.parse_args(argv)
        self.assertEqual(collect.parse_args(["--set-name", "test2", "--allow-test2"]).seed, 301)
        with tempfile.TemporaryDirectory() as d:
            for sealed in (os.path.join(d, "data", "test"), os.path.join(d, "obstacle", "test2")):
                with self.assertRaises(SystemExit):
                    collect.check_sealed(sealed)
            collect.check_sealed(os.path.join(d, "obstacle", "test2"), allow_test2=True)
            with self.assertRaises(SystemExit):
                collect.check_sealed(os.path.join(d, "data", "test"), allow_test2=True)
            collect.check_sealed(os.path.join(d, "obstacle", "dev2"))

    def test_jobs_noise_and_resume(self):
        with tempfile.TemporaryDirectory() as d:
            args = self.args(d, episodes="0-3", resume=True)
            scs = scenario.make_scenarios(args.seed, args.n_episodes, args.noise_frac)
            # ep 0 saved on its scenario, ep 1 saved on another scenario, ep 2 a broken file, ep 5 saved (not selected)
            for ep, sc in ((0, scs[0]), (1, scs[2]), (5, scs[5])):
                np.savez(os.path.join(d, f"ep_{ep:04d}.npz"), meta=json.dumps({"scenario": sc.to_dict(),
                                                                             "status": "reached"}))
            with open(os.path.join(d, "ep_0002.npz"), "wb") as f:
                f.write(b"truncated")
            rows, jobs = collect.plan_jobs(args, scs)
            self.assertEqual(sorted(row["episode"] for row in rows), [0, 5])
            self.assertEqual([job["info"]["episode"] for job in jobs], [1, 2, 3])
            for job in jobs:
                ep = job["info"]["episode"]
                self.assertEqual(job["kwargs"]["seed"], 3000 * 100_000 + ep)
                self.assertEqual(job["kwargs"]["noise_std"], {"v": 0.02, "w": 0.05, "torque": 2.0})  # all noisy
                self.assertIs(job["kwargs"]["scenario"], scs[ep])
            fresh, fresh_jobs = collect.plan_jobs(self.args(d, episodes="0-3"), scs)          # no --resume
            self.assertEqual((fresh, [j["info"]["episode"] for j in fresh_jobs]), ([], [0, 1, 2, 3]))
            with self.assertRaises(SystemExit):           # a fresh run must not mix with saved episodes
                collect.collect_batch(self.args(d, episodes="0-3"))

    def test_clean_set_has_no_noise(self):
        with tempfile.TemporaryDirectory() as d:
            args = collect.parse_args(["--set-name", "pilot2", "--n-episodes", "4", "--out-dir", d])
            _, jobs = collect.plan_jobs(args, scenario.make_scenarios(args.seed, 4, args.noise_frac))
            self.assertTrue(all(job["kwargs"]["noise_std"] is None for job in jobs))

    def test_no_obstacle_scenarios_follow_collect_data(self):
        args = collect.parse_args(["--no-obstacle", "--seed", "909", "--n-episodes", "5", "--noise-frac", "0.5",
                                   "--init-radius", "2", "--init-yaw-range", "-180", "180"])
        scs = collect.make_no_obstacle_scenarios(args)
        rng = np.random.default_rng(909)            # collect_data.collect_batch's draw order
        r, theta, z = collect_data.sample_targets(5, rng, [0.1, 3.0], [-180.0, 180.0], [0.2, 1.2])
        noisy = rng.random(5) < 0.5
        xy, yaw = collect_data.sample_init_poses(5, rng, 2.0, [-180, 180])
        for i, sc in enumerate(scs):
            np.testing.assert_array_equal(sc.target, collect_data.relative_to_world(xy[i], yaw[i], r[i], theta[i],
                                                                                    z[i]))
            self.assertEqual((sc.init_xy, sc.init_yaw, sc.noisy, sc.obstacles), (xy[i].tolist(), yaw[i], noisy[i], []))


# ==============================================================================
# Failure classes
# ==============================================================================
class TestObstacleClassifier(unittest.TestCase):
    def setUp(self):
        self.row = {"episode": 0, "success": False, "collision": False, "passed_obstacle": True,
                    "stop_reason": "stalled", "ring_dist_start": 3.0, "ring_dist_min": 0.02, "ring_dist_final": 0.10,
                    "base_path_length": 4.0, "base_turn_total_deg": 120.0, "min_error": 0.12,
                    "arm_extend_time": np.nan, "parked_time": 20.0}
        self.ref = {"navdone_ee_err": 0.12, "align_req_deg": 45.0, "orient_req_deg": 45.0}

    def classify(self, **updates):
        return classify.classify_row({**self.row, **updates}, self.ref)

    def test_precedence(self):
        self.assertEqual(self.classify(success=True, collision=True, stop_reason="reached"), "success")
        self.assertEqual(self.classify(stop_reason="harness_error", collision=True), "harness_error")
        for reason in ("tipped", "invalid_action", "policy_error"):
            self.assertEqual(self.classify(stop_reason=reason, collision=True, passed_obstacle=False), "crash")
        self.assertEqual(self.classify(collision=True, passed_obstacle=False), "collision")
        self.assertEqual(self.classify(stop_reason="collision"), "collision")
        self.assertEqual(self.classify(passed_obstacle=False, base_turn_total_deg=900.0), "no_pass")
        self.assertEqual(self.classify(collision="False", passed_obstacle="False"), "no_pass")   # raw CSV strings
        self.assertEqual(classify.FAILURE_CLASSES[:4], ("crash", "collision", "no_pass", "no_start"))

    def test_inherited_classes_after_passing(self):
        self.assertEqual(self.classify(), "parked_no_arm")
        self.assertEqual(self.classify(base_turn_total_deg=451.0), "circling")    # 120 + 360 - 45 - 45 below
        self.assertEqual(self.classify(ring_dist_final=0.31), "left_or_overshot")
        self.assertEqual(self.classify(min_error=0.03, arm_extend_time=20.0), "extended_not_reached")
        self.assertEqual(self.classify(min_error=0.09, arm_extend_time=20.0), "approached_other")
        self.assertEqual(self.classify(ring_dist_min=3.0, ring_dist_final=4.1), "runaway")

    def test_detour_required_turns(self):
        align, orient = classify.required_turns(synthetic_scenario())
        self.assertAlmostEqual(align, 135.0)     # 45 deg at the start + 90 deg at the waypoint
        self.assertAlmostEqual(orient, 45.0)     # from the waypoint->dock heading (-45 deg) to the dock bearing 0
        no_obstacle = classify.required_turns(replace(synthetic_scenario(), obstacles=[]))
        self.assertEqual(no_obstacle, (0.0, 0.0))  # A-to-B: start, dock and target on one line

    def test_validation_on_expert_like_rows(self):
        ok = {**self.row, **self.ref, "success": True, "stop_reason": "reached", "min_error": 0.009,
              "arm_extend_time": 21.0, "navdone_time": 20.5, "ring_dist_final": 0.05}
        fail = {**self.row, **self.ref, "episode": 1, "navdone_time": 20.0, "fclass": "parked_no_arm"}
        lines, passed = classify.validate_expert([{**ok, "fclass": "success"}, fail])
        self.assertTrue(passed, lines)
        self.assertIn("ep 1: parked_no_arm", "\n".join(lines))
        lines, passed = classify.validate_expert([{**ok, "fclass": "success", "passed_obstacle": False}])
        self.assertFalse(passed)                                  # a success that never passed: no_pass fires
        self.assertIn("'no_pass': [0]", "\n".join(lines))
        _, passed = classify.validate_expert([{**ok, "fclass": "success", "arm_extend_time": 20.0}])
        self.assertFalse(passed)                                  # arm extends before nav_done

    def test_run_writes_one_class_per_episode(self):
        with tempfile.TemporaryDirectory() as temp:
            data_dir, run_dir = os.path.join(temp, "data2"), os.path.join(temp, "run")
            os.mkdir(data_dir)
            os.mkdir(run_dir)
            np.savez(os.path.join(data_dir, "ep_0000.npz"), nav_done=[False, True], t=[0.0, 0.001],
                     target=[1.0, 0.0, 0.0], ee_pos=[[0.0, 0.0, 0.0], [0.88, 0.0, 0.0]],
                     meta=json.dumps({"scenario": synthetic_scenario().to_dict()}))
            with open(os.path.join(data_dir, "index.csv"), "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode", "file"])
                writer.writeheader()
                writer.writerow({"episode": 0, "file": "ep_0000.npz"})
            with open(os.path.join(run_dir, "rollouts.csv"), "w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(self.row))
                writer.writeheader()
                writer.writerow(self.row)
            with open(os.path.join(run_dir, "config.json"), "w") as stream:
                json.dump({"args": {"eval_set": data_dir}}, stream)
            rows, counts, lines = classify.classify_run(run_dir, validate=True)
            self.assertEqual(counts, {"parked_no_arm": 1})
            self.assertAlmostEqual(rows[0]["navdone_ee_err"], 0.12)
            with open(os.path.join(run_dir, "classes.csv"), newline="") as stream:
                self.assertEqual(next(csv.DictReader(stream))["fclass"], "parked_no_arm")
            self.assertTrue(os.path.exists(os.path.join(run_dir, "validate_expert.txt")))
            with self.assertRaises(SystemExit):     # a test2 set is refused before anything is read
                classify.classify_run(run_dir, eval_dir=os.path.join(temp, "test2"))


if __name__ == "__main__":
    unittest.main()
