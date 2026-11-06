"""Unit tests of the experiment-B hybrid policy (policies/hybrid.py).

  (h1) base=expert;arm=expert reproduces the recorded expert step by step in closed loop on 5 DEV episodes
       (RUN_SLOW=1 only: runs evaluate_policy.py with 4 workers, about 1 min)
  (h2) base=M;arm=M gives exactly BCPolicy's commands; base=M1;arm=M2 takes each half from its model
  (h3) HYBRID_EVENT lines: JSON with the documented keys; phase / nav_done / arm_trigger times equal the expert's
       own state; nav_done trigger "threshold" for a normal episode and "timeout" for a 25 s navigation timeout;
       the "still" trigger
  (h4) the experiment-B cells: base=M;arm=expert gives BCPolicy's wheels and ExpertPolicy's torques, and
       base=expert;arm=M ExpertPolicy's wheels (0 after navigation done) and BCPolicy's torques, bit for bit at
       every step of a recorded episode
Also (offline, fast): base=expert;arm=expert gives bit-identical commands to ExpertPolicy on a recorded episode.

The offline tests feed recorded data/pilot observations to the policies (no simulation step); the time t is
accumulated in float64 as evaluate_policy.py does, not read from the float32 recording.

  python -m unittest tests/test_hybrid.py -v
  RUN_SLOW=1 python -m unittest tests/test_hybrid.py -v
"""
import csv
import io
import json
import math
import os
import subprocess
import sys
import unittest
from contextlib import redirect_stdout

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
POLICY_DIR = os.path.join(TASK_DIR, "policies")
for _d in (TASK_DIR, POLICY_DIR, TESTS_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

from bc_policy import BCPolicy  # noqa: E402
from hybrid import EVENT_PREFIX, HybridPolicy, make_policy, parse_arg  # noqa: E402
from test_policies import (  # noqa: E402
    PILOT_DIR, TEST_MODELS, TMP_DIR, is_clean, load_episode, make_test_models, obs_at, pilot_index,
)

EVENT_KEYS = {
    "config": {"event", "base", "arm", "trigger"},
    "phase": {"event", "t", "phase"},
    "nav_done": {"event", "t", "trigger"},
    "arm_trigger": {"event", "t", "source"},
}
PHASE_NAMES = ["align", "drive", "orient"]  # recorded phase codes 0-2 (3 = reach, i.e. nav_done)


def parse_events(text):
    """HYBRID_EVENT lines of a log -> list of dicts; checks the documented keys and value types."""
    events = []
    for line in text.splitlines():
        if line.startswith(EVENT_PREFIX):
            event = json.loads(line[len(EVENT_PREFIX):])
            kind = event.get("event")
            if kind not in EVENT_KEYS or set(event) != EVENT_KEYS[kind]:
                raise AssertionError(f"bad event keys: {line}")
            if "t" in event and not isinstance(event["t"], (int, float)):
                raise AssertionError(f"t is not a number: {line}")
            if kind == "phase" and event["phase"] not in PHASE_NAMES:
                raise AssertionError(f"bad phase: {line}")
            if kind == "nav_done" and event["trigger"] not in ("threshold", "timeout"):
                raise AssertionError(f"bad nav_done trigger: {line}")
            if kind == "arm_trigger" and event["source"] not in ("fsm", "still"):
                raise AssertionError(f"bad arm_trigger source: {line}")
            events.append(event)
    return events


def sim_info(episode):
    """info of evaluate_policy.setup_simulation at the start pose of an evaluation-set episode."""
    from evaluate_policy import setup_simulation
    with redirect_stdout(io.StringIO()):
        _, client, _, info = setup_simulation(episode["init_xy"], episode["init_yaw"])
    client.disconnect()
    return info


def pilot_eval_episodes():
    from evaluate_policy import load_eval_set
    with redirect_stdout(io.StringIO()):
        episodes, _ = load_eval_set(PILOT_DIR)
    return episodes


def feed(policies, ep, n_steps, time_step=0.001):
    """Feed recorded steps 0..n_steps-1 to every policy (each gets its own obs copy).
    Yields (k, t, [outputs per policy])."""
    t = 0.0
    for k in range(n_steps):
        obs = obs_at(ep, k, t=t)
        outs = [policy.act({key: np.copy(value) for key, value in obs.items()}) for policy in policies]
        yield k, t, outs
        t += time_step


class TestParseArg(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_arg("base=expert;arm=expert"), {"base": "expert", "arm": "expert", "trigger": "fsm"})
        self.assertEqual(parse_arg(" base = a.npz ; arm=expert;trigger=still "),
                         {"base": "a.npz", "arm": "expert", "trigger": "still"})
        for bad in (None, "", "base=expert", "arm=expert", "base=expert;arm=expert;trigger=clock",
                    "base=expert;arm=expert;foo=1", "base=expert;arm", "base=a.npz;arm=a.npz;trigger=still"):
            with self.assertRaises(ValueError, msg=bad):
                parse_arg(bad)


class TestModelHalves(unittest.TestCase):
    """(h2) A model side gives exactly BCPolicy's commands."""

    @classmethod
    def setUpClass(cls):
        make_test_models()
        index = pilot_index()
        i = next(i for i, row in sorted(index.items()) if not is_clean(row) and row["status"] == "reached")
        cls.ep = load_episode(i)
        cls.info = {"init_base_pos": cls.ep["base_pos"][0].astype(float),
                    "init_base_bearing": float(cls.ep["base_bearing"][0])}

    def run_cell(self, arg, n_steps=200):
        hyb = make_policy(arg)
        with redirect_stdout(io.StringIO()) as out:
            hyb.reset(np.asarray(self.ep["target"], dtype=float), dict(self.info))
            outputs = [outs[0] for _, _, outs in feed([hyb], self.ep, n_steps)]
        return hyb, outputs, parse_events(out.getvalue())

    def test_same_model_both_sides(self):
        path = TEST_MODELS["ego"]
        hyb, outputs, events = self.run_cell(f"base={path};arm={path}")
        self.assertIs(hyb.base_model, hyb.arm_model)  # loaded once
        ref = BCPolicy(path)
        for (k, _, (expected,)), (wheels, tau) in zip(feed([ref], self.ep, 200), outputs):
            self.assertTrue(np.array_equal(wheels, expected[0]) and np.array_equal(tau, expected[1]), f"step {k}")
        self.assertEqual(len(outputs), 200)
        self.assertEqual(events[0], {"event": "config", "base": path, "arm": path, "trigger": "fsm"})

    def test_different_models(self):
        base_path, arm_path = TEST_MODELS["raw"], TEST_MODELS["ego_novw"]
        hyb, outputs, _ = self.run_cell(f"base={base_path};arm={arm_path}")
        self.assertIsNot(hyb.base_model, hyb.arm_model)
        base_ref, arm_ref = BCPolicy(base_path), BCPolicy(arm_path)
        for (k, _, (b, a)), (wheels, tau) in zip(feed([base_ref, arm_ref], self.ep, 200), outputs):
            self.assertTrue(np.array_equal(wheels, b[0]), f"step {k}")
            self.assertTrue(np.array_equal(tau, a[1]), f"step {k}")


class TestExpertCells(unittest.TestCase):
    """base=expert;arm=expert == ExpertPolicy bit for bit, and (h3) the shadow FSM's events."""

    @classmethod
    def setUpClass(cls):
        make_test_models()
        cls.index = pilot_index()
        cls.eval_episodes = pilot_eval_episodes()
        # a normal clean episode (reached, no navigation timeout) and a clean one with a navigation timeout
        cls.normal = next(i for i, row in sorted(cls.index.items())
                          if is_clean(row) and row["status"] == "reached" and row["nav_timed_out"] == "False")
        cls.timeout = next(i for i, row in sorted(cls.index.items())
                           if is_clean(row) and row["nav_timed_out"] == "True")
        cls.info = {i: sim_info(cls.eval_episodes[i]) for i in (cls.normal, cls.timeout)}

    def test_expert_expert_identical_and_events(self):
        from evaluate_policy import ExpertPolicy
        ep, info = load_episode(self.normal), self.info[self.normal]
        target = np.asarray(ep["target"], dtype=float)
        expert, hyb = ExpertPolicy(), make_policy("base=expert;arm=expert")
        expert.reset(target.copy(), info)
        expert_phases, expert_done_t = [], None
        with redirect_stdout(io.StringIO()) as out:
            hyb.reset(target.copy(), info)
            for k, t, ((w_e, tau_e), (w_h, tau_h)) in feed([expert, hyb], ep, len(ep["t"])):
                self.assertTrue(np.array_equal(w_e, w_h), f"wheels differ at step {k}")
                self.assertTrue(np.array_equal(tau_e, tau_h), f"torques differ at step {k}")
                if expert_done_t is None:
                    if not expert_phases or expert_phases[-1][1] != expert.base_state.phase:
                        expert_phases.append((round(t, 6), expert.base_state.phase))
                    if expert.base_state.done:
                        expert_done_t = round(t, 6)
        events = parse_events(out.getvalue())
        self.assertEqual(events[0], {"event": "config", "base": "expert", "arm": "expert", "trigger": "fsm"})
        phases = [(e["t"], e["phase"]) for e in events if e["event"] == "phase"]
        self.assertEqual(phases, expert_phases)
        # the recorded phase timeline of the data collection
        codes = ep["phase"].astype(int)
        changes = [0] + list(np.flatnonzero(np.diff(codes)) + 1)
        recorded = [(round(k * 0.001, 6), PHASE_NAMES[codes[k]]) for k in changes if codes[k] < 3]
        self.assertEqual(phases, recorded)
        nav_done = [e for e in events if e["event"] == "nav_done"]
        arm = [e for e in events if e["event"] == "arm_trigger"]
        self.assertEqual(len(nav_done), 1)
        self.assertEqual(nav_done[0]["trigger"], "threshold")
        self.assertEqual(nav_done[0]["t"], expert_done_t)
        self.assertAlmostEqual(nav_done[0]["t"], ep["meta"]["nav_done_time"], places=6)
        self.assertEqual(arm, [{"event": "arm_trigger", "t": expert_done_t, "source": "fsm"}])

    def test_expert_model_cells(self):
        """(h4) The two experiment-B cells on a whole recorded episode, bit for bit at every step:
        base=M;arm=expert gives BCPolicy's wheels and ExpertPolicy's torques (expert arm state: q_des, hold, reach
        latch from the shadow FSM), base=expert;arm=M gives ExpertPolicy's wheels (exactly 0 once its navigation
        is done) and BCPolicy's torques."""
        from evaluate_policy import ExpertPolicy
        ep, info = load_episode(self.normal), self.info[self.normal]
        target = np.asarray(ep["target"], dtype=float)
        cells = ((TEST_MODELS["ego"], "arm"),         # base=M;arm=expert
                 (TEST_MODELS["ego_novw"], "base"))   # base=expert;arm=M
        for path, expert_side in cells:
            arg = f"base={path};arm=expert" if expert_side == "arm" else f"base=expert;arm={path}"
            with self.subTest(arg=arg):
                expert, model, hyb = ExpertPolicy(), BCPolicy(path), make_policy(arg)
                done_steps = 0
                with redirect_stdout(io.StringIO()):
                    for policy in (expert, model, hyb):
                        policy.reset(target.copy(), info)
                    for k, _, ((w_e, tau_e), (w_m, tau_m), (w_h, tau_h)) in feed([expert, model, hyb], ep,
                                                                                    len(ep["t"])):
                        w_ref, tau_ref = (w_m, tau_e) if expert_side == "arm" else (w_e, tau_m)
                        self.assertTrue(np.array_equal(w_h, w_ref), f"wheels differ at step {k}")
                        self.assertTrue(np.array_equal(tau_h, tau_ref), f"torques differ at step {k}")
                        if expert.base_state.done:
                            done_steps += 1
                            if expert_side == "base":
                                self.assertFalse(np.any(np.asarray(w_h) != 0.0), f"wheels not 0 at step {k}")
                # the expert's navigation finished within the episode, so the reach (arm=expert) and the stopped
                # base (base=expert) are both exercised
                self.assertGreater(done_steps, 1000)

    def test_timeout_trigger(self):
        """A clean episode whose navigation timed out: nav_done by the 25 s timeout, no arm event (model arm)."""
        ep = load_episode(self.timeout)
        self.assertGreater(ep["meta"]["nav_done_time"], 25.0)
        hyb = make_policy(f"base=expert;arm={TEST_MODELS['ego']}")
        with redirect_stdout(io.StringIO()) as out:
            hyb.reset(np.asarray(ep["target"], dtype=float), self.info[self.timeout])
            n = int(round(ep["meta"]["nav_done_time"] / 0.001)) + 100
            for _ in feed([hyb], ep, n):
                pass
        events = parse_events(out.getvalue())
        nav_done = [e for e in events if e["event"] == "nav_done"]
        self.assertEqual(len(nav_done), 1)
        self.assertEqual(nav_done[0]["trigger"], "timeout")
        self.assertAlmostEqual(nav_done[0]["t"], ep["meta"]["nav_done_time"], places=6)
        self.assertFalse([e for e in events if e["event"] == "arm_trigger"])

    def run_synthetic(self, arg, xy_rate, yaw_rate, n_steps, move=(300, 500)):
        """Synthetic observations: the base moves at (xy_rate m/s, yaw_rate rad/s) during steps move[0]..move[1],
        and is still otherwise. Returns the events."""
        info = self.info[self.normal]
        target = np.asarray(self.eval_episodes[self.normal]["target"], dtype=float)
        hyb = make_policy(arg)
        base0, yaw0, q0 = info["init_base_pos"], info["init_base_bearing"], info["init_q_mes"]
        with redirect_stdout(io.StringIO()) as out:
            hyb.reset(target, info)
            t = 0.0
            for k in range(n_steps):
                moved = 0.001 * (min(max(k, move[0]), move[1]) - move[0])
                pos = base0 + np.array([xy_rate * moved, 0.0, 0.0])
                yaw = yaw0 + yaw_rate * moved
                quat = np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)])
                hyb.act({"t": t, "base_pos": pos, "base_ori": quat, "base_bearing": yaw, "q_mes": q0.copy(),
                         "qd_mes": np.zeros(17), "target": target.copy()})
                t += 0.001
        return parse_events(out.getvalue())

    def test_still_trigger(self):
        arg = "base=expert;arm=expert;trigger=still"
        # still 0-0.3 s (too short), moving 0.3-0.5 s at 2 cm/s, still again: fires 0.5 s after the stop
        events = self.run_synthetic(arg, 0.02, 0.0, 1200)
        self.assertEqual(events[0]["trigger"], "still")
        self.assertEqual([e for e in events if e["event"] == "arm_trigger"],
                         [{"event": "arm_trigger", "t": 1.0, "source": "still"}])
        self.assertTrue([e for e in events if e["event"] == "phase"])  # the shadow FSM still runs
        # turning at 0.03 rad/s for the whole run: never still
        events = self.run_synthetic(arg, 0.0, 0.03, 1200, move=(0, 1200))
        self.assertFalse([e for e in events if e["event"] == "arm_trigger"])
        # 0.5 cm/s and 0.01 rad/s count as still: fires at t = 0.5 s
        events = self.run_synthetic(arg, 0.005, 0.01, 1200, move=(0, 1200))
        self.assertEqual([e["t"] for e in events if e["event"] == "arm_trigger"], [0.5])


@unittest.skipUnless(os.environ.get("RUN_SLOW") == "1", "closed-loop run, set RUN_SLOW=1")
class TestClosedLoopExpertExpert(unittest.TestCase):
    """(h1) base=expert;arm=expert in evaluate_policy.py: identical to the recorded expert on 5 DEV episodes
    (normal ones, one that reached after a 25 s navigation timeout, one the expert fails)."""

    EPISODES = [0, 1, 2, 77, 169]
    OUT = os.path.join(TMP_DIR, "hyb_EE_dev5")

    def test_reproduces_expert(self):
        cmd = [sys.executable, "evaluate_policy.py", "--policy", "policies/hybrid.py:make_policy",
               "--policy-arg", "base=expert;arm=expert", "--name", "hybrid-EE", "--eval-set", "data/dev",
               "--episodes", ",".join(map(str, self.EPISODES)), "--compare-recorded", "--no-early-stop",
               "--workers", "4", "--out-dir", self.OUT]
        result = subprocess.run(cmd, cwd=TASK_DIR, capture_output=True, text=True, timeout=600)
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])
        with open(os.path.join(self.OUT, "rollouts.csv")) as f:
            rows = {int(r["episode"]): r for r in csv.DictReader(f)}
        with open(os.path.join(TASK_DIR, "results", "expert_dev", "rollouts.csv")) as f:
            expert = {int(r["episode"]): r for r in csv.DictReader(f)}
        self.assertEqual(sorted(rows), self.EPISODES)
        for i in self.EPISODES:
            row, ref = rows[i], expert[i]
            self.assertEqual(row["cmp_identical"], "True", f"episode {i}")
            self.assertEqual(row["success"], ref["success"], f"episode {i}")
            t, t_ref = float(row["t_reach"]), float(ref["t_reach"])
            self.assertTrue((math.isnan(t) and math.isnan(t_ref)) or abs(t - t_ref) < 1e-9, f"episode {i}")
            with open(os.path.join(self.OUT, "logs", f"ep_{i:04d}.log")) as f:
                events = parse_events(f.read())
            nav_done = [e for e in events if e["event"] == "nav_done"]
            arm = [e for e in events if e["event"] == "arm_trigger"]
            self.assertEqual(len(nav_done), 1, f"episode {i}")
            expected = "timeout" if ref["expert_nav_timed_out_recorded"] == "True" else "threshold"
            self.assertEqual(nav_done[0]["trigger"], expected, f"episode {i}")
            self.assertEqual([(e["t"], e["source"]) for e in arm], [(nav_done[0]["t"], "fsm")], f"episode {i}")


if __name__ == "__main__":
    unittest.main()
