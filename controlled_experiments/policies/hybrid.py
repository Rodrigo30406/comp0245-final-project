"""Hybrid policy for experiment B (base / arm 2x2 split): the expert drives one half, a learned model the other.
Diagnostic only: the expert half uses info (robot model, start pose), which a learned policy does not get.

  python evaluate_policy.py --policy policies/hybrid.py:make_policy --no-early-stop \
      --policy-arg "base=expert;arm=trained/ego_s3.npz" --eval-set data/pilot --workers 4 --out-dir results/<run>

Argument: "base=<expert|path.npz>;arm=<expert|path.npz>[;trigger=fsm|still]" (default trigger fsm; paths must not
contain ';' or '='). When both sides name the same model file it is loaded once and evaluated once per step.

Shadow expert, run at every step whatever the cell:
  - the expert base FSM (BaseNavState + base_controller, goal from evaluate_policy.expert_base_goal with the start
    pose of info, BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)) on the actual measured base state, exactly as
    evaluate_policy.ExpertPolicy. Its wheel commands are applied only when base=expert;
  - when arm=expert, the expert arm (arm_controller, ArmControllerParams(reach_limit=ARM_REACH_LIMIT), same reference
    state as ExpertPolicy). Its reach_active is
      trigger=fsm:   the shadow FSM's done (as the expert; with a model base this is the FSM's verdict on the model's
                     base motion: orient tolerance reached, or the 25 s navigation timeout),
      trigger=still: latched once the base has been still for 0.5 s (planar speed < 0.01 m/s and |yaw rate| < 0.02
                     rad/s, finite differences of obs base_pos / base_bearing; diagnostic only). There is no
                     "has moved" requirement: a base that does not move for the first 0.5 s triggers at t = 0.5 s.
A model side uses BCPolicy (policies/bc_policy.py): same features, forward pass and (v, w) -> wheel conversion.
With base=expert;arm=expert the commands are bit-identical to ExpertPolicy (same operations in the same order).

Events, one line each on stdout (the harness writes each episode's stdout to <run>/logs/ep_XXXX.log):
  HYBRID_EVENT {"event": "config", "base": ..., "arm": ..., "trigger": ...}        at reset
  HYBRID_EVENT {"event": "phase", "t": <s>, "phase": "align"|"drive"|"orient"}     shadow FSM phase after the first
                                                                                   step, then at every change
  HYBRID_EVENT {"event": "nav_done", "t": <s>, "trigger": "threshold"|"timeout"}   shadow FSM done: by the orient
                                                                                   tolerance, or by the 25 s timeout
  HYBRID_EVENT {"event": "arm_trigger", "t": <s>, "source": "fsm"|"still"}         expert arm starts reaching
                                                                                   (only when arm=expert)
t is obs["t"] of the step at which the change is first seen (the command of that step comes from the new state).

The hybrid uses time (the 25 s navigation timeout of the shadow FSM), so it is always evaluated with
--no-early-stop (EVALUATION.md: a policy that uses t could wait for a time-triggered event).
"""
import json
import os
import sys

_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_DIR = os.path.dirname(_POLICY_DIR)
for _d in (_TASK_DIR, _POLICY_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402
import pinocchio as pin  # noqa: E402
from simulation_and_control import wrap_angle  # noqa: E402

from bc_policy import BCPolicy  # noqa: E402
from evaluate_policy import expert_base_goal  # noqa: E402
from robotnik_arm_control import (  # noqa: E402
    ArmControllerParams, BaseControllerParams, BaseNavState, arm_controller, base_controller,
)
from robotnik_arm_task import ARM_REACH_LIMIT, BASE_NAV_TIMEOUT  # noqa: E402

EVENT_PREFIX = "HYBRID_EVENT "
TRIGGERS = ("fsm", "still")
STILL_LIN = 0.01     # m/s, as evaluate_policy.STILL_LIN
STILL_ANG = 0.02     # rad/s, as evaluate_policy.STILL_ANG
STILL_HOLD = 0.5     # s the base must stay still before the "still" trigger fires


def parse_arg(arg):
    """'base=...;arm=...[;trigger=...]' -> {'base': 'expert' | path, 'arm': 'expert' | path, 'trigger': ...}."""
    if not arg:
        raise ValueError('hybrid policy needs --policy-arg "base=<expert|path.npz>;arm=<expert|path.npz>'
                         '[;trigger=fsm|still]"')
    spec = {"trigger": "fsm"}
    for part in str(arg).split(";"):
        if not part.strip():
            continue
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or key not in ("base", "arm", "trigger") or not value:
            raise ValueError(f"bad hybrid policy argument {arg!r}: expected base=...;arm=...[;trigger=fsm|still]")
        spec[key] = value
    if "base" not in spec or "arm" not in spec:
        raise ValueError(f"hybrid policy argument {arg!r} must give both base= and arm=")
    if spec["trigger"] not in TRIGGERS:
        raise ValueError(f"trigger must be one of {TRIGGERS}, got {spec['trigger']!r}")
    if spec["trigger"] != "fsm" and spec["arm"] != "expert":
        raise ValueError("trigger selects when the expert arm starts reaching: it needs arm=expert")
    return spec


def emit(event, **fields):
    print(EVENT_PREFIX + json.dumps({"event": event, **fields}), flush=True)


class HybridPolicy:
    def __init__(self, arg):
        self.spec = parse_arg(arg)
        self.trigger = self.spec["trigger"]
        self.base_expert = self.spec["base"] == "expert"
        self.arm_expert = self.spec["arm"] == "expert"
        models = {}  # realpath -> BCPolicy: a file named on both sides is loaded once
        for side in ("base", "arm"):
            if self.spec[side] != "expert":
                key = os.path.realpath(self.spec[side])
                if key not in models:
                    models[key] = BCPolicy(self.spec[side])
        self.base_model = None if self.base_expert else models[os.path.realpath(self.spec["base"])]
        self.arm_model = None if self.arm_expert else models[os.path.realpath(self.spec["arm"])]

    def reset(self, target, info):
        emit("config", base=self.spec["base"], arm=self.spec["arm"], trigger=self.trigger)
        self.target = target
        # Shadow expert base FSM: as ExpertPolicy.reset
        self.base_goal_pos, self.base_goal_bearing, _ = expert_base_goal(
            target, info["init_base_pos"], info["init_base_bearing"])
        self.base_state = BaseNavState()
        self.base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
        if self.arm_expert:
            self.model = info["dyn_model"]
            self.ee_frame_id = info["ee_frame_id"]
            self.shoulder_frame_id = info["shoulder_frame_id"]
            self.arm_lower = info["joint_lower"][4:11]
            self.arm_upper = info["joint_upper"][4:11]
            self.time_step = info["time_step"]
            self.arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
            self.q_init_arm = info["init_q_mes"][4:11].copy()
            self.q_des_arm = self.q_init_arm.copy()
            self.qd_des_arm = np.zeros(7)
        for model in {id(m): m for m in (self.base_model, self.arm_model) if m is not None}.values():
            model.reset(target, info)
        # Event bookkeeping
        self.last_phase = None
        self.nav_done_seen = False
        self.reach_active = False
        # "still" trigger: finite-difference base speeds and the time since the base became still
        self.prev_t, self.prev_xy, self.prev_bearing = None, None, None
        self.still_since = None

    def _base_still(self, obs):
        """Latched 'base still for STILL_HOLD s' trigger, from finite differences of the measured base pose."""
        t, xy, bearing = float(obs["t"]), np.array(obs["base_pos"], dtype=float)[:2], float(obs["base_bearing"])
        prev_t, still = self.prev_t, False
        if prev_t is not None and t > prev_t:
            dt = t - prev_t
            still = (np.linalg.norm(xy - self.prev_xy) / dt < STILL_LIN
                     and abs(wrap_angle(bearing - self.prev_bearing)) / dt < STILL_ANG)
        self.prev_t, self.prev_xy, self.prev_bearing = t, xy, bearing
        if not still:
            self.still_since = None
            return False
        if self.still_since is None:
            self.still_since = prev_t  # still over the whole interval since the previous sample
        return t - self.still_since >= STILL_HOLD - 1e-9  # tolerance for the accumulated float time

    def act(self, obs):
        t = obs["t"]
        if self.arm_expert:  # as ExpertPolicy.act
            q_mes, qd_mes = obs["q_mes"], obs["qd_mes"]
            q_pin = self.model.ReoderJoints2PinVec(np.concatenate((obs["base_pos"], obs["base_ori"], q_mes)), "pos")
            pin.forwardKinematics(self.model.pin_model, self.model.pin_data, q_pin)
            pin.updateFramePlacements(self.model.pin_model, self.model.pin_data)
            ee_pos = self.model.pin_data.oMf[self.ee_frame_id].translation.copy()
        expert_wheels = base_controller(self.base_state, obs["base_pos"], obs["base_bearing"], self.base_goal_pos,
                                        self.base_goal_bearing, t, self.base_params)
        self._log_fsm(obs)

        if self.arm_expert:
            if self.trigger == "fsm":
                reach_active = self.base_state.done
            else:
                reach_active = self.reach_active or self._base_still(obs)
            if reach_active and not self.reach_active:
                emit("arm_trigger", t=round(float(t), 6), source=self.trigger)
            self.reach_active = reach_active
            torque_arm, self.q_des_arm, self.qd_des_arm = arm_controller(
                self.model, q_pin, q_mes[4:11], qd_mes[4:11], self.q_des_arm, self.q_init_arm, self.target, ee_pos,
                self.ee_frame_id, self.shoulder_frame_id, self.arm_lower, self.arm_upper, self.time_step,
                self.arm_params, reach_active=reach_active,
            )

        if self.base_model is not None and self.base_model is self.arm_model:
            return self.base_model.act(obs)  # one model on both sides: exactly BCPolicy's commands
        wheel_cmds = expert_wheels if self.base_expert else self.base_model.act(obs)[0]
        if not self.arm_expert:
            torque_arm = self.arm_model.act(obs)[1]
        return wheel_cmds, torque_arm

    def _log_fsm(self, obs):
        """Events of the shadow FSM after this step's base_controller call."""
        t = round(float(obs["t"]), 6)
        state = self.base_state
        if state.phase != self.last_phase:  # base_controller no longer changes the phase once done
            emit("phase", t=t, phase=state.phase)
            self.last_phase = state.phase
        if state.done and not self.nav_done_seen:
            self.nav_done_seen = True
            # base_controller sets done in its orient branch (|bearing error| < ang_tol) or, after it, when
            # t > nav_timeout. The orient test is re-evaluated here on the same inputs.
            at_threshold = (state.phase == "orient" and abs(wrap_angle(self.base_goal_bearing - obs["base_bearing"]))
                            < self.base_params.ang_tol)
            by_threshold = at_threshold or not float(obs["t"]) > self.base_params.nav_timeout
            emit("nav_done", t=t, trigger="threshold" if by_threshold else "timeout")


def make_policy(arg):
    """arg: "base=<expert|path.npz>;arm=<expert|path.npz>[;trigger=fsm|still]" (the --policy-arg string)."""
    return HybridPolicy(arg)
