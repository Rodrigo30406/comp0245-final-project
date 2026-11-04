"""Declared test-time adversities for the closed-loop evaluation (the robustness axis of the brief).

Each adversity is a key of the frozen evaluation protocol, so it is part of the evaluated condition, not of the policy:

  action_delay_steps  : the executed command is the one issued this many physics steps earlier (zeros before the first);
                        expert and policies alike (worker.make_action_filter).
  base_push           : a horizontal force of force_n newtons at the centre of mass of the base body (robot_base_link,
                        25 kg of the robot's 60 kg) for duration_s seconds from time_s, along the heading at the push
                        start, from behind or head-on at random (scenario seed); expert and policies alike. Along the
                        heading because the wheels can answer it: the simulator's wheel contact has almost no lateral
                        friction, so a sideways push makes the base slide indefinitely with zero wheel commands (expert
                        included: 30 N for 0.3 s during the approach ends 3 m from the target); that is a physics
                        artefact, not a controller property, and is not used as an adversity.
  payload_mass_kg     : extra mass on the end-effector link, unknown to the expert's gravity compensation; expert and
                        policies alike.
  actuator_gain       : the executed command is the issued one times wheel_left / wheel_right / arm (a calibration error
                        of the actuators, e.g. the left wheels at 85%); expert and policies alike (make_action_filter),
                        the labels and the policy outputs untouched. Calibration measured 3 Nov: the base push is either
                        negligible (along the heading) or a lateral-friction artefact (sideways), so the gain error is the
                        base adversity of the programme.
  observation_noise   : N(0, s) added to the observation the POLICY reads: base x / y (base_m), bearing (yaw_rad), arm
                        angles (q_rad), arm velocities (qd_rad_s). The target is never noised. The lecturer's expert reads
                        the simulator state, not this vector, so under observation noise the expert row is the clean
                        reference and the policies are the perturbed ones; the summary says so.

All randomness comes from the scenario seed, so every policy and the expert meet the same realisation in a scenario.
"""
import numpy as np
import pybullet as p

ADVERSITY_KEYS = {"action_delay_steps", "base_push", "payload_mass_kg", "observation_noise", "actuator_gain"}
GAIN_KEYS = {"wheel_left", "wheel_right", "arm"}
NOISE_KEYS = {"base_m", "yaw_rad", "q_rad", "qd_rad_s"}
PUSH_KEYS = {"time_s", "duration_s", "force_n"}
END_EFFECTOR_LINK = "robot_j2s7s300_end_effector"
BASE_LINK = "robot_base_link"  # the 25 kg base body; the URDF root (robot_base_footprint) is a massless frame at the ground
SEED_OFFSET = 7919  # the collector/evaluator use np.random.seed(scenario seed); the adversity rng is a separate stream


def _finite(value, *, nonnegative=True):
    return type(value) in (int, float) and np.isfinite(value) and (value >= 0 or not nonnegative)


def validate_adversities(protocol):
    """Type-check the adversity keys of an evaluation protocol (called by cli.validate_protocol)."""
    delay = protocol.get("action_delay_steps", 0)
    if type(delay) is not int or delay < 0:
        raise ValueError("action_delay_steps must be an integer >= 0")
    push = protocol.get("base_push")
    if push is not None:
        if not isinstance(push, dict) or set(push) != PUSH_KEYS or not all(_finite(push[k]) and push[k] > 0 for k in PUSH_KEYS):
            raise ValueError(f"base_push needs positive finite {sorted(PUSH_KEYS)}")
        if push["time_s"] + push["duration_s"] > protocol["horizon_s"]:
            raise ValueError("base_push must end within the horizon")
    payload = protocol.get("payload_mass_kg", 0.0)
    if not _finite(payload):
        raise ValueError("payload_mass_kg must be a finite number >= 0")
    gain = protocol.get("actuator_gain")
    if gain is not None:
        if not isinstance(gain, dict) or not gain or set(gain) - GAIN_KEYS or not all(_finite(v) and v > 0 for v in gain.values()):
            raise ValueError(f"actuator_gain needs a nonempty subset of {sorted(GAIN_KEYS)} with finite positive values")
        if all(v == 1 for v in gain.values()):
            raise ValueError("actuator_gain of 1 everywhere is not a perturbation; drop the key")
    noise = protocol.get("observation_noise")
    if noise is not None:
        if not isinstance(noise, dict) or not noise or set(noise) - NOISE_KEYS or not all(_finite(v) for v in noise.values()):
            raise ValueError(f"observation_noise needs a nonempty subset of {sorted(NOISE_KEYS)} with finite nonnegative values")
        if not any(noise.values()):
            raise ValueError("observation_noise with every deviation zero is not a perturbation; drop the key")


class Adversity:
    """One scenario's realisation of the protocol adversities; see the module docstring."""

    def __init__(self, protocol, seed, *, is_expert):
        validate_adversities(protocol)
        self.delay_steps = int(protocol.get("action_delay_steps", 0))
        self.push = protocol.get("base_push")
        self.payload = float(protocol.get("payload_mass_kg", 0.0))
        gain = protocol.get("actuator_gain") or {}
        self.gain = np.ones(11)
        self.gain[0] = self.gain[3] = gain.get("wheel_right", 1.0); self.gain[1] = self.gain[2] = gain.get("wheel_left", 1.0)
        self.gain[4:] = gain.get("arm", 1.0)
        self.gain = None if np.all(self.gain == 1) else self.gain
        self.noise = None if is_expert else protocol.get("observation_noise")
        self.is_expert = is_expert
        self.rng = np.random.default_rng(int(seed) + SEED_OFFSET)
        self.realised = {"action_delay_steps": self.delay_steps, "payload_mass_kg": self.payload, "actuator_gain": protocol.get("actuator_gain"),
                         "observation_noise": self.noise, "base_push": None,
                         "note": ("expert reads the simulator state: observation noise does not apply to it"
                                  if is_expert and protocol.get("observation_noise") else None)}
        self._push_force = None
        self._push_link = None
        self._noise_scale = None
        if self.noise:
            scale = np.zeros(21)
            scale[0:2] = self.noise.get("base_m", 0.0); scale[3] = self.noise.get("yaw_rad", 0.0)
            scale[4:11] = self.noise.get("q_rad", 0.0); scale[11:18] = self.noise.get("qd_rad_s", 0.0)
            self._noise_scale = scale

    @property
    def active(self):
        return bool(self.delay_steps or self.push or self.payload or self.noise or self.gain is not None)

    @staticmethod
    def _link(episode, name):
        client, rid = episode.client, episode.robot_id
        link = next((j for j in range(client.getNumJoints(rid)) if client.getJointInfo(rid, j)[12].decode() == name), None)
        if link is None:
            raise ValueError(f"{name} link not found")
        return link

    def setup(self, episode):
        if self.push is not None:
            self._push_link = self._link(episode, BASE_LINK)
        if self.payload:
            client, rid = episode.client, episode.robot_id
            link = self._link(episode, END_EFFECTOR_LINK)
            original = client.getDynamicsInfo(rid, link)[0]
            client.changeDynamics(rid, link, mass=original + self.payload)
            self.realised["payload_mass_kg"] = {"added_kg": self.payload, "link": END_EFFECTOR_LINK,
                                                "link_mass_kg": original + self.payload}

    def before_step(self, episode):
        if self.push is None:
            return
        t = episode.state["time"]
        start = self.push["time_s"]
        if t + 1e-9 < start or t >= start + self.push["duration_s"] - 1e-9:
            return
        if self._push_force is None:  # first pushed step: fix the sense (from behind or head-on) and the world direction
            sense = int(self.rng.choice([-1, 1]))
            heading = float(episode.state["yaw"]) + (0.0 if sense > 0 else np.pi)
            self._push_force = [self.push["force_n"] * np.cos(heading), self.push["force_n"] * np.sin(heading), 0.0]
            self.realised["base_push"] = {**self.push, "start_time_s": float(t), "sense": "from_behind" if sense > 0 else "head_on",
                                          "world_direction_rad": float(heading),
                                          "base_xy_at_start": [float(v) for v in episode.state["base"][:2]]}
        # at the base body's centre of mass, so the push translates the robot rather than tipping it
        position = episode.client.getLinkState(episode.robot_id, self._push_link)[0]
        episode.client.applyExternalForce(episode.robot_id, self._push_link, self._push_force, list(position), p.WORLD_FRAME)

    def observe(self, observation):
        """The observation handed to a learned policy (noised when observation_noise is set and this is not the expert)."""
        if self._noise_scale is None:
            return observation
        return observation + self.rng.normal(size=21) * self._noise_scale

    def summary(self):
        return self.realised
