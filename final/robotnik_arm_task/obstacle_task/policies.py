"""Closed-loop policies of the step-9 learners (Track B), run by obstacle_task/evaluate.py (policy interface of
evaluate_policy.py: reset(target, info), act(obs) -> (wheel_cmds (4,), torque_arm (7,)), one call per 1 ms step).

  python obstacle_task/evaluate.py --policy obstacle_task/policies.py:make_two_net \
      --policy-arg "base=trained/step9/L1base_s0.npz,arm=trained/step9/L1arm_s0.npz" ...
  python obstacle_task/evaluate.py --policy obstacle_task/policies.py:make_config --policy-arg "L2,seed=0" ...

Policies (the plan's configs, STEP9_PLAN_PROMPT.md section 5; POLICY_MODELS maps them onto the trained networks):
  SinglePolicy       L0     one network single84 -> (v, w, tau1..tau7)
  TwoNetPolicy       L1     base network base70 -> (v, w) and arm network arm20 -> tau1..tau7, called on the same
                            observation; L1-K0 is the same class with the arm network trained without copies
  SideLatchedPolicy  L2     at the first act() of an episode the side classifier (base70) gives p(left) from the
                            first observation and the side is latched, +1 if p_left >= 0.5 else -1, for the whole
                            episode; the side-conditioned base network (base70_side) -> (v, w); arm network arm20
At every step the features are built from the observation by exactly the training functions
(features.obs_signals + features.build_features), the networks run in numpy (features.forward) and (v, w) becomes the
4 wheel speeds with collect_data.wheel_speeds and BaseControllerParams(nav_timeout=25.0), as policies/bc_policy.py.

Inputs: only obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"], obs["target"] and obs["lidar"] (the
step-9 harness's 64-beam scan). Nothing from info and no clock (obs["t"] is not read). SinglePolicy and TwoNetPolicy
are stateless; SideLatchedPolicy keeps the latched side (exposed as latched_side / p_left / episode_log() for logging)
and needs reset() at the start of every episode.

The harness imports this file as a module named 'policies' (evaluate_policy.load_policy), so it imports nothing from
the policies/ package and reaches the task modules through absolute imports.
"""
import os
import sys

_TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TASK_DIR not in sys.path:  # evaluate_policy.load_policy only adds obstacle_task/ to sys.path
    sys.path.insert(0, _TASK_DIR)

import numpy as np  # noqa: E402

from collect_data import wheel_speeds  # noqa: E402
from obstacle_task import features as obf  # noqa: E402
from robotnik_arm_control import BaseControllerParams  # noqa: E402

# Skid-steer model used to turn the predicted (v, w) into wheel speeds (EVALUATION.md, policy interface)
BASE_PARAMS = BaseControllerParams(nav_timeout=25.0)
TRAINED_DIR = os.path.join(_TASK_DIR, "trained", "step9")
LEFT_THRESHOLD = 0.5      # latched side = +1 (pass left) if p_left >= LEFT_THRESHOLD, else -1

# Policy -> its networks (role -> train.CONFIGS name), for make_config
POLICY_MODELS = {
    "L0": {"model": "L0"},
    "L1": {"base": "L1base", "arm": "L1arm"},
    "L1-K0": {"base": "L1base", "arm": "L1armK0"},
    "L2": {"cls": "L2cls", "base": "L2base", "arm": "L1arm"},
    # maze networks (train.py CONFIGS M*): deeper base networks, same policy classes
    "M1": {"base": "M1base", "arm": "L1arm"},
    "M2": {"cls": "M2cls", "base": "M2base", "arm": "L1arm"},
}


def load_net(path, kind, feature_set, outputs):
    """A model file of features.save_model, checked to be the expected network (kind, feature set, outputs)."""
    model = obf.load_model(os.path.abspath(path))
    got = (model["kind"], model["feature_set"], model["output_names"])
    if got != (kind, feature_set, obf.OUTPUTS[outputs]):
        raise ValueError(f"{path}: a {got[0]} {got[1]} -> {got[2]} network, expected {kind} {feature_set} -> "
                         f"{obf.OUTPUTS[outputs]}")
    return model


class SinglePolicy:
    """L0: one network single84 -> (v, w, tau1..tau7). Stateless."""

    def __init__(self, model_path):
        self.model_path = os.path.abspath(model_path)
        self.model = load_net(model_path, "regressor", "single84", "all")
        self.base_params = BASE_PARAMS

    def reset(self, target, info):
        """The target is read from obs at every step; info is not used."""

    def features(self, obs):
        """Network inputs (1, F) of one observation, by name of the network."""
        return {"model": obf.build_features("single84", obf.obs_signals(obs))}

    def predict(self, obs):
        """(9,): v (m/s), w (rad/s), tau1..tau7 (N m)."""
        return obf.forward(self.model, self.features(obs)["model"])[0]

    def act(self, obs):
        y = self.predict(obs)
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]


class TwoNetPolicy:
    """L1 / L1-K0: base network base70 -> (v, w) and arm network arm20 -> tau1..tau7. Stateless."""

    def __init__(self, base_path, arm_path):
        self.base_path, self.arm_path = os.path.abspath(base_path), os.path.abspath(arm_path)
        self.base = load_net(base_path, "regressor", "base70", "vw")
        self.arm = load_net(arm_path, "regressor", "arm20", "tau")
        self.base_params = BASE_PARAMS

    def reset(self, target, info):
        """The target is read from obs at every step; info is not used."""

    def features(self, obs):
        sig = obf.obs_signals(obs)
        return {"base": obf.build_features("base70", sig), "arm": obf.build_features("arm20", sig)}

    def predict(self, obs):
        """(9,): v, w from the base network, tau1..tau7 from the arm network."""
        f = self.features(obs)
        return np.concatenate([obf.forward(self.base, f["base"])[0], obf.forward(self.arm, f["arm"])[0]])

    def act(self, obs):
        y = self.predict(obs)
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]


class SideLatchedPolicy:
    """L2: side classifier (once per episode, at its first act()) + side-conditioned base network + arm network.

    latched_side (+1 pass left, -1 right; None before the first act() of an episode) and p_left are exposed for
    logging; episode_log() returns both."""

    def __init__(self, cls_path, base_path, arm_path):
        self.cls_path, self.base_path, self.arm_path = (os.path.abspath(p) for p in (cls_path, base_path, arm_path))
        self.cls = load_net(cls_path, "classifier", "base70", "side")
        self.base = load_net(base_path, "regressor", "base70_side", "vw")
        self.arm = load_net(arm_path, "regressor", "arm20", "tau")
        self.base_params = BASE_PARAMS
        self.latched_side = self.p_left = None

    def reset(self, target, info):
        """Start of an episode: forget the latched side (the next act() latches a new one). info is not used."""
        self.latched_side = self.p_left = None

    def latch(self, sig):
        """p(left) of the classifier on the signals of one observation -> the latched side."""
        self.p_left = float(obf.forward(self.cls, obf.build_features("base70", sig))[0, 0])
        self.latched_side = 1 if self.p_left >= LEFT_THRESHOLD else -1
        return self.latched_side

    def features(self, obs):
        sig = obf.obs_signals(obs)
        if self.latched_side is None:
            self.latch(sig)
        return {"base": obf.build_features("base70_side", sig, side=self.latched_side),
                "arm": obf.build_features("arm20", sig)}

    def predict(self, obs):
        f = self.features(obs)
        return np.concatenate([obf.forward(self.base, f["base"])[0], obf.forward(self.arm, f["arm"])[0]])

    def act(self, obs):
        y = self.predict(obs)
        return wheel_speeds(y[0], y[1], self.base_params), y[2:9]

    def episode_log(self):
        return {"latched_side": self.latched_side, "p_left": self.p_left}


# ==============================================================================
# Factories (--policy obstacle_task/policies.py:<factory> --policy-arg <arg>)
# ==============================================================================
def parse_policy_arg(arg, keys):
    """'key=value,key=value' -> dict with exactly the given keys (paths relative to the working directory)."""
    if not arg:
        raise ValueError(f"policy argument needs {', '.join(f'{k}=...' for k in keys)}")
    out = {}
    for part in str(arg).split(","):
        key, sep, value = part.partition("=")
        if not sep or not value.strip():
            raise ValueError(f"policy argument {arg!r}: expected key=value pairs separated by commas")
        out[key.strip()] = value.strip()
    if set(out) != set(keys):
        raise ValueError(f"policy argument {arg!r}: expected the keys {list(keys)}, got {list(out)}")
    return out


def make_single(arg):
    """L0: arg = 'model=path/to/L0_sN.npz' (or just the path)."""
    if arg and "=" not in arg:
        return SinglePolicy(arg)
    return SinglePolicy(parse_policy_arg(arg, ["model"])["model"])


def make_two_net(arg):
    """L1 / L1-K0: arg = 'base=path/to/L1base_sN.npz,arm=path/to/L1arm_sN.npz' (or L1armK0_sN.npz)."""
    a = parse_policy_arg(arg, ["base", "arm"])
    return TwoNetPolicy(a["base"], a["arm"])


def make_side_latched(arg):
    """L2: arg = 'cls=path/to/L2cls_sN.npz,base=path/to/L2base_sN.npz,arm=path/to/L1arm_sN.npz'."""
    a = parse_policy_arg(arg, ["cls", "base", "arm"])
    return SideLatchedPolicy(a["cls"], a["base"], a["arm"])


def config_paths(policy, seed, model_dir=TRAINED_DIR):
    """Model files of a policy of POLICY_MODELS for one seed: role -> <model_dir>/<config>_s<seed>.npz."""
    if policy not in POLICY_MODELS:
        raise ValueError(f"unknown policy {policy!r}, expected one of {list(POLICY_MODELS)}")
    return {role: os.path.join(model_dir, f"{config}_s{int(seed)}.npz") for role, config in POLICY_MODELS[policy].items()}


def make_config(arg):
    """A policy of POLICY_MODELS by name and seed: arg = 'L1,seed=0' (optionally ',dir=path', default trained/step9).
    L0 -> SinglePolicy, L1 / L1-K0 -> TwoNetPolicy, L2 -> SideLatchedPolicy."""
    name, _, rest = str(arg or "").partition(",")
    opts = parse_policy_arg(rest, ["seed", "dir"] if "dir=" in rest else ["seed"])
    paths = config_paths(name.strip(), opts["seed"], opts.get("dir", TRAINED_DIR))
    if name.strip() == "L0":
        return SinglePolicy(paths["model"])
    if name.strip() in ("L2", "M2"):
        return SideLatchedPolicy(paths["cls"], paths["base"], paths["arm"])
    return TwoNetPolicy(paths["base"], paths["arm"])
