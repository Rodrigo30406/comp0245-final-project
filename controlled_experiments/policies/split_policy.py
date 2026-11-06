"""Step 11 S: two separately trained networks on the A-to-B task (results/step11/step11.py train-s).

  python evaluate_policy.py --policy policies/split_policy.py:make_policy \
      --policy-arg "base=trained/step11/S-base_s0.npz;arm=trained/step11/S-arm-K16_s0.npz" --eval-set data/dev ...

The base network sees only the target geometry (feature set ego_target = the first six ego_novw columns) and outputs
(v, w); the arm network sees ego_novw (target geometry, arm q, arm qd) and outputs tau1..7. Features, forward pass and
(v, w) -> wheel conversion are BCPolicy's. Stateless; uses no info.
"""
import os
import sys

_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_TASK_DIR = os.path.dirname(_POLICY_DIR)
for _d in (_TASK_DIR, _POLICY_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import numpy as np  # noqa: E402

import bc_features  # noqa: E402
from bc_policy import BASE_PARAMS  # noqa: E402
from collect_data import wheel_speeds  # noqa: E402

def register_ego_target():
    """Add the feature set ego_target (the first six ego_novw columns: the target geometry, no arm state) to
    bc_features without editing that file: FEATURE_SETS gets the names and build_features the matching branch, so the
    registry stays consistent for any code that iterates over FEATURE_SETS. Idempotent."""
    if getattr(bc_features.build_features, "handles_ego_target", False):
        return
    bc_features.FEATURE_SETS["ego_target"] = list(bc_features.FEATURE_SETS["ego_novw"][:6])
    build = bc_features.build_features

    def build_features(feature_set, signals):
        if feature_set == "ego_target":
            return build("ego_novw", signals)[:, :6]
        return build(feature_set, signals)

    build_features.handles_ego_target = True
    build_features.__doc__ = build.__doc__
    bc_features.build_features = build_features


register_ego_target()


class SplitPolicy:
    def __init__(self, base_path, arm_path):
        self.base = bc_features.load_model(os.path.abspath(base_path))
        self.arm = bc_features.load_model(os.path.abspath(arm_path))
        if self.base["feature_set"] != "ego_target" or self.base["meta"].get("output_names") != ["v", "w"]:
            raise ValueError(f"{base_path}: not a step-11 base network (ego_target -> v, w)")
        if (self.arm["feature_set"] != "ego_novw"
                or self.arm["meta"].get("output_names") != bc_features.OUTPUT_NAMES[2:]):
            raise ValueError(f"{arm_path}: not a step-11 arm network (ego_novw -> tau1..7)")
        self.base_params = BASE_PARAMS

    def reset(self, target, info):
        pass

    def predict(self, obs):
        """(v, w) from the base network and tau1..7 from the arm network, physical units."""
        signals = bc_features.measured_signals(obs["base_pos"], obs["base_bearing"], obs["q_mes"], obs["qd_mes"],
                                               obs["target"])
        F = bc_features.build_features("ego_novw", signals)
        vw = bc_features.forward(self.base, F[:, :6])[0]
        tau = bc_features.forward(self.arm, F)[0]
        return vw, tau

    def act(self, obs):
        vw, tau = self.predict(obs)
        return wheel_speeds(vw[0], vw[1], self.base_params), np.asarray(tau)


def make_policy(arg):
    parts = dict(p.split("=", 1) for p in str(arg).split(";") if p.strip())
    if set(parts) != {"base", "arm"}:
        raise ValueError(f"--policy-arg must be 'base=<npz>;arm=<npz>', got {arg!r}")
    return SplitPolicy(parts["base"], parts["arm"])
