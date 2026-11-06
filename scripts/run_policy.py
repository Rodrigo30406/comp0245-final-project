"""Watch a trained policy (or the expert) drive the robot to one target, in the PyBullet GUI or headless.

The same closed loop as the evaluator (src/robot_learning/simulation/episode.py): the policy reads the 21 observations
every 1 ms step, its commands are clipped to the expert's envelope, no stop rule (the episode runs --seconds).

  python scripts/run_policy.py --model weights/v2_mlp28_trig144_seed0 --target 1.5 1.0 0.6
  python scripts/run_policy.py --model weights/v2_mlp28_control_seed0 --target 1.5 1.0 0.6 --start 1.0 -1.0 2.5
  python scripts/run_policy.py --expert --target 1.5 1.0 0.6
Options: --dead-band 0.45 (the +DB variant), --headless (no window, faster than real time), --seconds 40.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_learning.evaluation.worker import action_limits, load_policy, make_action_filter  # noqa: E402
from robot_learning.settings import get_roboenv_root, load_project_env  # noqa: E402
from robot_learning.simulation.config import resolve_episode_config  # noqa: E402
from robot_learning.simulation.episode import Episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="training run directory (model.npz + model_info.json), e.g. weights/<run>")
    who.add_argument("--expert", action="store_true", help="run the expert instead")
    ap.add_argument("--target", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"))
    ap.add_argument("--start", type=float, nargs=3, metavar=("X", "Y", "YAW"), help="base start pose (default: the origin, facing +x)")
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--dead-band", type=float, default=0.0, help="wheel dead band in rad/s (0.45 = the +DB variant)")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()
    load_project_env()
    cfg = resolve_episode_config({"headless": args.headless, "goal_reference": "start"})
    initial = None if args.start is None else {"x": args.start[0], "y": args.start[1], "yaw": args.start[2]}
    policy, model = None, None
    if args.model:
        run = Path(args.model)
        model = load_policy(json.loads((run / "model_info.json").read_text())["kind"], run / "model.npz")
        policy = model.predict_action
    steps = int(round(args.seconds / 0.001))
    with Episode(args.target, cfg, get_roboenv_root(), initial, policy=policy, horizon_steps=steps,
                 action_filter=make_action_filter(action_limits(cfg), args.dead_band if model else 0.0)) as episode:
        if model is not None:
            model.reset(episode.dt)
        closest, reached = np.inf, None
        for transition in episode.transitions():
            d = transition["next"]["distance"]
            closest = min(closest, d)
            if reached is None and d < 0.01:
                reached = transition["next"]["time"]
        print(f"{'expert' if args.expert else args.model}: end effector {episode.state['distance'] * 100:.1f} cm from the "
              f"target at {episode.state['time']:.1f} s; closest {closest * 100:.1f} cm"
              + (f"; first within 1 cm at {reached:.1f} s" if reached is not None else "; never within 1 cm"))


if __name__ == "__main__":
    main()
