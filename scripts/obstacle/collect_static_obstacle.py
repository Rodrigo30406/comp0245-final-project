"""Collect the static-obstacle task with the group's collector, but keep stepping the expert for HOLD_S seconds after
the first reach, so the demonstrations contain the hold (the group's `obstacle_task.collect` stops at the reach, and
policies cloned from it leave the target 2-3 s after arriving; docs/01_data_collection.md, section 1.6).

Nothing in the group's code is edited: this script imports `obstacle_task.collect` and replaces its `check_stop` name
with a wrapper. Before the first reach the wrapper is the template rule unchanged (stall, step cap). From the step the
template reports "Target reached" the wrapper returns None (the stall rule, which would fire 3 s into a hold, is
ignored) until HOLD_S have elapsed, then returns a "Target reached ... held" message, so the status is "reached" and the
episode's meta reads as before; the step cap still ends an episode. The per-episode state resets at t = 0 (every episode
starts there; the pool runs one episode per fresh process anyway).

OBSTACLE_HOLD_WHEEL_NOISE=<rad/s> (default 0 = off) adds this project's DART base perturbation during the hold: the
group's collector switches its base noise off once navigation is done, so its hold rows show a still base, whereas our
reaching demonstrations perturb the base throughout (0.5 rad/s per wheel side, right and left independent, each sample
held 50 ms). The wrapper replaces `collect.MotorCommands` by a subclass whose SetControlCmd adds the held noise to the
(executed) motor command, after the collector has logged wheel_cmds_exec, so that logged field omits this noise; the
motion of the base in base_pos shows it. The noise is added to the four executed wheel speeds whenever the expert's wheel command is exactly zero (its latched stop: during navigation the
executed wheels are never all exactly zero); the labels stay the clean zero. The noise stream is seeded from the
episode's command-noise seed (seed * 100000 + episode, the collector's convention) plus 1.

Same arguments as the group's collector (set B of the report: our noise, base noise in the hold):
  cd $OBSTACLE_TASK_DIR && OBSTACLE_HOLD_WHEEL_NOISE=0.5 pixi run python <project>/scripts/obstacle/collect_static_obstacle.py \
      --set-name train2 --workers 12 --noise-frac 1.0 --noise-v 0.0389 --noise-w 0.1178 --noise-torque 1.0 \
      --noise-hold 0.05 --noise-fade-base 0 --noise-fade-arm 0 --out-dir <project>/datasets/static_obstacle/raw/set_B
"""
import os
import sys

# The group's obstacle task code (branch feature/step9-obstacle-lidar, folder final/robotnik_arm_task).
TASK_DIR = os.environ.get("OBSTACLE_TASK_DIR",
                          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "final", "robotnik_arm_task"))
if not os.path.isdir(os.path.join(TASK_DIR, "obstacle_task")):
    raise SystemExit(f"obstacle_task not found in {TASK_DIR}: set OBSTACLE_TASK_DIR to the group's final/robotnik_arm_task "
                     "of branch feature/step9-obstacle-lidar")
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from obstacle_task import collect  # noqa: E402

HOLD_S = float(os.environ.get("OBSTACLE_HOLD_S", "10.0"))
_template_check_stop = collect.check_stop
_state = {"reached_at": None}


def check_stop_with_hold(criteria, current_time, time_step, ee_dist_error, base_navigation_done):
    if current_time == 0.0:
        _state["reached_at"] = None
    message = _template_check_stop(criteria, current_time, time_step, ee_dist_error, base_navigation_done)
    if message is not None and message.startswith("Reached step cap"):
        return message
    if _state["reached_at"] is None:
        if message is None or not message.startswith("Target reached"):
            return message                      # not reached yet: template rule (None, or a stall)
        _state["reached_at"] = current_time
    if current_time - _state["reached_at"] + 1e-9 >= HOLD_S:
        return (f"Target reached within {criteria.reach_tolerance * 1000:.0f}mm tolerance at "
                f"t={_state['reached_at']:.2f}s, held {HOLD_S:g}s to t={current_time:.2f}s!")
    return None


collect.check_stop = check_stop_with_hold      # also applied in the spawned workers (they re-import this module)

HOLD_WHEEL_NOISE = float(os.environ.get("OBSTACLE_HOLD_WHEEL_NOISE", "0"))
HOLD_WHEEL_NOISE_STEPS = 50                    # 50 ms per sample, as this project's collector
_noise = {"held": None}

if HOLD_WHEEL_NOISE > 0:
    import numpy as np
    from collect_data import HeldNoise

    _template_run_episode = collect.run_episode
    _template_motor_commands = collect.MotorCommands

    class MotorCommandsWithHoldNoise(_template_motor_commands):
        def SetControlCmd(self, values, control_modes):
            values = np.array(values, dtype=float)
            if _noise["held"] is not None and np.all(values[:4] == 0.0):
                right, left = _noise["held"].sample()
                values[:4] += np.array([right, left, left, right])
            return super().SetControlCmd(values, control_modes)

    def run_episode_with_hold_noise(*args, **kwargs):
        seed = int(kwargs.get("seed", 0))
        _noise["held"] = HeldNoise(np.random.default_rng(seed + 1), [HOLD_WHEEL_NOISE, HOLD_WHEEL_NOISE],
                                   HOLD_WHEEL_NOISE_STEPS)
        try:
            return _template_run_episode(*args, **kwargs)
        finally:
            _noise["held"] = None

    collect.MotorCommands = MotorCommandsWithHoldNoise
    collect.run_episode = run_episode_with_hold_noise

if __name__ == "__main__":
    raise SystemExit(collect.main())
