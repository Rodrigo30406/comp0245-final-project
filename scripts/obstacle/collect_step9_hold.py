"""Collect the obstacle task (step 9) with the group's collector, but keep stepping the expert for HOLD_S seconds after
the first reach, so the demonstrations contain the hold (the group's `obstacle_task.collect` stops at the reach, and
policies cloned from it leave the target 2-3 s after arriving; REPORT_PLAN §8, 5 Nov).

Nothing in the group's code is edited: this script imports `obstacle_task.collect` and replaces its `check_stop` name
with a wrapper. Before the first reach the wrapper is the template rule unchanged (stall, step cap). From the step the
template reports "Target reached" the wrapper returns None (the stall rule, which would fire 3 s into a hold, is
ignored) until HOLD_S have elapsed, then returns a "Target reached ... held" message, so the status is "reached" and the
episode's meta reads as before; the step cap still ends an episode. The per-episode state resets at t = 0 (every episode
starts there; the pool runs one episode per fresh process anyway).

Same arguments as the group's collector. Example (same scenarios, noisy flags and noise seeds as their train2):
  cd <step9 worktree>/final/robotnik_arm_task && pixi run python <project>/scripts/obstacle/collect_step9_hold.py \
      --set-name train2 --workers 16 --out-dir <project>/datasets/step9/raw/train2_hold10
"""
import os
import sys

TASK_DIR = os.environ.get("STEP9_TASK_DIR", "/home/labia10/repos/comp0245-step9/final/robotnik_arm_task")
if TASK_DIR not in sys.path:
    sys.path.insert(0, TASK_DIR)

from obstacle_task import collect  # noqa: E402

HOLD_S = float(os.environ.get("STEP9_HOLD_S", "10.0"))
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

if __name__ == "__main__":
    raise SystemExit(collect.main())
