"""Policy outputs: nine independent controls, expanded to the robot's eleven commands.

The expert commands the four wheels of the skid-steer base pairwise equal per side ([right, left, left, right]), so the
eleven recorded commands carry nine independent values: right wheel, left wheel (rad/s) and seven arm torques (N*m).
Models learn the nine; `expand_actions` rebuilds the eleven. Loading requires the exact pair identity in the data
(never averaged away).
"""
import numpy as np

OUTPUT_NAMES = ["wheel_right", "wheel_left"] + [f"arm_torque_{i}" for i in range(1, 8)]
ACTION_NAMES = ["wheel_front_right", "wheel_front_left", "wheel_back_left", "wheel_back_right"] + OUTPUT_NAMES[2:]


def independent_actions(actions):
    """(N, 11) recorded commands -> (N, 9) [right, left, 7 torques]; requires the expert's same-side wheel identity."""
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != 11 or not np.isfinite(a).all():
        raise ValueError("Expected finite (N, 11) expert actions")
    if not np.array_equal(a[:, 0], a[:, 3]) or not np.array_equal(a[:, 1], a[:, 2]):
        raise ValueError("Expert wheel commands do not have identical same-side pairs")
    return a[:, [0, 1, *range(4, 11)]]


def expand_actions(actions):
    """(..., 9) controls -> (..., 11) commands [right, left, left, right, 7 torques]."""
    a = np.asarray(actions)
    if a.shape[-1] != 9 or not np.isfinite(a).all():
        raise ValueError("Expected finite nine-output controls")
    return a[..., [0, 1, 1, 0, *range(2, 9)]]
