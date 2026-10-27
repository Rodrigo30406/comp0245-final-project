"""Dataset fields, units, and alignment contract shared by recorder and CLI."""

FIELDS = {
    "observation": (21, "float32"), "next_observation": (21, "float32"),
    "action": (11, "float32"), "previous_action": (11, "float32"),
    "previous_action_valid": (None, "bool"),
    "time": (None, "float64"), "next_time": (None, "float64"), "step": (None, "int64"),
    "base_position": (3, "float32"), "base_orientation_xyzw": (4, "float32"),
    "base_bearing": (None, "float32"), "base_linear_velocity": (3, "float32"),
    "base_angular_velocity": (3, "float32"), "joint_position": (17, "float32"),
    "joint_velocity": (17, "float32"), "joint_acceleration": (17, "float32"),
    "acceleration_valid": (None, "bool"), "target": (3, "float32"),
    "ee_position": (3, "float32"), "next_ee_position": (3, "float32"),
    "ee_error": (3, "float32"), "ee_distance": (None, "float32"),
    "shoulder_position": (3, "float32"), "arm_auxiliary_target": (3, "float32"),
    "expert_phase": (None, "U8"), "base_navigation_done": (None, "bool"),
    "expert_q_des_arm": (7, "float32"), "expert_qd_des_arm": (7, "float32"),
    "torque_saturated": (7, "bool"),
}


def dataset_schema():
    names = ([f"base_position_{axis}" for axis in "xyz"] + ["base_bearing"] +
             [f"arm_q_{i}" for i in range(1, 8)] + [f"arm_qd_{i}" for i in range(1, 8)] +
             [f"target_{axis}" for axis in "xyz"])
    return {
        "version": 2, "format": "npz; load with numpy.load(..., allow_pickle=False)",
        "observation_names": names,
        "action_names": ["wheel_front_right", "wheel_front_left", "wheel_back_left", "wheel_back_right"] +
                        [f"arm_torque_{i}" for i in range(1, 8)],
        "units": {"position": "m", "angles": "rad", "joint_velocity": "rad/s",
                  "joint_acceleration": "rad/s^2", "base_linear_velocity": "m/s",
                  "base_angular_velocity": "rad/s", "wheel_commands": "rad/s",
                  "arm_torques": "N*m", "time": "simulated seconds"},
        "frames": "base position, EE, shoulder, target and base velocities are world-frame; quaternion XYZW",
        "alignment": "observation/time -> action -> next_observation/next_time, exactly one physics step apart",
        "subsampling": "record_every_steps selects transitions; it does not change the control period. "
                       "previous_action is the immediately preceding physics-step action, not the last stored row.",
        "joint_order": "4 wheels + 7 arm + 6 fingers; exact names in episode metadata",
        "diagnostics_only": ["expert_phase", "base_navigation_done", "expert_q_des_arm",
                             "expert_qd_des_arm", "arm_auxiliary_target"],
        "fields": {name: {"shape": ["N"] if width is None else ["N", width], "dtype": dtype}
                   for name, (width, dtype) in FIELDS.items()},
        "terminal_fields": ["terminal_observation", "terminal_ee_position", "terminal_time"],
        "success_definition": "Expert check_stop termination after reaching original target within tolerance "
                              "and after min_time, with base navigation done; no dwell-time guarantee.",
        "stop_evaluation": "Lecturer order: action, physics step, GUI exit, check_stop using the PRE-step "
                           "state/time. Clock accumulates dt as in the starter. The cap is checked after "
                           "stepping; executed steps can exceed max_steps (including float rounding). "
                           "Terminal state and final metrics are post-step; metadata separately records "
                           "stop_decision_time_s and stop_decision_distance_m. Errors use the original target.",
    }
