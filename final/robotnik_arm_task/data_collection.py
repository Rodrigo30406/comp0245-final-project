import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pinocchio as pin

from simulation_and_control import MotorCommands, PinWrapper, pb
from robotnik_arm_control import (
    ArmControllerParams,
    BaseControllerParams,
    BaseNavState,
    StopCriteria,
    arm_controller,
    base_controller,
    check_stop,
    configure_gripper_fingers,
    quaternion2bearing,
)


# This data-collection script is always headless; no GUI is opened.
headless = True
max_steps = int(os.environ.get("ROBOENV_MAX_STEPS", "40000"))

# Five trajectories are collected with one target sampled per trajectory.
NUM_TRAJECTORIES = 100
NUM_WORKERS = min(NUM_TRAJECTORIES, max(1, os.cpu_count() or 1))
DATASET_FILENAME = "data.csv"
TARGET_EE_X_RANGE = (-0.45, 0.45)
TARGET_EE_Y_RANGE = (0.75, 1.35)
TARGET_EE_Z_RANGE = (0.20, 1.00)
TARGET_RANDOM_SEED = 42

# Fixed gripper finger configuration (angles in radians).
FIXED_FINGER_ANGLES = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

BASE_NAV_TIMEOUT = 25.0
ARM_REACH_LIMIT = 0.85
STANDOFF_Z_POINTS = [0.1, 0.5, 1.1, 1.25]
STANDOFF_POINTS = [0.80, 0.65, 0.65, 0.35]
REACH_TOLERANCE = 0.01


def sample_target(rng):
    """Sample a target in the reachable workspace for one trajectory."""
    return np.array(
        [
            rng.uniform(*TARGET_EE_X_RANGE),
            rng.uniform(*TARGET_EE_Y_RANGE),
            rng.uniform(*TARGET_EE_Z_RANGE),
        ]
    )


def dataset_fieldnames():
    return (
        ["trajectory_id", "target_ee_x", "target_ee_y", "target_ee_z"]
        + ["base_pos_x", "base_pos_y", "base_pos_z", "base_bearing"]
        + [f"q_mes_{i}" for i in range(17)]
        + [f"qd_mes_{i}" for i in range(17)]
        + ["lw_cmd", "rw_cmd"]
        + [f"torque_arm_{i}" for i in range(7)]
    )


def make_dataset_row(trajectory_id, target, base_pos, base_bearing, q_mes, qd_mes,
                     wheel_cmds, torque_arm):
    """Return one supervised-learning row with measurements and controller outputs."""
    return [
        trajectory_id,
        *target.tolist(),
        *base_pos.tolist(),
        base_bearing,
        *q_mes.tolist(),
        *qd_mes.tolist(),
        wheel_cmds[1],  # left-wheel command; base_controller returns [rw, lw, lw, rw].
        wheel_cmds[0],  # right-wheel command.
        *torque_arm.tolist(),
    ]


def collect_trajectory(trajectory_id, target):
    conf_file_name = "robotnik_arm.json"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface(conf_file_name, conf_file_path_ext=cur_dir, use_gui=not headless)

    sim.SetFloorFriction(1.0)
    client = sim.pybullet_client
    rid = sim.bot[0].bot_pybullet
    wheel_joint_ids = [
        j for j in range(client.getNumJoints(rid))
        if "wheel" in client.getJointInfo(rid, j)[1].decode()
    ]
    for joint_id in wheel_joint_ids:
        client.changeDynamics(
            rid,
            joint_id,
            contactStiffness=30000,
            contactDamping=1000,
            anisotropicFriction=[1.0, 0.01, 1.0],
        )
    sim.bot[0].servo_motor_model.set_motor_gains(400, 30)

    target_finger_angles = configure_gripper_fingers(sim, client, rid, FIXED_FINGER_ANGLES)

    ext_names = sim.getNameActiveJoints()
    ext_names_arr = np.expand_dims(np.array(ext_names), axis=0)
    dyn_model = PinWrapper(
        conf_file_name,
        "pybullet",
        ext_names_arr,
        ["pybullet"],
        False,
        0,
        cur_dir,
    )
    ee_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_end_effector")
    shoulder_frame_id = dyn_model.pin_model.getFrameId("robot_j2s7s300_link_2")

    init_joint_angles = np.array(sim.GetMotorAngles(0))
    init_base_pos = np.array(sim.GetBasePosition())
    init_base_ori = np.array(sim.GetBaseOrientation())
    init_q_all = np.concatenate((init_base_pos, init_base_ori, init_joint_angles))
    init_q_pin = dyn_model.ReoderJoints2PinVec(init_q_all, "pos")
    pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, init_q_pin)
    pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
    init_ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()

    print(f"Trajectory {trajectory_id}: target={np.round(target, 4)}")
    print(f"Initial EE position: {np.round(init_ee_pos, 4)}")

    lower_limits, upper_limits = sim.GetBotJointsLimit()
    time_step = sim.GetTimeStep()
    current_time = 0.0

    standoff = float(np.interp(target[2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
    target_xy = target[:2]
    target_bearing = np.arctan2(target_xy[1], target_xy[0]) if np.linalg.norm(target_xy) > 1e-4 else 0.0
    base_goal_pos = target_xy - standoff * np.array(
        [np.cos(target_bearing), np.sin(target_bearing)]
    )

    base_state = BaseNavState()
    base_params = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
    arm_params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    q_init_arm = init_joint_angles[4:11].copy()
    q_des_arm = q_init_arm.copy()
    qd_des_arm = np.zeros(7)
    arm_lower = np.array(lower_limits[4:11])
    arm_upper = np.array(upper_limits[4:11])
    stop_criteria = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=max_steps)
    cmd = MotorCommands()
    rows = []

    while True:
        base_pos = np.array(sim.GetBasePosition())
        base_ori = np.array(sim.GetBaseOrientation())
        base_bearing = quaternion2bearing(base_ori[3], base_ori[0], base_ori[1], base_ori[2])
        q_mes = np.array(sim.GetMotorAngles(0))
        qd_mes = np.array(sim.GetMotorVelocities(0))

        q_all = np.concatenate((base_pos, base_ori, q_mes))
        q_pin = dyn_model.ReoderJoints2PinVec(q_all, "pos")
        pin.forwardKinematics(dyn_model.pin_model, dyn_model.pin_data, q_pin)
        pin.updateFramePlacements(dyn_model.pin_model, dyn_model.pin_data)
        ee_pos = dyn_model.pin_data.oMf[ee_frame_id].translation.copy()
        ee_dist_error = np.linalg.norm(target - ee_pos)

        wheel_cmds = base_controller(
            base_state,
            base_pos,
            base_bearing,
            base_goal_pos,
            target_bearing,
            current_time,
            base_params,
        )
        base_navigation_done = base_state.done
        torque_arm, q_des_arm, qd_des_arm = arm_controller(
            dyn_model,
            q_pin,
            q_mes[4:11],
            qd_mes[4:11],
            q_des_arm,
            q_init_arm,
            target,
            ee_pos,
            ee_frame_id,
            shoulder_frame_id,
            arm_lower,
            arm_upper,
            time_step,
            arm_params,
            reach_active=base_navigation_done,
        )

        rows.append(
            make_dataset_row(
                trajectory_id,
                target,
                base_pos,
                base_bearing,
                q_mes,
                qd_mes,
                wheel_cmds,
                torque_arm,
            )
        )

        torque_fingers = np.zeros(6)
        cmd_all = np.concatenate((wheel_cmds, torque_arm, torque_fingers))
        control_modes = ["velocity"] * 4 + ["torque"] * 7 + ["torque"] * 6
        cmd.SetControlCmd(cmd_all, control_modes)
        sim.Step(cmd, "torque")

        stop_message = check_stop(
            stop_criteria,
            current_time,
            time_step,
            ee_dist_error,
            base_navigation_done,
        )
        if stop_message:
            print(f"Trajectory {trajectory_id}: {stop_message}")
            break

        if not headless:
            time.sleep(0.001)
        current_time += time_step

    return rows


def main():
    rng = np.random.default_rng(TARGET_RANDOM_SEED)
    trajectory_inputs = [
        (trajectory_id, sample_target(rng))
        for trajectory_id in range(1, NUM_TRAJECTORIES + 1)
    ]

    # Each worker owns a separate simulator. map preserves CSV trajectory order.
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        trajectory_rows = executor.map(
            lambda args: collect_trajectory(*args),
            trajectory_inputs,
        )
        all_rows = [row for rows in trajectory_rows for row in rows]

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_FILENAME)
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(dataset_fieldnames())
        writer.writerows(all_rows)
    print(f"Saved {len(all_rows)} data points from {NUM_TRAJECTORIES} trajectories to {output_path}")


if __name__ == "__main__":
    main()
