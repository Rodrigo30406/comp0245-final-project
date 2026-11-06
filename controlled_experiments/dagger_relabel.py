"""Sequential shadow-expert labels for step 5, experiment F.

The simulator is used only to obtain ExpertPolicy.reset's initial information.
Every command is then computed from the supplied observations, in order, with
the expert's navigation state and arm reference carried between steps.
"""
import numpy as np

from evaluate_policy import ExpertPolicy, setup_simulation


OBS_FIELDS = ("t", "base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes")


def replay_expert(traj, episode):
    """Return shadow wheel commands, arm torques, nav_done and q_des_arm.

    ``traj`` contains every consecutive observation at the simulation time step;
    ``episode`` has the target, init_xy and init_yaw returned by load_eval_set.
    Observation values are promoted to float64, as in the evaluation harness.
    Recorded timestamps are used as supplied, without reconstructing measurements
    or replacing the expert's internal state with recorded diagnostic values.
    """
    arrays = {name: np.asarray(traj[name]) for name in OBS_FIELDS}
    n = len(arrays["t"])
    shapes = {"t": (), "base_pos": (3,), "base_ori": (4,), "base_bearing": (),
              "q_mes": (17,), "qd_mes": (17,)}
    for name, shape in shapes.items():
        if arrays[name].shape != (n, *shape):
            raise ValueError(f"{name}: expected {(n, *shape)}, got {arrays[name].shape}")
        if not np.all(np.isfinite(arrays[name])):
            raise ValueError(f"{name}: non-finite observation")
    if n == 0 or arrays["t"][0] != 0 or np.any(np.diff(arrays["t"]) <= 0):
        raise ValueError("replay needs a nonempty trajectory starting at t = 0, in time order")
    target = np.asarray(episode["target"], dtype=float).copy()
    _, client, _, info = setup_simulation(episode["init_xy"], episode["init_yaw"], target=target)
    client.disconnect()
    expert = ExpertPolicy()
    expert.reset(target.copy(), info)
    out = {"wheel_cmds": np.empty((n, 4)), "torque_arm": np.empty((n, 7)),
           "nav_done": np.empty(n, dtype=np.bool_), "q_des_arm": np.empty((n, 7))}
    for k in range(n):
        obs = {"t": float(arrays["t"][k]), "base_pos": arrays["base_pos"][k].astype(float),
               "base_ori": arrays["base_ori"][k].astype(float),
               "base_bearing": float(arrays["base_bearing"][k]),
               "q_mes": arrays["q_mes"][k].astype(float), "qd_mes": arrays["qd_mes"][k].astype(float),
               "target": target.copy()}
        out["wheel_cmds"][k], out["torque_arm"][k] = expert.act(obs)
        out["nav_done"][k] = expert.base_state.done
        out["q_des_arm"][k] = expert.q_des_arm
    return out
