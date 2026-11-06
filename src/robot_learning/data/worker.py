"""Persist one episode executed by the common simulation engine."""

from pathlib import Path
import sys
import time
import traceback

import numpy as np

from robot_learning.simulation.episode import Episode
from .storage import Recorder, read_json, sha256, utc_now, write_json


def make_perturbation(noise, scenario_seed):
    """Held Gaussian noise on the applied command; None reproduces the lecturer's loop exactly.

    Wheels get one draw per side, expanded [right, left, left, right] so the applied
    command keeps skid-steer pair identity. Deterministic per (noise seed, scenario seed).
    """
    if noise is None:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([noise["seed"], scenario_seed]))
    state = {"calls": 0, "offset": np.zeros(11)}

    def perturb(action):
        if state["calls"] % noise["hold_steps"] == 0:
            wheels = rng.normal(0.0, noise["wheel_std"], 2) if noise["wheel_std"] > 0 else np.zeros(2)
            torques = rng.normal(0.0, noise["torque_std"], 7) if noise["torque_std"] > 0 else np.zeros(7)
            state["offset"] = np.concatenate((wheels[[0, 1, 1, 0]], torques))
        state["calls"] += 1
        return np.asarray(action, dtype=float) + state["offset"]
    return perturb


def run_episode(directory, scenario, cfg, robo_root):
    episode_id = scenario["episode_id"]
    data_path = directory / "episodes" / f"{episode_id}.npz"
    metadata_path = data_path.with_suffix(".json")
    started = time.monotonic()
    metadata = {**scenario, "started_at": utc_now(), "status": "running", "success": False,
                "recorded_transitions": 0, "data_file": None}
    write_json(metadata_path, metadata)
    np.random.seed(scenario["seed"])
    # The original stop check can execute a final step beyond max_steps.
    # Recorder grows as needed, so no separate cap changes the lecturer's loop.
    recorder = Recorder(1 + cfg["max_steps"] // cfg["record_every_steps"])
    state = None
    try:
        perturbation = make_perturbation(cfg.get("action_noise"), scenario["seed"])
        with Episode(scenario["target"], cfg, robo_root, scenario["initial_base"],
                     action_filter=perturbation) as episode:
            state = episode.state
            dt = episode.dt
            target = episode.target
            metadata.update(dt=dt, configured_step_cap=cfg["max_steps"],
                            sample_period_s=cfg["record_every_steps"] * dt,
                            joint_names=episode.joint_names,
                            actual_initial_observation=state["observation"].tolist(),
                            actual_initial_orientation_xyzw=state["orientation"].tolist(),
                            actual_initial_joint_position=state["q"].tolist(),
                            physics_engine_parameters=episode.client.getPhysicsEngineParameters(),
                            base_goal_xy=episode.goal.tolist(), base_goal_bearing=episode.goal_bearing,
                            base_controller=cfg["base_controller"], arm_controller=cfg["arm_controller"],
                            action_noise=cfg.get("action_noise"))
            previous_action = np.zeros(11)
            previous_qd = state["qd"].copy()
            for transition in episode.transitions():
                current, state = transition["current"], transition["next"]
                step, applied = transition["step"], transition["action"]
                action = transition["raw_action"]  # clean expert command: the imitation label
                torque = action[4:]
                if step % cfg["record_every_steps"] == 0:
                    offset = target - current["shoulder"]
                    length = float(np.linalg.norm(offset))
                    auxiliary = current["shoulder"] + offset * min(1.0, episode.arm_params.reach_limit / max(length, 1e-15))
                    recorder.append({
                        "observation": current["observation"], "next_observation": state["observation"],
                        "action": action, "applied_action": applied,
                        "previous_action": previous_action, "previous_action_valid": step > 0,
                        "time": current["time"], "next_time": state["time"], "step": step,
                        "base_position": current["base"], "base_orientation_xyzw": current["orientation"],
                        "base_bearing": current["yaw"], "base_linear_velocity": current["velocity"],
                        "base_angular_velocity": current["angular_velocity"], "joint_position": current["q"],
                        "joint_velocity": current["qd"], "joint_acceleration": (current["qd"] - previous_qd) / dt,
                        "acceleration_valid": step > 0, "target": target, "ee_position": current["ee_position"],
                        "next_ee_position": state["ee_position"], "ee_error": target - current["ee_position"],
                        "ee_distance": current["distance"], "shoulder_position": current["shoulder"],
                        "arm_auxiliary_target": auxiliary, "expert_phase": transition["phase"], "base_navigation_done": transition["navigation_done"],
                        "expert_q_des_arm": transition["q_des"], "expert_qd_des_arm": transition["qd_des"],
                        "torque_saturated": np.abs(torque) >= episode.arm_params.max_torque - 1e-8,
                    })
                previous_action = applied.copy()
                previous_qd = current["qd"].copy()
            metadata.update(status="complete", **episode.result())
    except BaseException as error:
        metadata.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                        termination="interrupted" if isinstance(error, KeyboardInterrupt) else "error",
                        error_type=type(error).__name__, error_message=str(error))
        traceback.print_exc()
    finally:
        try:
            if state is not None and recorder.count:
                recorder.save(data_path, state)
                metadata.update(data_file=data_path.name, data_sha256=sha256(data_path))
        except Exception as error:
            metadata.update(status="error", success=False, termination="storage_error",
                            error_type=type(error).__name__, error_message=str(error))
            traceback.print_exc()
        finally:
            metadata.update(recorded_transitions=recorder.count, finished_at=utc_now(),
                            wall_duration_s=time.monotonic() - started)
            write_json(metadata_path, metadata)
    return metadata


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    episode = sys.argv[2]
    scenario = next(s for s in read_json(directory / "scenarios.json") if s["episode_id"] == episode)
    cfg = read_json(directory / "config.json")
    root = read_json(directory / "dataset.json")["last_roboenv_root"]
    result = run_episode(directory, scenario, cfg, Path(root))
    sys.exit(0 if result["status"] == "complete" else 1)
