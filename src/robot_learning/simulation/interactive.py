"""Interactive presentation of the shared expert episode engine."""

import argparse
import json
import os

import numpy as np

from robot_learning.settings import get_roboenv_root, load_project_env
from robot_learning.simulation.config import resolve_episode_config
from robot_learning.simulation.episode import Episode


def run_interactive(argv=None):
    load_project_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=float, nargs=3, metavar=("X", "Y", "Z"))
    parser.add_argument("--config", help="JSON overrides of experiment-wide episode rules")
    parser.add_argument("--headless", action="store_true", default=None)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args(argv)
    requested = {}
    if args.config:
        with open(args.config) as stream:
            requested = json.load(stream)
    requested.setdefault("headless", os.environ.get("ROBOENV_HEADLESS", "").strip().lower() in ("1", "true", "yes"))
    requested.setdefault("max_steps", int(os.environ.get("ROBOENV_MAX_STEPS", "40000")))
    if args.headless is not None:
        requested["headless"] = args.headless
    if args.max_steps is not None:
        requested["max_steps"] = args.max_steps
    config = resolve_episode_config(requested)
    # Rodrigo's existing interactive goal is retained; targets are episode inputs.
    target = np.asarray(args.target if args.target is not None else [0.0, 0.0, 0.1])
    headless = config["headless"]
    import matplotlib
    if headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time_all, ee_pos_all, ee_error_all, q_mes_all, q_d_all = [], [], [], [], []
    with Episode(target.tolist(), config, get_roboenv_root()) as episode:
        initial = episode.initial_state
        print(f"Target Cartesian EE Position : {target}")
        print(f"Initial Cartesian EE Position: {np.round(initial['ee_position'], 4)}")
        print(f"Initial Cartesian Error (m)  : {initial['distance']:.4f}")
        for transition in episode.transitions():
            current = transition["current"]
            # As in the starter, terminal-iteration measurements are not plotted.
            if episode.termination is not None:
                break
            if not headless:
                time_all.append(current["time"])
                ee_pos_all.append(current["ee_position"])
                ee_error_all.append(target - current["ee_position"])
                q_mes_all.append(current["q"])
                q_d_all.append(np.concatenate((current["q"][:4], transition["q_des"], episode.finger_angles)))
            if int(episode.state["time"] / episode.dt) % 500 == 0:
                print(f"t={episode.state['time']:.2f}s | EE error: {current['distance']:.4f}m | phase {transition['phase']}")
        result = episode.result()
        print(result["termination_message"])
        print(f"Stop decision: t={result['stop_decision_time_s']:.6f}s | EE error={result['stop_decision_distance_m']:.6f}m")
        print(f"Final Base Position          : {np.round(episode.state['base'][:2], 4)}")
        print(f"Final End-Effector Position  : {np.round(episode.state['ee_position'], 4)}")
        print(f"Final Cartesian Error (m)    : {result['final_distance_m']:.6f}")
        print(f"Executed physics steps       : {episode.steps}")
        print(f"Final simulation time (s)    : {result['duration_s']:.6f}")
        print(f"Max Finger Deviation (rad)   : {np.max(np.abs(episode.state['q'][11:17] - episode.finger_angles)):.6f}")
    if headless or not time_all:
        return result

    # --------------------------------------------------------------------------
    # Plotting Performance
    # --------------------------------------------------------------------------
    ee_pos_all = np.array(ee_pos_all)
    ee_error_all = np.array(ee_error_all)
    time_all = np.array(time_all)

    # 1. Cartesian end-effector coordinates tracking
    plt.figure(figsize=(10, 8))
    coords = ['X', 'Y', 'Z']
    for idx in range(3):
        plt.subplot(3, 1, idx + 1)
        plt.plot(time_all, ee_pos_all[:, idx], label=f'Actual EE {coords[idx]}', color='b')
        plt.axhline(y=target[idx], color='r', linestyle='--', label=f'Target EE {coords[idx]}')
        plt.ylabel(f'{coords[idx]} Position (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('End-Effector Cartesian Position Reaching')
    plt.tight_layout()
    plt.show()

    # 2. Cartesian distance error over time
    plt.figure(figsize=(8, 4))
    ee_dist_norm = np.linalg.norm(ee_error_all, axis=1)
    plt.plot(time_all, ee_dist_norm, label='EE Euclidean Error (m)', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (m)')
    plt.title('Cartesian End-Effector Reaching Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. Arm joint position tracking
    plt.figure(figsize=(10, 8))
    for i in range(7):
        plt.subplot(4, 2, i + 1)
        plt.plot(time_all, [q[4 + i] for q in q_mes_all], label=f'Measured J{i+1}')
        plt.plot(time_all, [q[4 + i] for q in q_d_all], label=f'Desired J{i+1}', linestyle='--')
        plt.ylabel(f'J{i+1} (rad)')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
    plt.xlabel('Time (s)')
    plt.suptitle('Arm Joint Position Profiles')
    plt.tight_layout()
    plt.show()

    return result


def main(argv=None):
    run_interactive(argv)


if __name__ == "__main__":
    main()
