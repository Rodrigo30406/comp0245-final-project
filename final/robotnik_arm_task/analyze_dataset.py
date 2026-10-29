"""Data-level analysis of the expert datasets (step 2 of the experiment plan): no learned model involved.

Every episode of the given datasets is scanned once, in parallel. Per episode the scan measures:
  - the task and the expert's behaviour mode (d0 = r - standoff(z), required turns, phase sequence),
  - phase sample counts, including the hold tail after the first reach,
  - the zero-parameter "copycat" predictor (measured wheel speeds used as the wheel command) around events,
  - how much of the base label is not a function of the current observation ("restarted expert"),
  - arm label quality: joint 6 starting below its limit, the torque jump at reach onset, period-2 torque
    chatter and its mechanism (joint-limit freezing in the IK, or damping near singular poses).
Tables go to <out>/summary.txt, one row per episode to <out>/episodes_<set>.csv, figures to <out>/*.png.
The predictions tested here were registered in report.txt before the analysis was run on data/train.

Examples:
  # calibrate the analysis settings on the pilot set (prints the values for CALIBRATION below)
  python analyze_dataset.py --calibrate data/pilot --workers 20
  # the analysis (train first: the mode and coverage tables use it as the reference)
  python analyze_dataset.py --sets data/train data/dev data/hard --workers 20 --out results/data_analysis
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import time
import warnings

import numpy as np
import pinocchio as pin
from scipy import stats
from scipy.spatial import cKDTree

from robotnik_arm_control import ArmControllerParams, BaseControllerParams, BaseNavState, StopCriteria, base_controller
from robotnik_arm_task import ARM_REACH_LIMIT, BASE_NAV_TIMEOUT, REACH_TOLERANCE

BASE = BaseControllerParams(nav_timeout=BASE_NAV_TIMEOUT)
ARM = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
MIN_TIME = StopCriteria(reach_tolerance=REACH_TOLERANCE, max_steps=0).min_time  # template: no stop before 8 s
SETTLE_TOL = 0.015  # m, "settled" threshold of evaluation protocol v1

PHASE_NAMES = ["align", "drive", "orient", "reach"]                 # recorded phase codes 0-3
CLASSES = ["align", "drive", "orient", "reach_move", "reach_hold"]  # reach split at the first reach
EVENT_TYPES = ["start", "align>drive", "drive>orient", "drive>reach", "orient>reach", "noise"]
OFFSETS = np.arange(-50, 501)    # steps (= ms) around an event, for the event-locked error curves
FIXED_WINDOWS = (0.05, 0.2, 0.4)  # s, window widths of the sensitivity analysis
STRIDES = (1, 5, 10, 20, 50)      # downsampling strides for the "decision samples" count
V_BINS = np.linspace(-0.1, 0.3, 401)  # m/s, histogram of the measured forward speed (calibration)
S_BINS = np.linspace(0.0, 20.0, 1001)  # N*m, histogram of the per-step chatter amplitude (calibration)

# Episodes of data/train looked at before the step-2 predictions were registered (scouting and plan review,
# see report.txt). The predictions are tested on the other train episodes; results on all are also given.
SEEN_TRAIN = set(range(200)) | set(range(1900, 1908))

# Pre-registered settings (report.txt, step 2)
E1_EVERY = 10                     # restarted-expert check on every 10th step (100 Hz)
INCONS_DV, INCONS_DW = 0.02, 0.05  # m/s, rad/s: a command differs from the label (= the DART noise std)
CHATTER_HF = 250.0                # Hz, chatter index = share of the reach torque power above this frequency
CHATTER_EPISODE = 0.05            # an episode "chatters" when its chatter index is above this
Z_SPLIT = 0.45                    # m, low / high target split for the arm (chatter is concentrated below it; pilot)
CHATTER_SENSITIVITY = (0.3, 1.0, 3.0)  # N m, step-amplitude thresholds of the mechanism test (1.0 is the main one)
E1_VARIANTS = [                   # (stop goal from the current position, start in drive when moving forward)
    ("true goal", False, False),
    ("current-position goal", True, False),
    ("true goal + wheel-speed rule", False, True),
    ("current-position goal + wheel-speed rule", True, True),
]

# Calibrated on data/pilot with --calibrate, before the analysis was run on data/train (see report.txt).
# settle: per event type, time after the event until the copycat error excess over the post-event level stays
#   below that level for 50 ms (s).
# chatter_s: per-step chatter amplitude above which a reach step counts as chattering. Smooth control changes
#   the torque by ~0.001 N m per step; the pilot shows no gap between that and the bursts, so the value is fixed
#   (the mechanism test is repeated at 0.3 and 3 N m). v_drive: measured forward speed that best separates drive
#   from align steps on the pilot (the "wheel-speed rule" of the restarted-expert check).
CALIBRATION = {
    "settle": {"start": 0.127, "align>drive": 0.055, "drive>orient": 0.234, "drive>reach": 0.072,
               "orient>reach": 0.38, "noise": 0.0},
    "chatter_s": 1.0,   # N m, fixed: the pilot amplitudes form a continuum without a gap (see report.txt)
    "v_drive": 0.065,
}


def wrap(angle):
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def vw(wheels):
    """Wheel speeds [front_right, front_left, back_left, back_right] (rad/s, (T, 4) or (4,)) -> (v, w).

    Inverse of the skid-steer model of base_controller (v = R (rw + lw) / 2, w = R (rw - lw) / W); the two wheels
    of a side are averaged, so measured speeds that differ slightly between front and back are handled too."""
    wheels = np.asarray(wheels, dtype=float)
    rw = 0.5 * (wheels[..., 0] + wheels[..., 3])
    lw = 0.5 * (wheels[..., 1] + wheels[..., 2])
    return BASE.wheel_radius * (rw + lw) / 2, BASE.wheel_radius * (rw - lw) / BASE.wheel_base_width


def mode_of(d0):
    """M0 'hardly drives' (|d0| < pos_tol: the FSM goes from align straight to orient), M1 'drive away, then turn
    around' (base goal on the far side of the start), M2 'normal' (drive towards the target)."""
    return "M0" if abs(d0) < BASE.pos_tol else ("M1" if d0 < 0 else "M2")


def required_turns(theta, d0):
    """In-place turns the expert needs from the start pose, (align, orient) in rad, for a straight path."""
    if d0 >= BASE.pos_tol:
        return abs(theta), 0.0
    if d0 <= -BASE.pos_tol:
        return abs(float(wrap(theta + np.pi))), np.pi
    return 0.0, abs(theta)


# ==============================================================================
# Arm IK, replicated from robotnik_arm_control.arm_controller to see which joints it freezes
# ==============================================================================
def ik_reference(model, data, q_pin, q_arm, target, ee_id, shoulder_id, lower, upper, params):
    """Joint velocity reference of the expert's differential IK, plus the joints frozen at a limit and the
    smallest singular value of the arm Jacobian. Same computation as the reach branch of arm_controller."""
    pin.forwardKinematics(model, data, q_pin)
    pin.updateFramePlacements(model, data)
    ee_pos = data.oMf[ee_id].translation.copy()
    pin.computeJointJacobians(model, data, q_pin)
    J_arm = pin.computeFrameJacobian(model, data, q_pin, ee_id, pin.LOCAL_WORLD_ALIGNED)[:3, 6:13]
    sigma_min = np.linalg.svd(J_arm, compute_uv=False)[-1]
    damping = np.interp(sigma_min, [0.0, params.singularity_threshold], [params.max_ik_damping, params.min_ik_damping])
    shoulder_pos = data.oMf[shoulder_id].translation
    offset = target - shoulder_pos
    if np.linalg.norm(offset) > params.reach_limit:
        offset *= params.reach_limit / np.linalg.norm(offset)
    v_cartesian = params.cartesian_kp * (shoulder_pos + offset - ee_pos)
    speed = np.linalg.norm(v_cartesian)
    if speed > params.max_cartesian_speed:
        v_cartesian *= params.max_cartesian_speed / speed
    has_limits = lower < upper
    free = np.ones(7, dtype=bool)
    qd_des = np.zeros(7)
    for _ in range(7):
        J_free = J_arm[:, free]
        qd_des = np.zeros(7)
        qd_des[free] = J_free.T @ np.linalg.solve(J_free @ J_free.T + damping**2 * np.eye(3), v_cartesian)
        pushing_lower = has_limits & (q_arm <= lower + params.limit_margin) & (qd_des < 0)
        pushing_upper = has_limits & (q_arm >= upper - params.limit_margin) & (qd_des > 0)
        newly_blocked = (pushing_lower | pushing_upper) & free
        if not newly_blocked.any():
            break
        free &= ~newly_blocked
        qd_des[:] = 0.0
    max_speed = np.max(np.abs(qd_des))
    if max_speed > params.max_joint_vel:
        qd_des *= params.max_joint_vel / max_speed
    near_limit = has_limits & ((q_arm <= lower + params.limit_margin) | (q_arm >= upper - params.limit_margin))
    return qd_des, ~free, near_limit, sigma_min


# ==============================================================================
# Per-episode scan (runs in the worker processes)
# ==============================================================================
_W = {}


def _init_worker(ext_names, settings):
    """Pinocchio model for the IK replication, built once per worker (no simulation is run)."""
    from simulation_and_control import PinWrapper
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)  # PinWrapper and pinocchio print model details: keep the console readable
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dyn = PinWrapper("robotnik_arm.json", "pybullet", np.expand_dims(np.array(ext_names), axis=0), ["pybullet"],
                     False, 0, cur_dir)
    _W.update(settings)
    _W.update(dyn=dyn, ee_id=dyn.pin_model.getFrameId("robot_j2s7s300_end_effector"),
              shoulder_id=dyn.pin_model.getFrameId("robot_j2s7s300_link_2"))


def scan_episode(job):
    path, set_name, episode, hold_steps = job["path"], job["set"], job["episode"], job["hold_steps"]
    cal = _W["calibration"]
    with np.load(path) as d:
        a = {k: d[k] for k in d.files}
    meta = json.loads(str(a["meta"]))
    assert meta["log_every"] == 1, "the analysis needs every simulation step"
    dt = float(meta["time_step"])
    t = a["t"].astype(float)
    n = len(t)
    ph = a["phase"].astype(int)
    pos3 = a["base_pos"].astype(float)
    pos = pos3[:, :2]
    yaw = a["base_bearing"].astype(float)
    q = a["q_mes"].astype(float)
    tau = a["torque_arm"].astype(float)
    target = a["target"].astype(float)
    err = np.linalg.norm(a["ee_pos"].astype(float) - target, axis=1)

    # ---------------- task, mode, outcome
    init = np.array(meta["init_base_pos"][:2])
    yaw0 = float(meta["init_base_yaw"])
    goal, goal_b, standoff = np.array(meta["base_goal_pos"]), float(meta["base_goal_bearing"]), float(meta["standoff"])
    rel = target[:2] - init
    r = float(np.linalg.norm(rel))
    theta = float(wrap(np.arctan2(rel[1], rel[0]) - yaw0))
    d0 = r - standoff
    align_req, orient_req = required_turns(theta, d0)
    noisy = float(meta["noise_v"]) > 0 or float(meta["noise_w"]) > 0 or float(meta["noise_torque"]) > 0
    keep = meta["status"] == "reached" and not meta["nav_timed_out"]
    row = {
        "set": set_name, "episode": episode, "seen": set_name == "train" and episode in SEEN_TRAIN,
        "noisy": noisy, "status": meta["status"], "nav_timed_out": bool(meta["nav_timed_out"]), "keep": keep,
        "duration": float(meta["duration"]), "r": r, "theta_deg": float(np.rad2deg(theta)), "z": float(target[2]),
        "standoff": standoff, "d0": d0, "mode": mode_of(d0),
        "align_req_deg": float(np.rad2deg(align_req)), "orient_req_deg": float(np.rad2deg(orient_req)),
        "init_x": float(init[0]), "init_y": float(init[1]), "init_yaw_deg": float(np.rad2deg(yaw0)),
        "target_x": float(target[0]), "target_y": float(target[1]),
    }

    # ---------------- phases (the reach phase split at the first reach)
    i_reach = int(np.argmax(ph == 3)) if (ph == 3).any() else n
    reached = np.flatnonzero(err[i_reach:] < REACH_TOLERANCE)
    k_reach = i_reach + int(reached[0]) if reached.size else None
    cls = ph.copy()
    if k_reach is not None:
        cls[k_reach + 1:] = 4
    counts = np.bincount(cls, minlength=5)
    changes = np.flatnonzero(ph[1:] != ph[:-1]) + 1
    dyaw = np.abs(wrap(np.diff(yaw)))
    turn = np.rad2deg(np.bincount(ph[:-1], weights=dyaw, minlength=4))
    step_len = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    path_len = np.bincount(ph[:-1], weights=step_len, minlength=4)
    row.update({f"n_{c}": int(k) for c, k in zip(CLASSES, counts)})
    row.update({
        "phase_seq": "".join(str(ph[k]) for k in [0, *changes]),
        "t_first_reach": float(t[k_reach]) if k_reach is not None else np.nan,
        "has_hold": k_reach is not None and k_reach < n - 1,
        "n_hold_exit": int(np.sum(err[k_reach + 1:] > SETTLE_TOL)) if k_reach is not None else 0,
        **{f"turn_{p}_deg": float(turn[i]) for i, p in enumerate(PHASE_NAMES[:3])},
        "path_drive": float(path_len[1]),
    })

    # ---------------- base mechanism variables
    align_idx = np.flatnonzero(ph == 0)
    row["align_disp"] = float(np.linalg.norm(pos[align_idx[-1]] - pos[0])) if align_idx.size else 0.0
    drive_idx = np.flatnonzero(ph == 1)
    if drive_idx.size:
        i_d = drive_idx[0]
        first = drive_idx[drive_idx < i_d + round(1.0 / dt)]
        heading = np.arctan2(goal[1] - pos[first, 1], goal[0] - pos[first, 0])
        row["d_drive"] = float(np.linalg.norm(pos[i_d] - goal))
        row["drive_head_err_1s_deg"] = float(np.rad2deg(np.max(np.abs(wrap(heading - yaw[first])))))
    else:
        row["d_drive"], row["drive_head_err_1s_deg"] = np.nan, np.nan
    after_drive = np.flatnonzero(ph >= 2)
    row["arrival_bearing_err_deg"] = float(np.rad2deg(abs(wrap(goal_b - yaw[after_drive[0]])))) if after_drive.size else np.nan
    if i_reach < n:
        to_t = target[:2] - pos[i_reach]
        mb = np.arctan2(to_t[1], to_t[0])
        goal_m = target[:2] - standoff * np.array([np.cos(mb), np.sin(mb)])
        row.update({"navdone_goal_dist": float(np.linalg.norm(pos[i_reach] - goal)),
                    "navdone_ee_err": float(err[i_reach]),
                    "navdone_markov_goal_offset": float(np.linalg.norm(goal_m - goal)),
                    "navdone_markov_bearing_offset_deg": float(np.rad2deg(abs(wrap(mb - goal_b))))})
    else:
        row.update({"navdone_goal_dist": np.nan, "navdone_ee_err": np.nan, "navdone_markov_goal_offset": np.nan,
                    "navdone_markov_bearing_offset_deg": np.nan})

    # ---------------- arm label quality
    lower, upper = _W["lower"], _W["upper"]
    clamp6 = lower[5] + ARM.limit_margin
    reach_idx = np.arange(i_reach, n)
    sat = (np.abs(tau) >= ARM.max_torque - 1e-3).any(axis=1)
    row.update({
        "q6_start": float(q[0, 9]),
        "j6_below_limit_frac": float(np.mean(q[:, 9] < lower[5])),
        "j6_clamp_frac": float(np.mean(np.abs(a["q_des_arm"][reach_idx, 5] - clamp6) < 1e-5)) if reach_idx.size else np.nan,
        "n_sat": int(sat.sum()),
        "n_sat_onset": int(sat[i_reach:i_reach + round(0.2 / dt)].sum()),
    })
    if 0 < i_reach < n:
        jump = tau[i_reach] - tau[i_reach - 1]
        row["onset_jump_max"], row["onset_jump_j6"] = float(np.max(np.abs(jump))), float(jump[5])
    else:
        row["onset_jump_max"], row["onset_jump_j6"] = np.nan, np.nan
    seg = tau[i_reach + 5:]
    if len(seg) >= 200:
        spec = np.abs(np.fft.rfft(seg - seg.mean(axis=0), axis=0))**2
        freq = np.fft.rfftfreq(len(seg), dt)
        row["chatter_index"] = float(spec[freq > CHATTER_HF].sum() / max(spec[freq > 0].sum(), 1e-12))
    else:
        row["chatter_index"] = np.nan
    # per-step chatter amplitude: distance of a torque from the mean of its two neighbours (largest joint)
    k_int = np.arange(i_reach + 3, n - 1)
    s = np.max(np.abs(tau[k_int] - 0.5 * (tau[k_int - 1] + tau[k_int + 1])), axis=1) if k_int.size else np.zeros(0)
    out = {"row": row, "s_hist": np.histogram(np.minimum(s, S_BINS[-1] - 1e-9), S_BINS)[0].astype(np.int32)}

    # ---------------- copycat: measured wheel speeds as the prediction of the wheel command
    v, w = vw(a["wheel_cmds"])
    vh, wh = vw(a["qd_mes"][:, :4])
    ve, we = vw(a["wheel_cmds_exec"])
    ve_prev, we_prev = np.r_[0.0, ve[:-1]], np.r_[0.0, we[:-1]]  # the robot starts at rest
    ev, ew = vh - v, wh - w
    phase_events = [("start", 0)] + [(f"{PHASE_NAMES[ph[k - 1]]}>{PHASE_NAMES[ph[k]]}", int(k)) for k in changes]
    noise_events, noise_overlap = [], 0
    if noisy:
        for k in range(hold_steps, i_reach, hold_steps):  # the base noise is redrawn every hold_steps while navigating
            if any(-50 <= k - kp <= 400 for _, kp in phase_events):
                noise_overlap += 1  # inside the window of a phase event: counted, not used for the noise curve
            else:
                noise_events.append(("noise", k))
    events = phase_events + noise_events
    row["n_other_events"] = sum(name not in EVENT_TYPES for name, _ in events)
    row["n_noise_overlap"] = noise_overlap
    curves = np.full((len(EVENT_TYPES), len(OFFSETS), 2), np.nan, dtype=np.float32)
    n_events = np.zeros(len(EVENT_TYPES), dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns at offsets outside the episode
        for j, name in enumerate(EVENT_TYPES):
            ks = np.array([k for nm, k in events if nm == name], dtype=int)
            n_events[j] = len(ks)
            if ks.size:
                idx = ks[:, None] + OFFSETS[None, :]
                valid = (idx >= 0) & (idx < n)
                idx = np.clip(idx, 0, n - 1)
                curves[j, :, 0] = np.nanmean(np.where(valid, ev[idx]**2, np.nan), axis=0)
                curves[j, :, 1] = np.nanmean(np.where(valid, ew[idx]**2, np.nan), axis=0)
    out.update(curves=curves, n_events=n_events)
    out["v_hist"] = np.stack([np.histogram(np.clip(vh[ph == c], V_BINS[0], V_BINS[-1] - 1e-9), V_BINS)[0]
                              for c in (0, 1)]).astype(np.int32)  # align, drive
    wheel_label, wheel_meas = a["wheel_cmds"].astype(float), a["qd_mes"][:, :4].astype(float)
    sst = np.sum((wheel_label - wheel_label.mean(axis=0))**2)
    row["copycat_r2_4wheels"] = float(1 - np.sum((wheel_meas - wheel_label)**2) / sst) if sst > 0 else np.nan

    if cal is not None:
        # windows after each event, [k, k + settle time(type)]; event types without one use the longest
        win = np.zeros(n, dtype=bool)
        win_by_type = {}
        for name, k in events:
            width = round(cal["settle"].get(name, max(cal["settle"].values())) / dt)
            win[k:k + width + 1] = True
            win_by_type.setdefault(name, []).append((k, width))
        if noisy:  # noise redraws next to a phase event (left out of the curves) also start a mismatch
            for k in range(hold_steps, i_reach, hold_steps):
                win[k:k + round(cal["settle"]["noise"] / dt) + 1] = True
        # sums per (class, outside / inside the windows): n, SSE v, SSE w, label-change + noise part,
        # tracking part (copycat error = (label - exec_prev) + (exec_prev - measured)), sums for R^2
        parts = np.stack([np.ones(n), ev**2, ew**2, (v - ve_prev)**2, (w - we_prev)**2, (ve_prev - vh)**2,
                          (we_prev - wh)**2, v, v**2, w, w**2], axis=1)
        sums = np.zeros((5, 2, parts.shape[1]))
        for c in range(5):
            for inside in (0, 1):
                m = (cls == c) & (win == bool(inside))
                sums[c, inside] = parts[m].sum(axis=0)
        fixed = np.zeros((len(FIXED_WINDOWS), 2, 3))  # (width, outside/inside, [n, SSE v, SSE w])
        for i, width_s in enumerate(FIXED_WINDOWS):
            fw = np.zeros(n, dtype=bool)
            starts = [k for _, k in events] + (list(range(hold_steps, i_reach, hold_steps)) if noisy else [])
            for k in starts:  # every event, including the noise redraws next to a phase event (as in the main windows)
                fw[k:k + round(width_s / dt) + 1] = True
            for inside in (0, 1):
                m = fw == bool(inside)
                fixed[i, inside] = [m.sum(), np.sum(ev[m]**2), np.sum(ew[m]**2)]
        win_idx = np.flatnonzero(win)
        decision = {s_: int(np.sum(win_idx % s_ == 0)) for s_ in STRIDES}
        decision_by_type = np.zeros((len(EVENT_TYPES), len(STRIDES)))
        for j, name in enumerate(EVENT_TYPES):
            for k, width in win_by_type.get(name, []):
                idx = np.arange(k, min(k + width + 1, n))
                decision_by_type[j] += [np.sum(idx % s_ == 0) for s_ in STRIDES]
        out.update(cc_sums=sums, cc_fixed=fixed, decision=decision, decision_by_type=decision_by_type)

        # ---------------- restarted expert: is the base label a function of the current observation?
        idx = np.arange(0, n, E1_EVERY)
        e1 = np.zeros((len(E1_VARIANTS), 4, 4))  # (variant, recorded phase, [n, inconsistent, SSE dv, SSE dw])
        thr = np.zeros((4, 4))  # (phase, [n, heading err to current-position goal >= ang_tol,
        #                                 same for the true goal, done tests (true / current-position goal) disagree])
        for i in idx:
            c = ph[i]
            to_t = target[:2] - pos[i]
            mb = np.arctan2(to_t[1], to_t[0])
            goal_m = target[:2] - standoff * np.array([np.cos(mb), np.sin(mb)])
            for j, (_, use_markov, use_rule) in enumerate(E1_VARIANTS):
                g, gb = (goal_m, mb) if use_markov else (goal, goal_b)
                state = BaseNavState(phase="drive" if use_rule and vh[i] > cal["v_drive"] else "align")
                cmd_v, cmd_w = vw(base_controller(state, pos3[i], yaw[i], g, gb, t[i], BASE))
                dv, dw = cmd_v - v[i], cmd_w - w[i]
                e1[j, c] += [1, abs(dv) > INCONS_DV or abs(dw) > INCONS_DW, dv**2, dw**2]
            head_m = np.arctan2(goal_m[1] - pos[i, 1], goal_m[0] - pos[i, 0])
            head_t = np.arctan2(goal[1] - pos[i, 1], goal[0] - pos[i, 0])
            done_t = abs(wrap(goal_b - yaw[i])) < BASE.ang_tol
            done_m = abs(wrap(mb - yaw[i])) < BASE.ang_tol
            thr[c] += [1, abs(wrap(head_m - yaw[i])) >= BASE.ang_tol, abs(wrap(head_t - yaw[i])) >= BASE.ang_tol,
                       done_t != done_m]
        out.update(e1=e1, e1_thr=thr)

        # ---------------- chatter mechanism (kept episodes with chattering reach steps)
        chat = s > cal["chatter_s"]
        row["n_chatter"] = int(chat.sum())
        if keep and (s > min(CHATTER_SENSITIVITY)).any():
            dyn, model, data = _W["dyn"], _W["dyn"].pin_model, _W["dyn"].pin_data
            base_ori = a["base_ori"].astype(float)
            qd_des_rec = a["qd_des_arm"].astype(float)
            ks = np.arange(i_reach + 2, n - 1)  # one step before the first tested step, for the toggle test
            frozen = np.zeros((len(ks), 7), dtype=bool)
            near = np.zeros((len(ks), 7), dtype=bool)
            sigma = np.zeros(len(ks))
            match = np.zeros(len(ks), dtype=bool)
            for m, k in enumerate(ks):
                q_pin = dyn.ReoderJoints2PinVec(np.concatenate((pos3[k], base_ori[k], q[k])), "pos")
                qd_ref, frozen[m], near[m], sigma[m] = ik_reference(
                    model, data, q_pin, q[k, 4:11], target, _W["ee_id"], _W["shoulder_id"], lower, upper, ARM)
                match[m] = np.max(np.abs(qd_ref - qd_des_rec[k])) < 1e-3
            toggle = np.r_[False, (frozen[1:] != frozen[:-1]).any(axis=1)]
            tested = slice(1, None)  # ks[1:] = the steps of k_int, aligned with s
            mech = {}
            for thr_s in CHATTER_SENSITIVITY:
                chat_t = s > thr_s
                groups = {}
                for name, m in (("chat", chat_t), ("calm", ~chat_t)):
                    groups[name] = [m.sum(), (near[tested].any(axis=1) & m).sum(), (toggle[tested] & m).sum(),
                                    ((sigma[tested] < ARM.singularity_threshold) & m).sum(), (match[tested] & m).sum()]
                mech[thr_s] = {"groups": groups, "near_joint_chat": near[tested][chat_t].sum(axis=0),
                               "frozen_joint_chat": frozen[tested][chat_t].sum(axis=0)}
            out["mech"] = mech
    return out


# ==============================================================================
# Statistics helpers
# ==============================================================================
def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return centre - half, centre + half


def wilson_half_width(p, n, z=1.96):
    """Half-width of the 95% Wilson interval of a proportion p estimated from n episodes (resolution of one rate)."""
    if n == 0:
        return np.nan
    return z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)


def boot_ci(stat, strata, n_boot=2000, seed=0):
    """Point estimate and 95% percentile interval of stat(episode indices), resampling episodes with replacement
    within each stratum (clean / noisy), so the clean/noisy mix stays fixed."""
    rng = np.random.default_rng(seed)
    strata = [np.asarray(s) for s in strata if len(s)]
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a ratio can be 0/0 (e.g. no step of a kind)
        point = stat(np.concatenate(strata))
        reps = [stat(np.concatenate([rng.choice(s, size=len(s)) for s in strata])) for _ in range(n_boot)]
        lo, hi = np.nanpercentile(np.array(reps, dtype=float), [2.5, 97.5], axis=0)
    return point, lo, hi


def fmt_ci(point, lo, hi, scale=1.0, digits=2):
    return f"{point * scale:.{digits}f} [{lo * scale:.{digits}f}, {hi * scale:.{digits}f}]"


def rcs_basis(x, knots):
    """Restricted cubic spline basis (linear beyond the outer knots), len(knots) - 1 columns including x."""
    k = np.asarray(knots)
    def pos3(u):
        return np.maximum(u, 0.0)**3
    cols = [x]
    for j in range(len(k) - 2):
        cols.append((pos3(x - k[j]) - pos3(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                     + pos3(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2])) / (k[-1] - k[0])**2)
    return np.column_stack(cols)


def logistic_fit(X, y, n_iter=100):
    """Maximum-likelihood logistic regression (Newton / IRLS) with an intercept. Returns (beta, cov, loglik)."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        p = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = np.clip(1 / (1 + np.exp(-X @ beta)), 1e-12, 1 - 1e-12)
    loglik = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    return beta, np.linalg.inv(H), loglik


# ==============================================================================
# Aggregation
# ==============================================================================
def column(rows, key, dtype=float):
    return np.array([row.get(key, np.nan) for row in rows], dtype=dtype)


def strata_of(rows, mask=None):
    noisy = column(rows, "noisy", bool)
    mask = np.ones(len(rows), dtype=bool) if mask is None else mask
    return [np.flatnonzero(mask & ~noisy), np.flatnonzero(mask & noisy)]


def section_modes(lines, sets):
    train = sets["train"]
    lines.append("== Behaviour modes (task parameters only; descriptive) ==")
    lines.append("M0 |d0| < pos_tol 0.08 m (no drive), M1 d0 <= -0.08 (drive away, turn around), M2 d0 >= 0.08 (normal); "
                 "'boundary' = 0.08 <= |d0| < 0.15 (both sides)")
    d0 = column(train, "d0")
    strata = [("M0", np.abs(d0) < 0.08), ("M1", d0 <= -0.08), ("M2 0.08-0.5", (d0 >= 0.08) & (d0 < 0.5)),
              ("M2 0.5-1.0", (d0 >= 0.5) & (d0 < 1.0)), ("M2 >= 1.0", d0 >= 1.0),
              ("boundary", (np.abs(d0) >= 0.08) & (np.abs(d0) < 0.15))]
    noisy, keep = column(train, "noisy", bool), column(train, "keep", bool)
    reached = np.array([row["status"] == "reached" for row in train])
    lines.append(f"{'train stratum':<14}{'group':<7}{'n':>5}{'kept':>6}  {'kept % [Wilson 95%]':<22}{'reached %':>10}{'nav t/o':>8}")
    for name, m in strata:
        for group, g in (("clean", ~noisy), ("noisy", noisy)):
            sel = m & g
            k, nn = int(keep[sel].sum()), int(sel.sum())
            lo, hi = wilson(k, nn)
            lines.append(f"{name:<14}{group:<7}{nn:>5}{k:>6}  {100 * k / max(nn, 1):5.1f} [{100 * lo:5.1f}, {100 * hi:5.1f}]"
                         f"{100 * reached[sel].mean() if nn else np.nan:>10.1f}"
                         f"{int(column(train, 'nav_timed_out', bool)[sel].sum()):>8}")
    # evaluation sets: stratum sizes and the resolution of one success rate (Wilson 95% half-width)
    for set_name in ("dev", "hard"):
        rows = sets.get(set_name)
        if not rows:
            continue
        d0s = column(rows, "d0")
        lines.append(f"{set_name.upper()} strata: n (Wilson 95% half-width of one success rate at p = 0.80 / 0.95)")
        parts = []
        for name, _ in strata:
            m = {"M0": np.abs(d0s) < 0.08, "M1": d0s <= -0.08, "M2 0.08-0.5": (d0s >= 0.08) & (d0s < 0.5),
                 "M2 0.5-1.0": (d0s >= 0.5) & (d0s < 1.0), "M2 >= 1.0": d0s >= 1.0,
                 "boundary": (np.abs(d0s) >= 0.08) & (np.abs(d0s) < 0.15)}[name]
            nn = int(m.sum())
            halves = [100 * wilson_half_width(p, nn) for p in (0.8, 0.95)]
            parts.append(f"{name} {nn} (+-{halves[0]:.0f} / +-{halves[1]:.0f})")
        lines.append("  " + ", ".join(parts))
    # mode check against the observed phases (kept train episodes)
    kept = [row for row in train if row["keep"]]
    bad = {"M0 with drive steps": sum(row["mode"] == "M0" and row["n_drive"] > 0 for row in kept),
           "M1 with orient turn <= 120 deg": sum(row["mode"] == "M1" and row["turn_orient_deg"] <= 120 for row in kept),
           "M2 with orient turn >= 60 deg": sum(row["mode"] == "M2" and row["turn_orient_deg"] >= 60 for row in kept)}
    lines.append("Mode check (kept train): " + ", ".join(f"{k} {v}" for k, v in bad.items()))
    absd = np.abs(column(kept, "d0"))
    diff = column(kept, "turn_align_deg") - column(kept, "align_req_deg")
    has_align = column(kept, "n_align") > 0
    bins = [(0.08, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 4.0)]
    lines.append("Measured - required align turn (deg), median [p10, p90] by |d0|: " + ", ".join(
        f"{lo}-{hi} m {np.median(diff[m]):.1f} [{np.percentile(diff[m], 10):.1f}, {np.percentile(diff[m], 90):.1f}]"
        for lo, hi in bins for m in [has_align & (absd >= lo) & (absd < hi)] if m.any()))
    disp = column(kept, "align_disp")
    lines.append(f"Base displacement during align (m): median {np.median(disp[has_align]):.3f}, "
                 f"p90 {np.percentile(disp[has_align], 90):.3f}, max {disp[has_align].max():.3f}")
    gap = column(kept, "d_drive") - absd
    has_drive = np.isfinite(gap)
    for lo, hi in ((0.08, 0.4), (0.4, 1.0), (1.0, 4.0)):
        m = has_drive & (absd >= lo) & (absd < hi)
        lines.append(f"  |d0| {lo}-{hi} m: distance to the goal at drive start - |d0| (m) median {np.median(gap[m]):+.3f}, "
                     f"p90 {np.percentile(gap[m], 90):+.3f} (n {m.sum()})")
    lines.append("")


def section_phases(lines, train):
    lines.append("== Phase shares (descriptive) ==")
    noisy, keep = column(train, "noisy", bool), column(train, "keep", bool)
    counts = np.stack([column(train, f"n_{c}") for c in CLASSES], axis=1)
    lines.append(f"{'subset':<22}{'per':<9}" + "".join(f"{c:>12}" for c in CLASSES))
    for name, m in (("all", np.ones(len(train), bool)), ("kept", keep), ("kept clean", keep & ~noisy),
                    ("kept noisy", keep & noisy)):
        by_sample = counts[m].sum(axis=0) / counts[m].sum()
        by_episode = np.mean(counts[m] / counts[m].sum(axis=1, keepdims=True), axis=0)
        lines.append(f"{name:<22}{'sample':<9}" + "".join(f"{100 * x:>11.1f}%" for x in by_sample))
        lines.append(f"{'':<22}{'episode':<9}" + "".join(f"{100 * x:>11.1f}%" for x in by_episode))
    lines.append("Duration per kept episode (s), median [p10, p90] over episodes with that class: " + ", ".join(
        f"{c} {np.median(x):.2f} [{np.percentile(x, 10):.2f}, {np.percentile(x, 90):.2f}] (n {len(x)})"
        for j, c in enumerate(CLASSES) for x in [counts[keep, j][counts[keep, j] > 0] / 1000.0] if len(x)))
    kept = [row for row in train if row["keep"]]
    has_hold = column(kept, "has_hold", bool)
    early = column(kept, "t_first_reach") < MIN_TIME
    exits = column(kept, "n_hold_exit")
    lines.append(f"Hold tail: {has_hold.sum()} of {len(kept)} kept episodes have steps after the first reach "
                 f"(first reach before {MIN_TIME:.0f} s: {early.sum()}); {int((exits[has_hold] > 0).sum())} of them leave "
                 f"{1000 * SETTLE_TOL:.0f} mm again ({int(exits.sum())} steps)")
    seqs = {}
    for row in kept:
        seqs[row["phase_seq"]] = seqs.get(row["phase_seq"], 0) + 1
    lines.append("Phase sequences (kept; 0 align, 1 drive, 2 orient, 3 reach): " + ", ".join(
        f"{k} {v}" for k, v in sorted(seqs.items(), key=lambda kv: -kv[1])))
    lines.append("")


def section_copycat(lines, train, results, cal, out_dir, plt):
    lines.append("== Copycat baseline: measured wheel speeds as the wheel-command prediction (kept train) ==")
    keep = column(train, "keep", bool)
    kidx = np.flatnonzero(keep)
    sums = np.stack([results[i]["cc_sums"] for i in kidx])  # (episode, class, out/in, stat)
    tot = sums.sum(axis=(0, 1, 2))
    n_all = tot[0]
    sd_v = np.sqrt(tot[8] / n_all - (tot[7] / n_all)**2)
    sd_w = np.sqrt(tot[10] / n_all - (tot[9] / n_all)**2)
    r2_v, r2_w = 1 - tot[1] / (n_all * sd_v**2), 1 - tot[2] / (n_all * sd_w**2)
    r2_4 = column(train, "copycat_r2_4wheels")
    noisy = column(train, "noisy", bool)
    lines.append(f"Label std (pooled, kept train): v {sd_v:.4f} m/s, w {sd_w:.4f} rad/s. Pooled R^2: v {r2_v:.4f}, w {r2_w:.4f} "
                 f"(mostly variance between phases). Per-episode R^2 of the 4 wheel speeds, median: clean "
                 f"{np.nanmedian(r2_4[keep & ~noisy]):.4f}, noisy {np.nanmedian(r2_4[keep & noisy]):.4f} "
                 f"(earlier check on train ep 0-59: 0.995 / 0.986; {int(np.sum(keep & np.isnan(r2_4)))} kept episodes never move the "
                 f"wheels)")
    lines.append(f"settle time per event type (s, calibrated on data/pilot): " + ", ".join(f"{k} {v:.3f}" for k, v in cal["settle"].items()))
    lines.append(f"{'class':<12}{'group':<7}{'steps':>11}{'RMSE v':>9}{'RMSE w':>9}{'nRMSE v':>9}{'nRMSE w':>9}"
                 f"{'steady v':>10}{'steady w':>10}{'window v':>10}{'window w':>10}  RMS label change+noise / tracking (v; w)")
    for c, cname in enumerate(CLASSES):
        for group, g in (("clean", ~noisy[kidx]), ("noisy", noisy[kidx])):
            s = sums[g, c].sum(axis=0)  # (out/in, stat)
            a = s.sum(axis=0)
            if a[0] == 0:
                continue
            rm = lambda x, k: np.sqrt(x[k] / x[0]) if x[0] else np.nan
            lines.append(f"{cname:<12}{group:<7}{int(a[0]):>11}{rm(a, 1):>9.4f}{rm(a, 2):>9.4f}{rm(a, 1) / sd_v:>9.3f}"
                         f"{rm(a, 2) / sd_w:>9.3f}{rm(s[0], 1):>10.4f}{rm(s[0], 2):>10.4f}{rm(s[1], 1):>10.4f}{rm(s[1], 2):>10.4f}"
                         f"  {rm(a, 3):.4f} / {rm(a, 5):.4f}; {rm(a, 4):.4f} / {rm(a, 6):.4f}")
    lines.append("(RMSE v in m/s, w in rad/s; nRMSE = RMSE / pooled label std; steady = outside the event windows; copycat "
                 "error = (label - executed command of the previous step) + (previous executed command - measured))")

    # share of the (normalised) squared error inside the event windows, with episode-bootstrap CI
    ep_in = sums[:, :, 1, :].sum(axis=1)   # (episode, stat)
    ep_all = sums.sum(axis=(1, 2))

    def window_share(idx):
        e_in = ep_in[idx, 1].sum() / sd_v**2 + ep_in[idx, 2].sum() / sd_w**2
        e_all = ep_all[idx, 1].sum() / sd_v**2 + ep_all[idx, 2].sum() / sd_w**2
        return np.array([ep_in[idx, 0].sum() / ep_all[idx, 0].sum(), e_in / e_all])

    strata = [np.flatnonzero(~noisy[kidx]), np.flatnonzero(noisy[kidx])]
    for name, st in (("all kept", strata), ("kept clean", strata[:1]), ("kept noisy", strata[1:])):
        point, lo, hi = boot_ci(window_share, st, n_boot=1000)
        lines.append(f"Event windows [0, settle time], {name}: {fmt_ci(point[0], lo[0], hi[0], 100, 1)}% of the steps hold "
                     f"{fmt_ci(point[1], lo[1], hi[1], 100, 1)}% of the normalised squared error "
                     f"(ratio {point[1] / point[0]:.2f})")
    fixed = np.stack([results[i]["cc_fixed"] for i in kidx]).sum(axis=0)  # (width, out/in, [n, sse v, sse w])
    lines.append("Sensitivity, fixed windows after every event (incl. noise redraws): " + ", ".join(
        f"{wd * 1000:.0f} ms: {100 * fixed[i, 1, 0] / fixed[i].sum(axis=0)[0]:.1f}% steps, "
        f"{100 * (fixed[i, 1, 1] / sd_v**2 + fixed[i, 1, 2] / sd_w**2) / (fixed[i].sum(axis=0)[1] / sd_v**2 + fixed[i].sum(axis=0)[2] / sd_w**2):.1f}% error"
        for i, wd in enumerate(FIXED_WINDOWS)))
    n_ev = np.stack([results[i]["n_events"] for i in kidx]).sum(axis=0)
    # where the excess error after an event sits: share of sum(max(curve - post-event level, 0)) over 0-400 ms
    curves_all = np.stack([results[i]["curves"] for i in kidx]).astype(float)
    bands = [("0 ms", 0, 1), ("1-9 ms", 1, 10), ("10-99 ms", 10, 100), ("100-400 ms", 100, 401)]
    lines.append("Excess copycat error after an event (normalised v + w), share by time since the event:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for j, name in enumerate(EVENT_TYPES):
            c = curves_all[:, j]
            if np.isfinite(c).any(axis=(1, 2)).sum() < 3:
                continue
            comb = np.nanmean(c[..., 0], axis=0) / sd_v**2 + np.nanmean(c[..., 1], axis=0) / sd_w**2
            excess = np.maximum(comb - np.mean(comb[(OFFSETS >= 400) & (OFFSETS <= 500)]), 0)
            total = excess[(OFFSETS >= 0) & (OFFSETS <= 400)].sum()
            lines.append(f"  {name:<14} " + ", ".join(
                f"{label} {100 * excess[(OFFSETS >= lo) & (OFFSETS < hi)].sum() / total:.0f}%" for label, lo, hi in bands)
                + f" (event step = {comb[OFFSETS == 0][0] / max(np.mean(comb[(OFFSETS >= 400) & (OFFSETS <= 500)]), 1e-12):.0f}x the post-event level)")
    dec = np.stack([results[i]["decision_by_type"] for i in kidx]).sum(axis=0)
    lines.append("Decision steps kept per event after downsampling (mean over events), strides " + "/".join(map(str, STRIDES)) + ":")
    for j, name in enumerate(EVENT_TYPES):
        if n_ev[j]:
            lines.append(f"  {name:<14} {n_ev[j]:>6} events: " + " / ".join(f"{x:.1f}" for x in dec[j] / n_ev[j]))
    lines.append("")

    # event-locked error curves, RMS over episodes of the per-episode mean squared error, in label-std units
    curves = np.stack([results[i]["curves"] for i in kidx]).astype(float)  # (episode, type, offset, v/w)
    rng = np.random.default_rng(1)
    with plt.rc_context(REPORT_RC):
        fig, axes = plt.subplots(3, 2, figsize=(3.5, 4.0), sharex=True, sharey=True, layout="constrained")
        for j, (ax, name) in enumerate(zip(axes.flat, EVENT_TYPES)):
            n_epi = int(np.isfinite(curves[:, j]).any(axis=(1, 2)).sum())
            for ch, (label, color, style, sd) in enumerate((("v", SERIES[0], "-", sd_v), ("w", SERIES[2], (0, (4, 1.5)), sd_w))):
                c = curves[:, j, :, ch]
                c = c[np.isfinite(c).any(axis=1)]
                if len(c) < 3:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean = np.sqrt(np.nanmean(c, axis=0)) / sd
                    boots = np.array([np.sqrt(np.nanmean(c[rng.integers(0, len(c), len(c))], axis=0)) / sd
                                      for _ in range(300)])
                lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
                ax.fill_between(OFFSETS, np.maximum(lo, 1e-5), hi, color=color, alpha=0.25, linewidth=0)
                ax.plot(OFFSETS, mean, color=color, linestyle=style, linewidth=1.0, label=f"{label} error")
            settle = cal["settle"].get(name, 0.0)
            if settle > 0:
                ax.axvline(1000 * settle, color=MUTED, linewidth=0.8, linestyle=":")
            title = "noise redraw" if name == "noise" else name.replace(">", "\u2192")
            count = f"{n_ev[j]}" if n_ev[j] == n_epi else f"{n_ev[j]} in {n_epi} ep."
            ax.set_title(f"{title} (n {count})", color=INK_2, loc="left", fontsize=6.5)
            ax.set_yscale("log")
            ax.set_ylim(1e-4, 5)
            style_axes(ax)
        fig.supxlabel("time since event (ms)", fontsize=8, color=INK_2)
        fig.supylabel("RMS copycat error / label s.d.", fontsize=8, color=INK_2)
        finish(fig, out_dir, "copycat_events.png", legend_ax=axes.flat[0], where="upper")
    return sd_v, sd_w


def section_hidden_state(lines, train, results):
    lines.append("== Restarted expert: base commands that are not a function of the current observation (kept train) ==")
    lines.append(f"Every {E1_EVERY}th step, base_controller is called with a fresh FSM state and compared with the label; "
                 f"inconsistent = |dv| > {INCONS_DV} m/s or |dw| > {INCONS_DW} rad/s.")
    keep = column(train, "keep", bool)
    noisy = column(train, "noisy", bool)
    unseen = keep & ~column(train, "seen", bool)
    e1 = np.stack([results[i].get("e1", np.zeros((4, 4, 4))) for i in range(len(train))])  # (ep, var, phase, stat)
    thr = np.stack([results[i].get("e1_thr", np.zeros((4, 4))) for i in range(len(train))])
    absd = np.abs(column(train, "d0"))

    def rate(idx, var, phases):
        s = e1[idx][:, var][:, phases].sum(axis=(0, 1))
        return s[1] / s[0] if s[0] else np.nan

    lines.append(f"{'variant':<44}" + "".join(f"{p:>22}" for p in PHASE_NAMES) + f"{'all':>22}")
    st_keep = strata_of(train, keep)
    for j, (name, _, _) in enumerate(E1_VARIANTS):
        cells = []
        for phases in ([0], [1], [2], [3], [0, 1, 2, 3]):
            point, lo, hi = boot_ci(lambda idx: rate(idx, j, phases), st_keep, n_boot=500)
            cells.append(f"{fmt_ci(point, lo, hi, 100, 1)}%")
        lines.append(f"{name:<44}" + "".join(f"{c:>22}" for c in cells))
    lines.append("(inconsistent steps in % [episode-bootstrap 95% CI])")
    t = thr[keep].sum(axis=0)
    lines.append(f"drive steps with heading error >= ang_tol to the current-position goal {100 * t[1, 1] / t[1, 0]:.1f}%, "
                 f"to the true goal {100 * t[1, 2] / t[1, 0]:.1f}% (fresh FSM state would turn on the spot)")
    lines.append(f"orient / reach steps where 'facing the goal' differs between the true and the current-position goal: "
                 f"{100 * t[2, 3] / max(t[2, 0], 1):.1f}% / {100 * t[3, 3] / max(t[3, 0], 1):.1f}%")
    kept = [row for row in train if row["keep"]]
    off, boff = column(kept, "navdone_markov_goal_offset"), column(kept, "navdone_markov_bearing_offset_deg")
    lines.append(f"At nav_done: current-position goal - true goal {100 * np.median(off):.1f} cm median, p90 "
                 f"{100 * np.percentile(off, 90):.1f} cm, > pos_tol in {100 * np.mean(off > BASE.pos_tol):.1f}%; bearing offset "
                 f"median {np.median(boff):.2f} deg, > ang_tol in {100 * np.mean(boff > np.rad2deg(BASE.ang_tol)):.1f}%")

    # Prediction 1 (pre-registered): on the unseen kept episodes. Main test with the current-position goal (only
    # single-frame information); the same tests with the true goal isolate the FSM memory.
    st = strata_of(train, unseen)
    for label, no_rule, rule in (("current-position goal (main)", 1, 3), ("true goal (FSM memory only)", 0, 2)):
        lines.append(f"Prediction 1, {label}, unseen kept train episodes:")

        def share_ratio(idx, no_rule=no_rule):
            s_ = e1[idx][:, no_rule].sum(axis=0)  # (phase, stat)
            return (s_[[1, 2], 1].sum() / s_[:, 1].sum()) / (s_[[1, 2], 0].sum() / s_[:, 0].sum())

        def d0_ratio(idx, no_rule=no_rule):
            near, far = idx[absd[idx] < 1.0], idx[absd[idx] >= 1.0]
            return rate(near, no_rule, [0, 1, 2, 3]) / rate(far, no_rule, [0, 1, 2, 3])

        tests = [("1a drive+orient share of inconsistent steps / their share of steps", share_ratio, "lo > 1"),
                 ("1b inconsistency rate |d0| < 1 / |d0| >= 1", d0_ratio, "lo > 1"),
                 ("1c drive: with rule / without", lambda idx, a=rule, b=no_rule: rate(idx, a, [1]) / rate(idx, b, [1]),
                  "hi < 0.5"),
                 ("1d orient: with rule / without", lambda idx, a=rule, b=no_rule: rate(idx, a, [2]) / rate(idx, b, [2]),
                  "in [0.8, 1.25]")]
        for name, fn, rule_text in tests:
            point, lo, hi = boot_ci(fn, st)
            if rule_text == "lo > 1":
                verdict = "supported" if lo > 1 else ("not supported" if hi <= 1 else "inconclusive")
            elif rule_text == "hi < 0.5":
                verdict = "supported" if hi < 0.5 else ("not supported" if lo > 0.5 else "inconclusive")
            else:
                verdict = ("supported" if 0.8 <= lo and hi <= 1.25 else
                           "not supported" if hi < 0.8 or lo > 1.25 else "inconclusive")
            point_all, lo_all, hi_all = boot_ci(fn, strata_of(train, keep), n_boot=500)
            lines.append(f"  {name}: {fmt_ci(point, lo, hi)}  (rule: CI {rule_text}) -> {verdict}; "
                         f"all kept: {fmt_ci(point_all, lo_all, hi_all)}")
    lines.append("")


def section_label_quality(lines, train, results, cal, out_dir, plt):
    lines.append("== Arm label quality (kept train) ==")
    kept = [row for row in train if row["keep"]]
    q6 = column(kept, "q6_start")
    lines.append(f"Joint 6 start angle {np.unique(np.round(q6, 4))} rad, URDF lower limit {LIMITS['lower'][5]:.4f} rad: "
                 f"below the limit in a median {100 * np.median(column(kept, 'j6_below_limit_frac')):.0f}% of the steps; "
                 f"IK reference clamped at lower + margin in a median {100 * np.nanmedian(column(kept, 'j6_clamp_frac')):.0f}% "
                 f"[p10 {100 * np.nanpercentile(column(kept, 'j6_clamp_frac'), 10):.0f}, p90 "
                 f"{100 * np.nanpercentile(column(kept, 'j6_clamp_frac'), 90):.0f}] of the reach steps")
    jump, jump6 = column(kept, "onset_jump_max"), column(kept, "onset_jump_j6")
    lines.append(f"Torque jump at reach onset: largest joint median {np.nanmedian(jump):.1f} N m [p10 {np.nanpercentile(jump, 10):.1f}, "
                 f"p90 {np.nanpercentile(jump, 90):.1f}]; joint 6 median {np.nanmedian(jump6):+.1f} N m")
    n_sat, n_sat_on = column(kept, "n_sat").sum(), column(kept, "n_sat_onset").sum()
    lines.append(f"Torque samples at the 30 N m limit: {int(n_sat)}, of which {int(n_sat_on)} ({100 * n_sat_on / max(n_sat, 1):.0f}%) "
                 f"within 0.2 s after reach onset")
    z, ci = column(kept, "z"), column(kept, "chatter_index")
    ok = np.isfinite(ci)
    lines.append(f"Chatter index (power share above {CHATTER_HF:.0f} Hz of the reach torques; white noise 0.5, pure period-2 1):")
    for lo, hi in ((0.2, 0.3), (0.3, 0.45), (0.45, 0.6), (0.6, 0.9), (0.9, 1.21)):
        m = ok & (z >= lo) & (z < hi)
        if m.any():
            lines.append(f"  z {lo:.2f}-{hi:.2f}: n {m.sum():>4}, median {np.median(ci[m]):.3f}, "
                         f"episodes > {CHATTER_EPISODE}: {int((ci[m] > CHATTER_EPISODE).sum())} ({100 * np.mean(ci[m] > CHATTER_EPISODE):.0f}%)")

    # Prediction 2 (pre-registered): mechanism of the chatter, on unseen kept episodes with chattering steps
    idx = [i for i, row in enumerate(train) if row["keep"] and "mech" in results[i]]
    unseen = [i for i in idx if not train[i]["seen"]]
    main_thr = cal["chatter_s"]
    lines.append(f"Prediction 2 (chatter step: amplitude > {main_thr:.1f} N m; {len(unseen)} unseen kept episodes with reach "
                 f"steps above {min(CHATTER_SENSITIVITY)} N m, {len(idx)} in all):")

    def groups_at(thr_s):
        # per episode: [steps, near a limit, frozen set toggled, sigma_min below threshold, IK replication matches]
        return {name: np.array([results[i]["mech"][thr_s]["groups"][name] if "mech" in results[i] else [0] * 5
                                for i in range(len(train))], dtype=float) for name in ("chat", "calm")}

    g = groups_at(main_thr)
    tested = g["chat"][:, 0].sum() + g["calm"][:, 0].sum()
    rep_all = (g["chat"][:, 4].sum() + g["calm"][:, 4].sum()) / max(tested, 1)
    rep = g["chat"][:, 4].sum() / max(g["chat"][:, 0].sum(), 1)
    lines.append(f"  IK replication: recomputed qd_des within 1e-3 rad/s of the recorded one in {100 * rep_all:.2f}% of "
                 f"{int(tested)} tested reach steps ({100 * rep:.2f}% of the chattering ones)")
    st = strata_of(train, np.isin(np.arange(len(train)), unseen))
    hypotheses = (("H-limit", 1, "a joint within the limit margin"),
                  ("H-limit (toggle)", 2, "the frozen joint set changed since the previous step"),
                  ("H-singular", 3, f"sigma_min < {ARM.singularity_threshold}"))
    for thr_s in (main_thr, *[x for x in CHATTER_SENSITIVITY if x != main_thr]):
        gs = groups_at(thr_s)

        def frac(idx_, col):
            return gs["chat"][idx_, col].sum() / gs["chat"][idx_, 0].sum()

        def enrich(idx_, col):
            return frac(idx_, col) / (gs["calm"][idx_, col].sum() / gs["calm"][idx_, 0].sum())

        if thr_s != main_thr:
            lines.append(f"  sensitivity, chatter step amplitude > {thr_s} N m ({int(gs['chat'][unseen, 0].sum())} steps):")
        for hyp, col, text in hypotheses:
            f_p, f_lo, f_hi = boot_ci(lambda i_: frac(i_, col), st, n_boot=2000 if thr_s == main_thr else 500)
            e_p, e_lo, e_hi = boot_ci(lambda i_: enrich(i_, col), st, n_boot=2000 if thr_s == main_thr else 500)
            if col == 2:
                verdict = "supported" if e_lo > 1 else ("not supported" if e_hi <= 1 else "inconclusive")
                rule_text = "enrichment CI lo > 1"
            else:
                verdict = ("supported" if f_lo >= 0.8 and e_lo > 1 else
                           "not supported" if f_hi < 0.8 or e_hi <= 1 else "inconclusive")
                rule_text = "fraction CI lo >= 0.8 and enrichment CI lo > 1"
            prefix = "  " if thr_s == main_thr else "    "
            lines.append(f"{prefix}{hyp}: chattering steps with {text}: {fmt_ci(f_p, f_lo, f_hi, 100, 1)}%, enrichment over "
                         f"calm reach steps of the same episodes {fmt_ci(e_p, e_lo, e_hi)} (rule: {rule_text}) -> {verdict}")
        if thr_s == main_thr:
            near_j = sum(results[i]["mech"][thr_s]["near_joint_chat"] for i in unseen)
            frozen_j = sum(results[i]["mech"][thr_s]["frozen_joint_chat"] for i in unseen)
            n_chat = gs["chat"][unseen, 0].sum()
            lines.append(f"  {int(n_chat)} chattering steps; per joint, share near a limit: "
                         + ", ".join(f"j{k + 1} {100 * x / n_chat:.0f}%" for k, x in enumerate(near_j))
                         + "; frozen by the IK: " + ", ".join(f"j{k + 1} {100 * x / n_chat:.0f}%" for k, x in enumerate(frozen_j)))
    lines.append("")

    noisy = column(kept, "noisy", bool)
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.4), layout="constrained")
        for name, m, marker in (("clean", ~noisy, "o"), ("noisy (DART)", noisy, "^")):
            sel = m & ok
            ax.scatter(z[sel], ci[sel], s=8, color=SERIES[0], marker=marker, linewidths=0, alpha=0.6,
                       label=f"{name} ({sel.sum()})", zorder=3)
        ax.axvline(Z_SPLIT, color=MUTED, linewidth=0.8, linestyle="--")
        ax.axhline(CHATTER_EPISODE, color=MUTED, linewidth=0.8, linestyle=":")
        ax.axhline(0.5, color=MUTED, linewidth=0.6)
        ax.text(1.21, 0.5 * 1.15, "white noise", ha="right", fontsize=6.5, color=MUTED)
        ax.text(1.21, CHATTER_EPISODE * 1.15, f"threshold {CHATTER_EPISODE}", ha="right", fontsize=6.5, color=MUTED)
        ax.set_yscale("log")
        ax.set_ylim(min(np.nanmin(ci[ok]) * 0.7, 2e-3), 8)
        for lo_z, hi_z, xm in ((0.0, Z_SPLIT, (0.2 + Z_SPLIT) / 2), (Z_SPLIT, np.inf, (Z_SPLIT + 1.2) / 2)):
            m = ok & (z >= lo_z) & (z < hi_z)
            k = int((ci[m] > CHATTER_EPISODE).sum())
            w_lo, w_hi = wilson(k, int(m.sum()))
            ax.text(xm, 1.3, f"{k}/{m.sum()} above\n({100 * w_lo:.0f}-{100 * w_hi:.0f}%)", ha="center", va="bottom",
                    fontsize=6.5, color=INK_2)
        ax.set_xlabel("target height z (m)")
        ax.set_ylabel("chatter index")
        style_axes(ax)
        finish(fig, out_dir, "chatter_by_z.png", legend_ax=ax)


def section_failure_model(lines, train):
    """Prediction 3 (exploratory): does the required align turn add to |d0| in explaining expert failures?"""
    lines.append("== Prediction 3 (exploratory): expert failure vs required align turn, 0.08 <= |d0| < 1 m (all train) ==")
    d0 = column(train, "d0")
    m = (np.abs(d0) >= BASE.pos_tol) & (np.abs(d0) < 1.0)  # M0 has no align phase before a drive: excluded
    y = (~column(train, "keep", bool))[m].astype(float)
    noisy = column(train, "noisy", bool)[m].astype(float)
    m1 = (d0[m] < 0).astype(float)
    absd = np.abs(d0[m])
    align = np.deg2rad(column(train, "align_req_deg")[m])
    knots = np.percentile(absd, [5, 35, 65, 95])
    spline = rcs_basis(absd, knots)
    X_red = np.column_stack([noisy, m1, spline])
    X_full = np.column_stack([X_red, align, noisy * align])
    b_full, cov_full, ll_full = logistic_fit(X_full, y)
    _, _, ll_red = logistic_fit(X_red, y)
    lr = 2 * (ll_full - ll_red)
    p = stats.chi2.sf(lr, df=2)
    se = np.sqrt(np.diag(cov_full))
    names = ["align_req (per rad)", "noisy x align_req"]
    coef = "; ".join(f"{nm} {b_full[-2 + k]:+.2f} [{b_full[-2 + k] - 1.96 * se[-2 + k]:+.2f}, {b_full[-2 + k] + 1.96 * se[-2 + k]:+.2f}]"
                     for k, nm in enumerate(names))
    lines.append(f"n {m.sum()} episodes, {int(y.sum())} filtered. Logistic fail ~ noisy + M1 + rcs(|d0|, 4 knots) + align_req + "
                 f"noisy:align_req. LR test of the two align terms: chi2(2) = {lr:.2f}, p = {p:.2g} "
                 f"({'significant' if p < 0.05 else 'not significant'} at 0.05). Log-odds: {coef}")
    for group, g in (("clean", noisy == 0), ("noisy", noisy == 1)):
        cuts = np.percentile(align[g], [100 / 3, 200 / 3])
        parts = []
        for k, (lo, hi) in enumerate(((-np.inf, cuts[0]), (cuts[0], cuts[1]), (cuts[1], np.inf))):
            sel = g & (align >= lo) & (align < hi)
            parts.append(f"tercile {k + 1}: {int(y[sel].sum())}/{sel.sum()} ({100 * y[sel].mean():.1f}%)")
        lines.append(f"  {group} filtered by align_req tercile: " + ", ".join(parts))
    kept = [row for row in train if row["keep"] and BASE.pos_tol <= abs(row["d0"]) < 1.0]
    rho1 = stats.spearmanr(column(kept, "align_req_deg"), column(kept, "align_disp")).statistic
    gap = column(kept, "d_drive") - np.abs(column(kept, "d0"))
    ok = np.isfinite(gap)
    rho2 = stats.spearmanr(column(kept, "align_disp")[ok], np.abs(gap[ok])).statistic
    lines.append(f"  mechanism links (kept, descriptive): Spearman(align_req, align displacement) {rho1:.2f}; "
                 f"Spearman(align displacement, |distance at drive start - |d0||) {rho2:.2f}")
    lines.append("")


def section_coverage(lines, sets, out_dir, plt):
    """Task-level coverage: nearest training episode in the world frame (start pose + target, 7-D) and relative to
    the start pose (target in the start frame, 3-D). Reference = the kept train episodes (the training set)."""
    lines.append("== Coverage of the task space by the kept train episodes (descriptive) ==")

    def embed(rows):
        yaw0 = np.deg2rad(column(rows, "init_yaw_deg"))
        world = np.column_stack([column(rows, "init_x"), column(rows, "init_y"), np.cos(yaw0), np.sin(yaw0),
                                 column(rows, "target_x"), column(rows, "target_y"), column(rows, "z")])
        th = np.deg2rad(column(rows, "theta_deg"))
        rel = np.column_stack([column(rows, "r") * np.cos(th), column(rows, "r") * np.sin(th), column(rows, "z")])
        return {"world 7-D": world, "relative 3-D": rel}

    train = [row for row in sets["train"] if row["keep"]]
    ref = embed(train)
    trees = {k: cKDTree(v) for k, v in ref.items()}
    dists = {}
    for rep, tree in trees.items():
        dists[("train LOO", rep)] = tree.query(ref[rep], k=2)[0][:, 1]
    for set_name in ("dev", "hard"):
        rows = sets.get(set_name)
        if not rows:
            continue
        emb = embed(rows)
        for rep, tree in trees.items():
            d = tree.query(emb[rep], k=1)[0]
            dists[(set_name.upper(), rep)] = d
            loo = dists[("train LOO", rep)]
            pct = np.searchsorted(np.sort(loo), d, side="right") / len(loo)
            key = "cov_pct_world" if rep.startswith("world") else "cov_pct_rel"
            for row, value, dist in zip(rows, pct, d):
                row[key] = float(value)
                row[key.replace("pct", "nn")] = float(dist)
    lines.append("Nearest-neighbour distance (m; yaw as a unit vector, i.e. 1 m per rad), median [p90]:")
    for rep in trees:
        loo = dists[("train LOO", rep)]
        p95 = np.percentile(loo, 95)
        parts = []
        for name in ("train LOO", "DEV", "HARD"):
            d = dists.get((name, rep))
            if d is None:
                continue
            text = f"{name} {np.median(d):.3f} [{np.percentile(d, 90):.3f}]"
            if name != "train LOO":
                k = int(np.sum(d > p95))
                lo, hi = wilson(k, len(d))
                text += f", beyond train-LOO p95: {100 * k / len(d):.1f}% [{100 * lo:.1f}, {100 * hi:.1f}]"
            parts.append(text)
        lines.append(f"  {rep}: " + "; ".join(parts))
    w, r = np.median(dists[("train LOO", "world 7-D")]), np.median(dists[("train LOO", "relative 3-D")])
    lines.append(f"  median train-LOO distance world / relative: {w / r:.1f}x (same {len(train)} episodes)")
    lines.append("  (test: computed with the same code when the test set is opened at the end, see report.txt)")
    lines.append("")

    with plt.rc_context(REPORT_RC):
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), layout="constrained")
        for ax, rep in zip(axes, trees):
            for k, name in enumerate(("train LOO", "DEV", "HARD")):
                d = dists.get((name, rep))
                if d is None:
                    continue
                x = np.sort(d)
                ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=SERIES[k], linewidth=1.2,
                        label=f"{name} ({len(x)})")
            ax.axvline(np.percentile(dists[("train LOO", rep)], 95), color=MUTED, linestyle=":", linewidth=0.8)
            ax.set_title(f"{rep} (dotted: train-LOO p95)", color=INK_2, loc="left")
            ax.set_xlabel("distance to nearest kept train episode (m; yaw 1 m/rad)" if rep.startswith("world")
                          else "distance to nearest kept train episode (m)")
            style_axes(ax)
        axes[0].set_ylabel("fraction of episodes")
        finish(fig, out_dir, "coverage_ecdf.png", legend_ax=axes[0])


def plot_d0_theta(train, out_dir, plt):
    """Main data figure: where the filter removes episodes in the (d0, theta) plane. Grey: the required align turn
    is above 160 deg (turn direction nearly ambiguous); hatched: the required orient turn is above 160 deg (all of M1,
    which ends with a ~180 deg turn); vertical lines: |d0| = pos_tol (M0). The align turn is |theta| in M2 and
    180 deg - |theta| in M1."""
    d0, th = column(train, "d0"), column(train, "theta_deg")
    keep, noisy = column(train, "keep", bool), column(train, "noisy", bool)
    xg = np.linspace(d0.min() - 0.05, d0.max() + 0.05, 500)
    yg = np.linspace(-180, 180, 361)
    X, Y = np.meshgrid(xg, yg)
    align, orient = np.vectorize(required_turns)(np.deg2rad(Y), X)
    align, orient = np.rad2deg(align), np.rad2deg(orient)
    with plt.rc_context(REPORT_RC):
        fig, ax = plt.subplots(figsize=(3.5, 2.8), layout="constrained")
        ax.contourf(X, Y, (align > 160).astype(float), levels=[0.5, 1.5], colors=[GRID], zorder=0)
        ax.contourf(X, Y, ((orient > 160) & ~(align > 160)).astype(float), levels=[0.5, 1.5], colors="none",
                    hatches=["////"], zorder=0)
        for x in (-BASE.pos_tol, BASE.pos_tol):
            ax.axvline(x, color=MUTED, linewidth=0.6, zorder=1)
        ax.scatter(d0[keep], th[keep], s=2, color=SERIES[0], linewidths=0, alpha=0.35, label=f"kept ({keep.sum()})",
                   zorder=2)
        for group, g, marker in (("clean", ~noisy, "X"), ("noisy", noisy, "^")):
            sel = g & ~keep
            ax.scatter(d0[sel], th[sel], s=14, color=SERIES[1], marker=marker, linewidths=0,
                       label=f"filtered, {group} ({sel.sum()}/{g.sum()})", zorder=3)
        tr = ax.get_xaxis_transform()
        for x, text in (((xg[0] - BASE.pos_tol) / 2, "M1"), (0.0, "M0"), ((BASE.pos_tol + xg[-1]) / 2, "M2")):
            ax.text(x, 1.01, text, transform=tr, ha="center", va="bottom", color=INK_2, fontsize=7)
        ax.set_xlim(xg[0], xg[-1])
        ax.set_ylim(-180, 180)
        ax.set_yticks([-180, -90, 0, 90, 180])
        ax.set_xlabel(r"$d_0 = r - s(z)$ (m)")
        ax.set_ylabel(r"target bearing $\theta$ (deg)")
        style_axes(ax)
        finish(fig, out_dir, "fig_d0_theta.png", legend_ax=ax, ncol=2)


def plot_pairplot(train, out_dir, plt):
    """Exploration figure: task parameters and the measured align drift, kept vs filtered out. Diagonal: share of
    episodes filtered out per bin with Wilson 95% intervals."""
    names = [("d0", "d0 (m)"), ("theta_deg", "bearing (deg)"), ("z", "z (m)"), ("align_req_deg", "align turn (deg)"),
             ("align_disp", "align drift (m)")]
    keep = column(train, "keep", bool)
    k = len(names)
    with plt.rc_context(REPORT_RC):
        fig, axes = plt.subplots(k, k, figsize=(7.0, 7.0), layout="constrained")
        for i, (yi, ylab) in enumerate(names):
            for j, (xj, xlab) in enumerate(names):
                ax = axes[i, j]
                x, y = column(train, xj), column(train, yi)
                if j > i:
                    ax.axis("off")
                    continue
                if i == j:
                    bins = np.linspace(np.nanmin(x), np.nanmax(x), 13)
                    which = np.clip(np.digitize(x, bins) - 1, 0, len(bins) - 2)
                    centres = (bins[:-1] + bins[1:]) / 2
                    rate, lo, hi = [], [], []
                    for b in range(len(centres)):
                        m = which == b
                        kk, nn = int((~keep[m]).sum()), int(m.sum())
                        w_lo, w_hi = wilson(kk, nn)
                        rate.append(kk / nn if nn else np.nan)
                        lo.append(w_lo)
                        hi.append(w_hi)
                    rate, lo, hi = map(np.array, (rate, lo, hi))
                    ok = np.isfinite(rate)  # empty bins have no rate
                    ax.errorbar(centres[ok], rate[ok], yerr=[np.maximum(rate - lo, 0)[ok], np.maximum(hi - rate, 0)[ok]],
                                fmt="o", ms=2.5, color=SERIES[1], elinewidth=0.6, capsize=0)
                    ax.set_ylim(0, None)
                    ax.set_title("filtered share", color=INK_2, fontsize=6.5, loc="left")
                else:
                    ax.scatter(x[keep], y[keep], s=1, color=SERIES[0], linewidths=0, alpha=0.3)
                    ax.scatter(x[~keep], y[~keep], s=6, color=SERIES[1], marker="X", linewidths=0)
                if i == k - 1:
                    ax.set_xlabel(xlab)
                if j == 0 and i > 0:
                    ax.set_ylabel(ylab)
                style_axes(ax)
        finish(fig, out_dir, "pairplot_episodes.png")


# ==============================================================================
# Calibration on the pilot set
# ==============================================================================
def calibrate(rows, results):
    keep = column(rows, "keep", bool)
    kidx = np.flatnonzero(keep)
    curves = np.stack([results[i]["curves"] for i in kidx]).astype(float)
    print("Calibration on the pilot set (kept episodes: %d)" % len(kidx))
    settle_times = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # v and w errors are combined after dividing by their overall RMS around events (a relative weighting only)
        sd = np.sqrt(np.nanmean(curves[..., 0])), np.sqrt(np.nanmean(curves[..., 1]))
        for j, name in enumerate(EVENT_TYPES):
            c = curves[:, j]
            present = np.isfinite(c).any(axis=(1, 2))
            if present.sum() < 3:
                continue
            comb = np.nanmean(c[present, :, 0], axis=0) / sd[0]**2 + np.nanmean(c[present, :, 1], axis=0) / sd[1]**2
            # settle time: first time after which the excess over the type's own post-event level (400-500 ms)
            # stays below that level (curve <= 2x it) for 50 ms; capped at 0.4 s. (A 90% decay of the peak only
            # covers the one-step spike at the event, after which the error stays ~10x the steady level for tens of ms.)
            base = np.mean(comb[(OFFSETS >= 400) & (OFFSETS <= 500)])
            settle = 0.4
            for o in range(0, 401):
                span = (OFFSETS >= o) & (OFFSETS < o + 50)
                if np.all(comb[span] <= 2 * base):
                    settle = o / 1000.0
                    break
            settle_times[name] = float(settle)
            print(f"  {name:<14} events in {present.sum():>3} episodes, curve at 0/1/10/50/200 ms = "
                  + "/".join(f"{comb[OFFSETS == o][0] / base:.1f}" for o in (0, 1, 10, 50, 200))
                  + f" x steady, settle time {settle_times[name]:.3f} s")
    for name, hist in ((f"z >= {Z_SPLIT}", sum(results[i]["s_hist"] for i in kidx if rows[i]["z"] >= Z_SPLIT)),
                       (f"z < {Z_SPLIT}", sum(results[i]["s_hist"] for i in kidx if rows[i]["z"] < Z_SPLIT))):
        print(f"  per-step chatter amplitude, {name}: share of reach steps above 0.3 / 1 / 3 N m = " + " / ".join(
            f"{100 * hist[np.searchsorted(S_BINS[1:], x, side='right'):].sum() / hist.sum():.2f}%" for x in (0.3, 1.0, 3.0)))
    chatter_s = 1.0  # N m: no gap in the distribution (see above), so a fixed value far above smooth control
    vh = sum(results[i]["v_hist"] for i in kidx)  # (align, drive) histograms of the measured forward speed
    centres = V_BINS[1:]
    best = None
    for thr in np.arange(0.0, 0.1001, 0.0025):
        fp = vh[0][centres > thr].sum() / vh[0].sum()
        fn = vh[1][centres <= thr].sum() / vh[1].sum()
        if best is None or (fp + fn) / 2 < best[1]:
            best = (thr, (fp + fn) / 2, fp, fn)
    print(f"  v_drive {best[0]:.4f} m/s (align steps above: {100 * best[2]:.2f}%, drive steps at or below: {100 * best[3]:.2f}%)")
    print("CALIBRATION = " + repr({"settle": settle_times, "chatter_s": round(chatter_s, 3), "v_drive": round(float(best[0]), 4)}))


# ==============================================================================
# Main
# ==============================================================================
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#ffffff", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# categorical slots 1-3 (validated all-pairs). Fixed meanings in the report figures: blue = kept / v, orange = filtered
# out, aqua = w; clean / noisy only by marker shape
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
# print-size figures: drawn at IEEE column width (3.5 in) or page width (7 in), never scaled down
REPORT_RC = {"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
             "legend.fontsize": 7, "hatch.color": AXIS, "hatch.linewidth": 0.5, "font.family": "sans-serif"}
LIMITS = {}


def style_axes(ax):
    """Recessive chrome: hairline grid, light axes, muted tick labels."""
    import matplotlib
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(AXIS)
        spine.set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=matplotlib.rcParams["xtick.labelsize"], width=0.6)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def finish(fig, out_dir, name, legend_ax=None, ncol=4, where="lower"):
    """Legend in a row below the plots (never over the data); PNG + PDF. No title: the caption goes in the report.
    The figure must use layout="constrained" (needed for the 'outside' legend placement)."""
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        legend = fig.legend(handles, labels, loc=f"outside {where} center", ncol=min(len(labels), ncol), frameon=False)
        for text in legend.get_texts():
            text.set_color(INK_2)
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=300, facecolor=SURFACE)
    fig.savefig(os.path.splitext(path)[0] + ".pdf", facecolor=SURFACE)
    print("saved", path)


def robot_info():
    """Active joint names and joint limits, read once from a simulator instance (no episode is simulated)."""
    from simulation_and_control import pb
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sim = pb.SimInterface("robotnik_arm.json", conf_file_path_ext=cur_dir, use_gui=False)
    names = list(sim.getNameActiveJoints())
    lower, upper = sim.GetBotJointsLimit()
    sim.pybullet_client.disconnect()
    return names, np.array(lower[4:11], dtype=float), np.array(upper[4:11], dtype=float)


def scan_set(set_dir, workers, ext_names, settings):
    set_name = os.path.basename(os.path.normpath(set_dir))
    with open(os.path.join(set_dir, "config.json")) as f:
        cfg = json.load(f)
    with open(os.path.join(set_dir, "index.csv")) as f:
        index = list(csv.DictReader(f))
    hold_steps = round(cfg.get("noise_hold", 0.3) / 0.001)
    jobs = [{"path": os.path.join(set_dir, row["file"]), "set": set_name, "episode": int(row["episode"]),
             "hold_steps": hold_steps} for row in index]
    start = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(ext_names, settings)) as pool:
        results = pool.map(scan_episode, jobs, chunksize=2)
    print(f"scanned {len(jobs)} episodes of {set_dir} in {time.time() - start:.0f} s")
    return results


def write_csv(rows, path):
    keys = []
    for row in rows:
        keys += [k for k in row if k not in keys]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["data/train", "data/dev", "data/hard"],
                        help="dataset folders; the first must be the training set")
    parser.add_argument("--calibrate", metavar="SET", help="only calibrate the analysis settings on this set (data/pilot)")
    parser.add_argument("--out", default="results/data_analysis", help="output folder")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--reuse-cache", action="store_true", help="reuse the scans saved in <out>/cache/")
    args = parser.parse_args()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

    ext_names, lower, upper = robot_info()
    LIMITS.update(lower=lower, upper=upper)
    if args.calibrate:
        results = scan_set(args.calibrate, args.workers, ext_names,
                           {"lower": lower, "upper": upper, "calibration": None})
        calibrate([res["row"] for res in results], results)
        return
    if CALIBRATION is None:
        raise SystemExit("CALIBRATION is not set: run --calibrate data/pilot first and copy the printed values")
    settings = {"lower": lower, "upper": upper, "calibration": CALIBRATION}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "sans-serif"
    os.makedirs(os.path.join(args.out, "cache"), exist_ok=True)
    sets, results = {}, {}
    for set_dir in args.sets:
        name = os.path.basename(os.path.normpath(set_dir))
        if name == "test":
            raise SystemExit("data/test is sealed until the final evaluation (report.txt, protocol v1)")
        cache = os.path.join(args.out, "cache", f"scan_{name}.pkl")
        if args.reuse_cache and os.path.exists(cache):
            with open(cache, "rb") as f:
                results[name] = pickle.load(f)
        else:
            results[name] = scan_set(set_dir, args.workers, ext_names, settings)
            with open(cache, "wb") as f:
                pickle.dump(results[name], f)
        results[name].sort(key=lambda res: res["row"]["episode"])
        sets[name] = [res["row"] for res in results[name]]
    train_name = os.path.basename(os.path.normpath(args.sets[0]))
    sets["train"] = sets.pop(train_name)
    results["train"] = results.pop(train_name)

    lines = [f"Step-2 data-level analysis ({time.strftime('%Y-%m-%d %H:%M')}), sets: {', '.join(args.sets)}",
             f"Calibration (data/pilot): {CALIBRATION}", ""]
    section_modes(lines, sets)
    section_phases(lines, sets["train"])
    section_copycat(lines, sets["train"], results["train"], CALIBRATION, args.out, plt)
    section_hidden_state(lines, sets["train"], results["train"])
    section_label_quality(lines, sets["train"], results["train"], CALIBRATION, args.out, plt)
    section_failure_model(lines, sets["train"])
    section_coverage(lines, sets, args.out, plt)
    plot_d0_theta(sets["train"], args.out, plt)
    plot_pairplot(sets["train"], args.out, plt)
    for name, rows in sets.items():
        write_csv(rows, os.path.join(args.out, f"episodes_{name}.csv"))
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
