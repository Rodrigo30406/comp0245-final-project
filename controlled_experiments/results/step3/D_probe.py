"""Experiment D, H-coverage offline probe (report only; STEP4_PLAN_PROMPT.md 2.4). Does a model push the arm towards the
target from rest when the end effector is 1-4 cm away?

Frames: validation episodes (results/step3/cache/val, 1 kHz), reach segment (nav_done, before the first step with
EE error < 10 mm) with EE error in [10, 40] mm; at most MAX_FRAMES, drawn with numpy default_rng(0). The base pose,
q_mes and ee_pos of each frame come from its recording.
For each frame and model: the arm qd is set to 0 (the learner's resting state), the model predicts the 7 torques;
  push = (tau_pred - g(q)) . u,   u = J^+ e / |J^+ e|,
g(q) = pinocchio's generalized gravity (arm joints, the expert's own compensation term), J = the end-effector
translational Jacobian of the arm joints (world-aligned), e = target - ee_pos. push > 0: the torque beyond gravity
compensation drives the arm along the joint direction that reduces the error. A second direction follows the expert's
IK (robotnik_arm_control.arm_controller): damped least squares on the free joints, joints within the limit margin that
the motion would push further into are frozen. Reference: the expert's recorded label at the same frame (with its real
qd), split by target z and for frames where the arm is at rest; per model also the median per-frame difference to the
expert label. Caveat (pre-freeze review): at low z the expert labels themselves give push < 0 in the plain projection
(gravity assists the downward reach; the sign mostly follows the target height), so the probe cannot tell a learner's
deficit from the expert's behaviour there; it is descriptive only.

  python results/step3/D_probe.py [model.npz ...]   (default: ego-novw-trig_s0-4 and, when they exist, ego-novw-trig-ee_s0-4)
Output: results/step3/D_probe.txt
"""
import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK)
import bc_features as bcf  # noqa: E402
import train_policy as tp  # noqa: E402

MAX_FRAMES = 20000
E_LO, E_HI = 0.010, 0.040


def frames():
    va = tp.Cache("val")
    c = va.cols
    nav, err = np.asarray(c["nav_done"], dtype=bool), np.asarray(c["ee_err"])
    sel = []
    for i in range(len(va.episodes)):
        sl = va.episode_slice(i)
        nd, er = nav[sl], err[sl]
        if not nd.any():
            continue
        k0 = int(np.argmax(nd))
        hit = np.flatnonzero(er[k0:] < 0.010)
        stop = k0 + (int(hit[0]) if hit.size else len(er) - k0)
        idx = np.arange(k0, stop)
        idx = idx[(er[idx] >= E_LO) & (er[idx] <= E_HI)]
        sel += [(i, sl.start + int(k)) for k in idx]
    rng = np.random.default_rng(0)
    pick = sorted(rng.choice(len(sel), size=min(MAX_FRAMES, len(sel)), replace=False))
    sel = [sel[j] for j in pick]
    # recorded signals of the selected frames
    rec = {k: [] for k in ("base_pos", "base_ori", "base_bearing", "q_mes", "qd_mes", "ee_pos", "target", "tau")}
    by_ep = {}
    for i, row in sel:
        by_ep.setdefault(i, []).append(row)
    split = va.split
    for i, rows in by_ep.items():
        e = va.episodes[i]
        steps = np.asarray(c["step"][np.array(rows)])
        with np.load(os.path.join(tp.TRAIN_DIR, split["episodes"][str(e)]["file"])) as d:
            for k, src in (("base_pos", "base_pos"), ("base_ori", "base_ori"), ("base_bearing", "base_bearing"),
                           ("q_mes", "q_mes"), ("qd_mes", "qd_mes"), ("ee_pos", "ee_pos"), ("tau", "torque_arm")):
                rec[k].append(d[src][steps].astype(float))
            rec["target"].append(np.broadcast_to(np.asarray(d["target"], dtype=float), (len(steps), 3)))
    va.release()
    return {k: np.concatenate(v) for k, v in rec.items()}, len(sel), len(by_ep)


def ik_direction(J, e, q_arm, lower, upper, params):
    """Unit joint direction of the expert's IK for the Cartesian error e: damped least squares on the free joints,
    joints within params.limit_margin of a limit that the motion would push further into are frozen."""
    sigma_min = np.linalg.svd(J, compute_uv=False)[-1]
    damping = np.interp(sigma_min, [0.0, params.singularity_threshold], [params.max_ik_damping, params.min_ik_damping])
    has = lower < upper
    free = np.ones(7, dtype=bool)
    qd = np.zeros(7)
    for _ in range(7):
        Jf = J[:, free]
        qd = np.zeros(7)
        qd[free] = Jf.T @ np.linalg.solve(Jf @ Jf.T + damping ** 2 * np.eye(3), e)
        blocked = (has & (q_arm <= lower + params.limit_margin) & (qd < 0)) | \
                  (has & (q_arm >= upper - params.limit_margin) & (qd > 0))
        blocked &= free
        if not blocked.any():
            break
        free &= ~blocked
    return qd / max(np.linalg.norm(qd), 1e-12)


def projections(rec):
    import pinocchio as pin
    from evaluate_policy import setup_simulation
    from robotnik_arm_control import ArmControllerParams
    from robotnik_arm_task import ARM_REACH_LIMIT
    with redirect_stdout(io.StringIO()):
        _, client, _, info = setup_simulation(np.zeros(2), 0.0)
    client.disconnect()
    m, fid = info["dyn_model"], info["ee_frame_id"]
    lower, upper = np.asarray(info["joint_lower"])[4:11], np.asarray(info["joint_upper"])[4:11]
    params = ArmControllerParams(reach_limit=ARM_REACH_LIMIT)
    n = len(rec["q_mes"])
    g, u, u_ik = np.empty((n, 7)), np.empty((n, 7)), np.empty((n, 7))
    for j in range(n):
        q_pin = m.ReoderJoints2PinVec(np.concatenate((rec["base_pos"][j], rec["base_ori"][j], rec["q_mes"][j])), "pos")
        pin.forwardKinematics(m.pin_model, m.pin_data, q_pin)
        pin.updateFramePlacements(m.pin_model, m.pin_data)
        pin.computeJointJacobians(m.pin_model, m.pin_data, q_pin)
        J = pin.computeFrameJacobian(m.pin_model, m.pin_data, q_pin, fid, pin.LOCAL_WORLD_ALIGNED)[:3, 6:13]
        g[j] = pin.computeGeneralizedGravity(m.pin_model, m.pin_data, q_pin)[6:13]
        e = rec["target"][j] - rec["ee_pos"][j]
        dq = np.linalg.pinv(J) @ e
        u[j] = dq / max(np.linalg.norm(dq), 1e-12)
        u_ik[j] = ik_direction(J, e, rec["q_mes"][j][4:11], lower, upper, params)
    return g, u, u_ik


def main(paths):
    rec, n, n_ep = frames()
    g, u, u_ik = projections(rec)
    z_low = rec["target"][:, 2] < 0.45
    rest = np.abs(rec["qd_mes"][:, 4:11]).max(axis=1) < 0.05
    masks = (("all", np.ones(len(z_low), dtype=bool)), ("z >= 0.45", ~z_low), ("z < 0.45", z_low))
    lines = [f"D probe (H-coverage, report only): {n} validation reach frames with EE error in [{1000 * E_LO:.0f}, "
             f"{1000 * E_HI:.0f}] mm from {n_ep} episodes (low z {int(z_low.sum())}, arm at rest {int(rest.sum())}); "
             f"arm qd set to 0 for the models; push = (tau - g) . u; u = unit J^+ e (pinv) or the expert's IK direction "
             f"(ik). Share of frames with push > 0 (median push, N m). Descriptive: at low z the expert labels also push < 0."]
    lab = {k: np.einsum("ij,ij->i", rec["tau"] - g, uu) for k, uu in (("pinv", u), ("ik", u_ik))}

    def summary(push):
        return "; ".join(f"{name} {100 * np.mean(push[mk] > 0):5.1f}% ({np.median(push[mk]):+.2f})" for name, mk in masks)
    for k in ("pinv", "ik"):
        lines.append(f"  expert label (real qd), {k:4s}: " + summary(lab[k]))
        lines.append(f"  expert label at rest (max|qd_arm| < 0.05), {k}: " + "; ".join(
            f"{name} {100 * np.mean(lab[k][mk & rest] > 0):5.1f}% (n {int((mk & rest).sum())})" for name, mk in masks))
    qd0 = rec["qd_mes"].copy()
    qd0[:, bcf.ARM] = 0.0
    for path in paths:
        model = bcf.load_model(path)
        sig = bcf.measured_signals(rec["base_pos"], rec["base_bearing"], rec["q_mes"], qd0, rec["target"],
                                   ee_pos=rec["ee_pos"])
        tau = bcf.forward(model, bcf.build_features(model["feature_set"], sig))[:, 2:]
        for k, uu in (("pinv", u), ("ik", u_ik)):
            push = np.einsum("ij,ij->i", tau - g, uu)
            diff = push - lab[k]
            lines.append(f"  {os.path.basename(path)[:-4]:24s} {k:4s}: " + summary(push) + "; minus the expert label "
                         "(median per frame): " + ", ".join(f"{name} {np.median(diff[mk]):+.2f}" for name, mk in masks))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(TASK, "results", "step3", "D_probe.txt"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [os.path.join(TASK, "trained", f"ego-novw-trig_s{s}.npz") for s in range(5)]
        args += [p for p in (os.path.join(TASK, "trained", f"ego-novw-trig-ee_s{s}.npz") for s in range(5))
                 if os.path.exists(p)]
    main([os.path.abspath(a) for a in args])
