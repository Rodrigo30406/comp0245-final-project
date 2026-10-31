"""Evidence for the next-step discussion (descriptive, after the step-3 results): DEV/HARD runs + validation cache."""
import csv, sys, os
import numpy as np
sys.path.insert(0, os.getcwd())
import bc_features as bf
from robotnik_arm_task import STANDOFF_Z_POINTS, STANDOFF_POINTS

def rows(p): return list(csv.DictReader(open(p)))
f = lambda r, k: float(r[k]) if r[k] not in ("", "nan") else np.nan
ref = {int(r["episode"]): r for r in rows("results/data_analysis/episodes_dev.csv")}

print("== 1) Where do ego-novw's reach_5cm come from? (DEV, 5 seeds) ==")
hits = []
for s in range(5):
    for r in rows(f"results/ego-novw_s{s}_dev/rollouts.csv"):
        if f(r, "min_error") < 0.05:
            hits.append((float(ref[int(r["episode"])]["navdone_ee_err"]), np.isfinite(f(r, "arm_extend_time")), f(r, "min_error")))
h = np.array(hits)
print(f"  {len(h)} seed x episode with min_error < 5 cm; expert hold-posture error at parking (navdone_ee_err): median {np.median(h[:,0])*100:.1f} cm, "
      f"< 7 cm in {np.mean(h[:,0] < 0.07)*100:.0f}%; arm extended in {int(h[:,1].sum())}")
print(f"  DEV episodes whose expert hold posture is already within 5 cm / 7 cm of the target: "
      f"{sum(float(r['navdone_ee_err']) < 0.05 for r in ref.values())} / {sum(float(r['navdone_ee_err']) < 0.07 for r in ref.values())} of 300")

print("== 2) ego-novw failures: is the base parked and still? (DEV, seed x episode) ==")
cls = {}
for s in range(5):
    c = {int(r["episode"]): r["fclass"] for r in rows(f"results/step3/classes_ego-novw_s{s}_dev.csv")}
    for r in rows(f"results/ego-novw_s{s}_dev/rollouts.csv"):
        k = c[int(r["episode"])]
        d = cls.setdefault(k, {"n": 0, "parked": 0, "ring": [], "turn": [], "path": []})
        d["n"] += 1; d["parked"] += np.isfinite(f(r, "parked_time")); d["ring"].append(f(r, "ring_dist_final"))
        d["turn"].append(f(r, "base_turn_total_deg")); d["path"].append(f(r, "base_path_length"))
for k, d in sorted(cls.items(), key=lambda kv: -kv[1]["n"]):
    print(f"  {k:22s} n {d['n']:4d}  parked (still 1 s on the ring) {100*d['parked']/d['n']:5.1f}%  ring_final median {100*np.median(d['ring']):5.1f} cm"
          f"  turn median {np.median(d['turn']):6.0f} deg  path median {np.median(d['path']):.2f} m")

print("== 3) Why the arm never triggers: labels near the trigger in the validation data (1 kHz, clean + noisy) ==")
C = "results/step3/cache/val/"
ep, step = np.load(C + "episode.npy"), np.load(C + "step.npy")
nav, phase = np.load(C + "nav_done.npy"), np.load(C + "phase.npy")
q = np.load(C + "q_arm.npy").astype(float); tau = np.load(C + "label_tau.npy").astype(float)
xy, yaw, tgt = np.load(C + "base_xy.npy").astype(float), np.load(C + "base_bearing.npy").astype(float), np.load(C + "target.npy")
dx, dy = bf.ego_target(xy, yaw, tgt[:, :2]); beta = np.arctan2(dy, dx)
ring = np.abs(np.hypot(dx, dy) - np.interp(tgt[:, 2], STANDOFF_Z_POINTS, STANDOFF_POINTS))
q0 = np.zeros_like(q)
for e in np.unique(ep):
    m = ep == e; q0[m] = q[m][0]
qdisp = np.abs(q - q0).max(axis=1)
# "parked-looking" frames: on the ring, facing the target (|beta| < 0.1 rad), arm still at its hold posture (< 0.06 rad)
look = (ring < 0.2) & (np.abs(beta) < 0.10) & (qdisp < 0.06)
print(f"  frames that look parked with the arm at hold: {look.sum()} ; of them labelled reach (nav_done) {100*nav[look].mean():.1f}%")
for tol in (0.05, 0.02):
    lk = (ring < 0.2) & (np.abs(beta) < tol) & (qdisp < 0.06)
    print(f"    with |beta| < {tol} rad: {lk.sum()} frames, reach-labelled {100*nav[lk].mean():.1f}%")
# time the arm stays near hold after nav_done, per episode
dt_hold = []
for e in np.unique(ep):
    m = ep == e
    k = np.flatnonzero(nav[m])
    if k.size == 0: continue
    after = qdisp[m][k[0]:]
    j = np.flatnonzero(after >= 0.06)
    dt_hold.append((j[0] if j.size else after.size) / 1000.0)
orient_dur = []
for e in np.unique(ep):
    m = ep == e
    orient_dur.append(np.sum((phase[m] == 2) & (~nav[m].astype(bool))) / 1000.0)
print(f"  after nav_done the arm stays within 0.06 rad of hold for median {np.median(dt_hold)*1000:.0f} ms (p90 {np.percentile(dt_hold,90)*1000:.0f} ms); "
      f"the orient phase before it lasts median {np.median(orient_dur)*1000:.0f} ms")

print("== 4) Does base stillness (measured wheel speeds) remove the ambiguity? ==")
vm, wm = bf.wheels_to_vw(np.load(C + "qd_wheels.npy").astype(float))
still = (np.abs(vm) < 0.01) & (np.abs(wm) < 0.02)
for name, m in (("parked-looking (ring < 0.2, |beta| < 0.1, arm at hold)", look),
                ("  ... and base still (|v| < 0.01, |w| < 0.02)", look & still),
                ("  ... and base moving", look & ~still)):
    print(f"  {name}: {m.sum()} frames, reach-labelled {100*nav[m].mean():.1f}%")
noisy = np.load(C + "noisy.npy").astype(bool)
m = look & still
print(f"  still + parked-looking frames that are NOT reach: phase codes {np.unique(phase[m & ~nav.astype(bool)], return_counts=True)}")
# where do the non-reach 'still + parked-looking' frames come from? time to nav_done
ttn = []
for e in np.unique(ep[m & ~nav.astype(bool)]):
    me = ep == e
    k = np.flatnonzero(nav[me])
    idx = np.flatnonzero((m & ~nav.astype(bool))[me])
    if k.size: ttn += list((k[0] - idx) / 1000.0)
ttn = np.array(ttn)
print(f"  ... their time before nav_done: median {np.median(ttn)*1000:.0f} ms, p90 {np.percentile(ttn,90)*1000:.0f} ms")
