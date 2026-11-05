"""The expert's 129 failures in the nominal DART collection (dart_stratified1024_low), re-run with the same DART noise
realisation and the navigation timeout raised from 25 s to 38 s: does more time let it park and arrive?"""
import sys, json, glob, numpy as np
sys.path.insert(0, 'src')
from concurrent.futures import ProcessPoolExecutor
from robot_learning.simulation.episode import Episode
from robot_learning.settings import get_roboenv_root
from robot_learning.data.worker import make_perturbation
N = "datasets/dart_stratified1024_low/20261027T060538Z_dart_stratified1024_wheel0_5_torque1_0_hold50_97ab75da"
CFG = json.load(open(f"{N}/config.json"))
def run(args):
    e, timeout = args
    cfg = json.loads(json.dumps(CFG)); cfg["base_controller"]["nav_timeout"] = timeout
    np.random.seed(e["seed"])
    with Episode(e["target"], cfg, get_roboenv_root(), e["initial_base"], action_filter=make_perturbation(cfg["action_noise"], e["seed"])) as ep:
        d = []; nav_t = None
        for tr in ep.transitions():
            d.append(tr["next"]["distance"])
            if nav_t is None and tr["navigation_done"]: nav_t = tr["next"]["time"]
        stop = ep.first_lecturer_stop or {}; d = np.array(d)
        inside = d <= 0.01; run_len = 0; best = 0
        for v in inside: run_len = run_len + 1 if v else 0; best = max(best, run_len)
    return (e["episode_id"], timeout, e["navigation_timeout"], round(float(e["minimum_distance_m"]), 3), -1 if nav_t is None else round(nav_t, 1), round(float(d[-1]), 3), round(float(d.min()), 3), best >= 500, round(stop.get("time_s", -1), 1), stop.get("message", "")[:30])
if __name__ == "__main__":
    fails = [json.load(open(f)) for f in sorted(glob.glob(f"{N}/episodes/*.json"))]
    fails = [e for e in fails if not e["success"]]
    jobs = [(e, 25.0) for e in fails[:3]] + [(e, 38.0) for e in fails]
    with ProcessPoolExecutor(6) as ex:
        for r in ex.map(run, jobs):
            print("ep %s timeout %.0f s (was timeout %s, min %.3f): nav_done at %.1f s  final %.3f min %.3f stable1cm %s stop %.1f s '%s'" % r, flush=True)
    print("done")
