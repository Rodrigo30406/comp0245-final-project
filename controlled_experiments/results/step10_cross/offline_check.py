"""Offline sanity check of policies/rodrigo_policy.py: feed the recorded observations of clean pilot episodes to the
adapter (its own clock from reset) and compare with the recorded clean expert labels. A wrong feature layout gives
errors of the order of the label spread; a right one gives errors well below it (his validation NRMSE is 0.13-0.14)."""
import os, sys, json, csv
import numpy as np
TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TASK); sys.path.insert(0, os.path.join(TASK, "policies"))
import rodrigo_policy as rp

def run(model_path, episodes, data_dir="data/pilot"):
    pol = rp.make_policy(model_path)
    rows = {int(r["episode"]): r for r in csv.DictReader(open(os.path.join(TASK, data_dir, "index.csv")))}
    sq = np.zeros(9); n = 0; lab = []
    for ep in episodes:
        r = rows[ep]
        assert float(r["noise_v"]) == 0.0 and float(r["noise_torque"]) == 0.0, "use clean episodes"
        d = np.load(os.path.join(TASK, data_dir, r["file"]))
        meta = json.loads(str(d["meta"]))
        pol.reset(d["target"], {"time_step": meta["time_step"]})
        T = len(d["t"])
        pred = np.empty((T, 9))
        for k in range(T):
            obs = {"base_pos": d["base_pos"][k], "base_bearing": d["base_bearing"][k], "q_mes": d["q_mes"][k],
                   "qd_mes": d["qd_mes"][k], "target": d["target"]}
            pred[k] = pol.predict9(obs)
        y = np.concatenate([d["wheel_cmds"][:, [0, 1]], d["torque_arm"]], axis=1).astype(np.float64)
        sq += ((pred - y) ** 2).sum(axis=0); n += T; lab.append(y)
        print(f"ep {ep}: T={T} t0={d['t'][0]:.3f} t1={d['t'][1]:.3f} rmse wheels {np.sqrt(((pred-y)**2)[:, :2].mean()):.3f} rad/s, torques {np.sqrt(((pred-y)**2)[:, 2:].mean()):.3f} N m; pred wheel max {np.abs(pred[:, :2]).max():.2f}")
    lab = np.concatenate(lab); rmse = np.sqrt(sq / n); std = lab.std(axis=0)
    print("per-output RMSE :", np.round(rmse, 3)); print("label std       :", np.round(std, 3))
    print("NRMSE (rmse/std):", np.round(rmse / std, 3), " mean wheels", np.round((rmse / std)[:2].mean(), 3), "arm", np.round((rmse / std)[2:].mean(), 3))

if __name__ == "__main__":
    eps = [int(e) for e in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 2, 4, 6, 8]
    run(sys.argv[1], eps)
