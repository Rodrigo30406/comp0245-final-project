"""Why does the MLP's arm stay at home when the dead band stops the base? Offline test on the evaluation traces:
(a) the policy's arm output on the EXPERT's rows around the done instant (open loop), as a fraction on the hold->reach axis;
(b) the same on its own closed-loop stop rows; (c) which inputs differ between the two rows; (d) swap experiments.

Usage (pixi env, repository root): python scripts/analysis/arm_switch_probe.py EVAL_DIR TRAINING_RUN POLICY_LABEL
"""
import sys; sys.path.insert(0, "scripts/analysis"); sys.path.insert(0, "src")
from pathlib import Path; import numpy as np
from openloop_phase_error import load_policy
wrap=lambda a:(a+np.pi)%(2*np.pi)-np.pi
def spells(zero,t):
    out=[];i=0
    while i<len(t):
        if zero[i]:
            j=i
            while j+1<len(t) and zero[j+1]: j+=1
            out.append((i,j)); i=j+1
        else: i+=1
    return out
ev=Path(sys.argv[1]); run=Path(sys.argv[2]); label=sys.argv[3]; pol=load_policy(run)
names=["x","y","z","bearing"]+[f"q{i}" for i in range(1,8)]+[f"qd{i}" for i in range(1,8)]+["tx","ty","tz"]
res=[]; deltas=[]
for p in sorted((ev/"traces").glob(f"{label}_episode_*.npz"))[:80]:
    ep=p.stem.rsplit("_",1)[1]; e=np.load(ev/"traces"/f"expert_episode_{ep}.npz"); z=np.load(p)
    k=np.flatnonzero(e["expert_navigation_done"])
    if not len(k): continue
    k0=k[0]; eo=e["observation"]; et=e["time"]
    hold=e["raw_action"][:k0,4:11].mean(axis=0); reach=e["raw_action"][k0:k0+10,4:11].mean(axis=0); axis=reach-hold
    frac=lambda a: float(np.dot(a-hold,axis)/np.dot(axis,axis))
    pol.reset(); pe=np.asarray(pol.predict(eo, et))           # open loop on the expert's rows
    f_before=np.median([frac(pe[i,2:9]) for i in range(max(0,k0-5),k0)]); f_after=np.median([frac(pe[i,2:9]) for i in range(k0,min(len(pe),k0+10))]); f_later=np.median([frac(pe[i,2:9]) for i in range(min(len(pe)-1,k0+10),min(len(pe),k0+30))])
    t=z["time"]; obs=z["observation"]; act=z["action"]
    sp=[s for s in spells(np.abs(act[:,:4]).max(axis=1)==0,t) if t[s[1]]-t[s[0]]>=0.5]
    if not sp: continue
    a,b=sp[0]
    pol.reset(); pz=np.asarray(pol.predict(obs[:b+1], t[:b+1])); f_stop=np.median([frac(pz[i,2:9]) for i in range(a,b+1)])
    # swap: expert done row with the policy's base pose / clock
    row=eo[k0].copy(); tt=et[k0]
    def one(r, time): pol.reset(); return frac(np.asarray(pol.predict(r[None], np.array([time])))[0,2:9])
    f_done=one(row,tt)
    r1=row.copy(); r1[:4]=obs[a,:4]; f_pose=one(r1,tt)              # base pose from the policy's stop
    r2=row.copy(); f_clock=one(r2,t[a])                              # policy's clock
    r3=row.copy(); r3[4:18]=obs[a,4:18]; f_arm=one(r3,tt)            # arm state from the policy's stop
    r4=obs[a].copy(); r4[:4]=row[:4]; f_stop_expose=one(r4,t[a])     # policy stop row with the expert's base pose
    res.append((f_before,f_after,f_later,f_stop,f_done,f_pose,f_clock,f_arm,f_stop_expose, et[k0], t[a]))
    deltas.append(np.abs(obs[a]-row))
r=np.array(res); m=np.median(r,axis=0)
print(f"{label}: n={len(r)}  arm output on hold->reach axis (0=hold, 1=reach), medians")
print(f"  open loop on expert rows: 0.5 s before done {m[0]:+.2f} | first 1 s after done {m[1]:+.2f} | 1-3 s after {m[2]:+.2f}   (expert done at {m[9]:.1f} s)")
print(f"  closed loop, own stop rows under the band: {m[3]:+.2f}   (stop at {m[10]:.1f} s)")
print(f"  single expert done row: {m[4]:+.2f};  same row with the policy's base pose: {m[5]:+.2f};  with the policy's clock: {m[6]:+.2f};  with the policy's arm state: {m[7]:+.2f}")
print(f"  policy's own stop row but with the expert's base pose: {m[8]:+.2f}")
d=np.median(np.array(deltas),axis=0)
print("  median |input difference| stop row vs expert done row:", {n:round(float(v),3) for n,v in zip(names,d) if v>1e-3})
