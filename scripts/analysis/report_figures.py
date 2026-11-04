"""Figures for the IEEE report (docs/figures/report, copied to the Overleaf project). Usage: report_figures.py [EPISODE_INDEX] [OUT_DIR].
Okabe-Ito palette, thin marks, one axis per panel."""
import json, glob, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys, os
OUT=(sys.argv[2] if len(sys.argv)>2 else 'docs/figures/report') + '/'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.size':7,'axes.titlesize':7.5,'axes.labelsize':7,'legend.fontsize':6.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':1.0,'pdf.fonttype':42})
C_EXPERT='#000000'; C_CTRL='#E69F00'; C_K144='#0072B2'; C_GRU='#009E73'; C_BAND='#56B4E9'; C_CAT='#CC79A7'
# ---- Figure: one geometry, expert vs control vs K 144 -------------------------------------------------------------
C='runs/evaluation/v2_mlp28_control_on_clean_validation_20261102'; K='runs/evaluation/v2_mlp28_trig144_on_clean_validation_20261102'
idx='000024'
def tr(d,pol): return np.load(f'{d}/traces/{pol}_episode_{idx}.npz', allow_pickle=True)
E=tr(K,'expert'); Ct=tr(C,'mlp28_control_seed0'); Kt=tr(K,'mlp28_trig144_seed0')
fig,ax=plt.subplots(1,3,figsize=(7.1,1.75))
for t,c,l in [(E,C_EXPERT,'expert'),(Ct,C_CTRL,'MLP, no device'),(Kt,C_K144,'MLP + ignition copies')]:
    ax[0].plot(t['observation'][:,0],t['observation'][:,1],color=c,label=l)
    ax[1].plot(t['time'],t['action'][:,1],color=c,label=l)   # left wheel command (rad/s)
    ax[2].plot(t['time'],t['distance'],color=c,label=l)
tgt=json.load(open(f'{K}/episodes/expert_episode_{idx}.json'))['scenario']['target']
ax[0].plot(tgt[0],tgt[1],marker='x',color='#D55E00',ms=5,ls='none',label='target'); ax[0].plot(0,0,marker='o',color='0.5',ms=3,ls='none')
ax[0].set_aspect('equal'); ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)'); ax[0].set_title('base path (40 s)')
ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('left wheel (rad/s)'); ax[1].set_title('wheel command'); ax[1].axhline(0,color='0.8',lw=0.5,zorder=0)
ax[2].set_xlabel('time (s)'); ax[2].set_ylabel('end effector to target (m)'); ax[2].set_title('reach error'); ax[2].set_yscale('log'); ax[2].set_ylim(2e-3,3); ax[2].axhline(0.01,color='0.8',lw=0.5,zorder=0); ax[2].text(39,0.0115,'1 cm',ha='right',va='bottom',color='0.4',fontsize=6)
ax[1].legend(frameon=False,loc='upper right'); fig.tight_layout(pad=0.4); fig.savefig(OUT+'trajectories.pdf'); fig.savefig(OUT+'trajectories.png',dpi=160); plt.close(fig)
print('trajectories: control final', float(Ct['distance'][-1]), 'K144 final', float(Kt['distance'][-1]), 'target', tgt)
# ---- Figure: generalisation banks and adversities, arrivals at 1 cm and 5 cm ---------------------------------------
def cnt(d,prefix,tol):
    s=json.load(open(f'runs/evaluation/{d}/summary.json')); return [v['success_by_tolerance'][tol] for k,v in s.items() if k.startswith(prefix)], s['expert']['success_by_tolerance'][tol]
groups=[('nominal','v2_mlp28_trig144_on_clean_validation_20261102','v2_gru28_patience40_on_clean_validation_20261103'),
        ('random\nstart','v2_mlp28_trig144_on_e1v2_random_validation_20261102','v2_gru28_patience40_on_e1v2_random_validation_20261103'),
        ('far\ntargets','ood_far_targets_mlp28_trig144_20261103','ood_far_targets_gru28_patience40_20261103'),
        ('far\nstarts','ood_far_start_mlp28_trig144_20261103','ood_far_start_gru28_patience40_20261103'),
        ('target\nheight','ood_target_height_mlp28_trig144_20261103','ood_target_height_gru28_patience40_20261103'),
        ('wheels\nat 95%','adv_gain_left095_mlp28_trig144_20261103','adv_gain_left095_gru28_patience40_20261103'),
        ('noise\n1/2','adv_obsnoise_half_mlp28_trig144_20261104','adv_obsnoise_half_gru28_patience40_20261104'),
        ('noise\nfull','adv_obsnoise_mlp28_trig144_20261103','adv_obsnoise_gru28_patience40_20261103')]
fig,axes=plt.subplots(1,2,figsize=(7.1,1.9),sharey=True)
for ax,tol,title in [(axes[0],'0.01','stable arrivals at 1 cm'),(axes[1],'0.05','stable arrivals at 5 cm')]:
    for i,(name,dm,dg) in enumerate(groups):
        m,ex=cnt(dm,'mlp28',tol); g,_=cnt(dg,'gru28',tol)
        ax.plot([i-0.18]*3,[100*x/205 for x in m],marker='o',ms=3,ls='none',color=C_K144,label='MLP + ignition copies' if i==0 else None)
        ax.plot([i+0.18]*3,[100*x/205 for x in g],marker='s',ms=3,ls='none',color=C_GRU,label='GRU, no device' if i==0 else None)
        ax.hlines(100*ex/205,i-0.35,i+0.35,color=C_EXPERT,lw=1.0,label='expert' if i==0 else None)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_title(title); ax.set_ylim(-3,116)
    ax.axvline(4.5,color='0.85',lw=0.6); ax.axvline(1.5,color='0.85',lw=0.6)
axes[0].set_ylabel('% of 205 geometries'); axes[1].legend(frameon=False,loc='lower left',ncol=1)
[(a.text(0.5,108,'validation',ha='center',color='0.4',fontsize=6), a.text(3,108,'new distributions',ha='center',color='0.4',fontsize=6), a.text(6,108,'adversities',ha='center',color='0.4',fontsize=6)) for a in axes]
fig.tight_layout(pad=0.4); fig.savefig(OUT+'generalisation.pdf'); fig.savefig(OUT+'generalisation.png',dpi=160); plt.close(fig)
print('done')
