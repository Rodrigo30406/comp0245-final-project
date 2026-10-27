"""Export trajectory comparisons in world metres, with separate early-time views."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS={'expert':'#2463a5','linear':'#df811c','mlp':'#2a9157'}
COLORS.update({f'{kind}{n}': color for kind in ('linear', 'mlp')
               for n, color in zip((64, 128, 256, 512),
                                   ('#d97706', '#059669', '#9333ea', '#dc2626'))})


def _save(fig,output,name,pdf=False):
    fig.savefig(output/f'{name}.png',dpi=150)
    if pdf:fig.savefig(output/f'{name}.pdf')
    plt.close(fig)


def plot_case(output,eid,items,curves,zoom_s):
    labels=[label for label in items if label!='expert']
    reference=items['expert'][1];t=reference['time'];zoom=t<=zoom_s+1e-10
    fig,axes=plt.subplots(2,len(labels),figsize=(6*len(labels),8),squeeze=False,layout='constrained')
    for j,label in enumerate(labels):
        record,trace=items[label];color=COLORS.get(label,'#9955aa')
        goal=np.array(record['subsystems']['base_navigation']['goal_xy_m'])
        for row,mask,title in [(0,zoom,f'First {zoom_s:g} s'),(1,np.ones(len(t),dtype=bool),'Full horizon')]:
            ax=axes[row,j]
            for name,tr,col in [('Expert',reference,COLORS['expert']),(label,trace,color)]:
                xy=tr['observation'][mask,:2]
                ax.plot(xy[:,0],xy[:,1],color=col,label=name)
                # Fixed-length heading arrows, independent of runaway path scale.
                idx=np.flatnonzero(mask);chosen=idx[np.unique(np.linspace(0,len(idx)-1,6,dtype=int))]
                yaw=tr['observation'][chosen,3]
                ax.quiver(tr['observation'][chosen,0],tr['observation'][chosen,1],
                          np.cos(yaw),np.sin(yaw),color=col,angles='xy',
                          scale_units='inches',scale=7,width=.004,alpha=.7)
                ax.scatter(xy[-1,0],xy[-1,1],marker='x',color=col,s=35)
            ax.scatter(*reference['observation'][0,:2],marker='o',facecolors='none',edgecolors='black',label='Start')
            if row==1:ax.scatter(*goal,marker='*',color='black',s=70,label='Navigation goal')
            ax.set(title=f'{label}: {title}',xlabel='World X (m)',ylabel='World Y (m)')
            ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{eid}: base trajectory, expert versus each model\nLinear spatial axes with equal scale; X marks the end; arrows show base heading')
    _save(fig,output,f'base_trajectories_{eid}',pdf=True)

    fig,axes=plt.subplots(4,2,figsize=(12,12),layout='constrained')
    for j,mask in enumerate([np.ones(len(t),dtype=bool),zoom]):
        for label in labels:
            c=curves[label];color=COLORS.get(label,'#9955aa')
            for row,key in [(0,'base_xy'),(1,'base_yaw_deg'),(2,'ee_world')]:
                axes[row,j].plot(t[mask],np.maximum(c[key][mask],1e-9),label=label,color=color)
        for label,(record,trace) in items.items():
            axes[3,j].plot(t[mask],np.maximum(trace['distance'][mask],1e-9),label=label,color=COLORS.get(label))
        for row,(title,unit) in enumerate([('Base separation from expert','m'),('Yaw difference from expert','degrees'),
                                          ('EE trajectory separation from expert','m'),('EE error to original target','m')]):
            ax=axes[row,j];ax.set(yscale='log',ylabel=unit,xlabel='Simulation time (s)',title=title+(' — full horizon' if j==0 else ' — start'))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[0,j].axhline(.001,color='gray',ls=':',alpha=.5)
        axes[0,j].axhline(.01,color='gray',ls='--',alpha=.5)
        axes[3,j].axhline(.01,color='gray',ls='--',alpha=.5)
    fig.suptitle(f'{eid}: trajectory divergence and task outcome\nLog curves; values below 1e-9 are drawn at 1e-9 for display only')
    _save(fig,output,f'trajectory_differences_{eid}',pdf=True)

    plot_orientation_and_saturation(output,eid,items,zoom_s)

    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for col,end in enumerate([zoom_s,min(.25,zoom_s)]):
        for label,(record,trace) in items.items():
            times=trace.get('action_time',trace['time']-record['dt'])
            mask=trace.get('has_action',trace['step']>0)&(times<=end+1e-10)
            for row,index in enumerate([0,1]):
                axes[row,col].plot(times[mask],trace['action'][mask,index],label=label,color=COLORS.get(label))
        for row,side in enumerate(['Right','Left']):
            ax=axes[row,col];ax.set(xlabel='PRE-step decision time (s)',ylabel='Applied command (rad/s)',title=f'{side.lower()} wheel, first {end:g} s')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle(f'{eid}: wheel commands on each rollout\'s own states\nNot supervised error on shared states, except initially; clipping included')
    _save(fig,output,f'initial_commands_{eid}')


def plot_overview(output,grouped,curves_by_episode,summary,zoom_s):
    labels=[label for label in summary if label!='expert']
    first=next(iter(grouped.values()));t=first['expert'][1]['time'];mask=t<=zoom_s+1e-10
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for label in labels:
        y=np.stack([curves_by_episode[eid][label]['base_xy'] for eid in grouped])
        median=np.maximum(np.median(y,axis=0),1e-9);color=COLORS.get(label)
        axes[0,0].plot(t,median,label=label,color=color)
        axes[0,0].fill_between(t,np.maximum(np.percentile(y,10,axis=0),1e-9),np.maximum(np.percentile(y,90,axis=0),1e-9),alpha=.15,color=color)
        axes[0,1].plot(t[mask],median[mask],label=label,color=color)
    for label in summary:
        y=np.stack([grouped[eid][label][1]['distance'] for eid in grouped])
        axes[1,0].plot(t,np.maximum(np.median(y,axis=0),1e-9),label=label,color=COLORS.get(label))
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set(yscale='log',xlabel='Simulation time (s)',ylabel='m');ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0,0].set_title('XY separation from expert: median and P10–P90')
    axes[0,1].set_title(f'XY separation: first {zoom_s:g} s')
    axes[1,0].set_title('EE target error: median at each time')
    names=list(summary);x=np.arange(len(names))
    for offset,key,title,color in [(-.18,'base_stable_arrivals','Base','#467fb5'),(.18,'system_stable_arrivals_1cm','System: 1 cm','#d18b35')]:
        vals=[summary[name][key] for name in names]
        axes[1,1].bar(x+offset,vals,width=.36,label=title,color=color)
        for pos,value in zip(x+offset,vals):axes[1,1].text(pos,value+.15,str(value),ha='center',fontsize=8)
    n=summary[names[0]]['episodes'];axes[1,1].set_xticks(x,names)
    axes[1,1].set(ylim=(0,n+2),ylabel=f'Episodes out of {n}',title='Arrivals with a 0.5 s dwell')
    axes[1,1].legend(fontsize=8);axes[1,1].grid(axis='y',alpha=.2)
    fig.suptitle('Nominal diagnosis: base and whole system, without an isolated arm score\nSame scenarios and frozen models; extreme excursions indicate simulated divergence')
    _save(fig,output,'trajectory_overview',pdf=True)


def plot_orientation_and_saturation(output,eid,items,zoom_s):
    from .subsystems import wrapped_angle
    reference=items['expert'][1];t=reference['time']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for col,end in enumerate([t[-1],zoom_s]):
        mask=t<=end+1e-10
        for label,(record,trace) in items.items():
            # Unwrap on the entire trace before slicing; preserve accumulated turns.
            yaw=np.rad2deg(np.unwrap(trace['observation'][:,3]))
            axes[0,col].plot(t[mask],yaw[mask],label=label,color=COLORS.get(label))
            if label!='expert':
                delta=np.rad2deg(wrapped_angle(trace['observation'][:,3]-reference['observation'][:,3]))
                axes[1,col].plot(t[mask],delta[mask],label=label,color=COLORS.get(label))
        for row,title in enumerate(['Yaw heading (unwrapped to show turns)','Signed angular difference from expert [-180, 180]']):
            axes[row,col].set(title=title+(' — full horizon' if col==0 else ' — start'),
                              xlabel='Simulation time (s)',ylabel='degrees')
            axes[row,col].grid(alpha=.2);axes[row,col].legend(fontsize=8)
        axes[1,col].axhline(0,color='gray',lw=.7)
    fig.suptitle(f'{eid}: base heading; positive = counterclockwise viewed from above\nYaw unwrapped from samples may miss turns during rapid divergence. Angular difference wraps at ±180°.')
    _save(fig,output,f'orientation_{eid}',pdf=True)

    fig,axes=plt.subplots(3,2,figsize=(12,10),layout='constrained')
    for col,end in enumerate([t[-1],zoom_s]):
        for label,(record,trace) in items.items():
            if 'limits' not in record:continue
            times=trace.get('action_time',trace['time']-record['dt'])
            valid=trace.get('has_action',trace['step']>0)&(times<=end+1e-10)
            times=times[valid];raw=trace['raw_action'][valid];applied=trace['action'][valid]
            limits=np.asarray(record['limits']);color=COLORS.get(label)
            for row,sl in enumerate([slice(0,4),slice(4,11)]):
                ratio=np.max(np.abs(raw[:,sl])/limits[sl],axis=1)
                axes[row,col].plot(times,ratio,label=label,color=color)
                clipped=(np.abs(raw[:,sl]-applied[:,sl])>1e-9).any(axis=1)
                hit=np.flatnonzero(clipped)
                if len(hit):axes[row,col].scatter(times[hit[0]],ratio[hit[0]],marker='x',color=color,s=45)
            # Show expert memory only as annotation, never as learned-model input.
            if label=='expert' and 'expert_phase' in trace:
                phase=trace['expert_phase'][valid]
                names=['align','drive','orient','done']
                codes=np.array([names.index(v) if v in names else np.nan for v in phase])
                if 'expert_navigation_done' in trace:
                    codes[trace['expert_navigation_done'][valid]]=3
                axes[2,col].step(times,codes,where='post',color=color,label='Expert phase')
                axes[2,col].set_yticks(range(4),names)
        for row,title in enumerate(['Maximum |raw command| / limit: wheels','Maximum |raw command| / limit: torques','Expert navigation (done overrides phase)']):
            ax=axes[row,col];ax.set(title=title+(' — full horizon' if col==0 else ' — start'),xlabel='PRE-step decision time (s)')
            ax.grid(alpha=.2)
            if row<2:
                ax.axhline(1,color='black',ls='--',lw=.8);ax.set_yscale('symlog',linthresh=1);ax.set_ylim(bottom=0);ax.legend(fontsize=8)
    fig.suptitle(f'{eid}: saturation and reference phases\nRatio >1 requires clipping; X = first recorded clip in the interval. Sampling follows the saved trace; sparse samples may miss transients.')
    _save(fig,output,f'saturation_{eid}',pdf=True)
