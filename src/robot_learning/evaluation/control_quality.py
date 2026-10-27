"""Post-hoc route metrics and robust episode summaries; no simulator changes."""
import numpy as np


def discrete_frechet(first, second):
    """Exact Eiter-Mannila recurrence on the supplied discrete vertices."""
    p=np.asarray(first,dtype=float);q=np.asarray(second,dtype=float)
    if p.ndim!=2 or q.ndim!=2 or p.shape[1]!=q.shape[1] or not len(p) or not len(q):
        raise ValueError('Expected two nonempty curves with matching dimensions')
    if not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError('Nonfinite curve')
    previous=np.full(len(q),np.inf)
    for i,point in enumerate(p):
        distances=np.linalg.norm(q-point,axis=1);current=np.empty(len(q))
        for j,distance in enumerate(distances):
            if i==0 and j==0:current[j]=distance
            elif i==0:current[j]=max(current[j-1],distance)
            elif j==0:current[j]=max(previous[j],distance)
            else:current[j]=max(min(previous[j],previous[j-1],current[j-1]),distance)
        previous=current
    return float(previous[-1])


def robust_summary(values):
    """Each episode has equal weight; retain extremes beside robust location."""
    supplied=list(values);a=np.array([v for v in supplied if v is not None and np.isfinite(v)],dtype=float)
    result={'count':len(supplied),'finite':len(a),'missing_or_nonfinite':len(supplied)-len(a)}
    if not len(a):return result
    q=np.percentile(a,[25,50,75,90,95]);total=float(a.sum());top=min(2,len(a))
    result.update(mean=float(a.mean()),q25=float(q[0]),median=float(q[1]),q75=float(q[2]),
        iqr=float(q[2]-q[0]),p90=float(q[3]),p95=float(q[4]),maximum=float(a.max()),
        minimum=float(a.min()),top_two_episodes_share_of_sum=float(np.sort(a)[-top:].sum()/total) if total>0 and np.all(a>=0) else None)
    return result
