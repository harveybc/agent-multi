"""Fixed-checkpoint action-switch accounting; never tunes a decision rule."""
import numpy as np

TOL=1e-8


def _values(value):
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or a.shape[1]!=3 or not len(a) or not np.isfinite(a).all():
        raise ValueError('DECISION_VALUES')
    return a


def distinct_gaps(target):
    y=_values(target);out=[]
    for r in y:
        lower=r[r<r.max()-TOL]
        out.append(float(r.max()-lower.max()) if len(lower) else 0.)
    return np.array(out)


def fit_gap_bins(fit_target):
    gaps=distinct_gaps(fit_target);positive=gaps[gaps>TOL]
    return np.quantile(positive,[1/3,2/3]).tolist() if len(positive) else []


def gap_labels(target,bins):
    if len(bins) not in (0,2) or (bins and (not np.isfinite(bins).all() or bins[0]>bins[1])):
        raise ValueError('DECISION_BINS')
    return ['all_tied' if g<=TOL else 'unbinned' if not bins else
            ('low','mid','high')[int(np.searchsorted(bins,g,side='left'))] for g in distinct_gaps(target)]


def decision_ledger(control,candidate,target):
    old,new,y=map(_values,(control,candidate,target))
    if old.shape!=new.shape or old.shape!=y.shape:raise ValueError('DECISION_SHAPE')
    rows=[]
    for i,(a,b,t) in enumerate(zip(old,new,y)):
        aa,bb=int(a.argmax()),int(b.argmax());best=float(t.max())
        ro,rn=float(best-t[aa]),float(best-t[bb]);delta=float(t[aa]-t[bb])
        r=dict(index=i,old_action=aa,new_action=bb,switched=aa!=bb,
               old_regret=ro,new_regret=rn,regret_delta=delta,
               old_optimal=ro<=TOL,new_optimal=rn<=TOL,
               target_optimal_actions=np.flatnonzero(best-t<=TOL).tolist(),
               true_selected_gap=float(t[bb]-t[aa]),
               old_margin=float(np.sort(a)[-1]-np.sort(a)[-2]),
               new_margin=float(np.sort(b)[-1]-np.sort(b)[-2]),
               harmful=delta>TOL,helpful=delta<-TOL,
               old_values=a.tolist(),new_values=b.tolist(),target=t.tolist())
        for label,q in (('old',a),('new',b)):
            error=q-t;level=float(error.mean())
            r[label+'_mse']=float(np.mean(error**2))
            r[label+'_level_mse']=level**2
            r[label+'_centered_mse']=float(np.mean((error-level)**2))
            if not np.isclose(r[label+'_mse'],r[label+'_level_mse']+r[label+'_centered_mse'],rtol=1e-12,atol=1e-12):
                raise ValueError('DECISION_MSE_IDENTITY')
        if not np.isclose(delta,rn-ro,atol=1e-12,rtol=1e-12):raise ValueError('DECISION_REGRET_IDENTITY')
        rows.append(r)
    return rows


def summarize(rows):
    if not rows:return None
    n=len(rows);average=lambda k:float(np.mean([r[k] for r in rows]))
    counts=lambda k:int(sum(bool(r[k]) for r in rows))
    delta=average('regret_delta')
    if not np.isclose(delta,average('new_regret')-average('old_regret'),atol=1e-12,rtol=1e-12):
        raise ValueError('DECISION_SUM_IDENTITY')
    return dict(n=n,switches=counts('switched'),harmful=counts('harmful'),helpful=counts('helpful'),
        neutral_switches=sum(r['switched'] and not r['harmful'] and not r['helpful'] for r in rows),
        old_regret=average('old_regret'),new_regret=average('new_regret'),regret_delta=delta,
        harmful_regret_sum=float(sum(r['regret_delta'] for r in rows if r['harmful'])),
        helpful_regret_sum=float(sum(r['regret_delta'] for r in rows if r['helpful'])),
        lower_mse_but_harmful=sum(r['harmful'] and r['new_mse']<r['old_mse'] for r in rows),
        wrong_to_right=sum(not r['old_optimal'] and r['new_optimal'] for r in rows),
        right_to_wrong=sum(r['old_optimal'] and not r['new_optimal'] for r in rows),
        **{label+'_'+key:average(label+'_'+key) for label in ('old','new') for key in ('mse','level_mse','centered_mse')})


def layer_delta(h0,h1,w0,w1,b0,b1):
    h0,h1,w0,w1,b0,b1=[np.asarray(a,dtype=float) for a in (h0,h1,w0,w1,b0,b1)]
    if (h0.ndim!=3 or w0.ndim!=3 or b0.ndim!=2 or h0.shape!=h1.shape or w0.shape!=w1.shape
            or b0.shape!=b1.shape or h0.shape[0]!=w0.shape[0] or w0.shape[:2]!=b0.shape
            or h0.shape[-1]!=w0.shape[-1] or not all(np.isfinite(a).all() for a in (h0,h1,w0,w1,b0,b1))):
        raise ValueError('DECISION_LAYER_SHAPE')
    hidden=np.einsum('tni,tai->tna',h1-h0,w0)
    output=np.einsum('tni,tai->tna',h1,w1-w0)+(b1-b0)[:,None]
    return hidden,output
