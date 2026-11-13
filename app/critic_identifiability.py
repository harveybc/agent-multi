"""FIT-only baselines and finite same-market state ambiguity measures."""
import numpy as np

from app.critic_representation_control import state_kind
from app.decision_attribution import fork_env, state_view, continuation


def baseline_predictions(windows, records, fit, held):
    fit,held=np.asarray(fit,int),np.asarray(held,int)
    x=np.asarray(windows,float).reshape(len(records),-1)
    if (not len(fit) or not len(held) or set(fit)&set(held) or not np.isfinite(x).all()
            or any(records[i]['role']!='fit' or records[i]['year']>=2023 for i in fit)):
        raise ValueError('IDENTIFIABILITY_FIT')
    y=np.array([[o['reward_return'] for o in records[i]['outcomes']] for i in fit])
    if y.shape!=(len(fit),3) or not np.isfinite(y).all():raise ValueError('IDENTIFIABILITY_LABEL')
    distance=np.sqrt(np.mean((x[held,None,:]-x[fit][None,:,:])**2,axis=-1))
    order=np.argsort(distance,axis=1,kind='stable')
    out={'fit_mean':np.tile(y.mean(0),(len(held),1))}
    for k in (1,5):out[f'nearest_{k}']=y[order[:,:min(k,len(fit))]].mean(1)
    grouped=np.full((len(held),3),np.nan)
    for j,i in enumerate(held):
        same=np.array([n for n,f in enumerate(fit) if state_kind(records[f])==state_kind(records[i])],int)
        if len(same):grouped[j]=y[same].mean(0)
    out['fit_mean_by_state_diagnostic_only']=grouped
    return out,dict(nearest_fit_rms=distance[np.arange(len(held)),order[:,0]].tolist(),
        nearest_fit_record=fit[order[:,0]].tolist(),
        missing_state_predictions=int((~np.isfinite(grouped).all(1)).sum()))


def ambiguity(values, tolerance=1e-8):
    y=np.asarray(values,float)
    if y.ndim!=2 or y.shape[1]!=3 or not len(y) or not np.isfinite(y).all():
        raise ValueError('IDENTIFIABILITY_VALUES')
    regret=y.max(1)[:,None]-y
    optimal=regret<=tolerance
    gaps=np.stack([y[:,a]-y[:,b] for a,b in ((0,1),(0,2),(1,2))],axis=1)
    return dict(states=len(y),common_optimal_actions=np.flatnonzero(optimal.all(0)).tolist(),
        optimal_sets=[np.flatnonzero(r).tolist() for r in optimal],
        mean_regret_by_shared_action=regret.mean(0).tolist(),
        best_shared_action=int(regret.mean(0).argmin()),
        minimum_shared_mean_regret=float(regret.mean(0).min()),
        conflicting_pairs=int(((gaps.min(0)<-tolerance)&(gaps.max(0)>tolerance)).sum()),
        common_q_mse_floor=float(np.mean((y-y.mean(0))**2)),
        scope='equal_weight_reachable_states_at_one_market_window_not_population')


def select_origins(records, held, seed=316, per_year=4):
    rng=np.random.default_rng(seed);selected=[];counts={}
    for year in range(2020,2024):
        eligible=sorted([i for i in held if records[i]['year']==year
            and state_kind(records[i])=='flat_both_allowed'],key=lambda i:records[i]['timestamp'])
        pick=rng.permutation(eligible)[:per_year].tolist();selected.extend(pick)
        counts[str(year)]=dict(available=len(eligible),requested=per_year,selected=len(pick))
    return sorted(selected,key=lambda i:records[i]['timestamp']),counts


def forked_state_test(env, signals, repeat=False):
    before=state_view(env);forks=[];first=[]
    if env.engine.pos is not None:raise ValueError('IDENTIFIABILITY_ORIGIN_NOT_FLAT')
    for action in range(3):
        branch=fork_env(env)
        _,reward,done,duration=branch.step(action)
        first.append(dict(action=action,reward=float(reward),duration=duration,terminal=done,
            state=state_view(branch)))
        forks.append(branch)
    if state_view(env)!=before:raise ValueError('IDENTIFIABILITY_PARENT_MUTATION')
    if any(f.finished for f in forks):
        return dict(status='TERMINAL_BRANCH',first=first,parent_unchanged=True)
    rows=[int(f.rows[f.bar]) for f in forks]
    features=[f.inputs.observation(row)['features'] for f,row in zip(forks,rows)]
    if len(set(rows))!=1 or any(not np.array_equal(features[0],v) for v in features[1:]):
        raise ValueError('IDENTIFIABILITY_MARKET_MISMATCH')
    outcomes=[]
    for branch in forks:
        values=[continuation(branch,signals,a) for a in range(3)]
        if repeat and values!=[continuation(branch,signals,a) for a in range(3)]:
            raise ValueError('IDENTIFIABILITY_REPEAT')
        outcomes.append(values)
    if state_view(env)!=before:raise ValueError('IDENTIFIABILITY_PARENT_MUTATION')
    values=[[v['reward_return'] for v in row] for row in outcomes]
    return dict(status='COMPLETE',first=first,next_row=rows[0],outcomes=outcomes,
        same_market_tensor=True,parent_unchanged=True,repeat_checked=repeat,
        distinct_positions=sorted({0 if f.engine.pos is None else f.engine.pos.direction for f in forks}),
        ambiguity=ambiguity(values))
