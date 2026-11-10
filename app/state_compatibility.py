"""Counterfactual account-state compatibility at an identical market observation."""
import numpy as np

from app.decision_attribution import fork_env, state_view


def compatibility(target,tolerance=1e-8):
    y=np.asarray(target,dtype=float)
    if y.ndim!=2 or y.shape[0]<2 or y.shape[1]!=3 or not np.isfinite(y).all():
        raise ValueError('COMPATIBILITY_TARGET')
    centered=y-y.mean(1,keepdims=True)
    pairs=((0,1),(0,2),(1,2))
    gaps=np.stack([y[:,a]-y[:,b] for a,b in pairs],axis=1)
    regret=y.max(1,keepdims=True)-y
    return dict(states=len(y),mean_targets=y.mean(0).tolist(),
        raw_rmse_floor=float(np.sqrt(np.mean((y-y.mean(0))**2))),
        centered_rmse_floor=float(np.sqrt(np.mean((centered-centered.mean(0))**2))),
        gap_rmse_floor=float(np.sqrt(np.mean((gaps-gaps.mean(0))**2))),
        strict_reversal_pairs=[list(pair) for j,pair in enumerate(pairs)
            if gaps[:,j].min() < -tolerance and gaps[:,j].max()>tolerance],
        optimal_action_sets=[np.flatnonzero(r<=tolerance).tolist() for r in regret],
        common_optimal_actions=np.flatnonzero((regret<=tolerance).all(0)).tolist(),
        best_common_action=int(regret.mean(0).argmin()),
        best_common_regret=float(regret.mean(0).min()),
        deterministic_minimax_regret=float(regret.max(0).min()))


def history_banks(snapshot,lags=(1,4,12)):
    if snapshot.finished or snapshot.engine.pos is not None:
        raise ValueError('COMPATIBILITY_ORIGIN_NOT_FLAT')
    if not lags or tuple(sorted(set(lags)))!=tuple(lags) or any(type(v)!=int or v<1 for v in lags):
        raise ValueError('COMPATIBILITY_LAGS')
    before=state_view(snapshot)
    forks={k:fork_env(snapshot) for k in ('flat','long','short')}
    for step in range(1,max(lags)+1):
        for name,env in forks.items():
            env.step({'flat':0,'long':1,'short':2}[name] if step==1 else 0)
        if len({(e.bar,e.finished) for e in forks.values()})!=1:
            raise ValueError('COMPATIBILITY_TIME_MISMATCH')
        if forks['flat'].finished:break
        if step in lags:
            ref=forks['flat'];states={};histories={}
            for name,env in forks.items():
                if (not np.array_equal(env.last_observation['features'],ref.last_observation['features'])
                        or not np.array_equal(env.last_observation['mask'],ref.last_observation['mask'])):
                    raise ValueError('COMPATIBILITY_OBSERVATION_MISMATCH')
                actual=0 if env.engine.pos is None else env.engine.pos.direction
                histories[name]=state_view(env)
                if actual=={'flat':0,'long':1,'short':-1}[name]:states[name]=fork_env(env)
            if state_view(snapshot)!=before:raise ValueError('COMPATIBILITY_PARENT_MUTATED')
            yield step,states,dict(histories=histories,observation_equal=True,
                elapsed_hours=(state_view(ref)['timestamp']-before['timestamp'])/(3600*10**9))
    if state_view(snapshot)!=before:raise ValueError('COMPATIBILITY_PARENT_MUTATED')
