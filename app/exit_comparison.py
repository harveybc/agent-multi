"""Paired cluster bootstrap for the synthetic exit subproblem, not annual trading."""
import numpy as np

from app.optimal_exit import outcome


def paired_comparison(genes, controls, rows, seed=113, replicates=2000):
    def metrics(g):
        result=[]
        for case, _, logits in rows:
            value=outcome(case,logits+g[0 if case['direction']==1 else 1]>=0)
            result.append([value['regret_risk'],value['net_usd']/10000.])
        return np.asarray(result)
    keys=[tuple(case[k] for k in ('peak','gain','offset','spread')) for case,_,_ in rows]
    clusters={key:np.array([i for i,k in enumerate(keys) if k==key]) for key in sorted(set(keys))}
    if len(clusters)<2 or replicates<2:raise ValueError('EXIT_COMPARISON_CLUSTERS')
    groups=list(clusters.values());candidate=metrics(genes);reports=[]
    def summary(a):
        std=float(a[:,1].std())
        return np.array([a[:,0].mean(),a[:,1].mean(),a[:,1].mean()/std if std>0 else np.nan])
    for control in controls:
        baseline=metrics(control);observed=summary(candidate)-summary(baseline)
        rng=np.random.default_rng(seed);samples=[]
        for _ in range(replicates):
            ix=np.concatenate([groups[j] for j in rng.integers(len(groups),size=len(groups))])
            samples.append(summary(candidate[ix])-summary(baseline[ix]))
        samples=np.asarray(samples)
        if not np.isfinite(samples).all() or not np.isfinite(observed).all():
            raise ValueError('EXIT_COMPARISON_UNDEFINED_METRIC')
        intervals=np.quantile(samples,[.025,.975],axis=0)
        reports.append(dict(control=control,delta_candidate_minus_control={name:dict(
            mean=float(observed[j]),ci95=intervals[:,j].tolist())
            for j,name in enumerate(('regret_risk','return_fraction','sharpe_episode_unannualized'))}))
    return dict(clusters=len(groups),resamples=replicates,seed=seed,
        clustering='same peak/gain/offset/spread keeps both directions and signal-quality variants together',
        comparisons=reports,scope='paired synthetic diagnostic intervals; no multiple-comparison winner declaration')
