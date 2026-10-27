"""DEAP calibration of two categorical close biases on frozen synthetic menus."""
import copy
import random
import math

import numpy as np
from deap import base,tools

from app.optimal_exit import targets,outcome


def panel(cases,seed):
    rng=np.random.default_rng(seed);rows=[]
    for case in cases:
        t=targets(case['close_net'],case['terminal_net'],case['risk_usd'])
        ideal=np.where(t['oracle_close'],6.,-6.)
        for quality in ('ideal','noise2','noise4','delay1'):
            logits=ideal.copy()
            if quality.startswith('noise'):logits+=rng.normal(0,float(quality[-1]),len(logits))
            if quality=='delay1':logits=np.r_[-6.,logits[:-1]]
            rows.append((case,quality,logits))
    return rows


def score(genes,rows):
    outcomes=[];ideal_errors=[]
    for case,quality,logits in rows:
        result=outcome(case,logits+genes[0 if case['direction']==1 else 1]>=0)
        outcomes.append(result)
        if quality=='ideal':ideal_errors.append(result['regret_risk'])
    returns=np.array([r['net_usd']/10000. for r in outcomes])
    std=float(returns.std())
    return dict(mean_regret_risk=float(np.mean([r['regret_risk'] for r in outcomes])),
        sharpe_episode_unannualized=float(returns.mean()/std) if std>0 else 0.,
        mean_return_fraction=float(returns.mean()),max_ideal_regret=max(ideal_errors),
        early_close_share=float(np.mean([r['early_close'] for r in outcomes])),
        mean_regret_long=float(np.mean([r['regret_risk'] for r in outcomes if r['direction']==1])),
        mean_regret_short=float(np.mean([r['regret_risk'] for r in outcomes if r['direction']==-1])))


class Fitness(base.Fitness):
    weights=(1.,)


class Genome(list):
    def __init__(self,values):
        super().__init__(values);self.fitness=Fitness()


def search(rows,objective,seed,population=20,generations=100,patience=10,tolerance=1e-8):
    if objective not in ('negative_exit_regret','sharpe','sharpe_return'):raise ValueError('EXIT_OBJECTIVE')
    rng_state=random.getstate();random.seed(seed)
    history=[];evaluated={};best=None;stale=0
    def evaluate(g):
        key=tuple(g)
        if key not in evaluated:evaluated[key]=score(g,rows)
        m=evaluated[key]
        value=-m['mean_regret_risk'] if objective=='negative_exit_regret' else m['sharpe_episode_unannualized']
        if objective=='sharpe_return':value+=m['mean_return_fraction']
        if m['max_ideal_regret']>1e-10:value=-1e9-m['max_ideal_regret']
        g.fitness.values=(value,)
    try:
        pop=[Genome([0.,0.]),Genome([1.,1.]),Genome([-1.,-1.])]
        pop += [Genome([random.uniform(-2,2),random.uniform(-2,2)]) for _ in range(population-len(pop))]
        for generation in range(generations+1):
            for g in pop:
                if not g.fitness.valid:evaluate(g)
            winner=tools.selBest(pop,1)[0]
            improved=best is None or winner.fitness.values[0]>best.fitness.values[0]+tolerance*max(abs(best.fitness.values[0]),1e-12)
            if improved:best=copy.deepcopy(winner);stale=0
            else:stale+=1
            history.append(dict(generation=generation,fitness=best.fitness.values[0],patience=stale))
            if stale>=patience or generation==generations:break
            children=[copy.deepcopy(x) for x in tools.selTournament(pop,population-1,tournsize=3)]
            for i in range(0,len(children)-1,2):
                if random.random()<.7:tools.cxBlend(children[i],children[i+1],alpha=.3)
            for g in children:
                if random.random()<.3:tools.mutGaussian(g,mu=0,sigma=.3,indpb=.5)
                g[:]=[max(-2.,min(2.,v)) for v in g]
                del g.fitness.values
            pop=[copy.deepcopy(best)]+children
        return dict(objective=objective,seed=seed,genes=list(best),calibration=score(best,rows),
            stop_reason='PATIENCE' if stale>=patience else 'GENERATION_CAP',history=history,
            evaluated=[dict(genes=list(g),metrics=m) for g,m in evaluated.items()],
            bound_hits=[abs(v)>=2 for v in best],population=population,generation_cap=generations,
            patience_limit=patience,relative_tolerance=tolerance)
    finally:random.setstate(rng_state)
