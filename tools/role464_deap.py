"""Oracle account strategy search. No noise and no learned predictions."""
import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, tools
from tools.role461_seasonal_account import DEFAULT,H,NS,load,cost_arrays,simulate
from tools.role463_noise import run


def decode(spec,x):
    if not math.isfinite(x) or not 0<=x<=1:raise ValueError(x)
    if spec['kind']=='choice':
        choices=spec['values'];return choices[min(int(x*len(choices)),len(choices)-1)]
    lo,hi=spec['range']
    if x==0:return float(lo)
    if x==1:return float(hi)
    if spec['kind']=='log':return lo*(hi/lo)**x
    assert spec['kind']=='linear'
    return lo+(hi-lo)*x


def encode(spec,value):
    if spec['kind']=='choice':return (spec['values'].index(value)+.5)/len(spec['values'])
    lo,hi=spec['range']
    return (math.log(value/lo)/math.log(hi/lo) if spec['kind']=='log' else (value-lo)/(hi-lo))


def prepare(a,cfg):
    data=list(load(a.src,a.bars,DEFAULT,a.prefix))
    t,o,hi,lo,cl,atr,ds,dp,gate,dead=data
    profile=json.loads(a.recipe.read_text())['business']['evaluation_and_policy']['profile']
    spread,roll=cost_arrays(t,json.loads((a.prefix/'CLOSURES.json').read_text()),profile)
    fees=np.array([profile[k] for k in ('slippage_market','slippage_stop','limit_through')]+
                  [profile[k]/profile['lot_units'] for k in ('commission_per_lot_side','swap_long_per_lot','swap_short_per_lot')])
    tr=np.maximum(hi-lo,np.maximum(abs(hi-np.r_[cl[0],cl[:-1]]),abs(lo-np.r_[cl[0],cl[:-1]])))
    windows={w:pd.Series(tr).rolling(w).mean().to_numpy() for w in cfg['genes']['atr_window']['values']}
    yearly={};counts={}
    for yr in cfg['train_years']+[cfg['validation_year'],cfg['report_year']]:
        end=pd.Timestamp(str(yr+1),tz='UTC').value
        mask=(t>=pd.Timestamp(str(yr),tz='UTC').value)&(t<end)
        signal=ds[mask].copy()
        signal[t[mask,None]+(1+H)*NS>=end]=np.nan
        ds[mask]=signal
        counts[yr]=dict(bars=int(mask.sum()),long_complete=int(np.isfinite(signal[:,5:]).all(axis=1).sum()))
        for w,ar in windows.items():
            for paid in (False,True):
                cols=(t,o,hi,lo,cl,ar,gate,dead,spread if paid else spread*0,roll)
                yearly[yr,w,paid]=(*[np.ascontiguousarray(col[mask]) for col in cols],signal,fees if paid else fees*0)
    def evaluate(par,yr,paid):
        w,*others=par
        ret,dd,orders,buy,wins,sh=run(*yearly[yr,w,paid],*others)
        return dict(ret=ret-100,dd=dd,orders=int(orders),buy=int(buy),wins=int(wins),
                    sharpe=float(sh) if np.isfinite(sh) else -1e6)
    def reference(par,yr,paid,with_book=False):
        named=dict(zip(cfg['genes'],par)); yy=data.copy();yy[5]=windows[par[0]]
        v,book=simulate(yy,(None,None),dict(DEFAULT,**named),yr,'bridge','sl_only',
                     (profile,spread,roll) if paid else None)
        fast=evaluate(par,yr,paid)
        np.testing.assert_allclose([fast[k] for k in ('ret','dd','orders','buy','wins')],
               [v[k] for k in ('return_pct','dd_close_pct','trades','buy','wins')],atol=1e-7,rtol=0)
        return (v,book) if with_book else v
    return evaluate,reference,counts


def search(cfg,ev,paid,seed,out,*,score_fn=None,warm=False):
    random.seed(seed); specs=list(cfg['genes'].values()); n=len(specs)
    actual=lambda ind:tuple(decode(s,x) for s,x in zip(specs,ind,strict=True))
    toolbox=base.Toolbox()
    toolbox.register('individual',tools.initRepeat,creator.Strategy464,random.random,n=n)
    pop=[toolbox.individual() for _ in range(cfg['population'])]
    pop[0][:]=[encode(s,s['baseline']) for s in specs]
    if warm:
        for ind in pop[1:len(pop)//2+1]:
            ind[:]=[min(1.,max(0.,random.gauss(x,.055))) for x in pop[0]]
    cache={};mon={};best=-math.inf;counter=0;champ=None;history=[]
    dest=out/f'{"cost" if paid else "free"}-{seed}';dest.mkdir(exist_ok=True)
    with gzip.open(dest/'candidates.jsonl.gz','wt') as stream:
        for gen in range(cfg['max_generations']+1):
            for ind in pop:
                p=actual(ind)
                if p not in cache:
                    metrics=[ev(p,y,paid) for y in cfg['train_years']]
                    score=score_fn(metrics,False) if score_fn else float(np.mean([v['sharpe'] for v in metrics]))
                    cache[p]=len(cache),score
                    stream.write(json.dumps(dict(id=cache[p][0],first_gen=gen,genome=list(ind),
                        effective=dict(zip(cfg['genes'],p)),train_by_year=metrics,fitness=score))+'\n')
                ind.fitness.values=(cache[p][1],)
            leader=tools.selBest(pop,1)[0];p=actual(leader)
            if p not in mon:mon[p]=ev(p,cfg['validation_year'],paid)
            val=score_fn([mon[p]],True) if score_fn else mon[p]['sharpe'];score=leader.fitness.values[0]
            if val>best+cfg['min_delta']:
                best=val;counter=0
                champ=dict(gen=gen,effective=dict(zip(cfg['genes'],p)),train=score,val=mon[p],seed=seed)
            else:counter+=1
            history.append(dict(gen=gen,ids=[cache[actual(i)][0] for i in pop],
                                leader=cache[p][0],train=score,val=val,wait=counter))
            if gen>=cfg['min_generations'] and counter>=cfg['patience']:break
            if gen==cfg['max_generations']:break
            nxt=[toolbox.clone(leader)]
            while len(nxt)<len(pop)-cfg['immigrants']:
                x,y=[toolbox.clone(z) for z in tools.selTournament(pop,2,tournsize=cfg['tournament'])]
                if random.random()<cfg['crossover_probability']:
                    tools.cxSimulatedBinaryBounded(x,y,eta=cfg['sbx_eta'],low=0,up=1)
                for z in (x,y):
                    tools.mutPolynomialBounded(z,eta=cfg['mutation_eta'],low=0,up=1,indpb=cfg['per_gene_mutation'])
                    if len(nxt)<len(pop)-cfg['immigrants']:nxt.append(z)
            pop=nxt+[toolbox.individual() for _ in range(cfg['immigrants'])]
    (dest/'generations.json').write_text(json.dumps(history)+'\n')
    champ.update(last_generation=gen,distinct=len(cache),wait=counter,
                 stop='patience' if counter>=cfg['patience'] else 'cap',
                 distinct_values={k:len({x[j] for x in cache}) for j,k in enumerate(cfg['genes'])})
    return champ


def side_report(out,cfg,winners,reference,panels):
    rows=[]
    for name in panels:
        par=tuple(winners[name]['effective'][k] if name!='baseline' else s['baseline']
                  for k,s in cfg['genes'].items())
        for arm,annual in panels[name].items():
            train=np.zeros(5,dtype=int)
            def add(period,counts):
                total,buy,sell,bw,sw=map(int,counts)
                pct=lambda x,n:100*x/n if n else None
                rows.append(dict(profile=name,execution=arm,period=period,orders=total,
                    buys=buy,sells=sell,buy_wins=bw,sell_wins=sw,wins=bw+sw,
                    win_pct=pct(bw+sw,total),buy_pct=pct(buy,total),sell_pct=pct(sell,total),
                    buy_win_pct=pct(bw,buy),sell_win_pct=pct(sw,sell)))
            for yr,rec in annual.items():
                v=reference(par,int(yr),arm=='cost')
                n,b,w,bw=(v[k] for k in ('trades','buy','wins','wins_buy'))
                assert (n,b,w)==(rec['orders'],rec['buy'],rec['wins'])
                counts=np.array([n,b,n-b,bw,w-bw])
                assert 0<=bw<=b and 0<=w-bw<=n-b
                add(yr,counts)
                if int(yr) in cfg['train_years']:train+=counts
            add('TRAIN',train)
    with (out/'side_report.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main(a):
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True)
    ev,ref,counts=prepare(a,cfg);genes=cfg['genes']
    if a.side_report_only:
        side_report(a.out,cfg,json.loads((a.out/'locked.json').read_text()),ref,
                    json.loads((a.out/'results.json').read_text())['panels'])
        return
    baseline=tuple(s['baseline'] for s in genes.values())
    checks={}
    for paid in (False,True):
        ref(baseline,2020,paid);ref(baseline,2023,paid)
    for j,(key,s) in enumerate(genes.items()):
        assert np.isclose(decode(s,encode(s,baseline[j])),baseline[j])
        variants=[]
        for x in (0.,1.):
            p=list(baseline);p[j]=decode(s,x);ref(p,2021,True);variants.append(ev(p,2021,True))
        assert variants[0]!=variants[1],key
        checks[key]=variants
    creator.create('Fit464',base.Fitness,weights=(1.,));creator.create('Strategy464',list,fitness=creator.Fit464)
    winners={};runs={}
    for name,paid in [('cost',True),('free',False)]:
        runs[name]=[search(cfg,ev,paid,seed,a.out) for seed in cfg['seeds']]
        winners[name]=max(runs[name],key=lambda x:x['val']['sharpe'])
    # Freeze both policies before scoring the 2024 reporting panel.
    (a.out/'locked.json').write_text(json.dumps(winners,indent=2)+'\n')
    report={}
    for name,p in [('baseline',baseline)]+[(k,tuple(v['effective'].values())) for k,v in winners.items()]:
        report[name]={}
        for arm,paid in [('cost',True),('free',False)]:
            ref(p,2024,paid)
            report[name][arm]={y:ev(p,y,paid) for y in cfg['train_years']+[2023,2024]}
    (a.out/'results.json').write_text(json.dumps(dict(counts=counts,runs=runs,panels=report,gene_checks=checks),indent=2)+'\n')
    side_report(a.out,cfg,winners,ref,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    for k in ('src','bars','prefix','recipe','config','out'):ap.add_argument(k,type=Path)
    ap.add_argument('--side-report-only',action='store_true')
    main(ap.parse_args())
