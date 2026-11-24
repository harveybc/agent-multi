"""Attribution of the locked 471 context policy. No search, 2024 or broker."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.role471_net_selector import NS, prepare, audit_book
from tools.role461_seasonal_account import DEFAULT,load,cost_arrays
from tools.role470_barriers import probe

WEEK=168*NS
MON=pd.Timestamp('1970-01-05',tz='UTC').value
def week(ns):
    return (ns-MON)//WEEK


def replay(schedule,arr,t):
    ii,_,pnl,ends=arr
    bal=peak=10000.;dd=exposure=volume=0.;wins=n=skips=0;last=-1
    for j,k,q in sorted(schedule):
        if ii[j]<=last:
            skips+=1
            continue
        last=ends[j,k];delta=q*pnl[j,k]
        bal*=1+delta/10000;peak=max(peak,bal);dd=max(dd,1-bal/peak)
        # Occupied H1 bars inclusive; this is not an intrabar time estimate.
        exposure+=q*(last-ii[j]+1);volume+=q;wins+=int(delta>0);n+=1
    return dict(ret=bal/100-100,dd=dd*100,n=n,wins=wins,skips=skips,q_sum=volume,q_bars=exposure)


def month_permutations(arr,t,template):
    ii,_,_,ends=arr
    ns=t[ii];positions={int(v):j for j,v in enumerate(ns)}
    weeks=np.unique(week(ns));source=week(ns)
    month=pd.to_datetime(ns,utc=True).month.to_numpy()
    blocks=pd.to_datetime(MON+weeks*WEEK+3*24*NS,utc=True).strftime('%Y-%m')
    choices=[]
    for block in np.unique(blocks):
        group=weeks[blocks==block]
        matrix={}
        for src in group:
            events=[p for p in template if source[p[0]]==src]
            for dst in group:
                moved=[];last=-1
                for j,_,q in events:
                    k=positions.get(ns[j]+(dst-src)*WEEK)
                    if k is None or ii[k]<=last or month[k]!=month[j]:break
                    moved.append((k,5,q));last=ends[k,5]
                else:
                    matrix[src,dst]=(moved,len(events)*int(src!=dst))
        options=[]
        for perm in itertools.permutations(group):
            links=list(zip(group,perm))
            if all(link in matrix for link in links):
                options.append(([v for key in links for v in matrix[key][0]],
                                sum(matrix[key][1] for key in links)))
        if not options:
            raise ValueError(f'No feasible paired schedules for {block}; do not relax controls silently')
        choices.append(options)
    return choices


def quantiles(df):
    return dict(n=len(df),median=float(df.ret.median()),p025=float(df.ret.quantile(.025)),
                p975=float(df.ret.quantile(.975)),mean_orders=float(df.n.mean()),
                mean_q_bars=float(df.q_bars.mean()))


def main(a):
    cfg=json.loads((a.prev/'config.json').read_text())
    genes=tuple(json.loads((a.prev/'locked.json').read_text())['context']['effective'].values())
    arrays,_,checks,t,_=prepare(a,cfg)
    data=load(a.src,a.bars,DEFAULT,a.prefix)
    prof=json.loads((a.bars/'RECIPE.json').read_text())['business']['evaluation_and_policy']['profile']
    costs=(prof,*cost_arrays(t,json.loads((a.prefix/'CLOSURES.json').read_text()),prof))
    fixed=pd.read_csv(a.prev/'accounts.csv.gz').query('selector=="context" and arm=="deap"')
    sample=lambda y: arrays['context',*genes[:2],y]
    # Estimate clock-only entry propensity per eligible bar and typical q, without 2023.
    counts=np.zeros(168);opps=np.zeros(168);qtotal=np.zeros(168)
    def slot(ns):
        date=pd.to_datetime(ns,utc=True)
        return np.asarray(date.dayofweek*24+date.hour)
    for y in cfg['train_years']:
        b=fixed[fixed.fold==y];hours=slot(b.entry_ns.to_numpy())
        opps+=np.bincount(slot(t[sample(y)[0]]),minlength=168)
        counts+=np.bincount(hours,minlength=168)
        qtotal+=np.bincount(hours,weights=b.q,minlength=168)
    # Count free opportunities, including the entry decision but not the exit bar.
    for y in cfg['train_years']:
        dates=t[sample(y)[0]]
        for r in fixed[fixed.fold==y].itertuples():
            blocked=dates[(dates>r.entry_ns)&(dates<=r.exit_bar_open_ns)]
            opps-=np.bincount(slot(blocked),minlength=168)
    assert np.all(counts<=opps)
    rates=(counts+counts.sum()/opps.sum())/(opps+1)
    sizes=(qtotal+qtotal.sum()/counts.sum())/(counts+1)
    a.out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(dict(hour_of_week=range(168),train_free=opps,entries=counts,
                      probability=rates,q=sizes)).to_csv(a.out/'train_clock.csv',index=False)
    summary={};rows=[];fills=[]
    for y in (20231,20232):
        arr=sample(y);ns=t[arr[0]]
        book,ret=audit_book(arr,genes,t)
        b=fixed[fixed.fold==y]
        np.testing.assert_array_equal([v['entry_ns'] for v in book],b.entry_ns)
        template=[(int(np.searchsorted(ns,v['entry_ns'])),(0 if v['side']=='buy' else 3)+(6,24,48).index(v['hours']),v['q']) for v in book]
        original=replay(template,arr,t);np.testing.assert_allclose(original['ret'],ret,atol=1e-9)
        same=replay([(j,5,q) for j,_,q in template],arr,t)
        for j,k,q in template:
            i=arr[0][j]
            for tag,action in [('selector',k),('sell_same_proposal',5)]:
                h=(6,24,48)[action%3];w=data[5][i-1]*np.sqrt(h)
                record=probe(i,1 if action<3 else -1,genes[0]*w,genes[1]*w,q,h,data,costs)
                assert record['end']==arr[3][j,action]
                fills.append(dict(fold=y,arm=tag,entry_ns=int(t[i]),**record))
        choices=month_permutations(arr,t,template)
        rng=np.random.default_rng(472+y);hrs=slot(ns);rr=[];clock=[]
        for seed in range(2048):
            parts=[opts[rng.integers(len(opts))] for opts in choices]
            rec=replay([i for s,_ in parts for i in s],arr,t)
            assert rec['skips']==0 and rec['n']==original['n']
            np.testing.assert_allclose(rec['q_sum'],original['q_sum'])
            rr.append(dict(fold=y,arm='paired',draw=seed,moved=sum(n for _,n in parts),**rec))
            selected=np.flatnonzero(rng.random(len(ns))<rates[hrs])
            clock.append(dict(fold=y,arm='historical_clock',draw=seed,
                         **replay([(j,5,sizes[hrs[j]]) for j in selected],arr,t)))
        df=pd.DataFrame(rr);live=pd.DataFrame(clock)
        similar=df[abs(df.q_bars/original['q_bars']-1)<=.05]
        rows.extend(rr+clock)
        summary[y]=dict(selector=original,sell_same=same,paired=quantiles(df),clock=quantiles(live),
            moved_mean=float(df.moved.mean()),distinct_returns_at_8dp=int(df.ret.round(8).nunique()),
            permutation_options_per_month=[len(o) for o in choices],
            paired_at_least_selector=float((df.ret>=ret).mean()),paired_qbars_5pct=quantiles(similar) if len(similar) else None,
            qbars_at_least_selector=float((similar.ret>=ret).mean()) if len(similar) else None)
    pd.DataFrame(rows).to_csv(a.out/'accounts.csv.gz',index=False)
    pd.DataFrame(fills).to_csv(a.out/'barrier_audit.csv',index=False)
    (a.out/'report.json').write_text(json.dumps(dict(reference_checks=checks,panels=summary),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('src','bars','prefix','signals','prev','out'):p.add_argument(k,type=Path)
    main(p.parse_args())
