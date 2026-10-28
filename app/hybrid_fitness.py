"""Common-scale fitness diagnostics; observed year never selects or calibrates."""
import math
import statistics
from app.recipe_activity import activity


def components(r):
    return {'S':r['sharpe_daily_sqrt365'],'R':r['net_profit_usd']/r['initial_capital'],
            'D':r['max_drawdown_pct']/100,'A':r['average_entries_per_business_day']}


def compare(rows):
    rows=[r for r in rows if r['period'] in ('TRAIN','INNER_A')]
    common=[]
    for r in rows:
        if r['period']=='TRAIN' and activity(r)['activity_eligible']:
            c=components(r)
            if all(v is not None and math.isfinite(v) for v in c.values()): common.append(c)
    fallback={'S':1.,'R':.1,'D':.1,'A':.5}
    scales={}
    for key in fallback:
        values=[abs(c[key]) for c in common]
        median=statistics.median(values) if values else 0.
        scales[key]=median if median>=1e-6 else fallback[key]
    def score(c):
        s,r,d,a=(c[k]/scales[k] for k in ('S','R','D','A'))
        return {'sharpe':c['S'],'sharpe_return':s+r,'plus_activity025':s+r+.25*a,
                'plus_activity100':s+r+a,'balanced_drawdown':s+r-d,
                'return_minus_dd':c['R']-c['D'],
                'return_weight025':.75*s+.25*r,'return_weight075':.25*s+.75*r}
    rankings={k:[] for k in score({'S':0.,'R':0.,'D':0.,'A':0.})}
    for arm in sorted({r['arm'] for r in rows}):
        group=[r for r in rows if r['arm']==arm]
        if len(group)!=6 or {(r['seed'],r['period']) for r in group}!={(s,p) for s in (7,19,43) for p in ('TRAIN','INNER_A')}:
            continue
        if not all(activity(r)['activity_eligible'] and r.get('business_behavior_eligible',True) for r in group): continue
        cs=[components(r) for r in group]
        if not all(v is not None and math.isfinite(v) for c in cs for v in c.values()): continue
        inner=[score(components(r)) for r in group if r['period']=='INNER_A']
        for key in rankings:
            values=[c[key] for c in inner]
            rankings[key].append(dict(arm=arm,mean=statistics.mean(values),std=statistics.stdev(values),seeds=3))
    for v in rankings.values(): v.sort(key=lambda r:(-r['mean'],r['arm']))
    return dict(rankings=rankings,scales=scales,scope='DEVELOPMENT_DIAGNOSTIC_NOT_FINAL_PROMOTION',
                scale_fit='COMMON_TRAIN_CANDIDATES_ONLY',selection='INNER_A; all three seeds eligible in TRAIN and INNER_A',
                undefined_ratio_metrics='Calmar/uncertainty not used as governing selection in this version')
