"""F gene reach, P2/P3 action identity, H games, G paired bootstrap and dominance."""
import json, sys
from collections import Counter
import numpy as np
sys.path.insert(0, sys.argv[1]); from common import *
from app.dynamic_account_panel import World, panel_specs, evaluate, BASE
from app.dynamic_account_arbiter import decode, bounds, HORIZONS, choose
run=sys.argv[2]; R=replays(run); rep=json.loads((Path(run)/'REPORT.json').read_text())
front=json.loads((Path(run)/'FRONTIER.json').read_text()); G={r['key'][:12]:r['genes'] for r in front['rows']}
key={a:[k for k in R if k.startswith(p)][0] for p,a in ALIAS.items()}
out={}
cal=[World(s,CFG) for s in panel_specs(7)]
def scores(g,bids,eff):
    return np.array([g[HORIZONS.index(b['horizon'])]*b['gain_risk']-g[5]*b['uncertainty_risk']+g[4]*b['direction']*eff*b['horizon']/24-g[6]*b['duration']/24-g[7] for b in bids])
NAMES=['w3','w6','w12','w24','trend','uncert','time','threshold','size','margin','persist']
for al in ('P1','P3'):
    g0=decode(G[key[al][:12]])
    base_obj=evaluate(cal,g0)['objectives']
    res={}
    # baseline traces for flat & occupied ticks (calibration)
    for i,n in enumerate(NAMES):
        lo,hi=bounds()[i]
        step=1 if i==10 else .05
        gp=list(g0); gp[i]=g0[i]+step if g0[i]+step<=hi else g0[i]-step
        gp=decode(gp)
        # computation-level effect on entry scores (all flat gate ticks, all worlds)
        dscore=Counter(); dchoice=0; ticks=0
        dz=0; dq=0; dact=0; occ=0; dunits=0
        for w,rec in zip(cal,R[key[al]]['calibration']['worlds']):
            e=w.env
            for t in rec['trace']:
                b=t['bar']
                pos=next((x for x in rec['ledger'] if x['entry_bar']<=b<x['exit_bar']),None)
                if pos is None and b+1<=e.last and e.v[b+1]>=e.op[0] and e.v[b+1]<e.ff[0]:
                    bids=w.candidate_bids(b); x=np.diff(e.c[max(0,b-6):b+1]);den=np.abs(x).sum();eff=float(x.sum()/den) if den else 0.
                    s0=scores(g0,bids,eff); s1=scores(gp,bids,eff); ticks+=1
                    for bid,a,c in zip(bids,s0,s1):
                        if abs(a-c)>1e-12: dscore[bid['horizon']]+=1
                    c0=choose(bids,g0,eff); c1=choose(bids,gp,eff)
                    if (c0 and (c0['horizon'],c0['direction']))!=(c1 and (c1['horizon'],c1['direction'])): dchoice+=1
        # close-controller reach: replay controller on stored occupied ticks with both configs
        from app.close_hold_controller import CloseController, CloseConfig
        for w,rec in zip(cal,R[key[al]]['calibration']['worlds']):
            e=w.env
            c0=CloseController(CloseConfig(g0[5],g0[9],g0[10],.25),allow_synthetic=True)
            c1=CloseController(CloseConfig(gp[5],gp[9],gp[10],.25),allow_synthetic=True)
            for t in rec['trace']:
                b=t['bar'];pos=next((x for x in rec['ledger'] if x['entry_bar']<=b<x['exit_bar']),None)
                if pos is None or b+1>=w.end: continue
                fake=type('E',(),{'pos':None,'v':e.v})()
                k=pos['entry_bar'];d=pos['direction'];s=.00005 if w.spec['volatility']=='decreasing' else .0003
                E=e.o[k]+d*s/2;u=w.position(pos['horizon'],d,E,k)
                fake.pos=type('P',(),dict(direction=d,entry_bar=k,entry_price=E,sl=u.sl,tp=u.tp))()
                unit,vals,_,_=w.close_values(fake,pos['horizon'],b);r=w.report(b)
                a0=c0.decide_synthetic_values(unit.identity,int(r.issued_at),3600*10**9,vals,r.weights,unit.risk)
                a1=c1.decide_synthetic_values(unit.identity,int(r.issued_at),3600*10**9,vals,r.weights,unit.risk)
                occ+=1; dz+=abs(a0['advantage_risk']-a1['advantage_risk'])>1e-12; dq+=a0['qualifies']!=a1['qualifies']; dact+=a0['action']!=a1['action']
        obj=evaluate(cal,gp)['objectives']
        # size: units ratio with identical actions
        r0=cal[0].run(g0,details=True); r1=cal[0].run(gp,details=True)
        res[n]=dict(value=g0[i],perturbed=gp[i],entry_scores_changed_by_horizon=dict(dscore),flat_ticks=ticks,entry_choices_changed=dchoice,
                    occupied_ticks=occ,close_adv_changed=int(dz),close_qualifies_changed=int(dq),close_actions_changed=int(dact),
                    world0_trace_equal=r0['trace']==r1['trace'],world0_units=[(a['units'],b['units']) for a,b in zip(r0['ledger'],r1['ledger'])][:3],
                    cal_objectives=obj,cal_objective_changed=obj!=base_obj)
    out[f'F|{al}']=dict(base=base_obj,genes=list(g0),perturb=res)
# P2 vs P3 identity
for split in ('calibration','confirmation'):
    a=R[key['P2']][split]['worlds'];b=R[key['P3']][split]['worlds']
    same_tr=all([(x['bar'],x['action'],x['horizon']) for x in u['trace']]==[(x['bar'],x['action'],x['horizon']) for x in v['trace']] for u,v in zip(a,b))
    same_led=all([(x['entry_bar'],x['exit_bar'],x['reason'],x['direction'],x['horizon']) for x in u['ledger']]==[(x['entry_bar'],x['exit_bar'],x['reason'],x['direction'],x['horizon']) for x in v['ledger']] for u,v in zip(a,b))
    ratios=[y['units']/x['units'] for u,v in zip(a,b) for x,y in zip(u['ledger'],v['ledger'])]
    qr=[y['q']/x['q'] for u,v in zip(a,b) for x,y in zip(u['ledger'],v['ledger'])]
    out[f'P2P3|{split}']=dict(same_trace=same_tr,same_ledger_events=same_led,units_ratio_min=min(ratios),units_ratio_max=max(ratios),q_ratio_min=min(qr),q_ratio_max=max(qr),
        ret=[np.mean([w['return_pct'] for w in a]),np.mean([w['return_pct'] for w in b])])
# H games at P3 (diagnostic representative)
dk=rep['diagnostic_representative']; g=G[dk[:12]]
full=evaluate(cal,g); phi=sum(full['objectives']); allp={(h,d) for h in HORIZONS for d in (1,-1)}
dev=[]
for pair in sorted(allp):
    r=evaluate(cal,g,allp-{pair}); dev.append(dict(disabled=list(pair),gain=sum(r['objectives'])-phi,eligible=r['eligible'],objectives=r['objectives']))
mats=[]
for h in HORIZONS:
    o={}
    for bu in (0,1):
        for se in (0,1):
            en=({(h,1)} if bu else set())|({(h,-1)} if se else set()); o[bu,se]=evaluate(cal,g,en)['objectives']
    mats.append({f'{bu}{se}':o[bu,se] for (bu,se) in o})
stored={tuple(x['disabled']):x['unilateral_gain'] for x in rep['unilateral_deviations']}
out['H']=dict(representative=dk[:12],is_P3=dk.startswith('18c72e613b22'),deviations=dev,
    max_dev_diff_vs_report=max(abs(x['gain']-stored[tuple(x['disabled'])]) for x in dev),
    epsilon=max(0.,max(x['gain'] for x in dev)),epsilon_report=rep['epsilon_on_off'],epsilon_feasible_report=rep['epsilon_activity_feasible'],
    epsilon_feasible=max([0.]+[x['gain'] for x in dev if x['eligible']]),
    matrices=mats,matrix_max_diff=max(abs(a-b) for m,sm in zip(mats,rep['payoff_matrices']) for c in sm['cells'] for a,b in zip(m[f"{c['buy']}{c['sell']}"],c['account_objectives'])))
# paired bootstrap P1 - only_3
rng=np.random.default_rng(20261029)
for split in ('calibration','confirmation'):
    a=np.array([w['return_pct'] for w in R[key['P1']][split]['worlds']]); b=np.array([w['return_pct'] for w in R['only_3'][split]['worlds']])
    d=a-b; idx=rng.integers(0,12,(100000,12)); bs=d[idx].mean(1)
    out[f'BOOT|{split}']=dict(diffs=d.tolist(),mean=float(d.mean()),ci95=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
        positive_weeks=int((d>0).sum()),zero_weeks=int((d==0).sum()),negative_weeks=int((d<0).sum()),
        dd_p1=[w['drawdown_pct'] for w in R[key['P1']][split]['worlds']],dd_o3=[w['drawdown_pct'] for w in R['only_3'][split]['worlds']],
        specs=[w['spec'] for w in R['only_3'][split]['worlds']])
def obj(n,s): return R[n][s]['objectives']
out['DOM']=dict(P2=obj(key['P2'],'confirmation'),only_6=obj('only_6','confirmation'),P3=obj(key['P3'],'confirmation'),greedy=obj('greedy','confirmation'),
   all_conf={ALIAS.get(k[:12],k):R[k]['confirmation']['objectives']+[R[k]['confirmation']['eligible']] for k in R})
print(json.dumps(out,indent=1,default=float))
