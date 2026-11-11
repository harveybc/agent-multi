"""Outcome-blind boundary sampling and shared-head gradient measurements."""
from collections import defaultdict

import numpy as np
import torch


def choose_boundary_panel(rows):
    years=defaultdict(list);seen=set()
    for r in rows:
        z=np.asarray(r['logits'],dtype=float)
        if (r['timestamp'] in seen or r['year'] not in range(2020,2024)
                or r['position'] not in (-1,0,1) or type(r['allowed']) is not bool
                or z.shape!=(3,) or not np.isfinite(z).all()):
            raise ValueError('BOUNDARY_TRACE')
        seen.add(r['timestamp']);years[r['year']].append(r)
    chosen={}
    def add(r,kind):
        if r['timestamp'] not in chosen:chosen[r['timestamp']]=dict(r,boundary_strata=[])
        chosen[r['timestamp']]['boundary_strata'].append(kind)
    for year,group in sorted(years.items()):
        flat=[r for r in group if r['position']==0 and r['allowed']]
        for direction in (1,2):
            for winner in (0,direction):
                pool=[r for r in flat if int(np.argmax(r['logits']))==winner]
                pool.sort(key=lambda r:(abs(r['logits'][direction]-r['logits'][0]),r['timestamp']))
                selected=[]
                for r in pool:
                    if all(abs(r['timestamp']-t)>=24*3600*10**9 for t in selected):
                        add(r,f'boundary_{direction}_winner_{winner}');selected.append(r['timestamp'])
                    if len(selected)==6:break
        far=sorted((r for r in flat if r['timestamp'] not in chosen),
                   key=lambda r:(-min(abs(r['logits'][d]-r['logits'][0]) for d in (1,2)),r['timestamp']))
        for r in far[:4]:add(r,'far_flat')
        rng=np.random.default_rng(282+year)
        for kind,pos,count in [('restricted',0,2),('holding_long',1,4),('holding_short',-1,4)]:
            pool=sorted((r for r in group if r['position']==pos and (pos!=0 or not r['allowed'])),key=lambda r:r['timestamp'])
            for i in rng.permutation(len(pool))[:count]:add(pool[int(i)],kind)
    return sorted(chosen.values(),key=lambda r:r['timestamp'])


def margin_derivatives(logits, parameters, value_loss, entropy_loss):
    if logits.ndim!=2 or logits.shape[-1]!=3 or not torch.isfinite(logits).all():
        raise ValueError('BOUNDARY_GRADIENT_LOGITS')
    def flat(gs):return torch.cat([g.detach().flatten().double() for g in gs])
    gv=flat(torch.autograd.grad(value_loss,parameters,retain_graph=True))
    ge=flat(torch.autograd.grad(entropy_loss,parameters,retain_graph=True))
    gn=flat(torch.autograd.grad(value_loss+entropy_loss,parameters,retain_graph=True))
    if not torch.isfinite(torch.stack([gv,ge,gn])).all():raise ValueError('BOUNDARY_GRADIENT_FINITE')
    torch.testing.assert_close(gv+ge,gn,rtol=2e-5,atol=2e-5)
    output={name:[] for name in ('value','entropy','total')}
    for row in range(len(logits)):
        values={name:[] for name in output}
        for direction in (1,2):
            gm=flat(torch.autograd.grad(logits[row,direction]-logits[row,0],parameters,retain_graph=True))
            for name,g in [('value',gv),('entropy',ge),('total',gn)]:values[name].append(float(-gm.dot(g)))
        for name in output:output[name].append(values[name])
    return dict(output,norms=dict(value=float(gv.norm()),entropy=float(ge.norm()),total=float(gn.norm())),
                decomposition_max_error=float((gv+ge-gn).abs().max()),
                meaning='infinitesimal shared-head SGD margin change; actual AdamW measured separately')


def pressure_summary(records, gradient, actual):
    n=len(records)
    expected=[r['row'] for r in records]
    if gradient['rows']!=expected or actual['rows']!=expected:
        raise ValueError('BOUNDARY_PRESSURE_ROWS')
    q,y,cash=[np.asarray(gradient[k],dtype=float) for k in ('q','native_reward','native_cash')]
    if any(a.shape!=(n,3) or not np.isfinite(a).all() for a in (q,y,cash)):
        raise ValueError('BOUNDARY_PRESSURE_TARGET')
    changes={k:np.asarray(gradient[k],dtype=float) for k in ('value','entropy','total')}
    changes['adam_margin']=np.asarray(actual['margin_delta'],dtype=float)
    if any(v.shape!=(n,2) or not np.isfinite(v).all() for v in changes.values()):
        raise ValueError('BOUNDARY_PRESSURE_VALUES')
    gap=y[:,1:]-y[:,:1];qgap=q[:,1:]-q[:,:1];cashgap=cash[:,1:]-cash[:,:1]
    result={}
    for role in ('fit','monitor'):
        for state in ('all','flat_allowed','flat_restricted','long','short'):
            ix=[]
            for i,r in enumerate(records):
                kind=('long' if r['position']==1 else 'short') if r['position'] else ('flat_allowed' if r['allowed'] else 'flat_restricted')
                if r['role']==role and state in ('all',kind):ix.append(i)
            if not ix:continue
            truth=gap[ix];positive=truth>1e-8;negative=truth< -1e-8
            result[f'{role}:{state}']=dict(rows=len(ix),beneficial=int(positive.sum()),harmful=int(negative.sum()),
                q_favors_harmful=int((negative & (qgap[ix]>1e-8)).sum()),
                reward_cash_opposed=int(((truth*cashgap[ix]<0)&(np.abs(truth)>1e-8)&(np.abs(cashgap[ix])>1e-6)).sum()),
                pressure={name:dict(favors_harmful=int((negative & (v[ix]>1e-10)).sum()),
                    suppresses_helpful=int((positive & (v[ix]< -1e-10)).sum()),
                    favors_helpful=int((positive & (v[ix]>1e-10)).sum())) for name,v in changes.items()})
    return result
