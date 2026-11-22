"""Prepare verified pairs, distribute fits, consume each result and gate integration."""
import fcntl
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import torch
import role429_worker as w
from role421_sell_value import loss, output
from role424_sell_bias import SOURCE, base, common, panel, readout, score, selected, atomic
from role426_sell_rank_control import contrasts, weighted
from role427_sell_rank_stages import OUT as PRIOR

OUT=Path('/var/tmp/role429-sell-ranking')
TOOLS=Path(__file__).resolve().parent
PY='/var/tmp/discrete-sac-probe-venv/bin/python'
HOST=os.environ.get('ROLE429_SSH_TARGET')
REMOTE='/var/tmp/role429-sell-ranking-worker'


def pretest():
    torch.manual_seed(429)
    z=torch.randn(200,12,requires_grad=True);y=torch.randn(200,4)
    torch.testing.assert_close(w.severity(z,y),loss('severity',z,y))
    torch.testing.assert_close(w.heads(z)[-1],output('severity',z,None))
    groups=np.arange(200).reshape(10,20)
    p=w.partners(groups,np.random.default_rng(1))
    assert np.all(p//20==np.arange(200)//20) and not np.any(p==np.arange(200))
    shuffled=y[torch.randperm(len(y))]
    def test(s):return w.pairloss(s,s[p],y,y[p]).item()
    assert test(y)<test(shuffled)<test(-y)
    a=torch.randn_like(y,requires_grad=True)
    w.pairloss(a,a[p],y,y).backward()
    assert not a.grad.any()


def outcomes(net,scale,bank,year,seed):
    x,t,valid=bank
    ids,_,_,y,z=common(year,seed,t)
    dates=t[ids];ts=z['origin_ns'];ii=np.searchsorted(t,ts);ix=np.searchsorted(ts,dates)
    assert np.array_equal(t[ii],ts) and np.array_equal(ts[ix],dates) and valid[ii].all()
    pn=panel(year,seed,*bank);assert np.array_equal(pn.p,z['p'])
    jul=base.ns(f'{year}-07-01');ev=dates>=jul
    p=readout(net,x[ii],scale)[0]
    res=score(p,pn,dates,ix,y,jul,ts,True)
    r=contrasts(p[ix[ev]],y[ev],dates[ev])
    assert abs(res['bp']-weighted(r,'model'))<.001
    res.update(random_bp=weighted(r,'control'),scoring_weeks=int(np.unique(pn.e.seg_of[pn.bars[ts>=jul]]).size))
    return res,p[ix[ev]],dates[ev],y[ev]


def prepare(year,seed,bank):
    x,t,valid=bank;past=[common(yr,seed,t) for yr in range(2020,year)]
    i,end,y=[np.concatenate([r[k] for r in past]) for k in (0,1,3)]
    cut=base.ns(f'{year-1}-07-01')
    tr=(t[i]<cut)&(end<cut);mon=(t[i]>=cut)&(end<base.ns(f'{year}-01-01'))
    assert valid[i].all() and end[tr].max()<t[i[mon]].min()
    scale=float(y[tr].std())
    tag=f'{year}-{seed}'
    ck=torch.load(SOURCE/f'{tag}-shared8.pt',map_location='cpu',weights_only=True)
    assert np.isclose(ck['scale'],scale)
    net=w.model();net.load_state_dict(ck['weights'])
    s,*_=outcomes(net,scale,bank,year,seed)
    ref=base.read(SOURCE/f'{tag}.json')['arms']['shared8']['positive']
    assert ref['account']==s['account'] and abs(s['bp']-ref['bp'])<.001
    r=prior[year,seed]
    assert abs(s['bp']-weighted(r,'model'))<.001 and abs(s['random_bp']-weighted(r,'control'))<.001
    path=OUT/f'{tag}.npz'
    if not path.exists():
        np.savez(path,x=x[i[tr]],y=(y[tr]/scale).astype('f'),v=x[i[mon]],t=(y[mon]/scale).astype('f'),
                 week=(t[i[tr]]//86400000000000+3)//7,seed=seed)
    atomic(OUT/f'{tag}-preflight.json',dict(fit=int(tr.sum()),monitor=int(mon.sum()),scale=scale,
                                           reference=s,sha=base.file_sha(path)))
    return path,scale


def run(args,**kw):
    subprocess.run(args,check=True,**kw)


def cell(year,seed,device,remote,bank):
    tag=f'{year}-{seed}';result=OUT/f'{tag}.json'
    if result.exists():return base.read(result)
    path,scale=prepare(year,seed,bank)
    sha=base.file_sha(path)
    prefix=['env',f'CUDA_VISIBLE_DEVICES={device}','CUBLAS_WORKSPACE_CONFIG=:4096:8','OMP_NUM_THREADS=1']
    with (OUT/f'{tag}.log').open('a',buffering=1) as log:
        if remote:
            run(['rsync','-a',str(path),f'{HOST}:{REMOTE}/'])
            cmd=[*prefix,PY,f'{REMOTE}/role429_worker.py',f'{REMOTE}/{path.name}',sha]
            run(['ssh','-o','ServerAliveInterval=15',HOST,shlex.join(cmd)],stdout=log,stderr=subprocess.STDOUT)
            run(['rsync','-a',f'{HOST}:{REMOTE}/{tag}.pt',str(OUT)+'/'])
        else:
            run([*prefix,PY,str(TOOLS/'role429_worker.py'),str(path),sha],stdout=log,stderr=subprocess.STDOUT)
    d=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
    assert d['sha']==sha
    arms=[];scores=[]
    for state in d['champs']:
        net=w.model();net.load_state_dict(state)
        s,p,dates,y=outcomes(net,scale,bank,year,seed)
        s['relative_weight_change']={k:float((v-d['initial'][k]).norm()/d['initial'][k].norm().clamp_min(1e-12))
                                     for k,v in state.items()}
        s['trades_per_scoring_week']=s['account']['trades']/s['scoring_weeks']
        arms.append(s);scores.append(p)
    # Evaluate order changes on identical non-self pairs, within week/horizon.
    wk=(dates//86400000000000+3)//7
    q=w.partners([np.flatnonzero(wk==v) for v in np.unique(wk)],np.random.default_rng(429))
    sign=np.sign(y-y[q]);m=sign!=0
    ok=[((p-p[q])*sign)>0 for p in scores]
    a,b=arms
    passed=(b['bp']>max(a['bp'],b['random_bp'],0) and b['account']['ret']>max(0,a['account']['ret'])
            and b['trades_per_scoring_week']>2)
    row=dict(year=year,seed=seed,device=d['device'],epochs=len(d['hist']),best_epochs=d['epochs'],stop=d['stop'],
             control=a,ranking=b,corrected_pairs=int((~ok[0]&ok[1]&m).sum()),damaged_pairs=int((ok[0]&~ok[1]&m).sum()),
             mean_absolute_score_change_bp=float(np.abs(scores[0]-scores[1]).mean()),
             top_changed=int((selected(scores[0],dates)!=selected(scores[1],dates)).sum()),passed=bool(passed))
    atomic(OUT/f'{tag}-history.json',d['hist']);atomic(result,row)
    return row


def lane(jobs,device,remote,bank):
    return [cell(y,s,device,remote,bank) for y,s in jobs]


def main():
    global prior
    torch.set_num_threads(1);OUT.mkdir(exist_ok=True)
    lock=(OUT/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pretest()
    market_sha=base.file_sha(base.DEST/'market.npz')
    assert market_sha==base.read(base.DEST/'CACHE.json')['sha']
    config=dict(code={k:base.file_sha(TOOLS/k) for k in ('role429_worker.py','role429_campaign.py','role405_plateau.py')},
                market_sha=market_sha,rule='429_SPECIALIST_CHECKPOINT',scope='research only')
    if (OUT/'CONFIG.json').exists():assert config==base.read(OUT/'CONFIG.json')
    atomic(OUT/'CONFIG.json',config)
    with np.load(base.DEST/'market.npz') as z:bank=z['x'],z['t'],z['valid']
    rows=base.read(PRIOR/'WEEKS.json')
    prior={(y,s):[r for r in rows if (r['year'],r['seed'],r['stage'],r['arm'])==(y,s,'held','shared8')]
           for y in (2022,2023) for s in (7,19,43)}
    if not HOST or HOST.startswith('-'):
        raise SystemExit('Set ROLE429_SSH_TARGET to the trusted worker SSH alias.')
    run(['ssh',HOST,f'mkdir -p {REMOTE}'])
    run(['rsync','-a',str(TOOLS/'role429_worker.py'),str(TOOLS/'role405_plateau.py'),f'{HOST}:{REMOTE}/'])
    with ThreadPoolExecutor(3) as pool:
        plans=[([(2023,7),(2023,43)],1,True), ([(2023,19),(2022,43)],0,True), ([(2022,7),(2022,19)],0,False)]
        runs=[pool.submit(lane,*plan,bank) for plan in plans]
        done=[r for f in as_completed(runs) for r in f.result()]
    assert len(done)==6
    atomic(OUT/'DONE.json',dict(pairs=6,passed=sum(r['passed'] for r in done),
                                decision='qualify_for_integration_only' if all(r['passed'] for r in done)
                                else 'reject_for_integration',promoted=False))


if __name__=='__main__':
    main()
