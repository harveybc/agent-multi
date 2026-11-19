"""Cold-reload each opener and replay the same native account with frozen closes."""
from collections import Counter
from math import fsum
import os
from pathlib import Path
import numpy as np
import torch
from advantage386 import choose, digest, DATA
from direct_data385 import context, arguments
from train_direct385 import head
from docs.work_plan.generalization.close_attribution380 import equal
from tools.run_discrete_curve_probe import read, write

CONTROL = 'fbcc186bf8fe9f88dd751fa4afb4358ba22b88d88c08e7d0c210eef3883976c2'

def account(make, sig, actions, period=('2023-01-01','2024-01-01')):
    env = make(*period)
    flat = Counter()
    while not env.finished:
        row = int(env.rows[env.bar])
        cmd = int(sig[row])
        if not env.engine.pos:
            legal = env.engine.entry_gate(env.bar)[0]
            if actions is not None:
                cmd = actions[row] if legal else 0
            if legal:
                flat[cmd] += 1
        env.step(cmd)
    trades = env.engine.trades
    net = env.engine.balance - env.engine.cfg.capital
    equal(net, fsum(t['net_usd'] for t in trades))
    eq = np.array([v for _, v in env.equity_trace])
    return dict(net_usd=net, ret=100*net/env.engine.cfg.capital,
                dd=float(100*max(1-eq/np.maximum.accumulate(eq))), trades=len(trades),
                flat_nop=flat[0], buy=sum(t['direction']==1 for t in trades),
                wins=sum(t['net_usd']>0 for t in trades),
                reasons=dict(Counter(t['close_reason'] for t in trades)))

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='': raise ValueError('CPU required')
    if digest(a.control_result)!=CONTROL or digest(a.data/'DATA.json')!=DATA:
        raise ValueError('experiment identity')
    control=read(a.control_result)
    _,_,sig,make,old=context(a)
    manifest=read(a.data/'DATA.json')
    if manifest['source']!=old['source_sha256']: raise ValueError('source')
    for name,sha in manifest['files'].items():
        if digest(a.data/name)!=sha: raise ValueError(name)
    with np.load(a.data/'market.npz') as d: x,ix=d['x'],d['ix']
    with np.load(a.data/'flat.npz') as d:
        mask=d['split']==3; held=d['idx'][mask]; values=d['bp'][mask]
    def replay(loc,k):
        net=head(x.shape[1],k)
        net.load_state_dict(torch.load(loc/'head.pt',map_location='cpu',weights_only=True));net.eval()
        with np.load(loc/'apply.npz') as f,torch.inference_mode():
            raw=net(torch.tensor((x-f['mu'])/f['sd'])).numpy()
            if k==3: return raw.argmax(1)
            decision=choose(raw*float(f['scale'])+f['offset'])
            np.testing.assert_array_equal(decision[held],f['held_actions'])
            return decision,raw*float(f['scale'])+f['offset']
    accounts={'source':account(make,sig,None)}
    if digest(a.control_fit/'flat/head.pt')!=control['weights']['flat']:raise ValueError('classifier')
    classifier=replay(a.control_fit/'flat',3)
    arms={'classifier':classifier}
    calibration_period=('2022-07-01','2023-01-01')
    veto={'source':account(make,sig,None,calibration_period)}
    for seed in a.seeds:
        loc=a.fit/str(seed);receipt=read(loc/'RESULT.json')
        if (receipt['seed'],receipt['data'],receipt['head'],receipt['apply'])!=(
            seed,DATA,digest(loc/'head.pt'),digest(loc/'apply.npz')):raise ValueError('weights')
        arms[str(seed)],pred=replay(loc,2)
        source=sig[ix].astype(int)
        keep=((source==1)&(pred[:,0]>0))|((source==2)&(pred[:,1]>0))
        filtered=np.where(keep,source,0)
        veto[str(seed)]=account(make,sig,dict(zip(ix.tolist(),filtered.tolist())),calibration_period)
    local={}
    for name,policy in arms.items():
        accounts[name]=account(make,sig,dict(zip(ix.tolist(),policy.tolist())))
        selected=values[np.arange(len(held)),policy[held]]
        incumbent=sig[ix[held]].astype(int)
        reference=values[np.arange(len(held)),incumbent]
        change={}
        for label,mask in (
            ('extra_entry',(incumbent==0)&(policy[held]!=0)),
            ('skipped_entry',(incumbent!=0)&(policy[held]==0)),
            ('opposite_entry',(incumbent!=0)&(policy[held]!=0)&(policy[held]!=incumbent)),
            ('unchanged',incumbent==policy[held])):
            change[label]=dict(n=int(mask.sum()),mean_delta_bp=(
                float((selected[mask]-reference[mask]).mean()) if mask.any() else None))
        if sum(v['n'] for v in change.values())!=len(held):raise ValueError('unmatched decision')
        local[name]=dict(mean_adv_bp=float(selected.mean()),
                         regret_bp=float((values.max(1)-selected).mean()),
                         n=int(len(held)),paired_against_source=change)
    for name,ref in [('source','source_source'),('classifier','direct_source')]:
        for key in ('net_usd','dd','trades','wins','buy'):
            equal(accounts[name][key],control['accounts'][ref][key])
    write(a.out,dict(accounts=accounts,local=local,data=DATA,control=CONTROL,
                     exploratory_veto_period=calibration_period,exploratory_veto=veto))
    print(accounts,flush=True)

if __name__=='__main__':
    p=arguments()
    for k in ('fit','data','control-fit','control-result'):p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--seeds',type=int,nargs='+',default=[19,7,43])
    run(p.parse_args())
