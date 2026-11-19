"""Cold CPU decisions and complete 2023 conditional account replays."""
from collections import Counter
from math import fsum
import os
from pathlib import Path
import numpy as np
import torch
from direct_data385 import context,arguments
from train_direct385 import head
from app.close_surface import scores
from app.frozen_recipe_inputs import file_sha
from docs.work_plan.generalization.close_attribution380 import PINS,equal
from docs.work_plan.generalization.role_surface379 import query,basis
from tools.run_discrete_curve_probe import read,write

def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU only')
    _,_,sig,make,old=context(a)
    receipt=read(a.fit/'DONE.json')
    if receipt['data_sha']!=file_sha(a.data/'DATA.json'):raise ValueError('dataset')
    data=read(a.data/'DATA.json')
    if data['source']!=old['source_sha256']:raise ValueError('source')
    for name,digest in data['files'].items():
        if file_sha(a.data/name)!=digest:raise ValueError('data hash')
    with np.load(a.data/'market.npz') as f:x,ix=f['x'],f['ix']
    actions={};surfaces={};hashes={}
    for role,k in [('flat',3),('long',2),('short',2)]:
        loc=a.fit/role
        net=head(x.shape[1],k);net.load_state_dict(torch.load(loc/'head.pt',map_location='cpu',weights_only=True));net.eval()
        hashes[role]=file_sha(loc/'head.pt')
        with np.load(loc/'apply.npz') as f,torch.inference_mode():
            probs=(net(torch.tensor((x-f['mu'])/f['sd']))/float(f['T'])).softmax(1).numpy()
        dec=probs.argmax(1)
        with np.load(a.data/(role+'.npz')) as d:
            mask=d['split']==3;idx=d['idx'][mask];y=d['y'][mask]
            got=np.bincount(dec[idx],minlength=k).tolist();stats=read(loc/'RESULT.json')
            if got!=stats['decisions']:raise ValueError('cold decision count')
            equal(float(-np.log(probs[idx,y]).mean()),stats['nll'])
        actions[role]=dict(zip(ix.tolist(),dec.tolist()))
    for side in (-1,1):
        name=f'HEAD_{side}.npz'
        if file_sha(a.table/name)!=PINS[name]:raise ValueError('table hash')
        with np.load(a.table/name) as f:surfaces[side]=dict(f)
    accounts={}
    for opens,closes in [('source','source'),('direct','source'),('source','direct'),('direct','direct'),('direct','table')]:
        e=make('2023-01-01','2024-01-01');counts=Counter()
        while not e.finished:
            row=int(e.rows[e.bar]);cmd=int(sig[row]);p=e.engine.pos
            if not p and opens=='direct':
                cmd=actions['flat'][row] if e.engine.entry_gate(e.bar)[0] else 0
            elif p and closes!='source':
                side=p.direction
                close=actions['long' if side==1 else 'short'][row]==1 if closes=='direct' else (
                    scores(surfaces[side],basis(e)[None,:],query(e)[None,:])[1][0]>0)
                cmd=(2 if side==1 else 1) if close else 0
            if not p and e.engine.entry_gate(e.bar)[0]:counts[str(cmd)]+=1
            e.step(cmd)
        tr=e.engine.trades
        net=e.engine.balance-e.engine.cfg.capital
        equal(net,fsum(t['net_usd'] for t in tr))
        eq=np.array([v for _,v in e.equity_trace])
        accounts[opens+'_'+closes]=dict(net_usd=float(net),ret=100*net/e.engine.cfg.capital,
            dd=float(100*max(1-eq/np.maximum.accumulate(eq))),trades=len(tr),eligible_flat_signals=dict(counts),
            buy=sum(int(t['direction']==1) for t in tr),wins=sum(int(t['net_usd']>0) for t in tr),
            reasons=dict(Counter(t['close_reason'] for t in tr)))
    if file_sha(a.table/'RESULT.json')!=PINS['RESULT.json']:raise ValueError('source benchmark identity')
    equal(accounts['source_source']['net_usd'],read(a.table/'RESULT.json')['accounts']['source']['net_usd'])
    write(a.out,dict(accounts=accounts,weights=hashes,fit=receipt))
    print(accounts,flush=True)

if __name__=='__main__':
    p=arguments()
    for k in ('data','fit','table'):p.add_argument('--'+k,required=True,type=Path)
    run(p.parse_args())
