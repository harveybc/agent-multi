"""Attribute the semester-7 switch using archived policies and the native validation ledger."""
import gzip
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role448_worker import Sell, attach_engine, bids, emit, file_sha, heads


@torch.inference_mode()
def main(seed=7):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path('/var/tmp/role448-temporal-4plus1')
    archived=seed in (19,43)
    assert archived or seed==7
    run=Path('/var/tmp/role450-full-policy' if archived else '/var/tmp/role451-activity')/f'run-{seed}'
    read=lambda p:json.loads(p.read_text())
    manifest=root/f'{seed}.json'
    spec=read(manifest)
    for name,key in [(f'{seed}.npz','inputs'),(f'{seed}.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/name)==spec[key]
    assert torch.cuda.get_device_name()==read(run/'PREFLIGHT.json')['device']
    attach_engine(root/'engine')
    with gzip.open(root/f'{seed}.sim.gz','rb') as f:panel,ref=cloudpickle.load(f)[2024]
    assert panel.account(panel.raw)==ref
    with np.load(root/f'{seed}.npz') as z:
        x=torch.from_numpy(z['x8']).cuda()
        ix=z['i8']
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    preds,refs,vals,changed,early={},{},{},{},{}
    stages=list(range(8)) if archived else [5,6]
    prev=None
    for s in stages:
        name=f's{s}-{"validation" if s else "common"}'
        checkpoint=torch.load(run/f'{name}.pt',map_location='cpu',weights_only=False)
        assert checkpoint['sig']==file_sha(manifest)
        net.load_state_dict(checkpoint['champ']['weights'])
        preds[s]=torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
        refs[s]=checkpoint['champ']['reference']
        vals[s]=read(run/f'{name}.json')['val']
        weights=checkpoint['champ']['weights']
        if prev is not None:changed[s]=any(not torch.equal(weights[k],prev[k]) for k in weights)
        prev=weights
        early[s+1]=sum(r['val']['ret']>0 and r['val']['trades']>2*r['val']['weeks']
                       for r in checkpoint['history'] if r['epoch']>0)
    rows=[]
    for old,new in zip(stages,stages[1:]):
        accounts={}
        for w in (old,new):
            for c in (old,new):
                scores=np.zeros_like(panel.raw)
                scores[ix,1,:]=bids(preds[w],refs[c])
                a=panel.account(scores)
                assert a['buy']==0
                if w==c:assert a==vals[w]
                accounts[f'weights_{w+1}_table_{c+1}']=a
        rows.append(dict(semester=new+1,weights_changed=changed[new],
                         table_changed=not np.array_equal(np.sort(refs[old],axis=0),np.sort(refs[new],axis=0)),
                         accounts=accounts))
    if archived:
        out=Path('/var/tmp/role452-cross')
        out.mkdir(exist_ok=True)
        emit(out/f'{seed}.json',dict(seed=seed,manifest=file_sha(manifest),rows=rows,active_positive_epochs=early))
    else:emit(run/'CROSS_6_7.json',rows[0]['accounts'])


if __name__=='__main__':main(int(sys.argv[1]) if len(sys.argv)>1 else 7)
