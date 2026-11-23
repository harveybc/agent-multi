"""Replay each matched checkpoint; deduplicate identical SELL decisions within seed."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role448_worker import Sell, attach_engine, bids, emit, file_sha, heads
from tools.role407_openings import decisions
from tools.role453_sell_audit import csv_rows
from tools.role454_paired_hold import paired_hold


@torch.inference_mode()
def main(seed):
    assert seed in (7,19,43)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path('/var/tmp/role448-temporal-4plus1')
    out=Path('/var/tmp/role455-matched-close')/str(seed);out.mkdir(exist_ok=True,parents=True)
    read=lambda p:json.loads(p.read_text())
    manifest=root/f'{seed}.json';sig=file_sha(manifest);spec=read(manifest)
    for n,k in [(f'{seed}.npz','inputs'),(f'{seed}.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/n)==spec[k]
    attach_engine(root/'engine')
    with gzip.open(root/f'{seed}.sim.gz','rb') as f:panel,ref=cloudpickle.load(f)[2024]
    assert panel.account(panel.raw)==ref
    with np.load(root/f'{seed}.npz') as z:x=torch.from_numpy(z['x8']).cuda();ix=z['i8']
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    catalog={};results={};pairs=[]
    for campaign in (['450-full-policy','451-activity'] if seed==7 else ['450-full-policy']):
        run=Path('/var/tmp/role'+campaign)/f'run-{seed}'
        assert torch.cuda.get_device_name()==read(run/'PREFLIGHT.json')['device']
        for path in sorted(run.glob('s*-*.pt')):
            label=campaign[:3]+'/'+path.stem;ck=torch.load(path,map_location='cpu',weights_only=False)
            assert ck['sig']==sig
            net.load_state_dict(ck['champ']['weights'])
            p=torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
            scores=np.zeros_like(panel.raw);scores[ix,1,:]=bids(p,ck['champ']['reference'])
            cmd,h=decisions(scores);assert np.isin(cmd,[0,2]).all()
            routes=np.where(cmd==2,h+1,0).astype(np.uint8)
            key=hashlib.sha256(routes.tobytes()).hexdigest()
            expected=read(path.with_suffix('.json'))['val']
            if key not in results:
                actual=panel.account(scores);assert actual==expected
                horizon={int(panel.bars[i]):int(h[i]) for i in np.flatnonzero(cmd)}
                r,rows=paired_hold(panel,scores,horizon,actual,list(panel.e.trades))
                results[key]=r
                pairs.extend(dict(seed=seed,route=key,**row) for row in rows)
            assert results[key]['factual']==expected
            catalog[label]=key
    assert len(catalog)==(19 if seed==7 else 15)
    if pairs:csv_rows(out/'pairs.csv',pairs)
    emit(out/'DONE.json',dict(seed=seed,manifest=sig,catalog=catalog,results=results))


if __name__=='__main__':main(int(sys.argv[1]))
