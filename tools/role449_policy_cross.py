"""Replay frozen calibration tables crossed with inherited and selected SELL weights."""
import gzip
import json
import sys
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role448_worker import Sell,attach_engine,bids,emit,file_sha,heads
from tools.role407_openings import decisions


def run(root,seed,out):
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    attach_engine(root/'engine')
    read=lambda p:json.loads(p.read_text())
    src=root/f'run-{seed}'; spec=read(root/f'{seed}.json')
    sig=file_sha(root/f'{seed}.json')
    assert read(src/'PREFLIGHT.json')['manifest']==sig and read(src/'DONE.json')['stages']==8
    assert read(src/'PREFLIGHT.json')['device']==torch.cuda.get_device_name()
    for name,k in [(f'{seed}.npz','inputs'),(f'{seed}.sim.gz','sim'),('arch.json','arch'),('recipe.json','recipe')]:
        assert file_sha(root/name)==spec[k]
    with np.load(root/f'{seed}.npz') as z:
        x=[torch.from_numpy(z[f'x{s}']).cuda() for s in range(9)]
        indices=z['i8']
    with gzip.open(root/f'{seed}.sim.gz','rb') as f:pn,ref=cloudpickle.load(f)[2024]
    assert pn.account(pn.raw)==ref
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    archive=out.parent/f'policies-{seed}';archive.mkdir(exist_ok=True)
    def surfaces(name,stage,extra):
        d=torch.load(src/f'{name}.pt',map_location='cpu',weights_only=False)
        assert d['sig']==sig
        weights=d['champ']['weights'];net.load_state_dict(weights)
        @torch.inference_mode()
        def pred(k):return torch.cat([heads(net(b))[-1] for b in x[k].split(512)]).cpu().numpy()
        cal=pred(stage)
        err=float(np.max(np.abs(np.quantile(cal,.9,axis=0)-read(src/f'{name}.json')['cut'])))
        assert err<1e-6,(name,err)
        table=archive/f'{name}.npz';checkpoint=src/f'{name}.pt'
        if not table.exists():
            tmp=table.with_suffix('.tmp.npz')
            np.savez_compressed(tmp,reference=np.sort(cal,axis=0),weight_file_sha=file_sha(checkpoint),
                                source_manifest=sig,stage=stage,policy='448_positive_q90_right_rank')
            tmp.replace(table)
        val=pred(8)
        with np.load(table) as saved:
            assert str(saved['weight_file_sha'])==file_sha(checkpoint) and str(saved['source_manifest'])==sig
            assert np.array_equal(saved['reference'],np.sort(cal,axis=0))
            assert np.array_equal(bids(val,cal),bids(val,saved['reference']))
        return weights,val,cal,pred(extra) if extra is not None else None,d['history'][0]['val'],err
    def replay(p,reference):
        scores=np.zeros_like(pn.raw);scores[indices,1,:]=bids(p,reference)
        ac=pn.account(scores)
        d,h=decisions(scores)
        trades={int(t['entry_bar_abs']) for t in pn.e.trades}
        assert len(trades)==ac['trades'] and ac['buy']==0
        return ac,np.where(d!=0,h+1,0),trades
    def compare(a,b):
        _,before,old=a;_,after,new=b
        return dict(enabled=int(((before==0)&(after!=0)).sum()),disabled=int(((before!=0)&(after==0)).sum()),
                    horizon=int(((before!=0)&(after!=0)&(before!=after)).sum()),
                    new_trades=len(new-old),lost_trades=len(old-new))
    rows=[]
    for arm in ('retention','validation'):
        for s in range(1,8):
            parent=f's{s-1}-{"common" if s==1 else arm}';child=f's{s}-{arm}'
            wa,pa,ra,at_new,_,ea=surfaces(parent,s-1,s)
            wb,pb,rb,_,entry,eb=surfaces(child,s,None)
            same=all(torch.equal(v,wb[k]) for k,v in wa.items())
            a,b,c,d,e=[replay(p,r) for p,r in [(pa,ra),(pa,rb),(pb,ra),(pb,rb),(pa,at_new)]]
            assert a[0]==read(src/f'{parent}.json')['val'] and d[0]==read(src/f'{child}.json')['val']
            assert e[0]==entry
            if same:
                assert a[0]==c[0] and b[0]==d[0]==e[0] and np.array_equal(pa,pb)
            wr=.5*((c[0]['ret']-a[0]['ret'])+(d[0]['ret']-b[0]['ret']))
            cr=.5*((b[0]['ret']-a[0]['ret'])+(d[0]['ret']-c[0]['ret']))
            assert abs(wr+cr-(d[0]['ret']-a[0]['ret']))<1e-10
            rows.append(dict(seed=seed,arm=arm,stage=s+1,equal_weights=same,max_cut_error=max(ea,eb),
                             accounts=dict(zip(('parent','cal_only','weights_only','child','epoch_zero'),[r[0] for r in (a,b,c,d,e)])),
                             calibration_changes=compare(a,b),weight_changes=compare(a,c),pure_segment_changes=compare(a,e),
                             weights_pp=wr,calibration_pp=cr))
    emit(out,rows)


if __name__=='__main__':run(Path(sys.argv[1]),int(sys.argv[2]),Path(sys.argv[3]))
