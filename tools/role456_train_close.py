"""Frozen active SELL book; expanding past-only linear veto of voluntary closes."""
import gzip
import json
import time
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role448_worker import Sell, attach_engine, bids, emit, file_sha, heads
from tools.role404_data import HS,HOUR
from tools.role407_openings import decisions
from tools.role454_paired_hold import paired_hold


def context(e,b,j):
    p=e.pos;now=int(e.v[b])+HOUR
    quote=e.quote_price(b,e.c[b],-p.direction,now)
    risk=abs(p.entry_price-p.sl)
    return [p.direction*(quote-p.entry_price)/risk,abs(p.tp-p.entry_price)/risk,
            np.log1p(b-p.entry_bar),np.log1p(max(0,(int(e.ff[e.seg_of[b]])-now)/HOUR)),
            *[float(j==i) for i in range(len(HS))]]


def fit(past):
    x=np.array([r[0] for r in past]);y=np.array([r[1] for r in past])
    mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd<1e-8]=1
    z=(x-mu)/sd
    w=np.linalg.solve(z.T@z+np.eye(x.shape[1]),z.T@(y-y.mean()))
    return lambda row:float(y.mean()+((np.array(row)-mu)/sd)@w)


@torch.inference_mode()
def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    root=Path('/var/tmp/role448-temporal-4plus1')
    run=Path('/var/tmp/role451-activity/run-7')
    out=Path('/var/tmp/role456-train-close');out.mkdir(exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    spec=read(root/'7.json')
    for n,k in [('7.npz','inputs'),('7.sim.gz','sim'),('recipe.json','recipe'),('arch.json','arch')]:
        assert file_sha(root/n)==spec[k]
    assert torch.cuda.get_device_name()==read(run/'PREFLIGHT.json')['device']
    ck=torch.load(run/'s7-validation.pt',map_location='cpu',weights_only=False)
    assert ck['sig']==file_sha(root/'7.json')
    attach_engine(root/'engine')
    with gzip.open(root/'7.sim.gz','rb') as f:panels=cloudpickle.load(f)
    net=Sell(read(root/'recipe.json'),read(root/'arch.json')).cuda().eval()
    net.load_state_dict(ck['champ']['weights'])
    prior=read(run/'s7-validation.json')['prior'];past=[];rows=[]
    with np.load(root/'7.npz') as z:
        for s in range(8):
            start=time.monotonic()
            predict=fit(past) if past else None
            panel=panels[2020+s//2][0];ix=z[f'i{s}'];x=torch.from_numpy(z[f'x{s}']).cuda()
            p=torch.cat([heads(net(b))[-1] for b in x.split(512)]).cpu().numpy()
            scores=np.zeros_like(panel.raw);scores[ix,1,:]=bids(p,ck['champ']['reference'])
            d,h=decisions(scores);horizon={int(panel.bars[i]):int(h[i]) for i in np.flatnonzero(d)}
            native=panel.close;features={}
            def capture(e,b,j):
                cmd=native(e,b,j)
                if cmd:features[e.pos.entry_bar]=context(e,b,j)
                return cmd
            panel.close=capture
            try:base=panel.account(scores)
            finally:del panel.close
            if s<len(prior):assert base==prior[s]
            diag,pairs=paired_hold(panel,scores,horizon,base,list(panel.e.trades))
            vetoed=set()
            def gated(e,b,j):
                if e.pos.entry_bar in vetoed:return 0
                cmd=native(e,b,j)
                if cmd and predict(context(e,b,j))<0:
                    vetoed.add(e.pos.entry_bar)
                    return 0
                return cmd
            test=None
            if predict is not None:
                panel.close=gated
                try:test=panel.account(scores)
                finally:del panel.close
            dropped=[r['delta_usd'] for r in pairs if predict is not None and predict(features[r['entry_bar']])<0]
            row=dict(stage=s+1,decisions=len(ix),prior_pairs=len(past),pairs=len(pairs),base=base,
                     hold=diag['broker_only'],trial=test,paired_vetoes=len(dropped),
                     avoided_harm=-sum(min(v,0) for v in dropped),lost_protection=sum(max(v,0) for v in dropped),
                     seconds=time.monotonic()-start)
            rows.append(row)
            for r in pairs:
                risk=r['units']*abs(r['price']-r['sl'])
                assert risk>0 and np.datetime64(r['hold_exit'][:19],'ns').astype('int64')<spec['blocks'][str(s)]['end']
                past.append((features[r['entry_bar']],r['delta_usd']/risk))
            emit(out/'PROGRESS.json',row)
    emit(out/'DONE.json',dict(stages=rows,checkpoint=file_sha(run/'s7-validation.pt'),promoted=False))


if __name__=='__main__':main()
