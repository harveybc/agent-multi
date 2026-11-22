"""Package real windows, purged labels and parity-checked native accounts."""
import gzip
import json
from pathlib import Path

import cloudpickle
import numpy as np
import torch
from role413_sequence import OLD, OUT as EARLY, labels
from role438_close_control import base, atomic

OUT=Path('/var/tmp/role448-temporal-4plus1')


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    bundle=base.ROOT/'model/61/bundle'
    joined,orig=base.assemble(base.PARENT,bundle,base.ROOT/'history397-61-r2')
    recipe=base.read(bundle/'RECIPE.json')
    assert recipe['split']['train']==['2020-01-01','2024-01-01']
    assert recipe['split']['validation']==['2024-01-01','2025-01-01']
    atomic(OUT/'recipe.json',recipe)
    atomic(OUT/'arch.json',base.read(bundle/'MODEL_CONFIG.json')['architecture'])
    assert base.file_sha(base.DEST/'market.npz')==base.read(base.DEST/'CACHE.json')['sha']
    with np.load(base.DEST/'market.npz') as z:
        assert np.array_equal(z['t'],joined.inputs.row_ts_ns)
        cached=z['x']
    old,_=base.load_bundle(bundle,'cpu')
    risk=base.read(base.PARENT/'RISK.json')
    for seed in (7,19,43):
        arrays={};panels={};spec={}
        for year in range(2020,2025):
            m=orig if year==2024 else joined
            ins=m.inputs; ts=ins.row_ts_ns
            ix=np.flatnonzero(ins.valid&(ts>=base.ns(f'{year}-01-01'))&(ts<base.ns(f'{year+1}-01-01')))
            base.OUT=OLD if year>=2022 else EARLY
            if year<2024: bank=torch.from_numpy(cached[ix])
            else:
                with torch.inference_mode(),old.loss.actor_network_params.to_module(old.actor):
                    enc=old.actor_net.encoder.eval()
                    bank=torch.cat([enc({'features':torch.from_numpy(ins.windows[ix[k:k+256]])}) for k in range(0,len(ix),256)])
            pn=base.Panel(m,year,seed,bank,ix,base.class_means(year),risk)
            z,rc=labels(year,seed)
            assert pn.hashes==rc['spec']['heads'] and np.array_equal(z['p'],pn.p)
            assert np.array_equal(z['bars'],pn.bars)
            positions=pn.envs[0].rows[pn.bars]
            origin=ts[positions]
            assert np.array_equal(origin,z['origin_ns']) and np.array_equal(origin,ins.slot_ts_ns[positions,-1])
            for half in range(1 if year==2024 else 2):
                s=2*(year-2020)+half
                lo=base.ns(f'{year}-07-01' if half else f'{year}-01-01')
                hi=base.ns(f'{year}-07-01' if s<8 and not half else f'{year+1}-01-01')
                # Earliest permissible slot and known session deadline, not future price.
                i=np.flatnonzero((ins.slot_ts_ns[positions,0]>=lo)&(origin<hi)&(pn.e.ff[pn.e.seg_of[pn.bars]]<hi))
                xx=ins.windows[positions[i]]; yy=z['y'][i,1,:]
                assert xx.shape==(len(i),13,61) and np.isfinite(xx).all()
                assert len(i)>1000 and np.isfinite(yy).all(axis=1).sum()>1000
                arrays.update({f'x{s}':xx,f'y{s}':yy.astype('f'),f'i{s}':i,f't{s}':origin[i]})
                spec[s]=dict(start=int(lo),end=int(hi),first=int(origin[i[0]]),last=int(origin[i[-1]]),
                             decisions=len(i),labelled=int(np.isfinite(yy).all(1).sum()),label_sha=rc['sha'])
            reference=pn.account(pn.raw)
            for env in pn.envs: env.inputs=None
            pn.e.reset()
            panels[year]=(pn,reference)
        assert len(spec)==9 and spec[7]['end']==spec[8]['start']
        path=OUT/f'{seed}.npz';np.savez_compressed(path,**arrays)
        sim=OUT/f'{seed}.sim.gz'
        with gzip.open(sim,'wb',compresslevel=1) as f:cloudpickle.dump(panels,f)
        atomic(OUT/f'{seed}.json',dict(inputs=base.file_sha(path),sim=base.file_sha(sim),blocks=spec,
                                       recipe=base.file_sha(OUT/'recipe.json'),arch=base.file_sha(OUT/'arch.json')))
        print(seed,[(s,r['decisions'],r['labelled']) for s,r in spec.items()],flush=True)


if __name__=='__main__':main()
