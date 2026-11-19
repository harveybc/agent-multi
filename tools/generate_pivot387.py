"""Produce nested 4/8/16 TRAIN-year cohorts from pivot-conditioned scenarios."""
import argparse
import hashlib
import json
import subprocess
import types
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
from app.pivot_segment_generator import generate,match,pivots_by_chunk
from app.frozen_recipe_inputs import FrozenRecipeInputs,file_sha
from app.g3_joint_sampler import innovations,distribution_gate
from app.g3_producer_audit import parity
from app.structural_refit import load_bundle
from tools.audit_g3_producers import authenticate,load_external,producers,rebuild,write


def fused(model,inputs):
    idx=np.flatnonzero(inputs.valid & (inputs.period=='TRAIN'));parts=[]
    net=model.actor_net; enc=net.encoder
    net.eval()
    with torch.inference_mode(),model.loss.actor_network_params.to_module(model.actor):
        for batch in np.array_split(idx,(len(idx)+255)//256):
            x=torch.from_numpy(inputs.windows[batch])
            h=enc.post_fusion.register_forward_pre_hook(lambda mod,args: parts.append(args[0].numpy().copy()))
            try: y=net.surface(x)
            finally:h.remove()
            replay=net.head(net.regularizer(enc.core(enc.post_fusion(torch.from_numpy(parts[-1]))).flatten(1)))
            torch.testing.assert_close(y,replay,rtol=0,atol=0)
    return inputs.row_ts_ns[idx],np.concatenate(parts)


def run(a):
    if file_sha(a.spec)!='a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813':raise ValueError('spec')
    spec=json.loads(a.spec.read_text())
    old=spec['producer_settings']['factor_state']
    spec['pins'][str(a.factor)]=spec['pins'].pop(old)
    spec['producer_settings']['factor_state']=str(a.factor);authenticate(spec)
    model,_=load_bundle(a.bundle,'cpu')
    if file_sha(a.bundle/'model.pt')!='69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536':raise ValueError('source')
    # Historical implementation used to select the 6h pivot parameters.
    code=subprocess.check_output(['git','-C',str(a.engine),'cat-file','blob','1aaea5cb2262130cf4ce8ff7c87a745ccbe82290'])
    if hashlib.sha256(code).hexdigest()!='c06592f6478f3887213e985a80de5823dd953d3887ce25bec0272197860affda':raise ValueError('pivot implementation')
    from app.zigzag_opportunities import engine
    engine(a.engine)
    zz=types.ModuleType('app.short_horizon.zig387');zz.__package__='app.short_horizon'
    sys.modules[zz.__name__]=zz
    exec(compile(code,'pinned_zigzag','exec'),zz.__dict__)
    params=zz.Params(depth=4,backstep=2,deviation_pips=5.)
    external=load_external(spec);loaded=producers(spec)
    settings=spec['producer_settings'];state=json.loads(Path(settings['scope']).read_text())
    closures=json.loads(Path(spec['closures']).read_text())
    cols=model.recipe['list']['columns'];start,end=map(lambda s:pd.Timestamp(s,tz='UTC'),model.recipe['split']['periods']['TRAIN'])
    filters=[('timestamp','>=',start),('timestamp','<',end)]
    frame=pd.read_parquet(settings['bars'],filters=filters).set_index('timestamp')
    frame=frame[['BidOpen','BidHigh','BidLow','BidClose']].set_axis(['OPEN','HIGH','LOW','CLOSE'],axis=1)
    orig=pd.read_parquet(settings['store'],columns=['timestamp',*cols],filters=filters).set_index('timestamp')
    control=rebuild(frame,cols,state,closures,loaded,external)
    if not all(v['passed'] for v in parity(control,orig).values()):raise ValueError('original 71 parity')
    crosses=pd.DataFrame({k:v.reindex(frame.index) for k,v in external['fx'].items()})
    piv=pivots_by_chunk(frame,zz,params);nref=sum(map(len,piv))
    real=innovations(frame,crosses)[:,[0,4,5,6,7,8,9]]
    a.output.mkdir(parents=True,exist_ok=False);results=[];previous=[]
    for seed in (19,7,43,71):
        path=a.output/str(seed);path.mkdir()
        bars,fx,covered,segments=generate(frame,crosses,piv,seed)
        syn=pivots_by_chunk(bars,zz,params);n=sum(map(len,syn));hits=match(piv,syn,1)
        changed=float(np.mean(np.abs(bars.CLOSE[covered]-frame.CLOSE[covered])>1e-10))
        diff=[float(np.mean(np.abs(other[covered]-bars.CLOSE.values[covered])>1e-10)) for other in previous]
        previous.append(bars.CLOSE.to_numpy(copy=True))
        dist=distribution_gate(real,innovations(bars,fx)[:,[0,4,5,6,7,8,9]],bars.index)
        err=float(np.max(np.abs(segments[:,2:5]-segments[:,5:8])))
        assert np.isfinite(bars.to_numpy()).all() and (bars.to_numpy()>0).all()
        assert (bars.HIGH>=bars[['OPEN','CLOSE']].max(axis=1)).all() and (bars.LOW<=bars[['OPEN','CLOSE']].min(axis=1)).all()
        pd.testing.assert_frame_equal(bars,generate(frame,crosses,piv,seed)[0],check_exact=True)
        feat=rebuild(bars,cols,state,closures,loaded,dict(external,fx={k:fx[k] for k in fx}))
        feat.index.name='timestamp'
        inputs=FrozenRecipeInputs.from_frame(model.recipe,feat.reset_index())
        ts,z=fused(model,inputs)
        assert np.isfinite(z).all() and np.isin(ts,bars.index.as_unit('ns').asi8+3600*10**9).all()
        bars.to_parquet(path/'OHLC.parquet');feat.to_parquet(path/'FEATURES71.parquet')
        np.savez(path/'FUSED.npz',available_at_ns=ts,z=z)
        np.save(path/'SEGMENTS.npy',segments)
        report=dict(seed=seed,bars=len(bars),covered=int(covered.sum()),segments=len(segments),shape=list(z.shape),
                    max_control_error=err,changed=changed,distinct_vs_previous=diff,
                    precision=hits/n,recall=hits/nref,exact_recall=match(piv,syn,0)/nref,
                    distribution=dist,passed=bool(min([changed,*diff])>=.25 and min(hits/n,hits/nref)>=.8 and dist['passed'] and err<1e-12),
                    sha256={p.name:file_sha(p) for p in sorted(path.iterdir())})
        write(path/'RESULT.json',report);results.append(report)
        print(seed,report['passed'],report['shape'],changed,hits/n,hits/nref,flush=True)
    write(a.output/'COHORTS.json',dict(seeds={str(y):[r['seed'] for r in results[:y//4]] for y in (4,8,16)},
        release_for_training=all(r['passed'] for r in results),source=file_sha(a.bundle/'model.pt'),
        receipts={str(r['seed']):file_sha(a.output/str(r['seed'])/'RESULT.json') for r in results}))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('spec','bundle','engine','factor','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
