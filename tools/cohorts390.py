"""Materialize separately identified 4/8/16 year scenario banks, never train."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tools.regime390 import (authenticate,fit_regimes,generate,load_external,macroblocks,
                             physics,producers,rebuild,sha,write)


def run(a):
    report = json.loads((a.screen/'RESULT.json').read_text())
    native = json.loads((a.screen/'NATIVE.json').read_text())
    if not all(report[m+k] for m in ('price','context') for k in ('_screen','_physical_screen')):
        raise ValueError('screen not passed')
    if any(abs(native[k]['cash_error'])>1e-6 for k in ('61_real','71_real','61_price7','61_price19','71_context7','71_context19')):
        raise ValueError('native bookkeeping')
    if sha(a.spec)!='a37df7503445d0eddff3207b08e6f9f5c424d4f64fef1b8dd309472f15d95813':
        raise ValueError('spec')
    spec=json.loads(a.spec.read_text()); s=spec['producer_settings']
    spec['pins'][str(a.factor)]=spec['pins'].pop(s['factor_state']);s['factor_state']=str(a.factor)
    authenticate(spec)
    with np.load(a.business/'19.npz') as z:
        bars,ts=z['real'],pd.to_datetime(z['ts'],utc=True)
    assert ts[0].year==2020 and ts[-1].year==2023
    ts.name='timestamp'
    cols=json.loads(a.recipe.read_text())['list']['columns']
    pricecols=pd.read_parquet(a.screen/'price7/FEATURES61.parquet').columns.tolist()
    ext=load_external(spec); loaded=producers(spec)
    fx=pd.DataFrame({k:v.reindex(ts) for k,v in ext['fx'].items()})
    macro=[c for c in cols if c.startswith(('cot.','dly.'))]
    context=pd.read_parquet(s['store'],columns=['timestamp',*macro],filters=[
        ('timestamp','>=',ts[0]),('timestamp','<=',ts[-1])]).set_index('timestamp').loc[ts].shift(1).to_numpy()
    blocks=macroblocks(pd.read_parquet(a.business/'real_catalog.parquet'),bars,ts,4)
    labels,_=fit_regimes(blocks,ts)
    state=json.loads(Path(s['scope']).read_text())
    closures=json.loads(Path(spec['closures']).read_text())
    a.out.mkdir(parents=True,exist_ok=False)
    ids=[7,19,43,71]
    rec=dict(schema='hierarchical390.banks.v1',seeds={y:ids[:y//4] for y in (4,8,16)},
             independent_accounts=True,calendar='2020-2023',donors_before='2023-01-01',
             training_released=False,paths={},inputs={p.name:sha(p) for p in
                 (a.recipe,a.spec,a.business/'19.npz',a.business/'real_catalog.parquet',
                  a.screen/'RESULT.json',a.screen/'NATIVE.json',Path(__file__),Path(__file__).with_name('regime390.py'))})
    for seed in ids:
        for mode in ('price','context'):
            name=mode+str(seed);d=a.out/name;d.mkdir()
            use=mode=='context'
            p,x,tr=generate(blocks,labels,ts,bars,fx,context,seed,False,use,2024)
            assert np.all(ts[tr[:,3]].year<2023) and np.isfinite(p).all() and np.all(p>0)
            assert (p[:,1]>=p[:,[0,3]].max(1)).all() and (p[:,2]<=p[:,[0,3]].min(1)).all()
            f=pd.DataFrame(p,index=ts,columns=['OPEN','HIGH','LOW','CLOSE'])
            x[~np.isfinite(fx.values)]=np.nan
            xx=pd.DataFrame(x,index=ts,columns=fx.columns)
            phys=physics(p[:,3],xx,ts,tr[1:,0][tr[1:,0]==tr[:-1,1]])
            phyref=pd.read_csv(a.screen/(mode+'7')/'physics.csv',index_col=0)
            flags=(phys[phyref.index]-phyref.held_2023).abs()<=phyref.tolerance
            joined=dict(ext,fx={k:xx[k] for k in xx})
            feats=rebuild(f,cols if use else pricecols,state,closures,loaded,joined if use else None)
            if use:
                pd.testing.assert_frame_equal(rebuild(f,pricecols,state,closures,loaded),feats[pricecols],check_exact=True)
            fn='FEATURES%d.parquet'%len(feats.columns)
            f.to_parquet(d/'OHLC.parquet');feats.to_parquet(d/fn);np.save(d/'DONORS.npy',tr)
            if seed in (7,19):
                old=pd.read_parquet(a.screen/name/'OHLC.parquet')
                pd.testing.assert_frame_equal(old,f.loc[old.index],check_exact=True)
            rec['paths'][name]=dict(rows=len(f),n_columns=len(feats.columns),blocks=len(tr),
                finite_rows=int(np.isfinite(feats).all(axis=1).sum()),physics_failed=flags.index[~flags].tolist(),
                sha256={n:sha(d/n) for n in ('OHLC.parquet',fn,'DONORS.npy')})
            write(a.out/'PROGRESS.json',rec)
    write(a.out/'COHORTS.json',rec)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for k in ('out','screen','spec','recipe','factor','business'):
        parser.add_argument('--'+k,type=Path,required=True)
    run(parser.parse_args())
