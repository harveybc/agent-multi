"""Propagate verified November price repairs through frozen selected FX producers."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

COLS=['fac.fx_f1.ret_4','fac.fx_f1.ret_24','wv.usd.A4','hb.eur.inst_freq']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def same(a,b,tol=1e-10):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return np.array_equal(np.isfinite(a),np.isfinite(b)) and np.allclose(a,b,rtol=tol,atol=tol,equal_nan=True)


def install_changes(store,old,new,allowed):
    if list(old.columns)!=COLS or list(new.columns)!=COLS or not old.index.equals(store.index) or not new.index.equals(store.index):
        raise ValueError('REPAIR_FEATURE_ORDER_OR_SUPPORT')
    if not same(old,store[COLS]):raise ValueError('REPAIR_BASELINE_MISMATCH')
    if not same(old.loc[~allowed],new.loc[~allowed]):raise ValueError('REPAIR_OUTSIDE_SUPPORT')
    result=store.copy()
    result.loc[allowed,COLS]=new.loc[allowed,COLS]
    if not result.drop(columns=COLS).equals(store.drop(columns=COLS)):
        raise ValueError('REPAIR_OTHER_COLUMNS_CHANGED')
    return result


def run(spec_path,output):
    spec=json.loads(Path(spec_path).read_text());out=Path(output)
    if out.exists() and (out/'DONE.json').exists():
        done=json.loads((out/'DONE.json').read_text())
        if done['spec_sha256']!=sha(spec_path) or any(sha(out/p)!=h for p,h in done['files'].items()):
            raise ValueError('REPAIR_RESUME_IDENTITY')
        return
    for p,h in spec['pins'].items():
        if sha(p)!=h:raise ValueError('REPAIR_INPUT_DIGEST')
    out.mkdir(parents=True,exist_ok=True)
    def emit(stage,**kw):
        body=dict(stage=stage,**kw)
        (out/'PROGRESS.json').write_text(json.dumps(body,indent=2)+'\n')
        print(json.dumps(body),flush=True)
    def module(name,path):
        s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
    sys.path.insert(0,spec['predictor_root'])
    from tools import layer2_pool_h1 as layer2
    cross=module('recovery_cross_asset',spec['cross_module'])
    state=json.loads(Path(spec['factor_state']).read_text())
    for k in ['mu','sd','W']:state[k]=np.array(state[k],float)
    if (state['W'].shape!=(6,3) or not np.isfinite(state['W']).all()
            or not (state['sd']>0).all()):raise ValueError('FROZEN_FACTOR_STATE')
    cutoff=pd.Timestamp('2025-01-01',tz='UTC')
    store=pd.read_parquet(spec['store'],filters=[('timestamp','<',cutoff)]).set_index('timestamp')
    if not store.index.is_unique or not store.index.is_monotonic_increasing:raise ValueError('REPAIR_GRID')
    old_prices={};new_prices={}
    receipt=json.loads(Path(spec['source_receipt']).read_text())
    for pair in state['columns']:
        base=Path(spec['old_prices'])/f'{pair}.parquet';new=Path(spec['new_prices'])/f'{pair}.parquet'
        proof=receipt['november'][pair]
        if sha(base)!=proof['recovered_sha256'] or sha(new)!=proof['interval_repair']['candidate_sha256']:
            raise ValueError('REPAIR_SOURCE_CANDIDATE')
        old_prices[pair]=pd.read_parquet(base,filters=[('timestamp','<',cutoff)]).set_index('timestamp')['close'].reindex(store.index)
        new_prices[pair]=pd.read_parquet(new,filters=[('timestamp','<',cutoff)]).set_index('timestamp')['close'].reindex(store.index)
    def compute(prices,index):
        per={p:cross.hourly_features(s.reindex(index),f'fx.{p}') for p,s in prices.items()}
        r=pd.DataFrame({p:per[p][f'fx.{p}.ret_1'] for p in state['columns']},index=index)
        f=cross.pca_apply(r,state)['f1']
        strength=pd.DataFrame({
            'usd_strength_ret':cross.composite(per,cross.USD_SIGN,'usd_strength')['fx.usd_strength.ret_1'],
            'eur_strength_ret':cross.composite(per,cross.EUR_SIGN,'eur_strength')['fx.eur_strength.ret_1']},index=index)
        decomposed,_=layer2.fam_strength(strength)
        return pd.DataFrame({COLS[0]:f.rolling(4,min_periods=4).sum(),
            COLS[1]:f.rolling(24,min_periods=24).sum(),COLS[2]:decomposed[COLS[2]],
            COLS[3]:decomposed[COLS[3]]},index=index)
    emit('BASELINE_REPRODUCTION')
    baseline=compute(old_prices,store.index)
    if not same(baseline,store[COLS]):
        detail={c:dict(mask_equal=np.array_equal(np.isfinite(baseline[c]),np.isfinite(store[c])),
            max_error=float(np.nanmax(abs(baseline[c]-store[c])))) for c in COLS}
        emit('BASELINE_FAILED',detail=detail);raise ValueError('REPAIR_BASELINE_MISMATCH')
    emit('REPAIRED_FEATURES')
    repaired=compute(new_prices,store.index)
    start=int(store.index.get_indexer([pd.Timestamp('2023-11-29T12:00Z')])[0])
    last=int(store.index.get_indexer([pd.Timestamp('2023-11-29T14:00Z')])[0])
    if start<0 or last<start:raise ValueError('REPAIR_EVENT_NOT_ON_GRID')
    # A changed close affects the next return and at most256 trailing return bars.
    allowed=np.zeros(len(store),bool);allowed[start:last+258]=True
    result=install_changes(store,baseline,repaired,allowed)
    cut=last+260
    emit('PREFIX_CHECK',rows=cut)
    prefix=compute(new_prices,store.index[:cut])
    if not same(prefix,repaired.iloc[:cut]):raise ValueError('REPAIR_PREFIX_FAILURE')
    rows={c:dict(recovered=int((~np.isfinite(store[c])&np.isfinite(result[c])).sum()),
        changed=int((~np.isclose(store[c],result[c],equal_nan=True,rtol=1e-12,atol=1e-12)).sum())) for c in COLS}
    target=out/'STORE_CANDIDATE.parquet';result.reset_index().to_parquet(target,index=False)
    for p,h in spec['pins'].items():
        if sha(p)!=h:raise ValueError('REPAIR_INPUT_CHANGED')
    report=dict(schema='selected_fx_source_repair.v1',columns=COLS,counts=rows,
        store_rows=len(result),feature_count=len(result.columns),grid_unchanged=True,
        frozen_pca=True,baseline_parity=True,prefix_pass=True,original_sources_unchanged=True,
        training_released=False,missing=['USDCHF May2022 qualification','pre-TRAIN history reconstruction'])
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    done=dict(spec_sha256=sha(spec_path),files={p:sha(out/p) for p in ['RESULT.json','STORE_CANDIDATE.parquet']})
    (out/'DONE.json').write_text(json.dumps(done,indent=2)+'\n');emit('COMPLETE',counts=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();run(a.spec,a.output)
