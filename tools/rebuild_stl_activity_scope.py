"""Separate activity-producer candidate with a purge sized for downstream STL."""
import argparse
import importlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.transformed_feature_scope import context_fit_overlap
from tools.recover_consumer_activity import replay
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write


def downstream_context(history, observations, stride):
    if any(type(v) is not int or v<1 for v in (history,observations,stride)):
        raise ValueError('ACTIVITY_CONTEXT')
    return history+stride*(observations-1)


def run(a):
    spec=read(a.spec);files={k:Path(v['path']) for k,v in spec['files'].items()}
    for k,p in files.items():
        if file_sha(p)!=spec['files'][k]['sha256']:raise ValueError('ACTIVITY_SOURCE:'+k)
    old_receipt=read(a.activity.parent/'RESULT.json')
    if file_sha(a.activity)!=old_receipt['weights_sha256'] or file_sha(a.spec)!=old_receipt['spec_sha256']:
        raise ValueError('ACTIVITY_ORIGINAL_RECEIPT')
    old=joblib.load(a.activity)
    context=downstream_context(a.history,a.observations,a.stride)
    embargo=max(old['metadata']['embargo_bars'],context-1)
    cfg=dict(source_spec_sha256=file_sha(a.spec),source_activity_sha256=file_sha(a.activity),
        history=a.history,observations=a.observations,stride=a.stride,context=context,embargo=embargo,
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'tools/rebuild_stl_activity_scope.py','app/transformed_feature_scope.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:
        raise ValueError('ACTIVITY_REPAIR_COLLISION')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    sys.path.insert(0,spec['predictor_root'])
    # Both repositories own a tools package. Resolve legacy producers explicitly
    # before importing them; the source digest checks above bind that directory.
    import tools
    tools.__path__=[str(Path(spec['predictor_root'])/'tools'),*tools.__path__]
    producer=importlib.import_module('tools.activity_producer')
    scopes=importlib.import_module('tools.fold_local_producers')
    if Path(producer.__file__).resolve()!=files['producer'].resolve() or Path(scopes.__file__).resolve()!=files['scope_producer'].resolve():
        raise ValueError('ACTIVITY_IMPORT_IDENTITY')
    bars,x,hit,resolution,features=producer.load_inputs(files['snapshot'].parent,files['bars'].parent,
        files['zigzag'].parent,files['pool'],files['decision'])
    if (not np.array_equal(bars.index.asi8,old['times_ns']) or features!=old['features']
            or (bars.index>=pd.Timestamp('2025-01-01',tz='UTC')).any()):
        raise ValueError('ACTIVITY_REPAIR_GRID')
    reference=pd.read_parquet(files['store'],columns=['timestamp','act.p_hit'])
    before=replay(old,x)
    if not np.array_equal(before,reference['act.p_hit'].to_numpy()):raise ValueError('ACTIVITY_ORIGINAL_PARITY')
    allowed=scopes.allowed_mask(bars.index,old['scope'])
    models=[];fit_rows=[];original=producer._fit
    def capture(values,labels,rows):
        model=original(values,labels,rows);models.append(model);fit_rows.append(rows.copy());return model
    producer._fit=capture
    try:
        probabilities,metadata=producer.scope_predictions(x,hit,resolution,allowed,
            k=len(old['prediction_rows']),embargo=embargo,
            lookback=old['metadata']['input_lookback_bars'],mode='purged_kfold')
    finally:producer._fit=original
    bundle=dict(old,estimators=models,fit_rows=fit_rows,metadata=metadata)
    if not np.array_equal(replay(bundle,x),probabilities):raise ValueError('ACTIVITY_REPAIR_REPLAY')
    validation=bars.index>=pd.Timestamp('2024-01-01',tz='UTC')
    if not np.array_equal(probabilities[validation],before[validation]):
        raise ValueError('ACTIVITY_VALIDATION_CHANGED')
    old_overlap,_,_=context_fit_overlap(old['fit_rows'],old['prediction_rows'],np.ones(len(x),bool),np.arange(context))
    new_overlap,_,_=context_fit_overlap(fit_rows,bundle['prediction_rows'],np.ones(len(x),bool),np.arange(context))
    if new_overlap.any():raise ValueError('ACTIVITY_REPAIR_STILL_OVERLAPS')
    target=a.output/'CONSUMER_ACTIVITY.joblib';joblib.dump(bundle,target)
    if not np.array_equal(replay(joblib.load(target),x),probabilities):raise ValueError('ACTIVITY_REPAIR_RELOAD')
    pd.DataFrame({'timestamp':bars.index,'act.p_hit':probabilities}).to_parquet(a.output/'ACTIVITY_CANDIDATE.parquet',index=False)
    for k,p in files.items():
        if file_sha(p)!=spec['files'][k]['sha256']:raise ValueError('ACTIVITY_SOURCE_CHANGED:'+k)
    result=dict(identity=cfg['identity'],context_rows=context,embargo_before=old['metadata']['embargo_bars'],
        embargo_after=embargo,overlap_before=int(old_overlap.sum()),overlap_after=int(new_overlap.sum()),
        changed_rows=int((probabilities!=before).sum()),validation_exact=True,reload_exact=True,
        source_unchanged=True,training_released=False,live_model_replaced=False,
        weights_sha256=file_sha(target),spec_sha256=file_sha(a.spec),
        limitation='Repairs direct target-row overlap for this bounded context, not missing source data or all upstream lineage')
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity']);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('spec','activity','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--history',type=int,default=480);p.add_argument('--observations',type=int,default=13)
    p.add_argument('--stride',type=int,default=2)
    a=p.parse_args()
    try:run(a)
    except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc)));raise
