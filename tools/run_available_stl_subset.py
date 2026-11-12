"""Availability-qualified STL subset and finite paired training queues."""
import argparse
import copy
import fcntl
from pathlib import Path

import numpy as np
import pandas as pd

from app.causal_stl_inputs import CausalSTLColumn, OnlineSTLInputs, encode_seasonal
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, seal_recipe, subset_recipe, support_hash
from app.stl_availability_subset import availability_subset
from tools import run_causal_stl_lane as lane
from tools.audit_stl_source_support import finite_windows
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write, now


def prepare(a):
    if a.root.resolve().is_relative_to(a.source_root.resolve()):
        raise ValueError('STL_SUBSET_SOURCE_OVERLAP')
    original=lane.authenticated(a.source_root)
    spec,parent_recipe,market=lane.source(a)
    if (file_sha(a.bundle/'model.pt')!=original['source_sha256'] or
            file_sha(a.bundle/'RECIPE.json')!=original['recipe_sha256']):
        raise ValueError('STL_SUBSET_SOURCE_BUNDLE')
    binding=dict(source=file_sha(a.source_root/'PREPARED.json'),
        tool=file_sha(Path(__file__)),selector=file_sha(lane.ROOT/'app/stl_availability_subset.py'))
    if (a.root/'prepared/DONE.json').exists():
        cfg=lane.authenticated(a.root)
        if cfg['availability_binding']!=binding:raise ValueError('STL_SUBSET_RESUME_BINDING')
        verify_done(a.root/'prepared',cfg['identity']);return
    with np.load(a.source_root/'RAW.npz') as f:
        raw=f['values'].copy();times=f['times'].copy();indices=f['parent_indices'].copy()
    if not np.array_equal(times[indices],market.inputs.row_ts_ns):
        raise ValueError('STL_SUBSET_CLOCK')
    cols=[];parameters=[];receipts={};valid=[]
    offsets=original['original_window']['slot_offsets_rows']
    for j,name in enumerate(original['columns']):
        folder=a.source_root/'columns'/f'{j:03d}'
        verify_done(folder,original['identity'])
        receipts[name]=file_sha(folder/'DONE.json')
        with np.load(folder/'SEASONAL.npz') as f:values=f['values'].copy()
        cols.append(values);parameters.append(read(folder/'RESULT.json')['parameters'])
        valid.append(finite_windows(values,indices,offsets) &
                     finite_windows(raw[:,j],indices,offsets) & market.inputs.valid)
    selection=availability_subset(original['columns'],np.column_stack(valid),
        market.inputs.row_ts_ns,parent_recipe['split']['periods'],
        {'act.p_hit':'unresolved_fitted_context_scope'},reference_valid=market.inputs.valid)
    selected=selection['selected'];take=[original['columns'].index(n) for n in selected]
    recipe=subset_recipe(parent_recipe,selected)
    values=np.column_stack([cols[j] for j in take]);raw=raw[:,take]
    parameters=[parameters[j] for j in take]
    train_start,train_end=(pd.Timestamp(t,tz='UTC').value for t in recipe['split']['periods']['TRAIN'])
    encoding=encode_seasonal(values,(times>=train_start)&(times<train_end))
    variants={'raw':(raw-np.array(recipe['scaler']['mean']))/np.array(recipe['scaler']['std']),
              **{k:encoding[k] for k in ('continuous','ordinal','binary')}}
    common=market.inputs.valid.copy()
    for v in variants.values():common &= lane.pack_inputs(market.inputs,recipe,v,common,indices).valid
    if not np.array_equal(common,market.inputs.valid):
        raise ValueError('STL_SUBSET_LOST_REFERENCE_SUPPORT')
    coverage=lane.monthly_coverage(market.inputs.row_ts_ns,common,recipe['split']['periods'])
    lane.require_coverage(coverage)
    # Independent trailing-prefix and restarted online checks, without future values.
    eligible=np.flatnonzero(np.isfinite(values).all(1));checks=[]
    if not len(eligible):raise ValueError('STL_SUBSET_NO_FINITE_OUTPUT')
    for t in np.unique(eligible[np.linspace(0,len(eligible)-1,5,dtype=int)]):
        history=[]
        for j,p in enumerate(parameters):
            h=raw[t-p['history']+1:t,j].tolist();history.append(h)
            if CausalSTLColumn(p,h).update(raw[t,j])!=values[t,j]:
                raise ValueError('STL_SUBSET_ENDPOINT_PARITY')
        for k in ('continuous','ordinal','binary'):
            if not np.array_equal(OnlineSTLInputs(parameters,encoding['parameters'],k,history).update(raw[t]),variants[k][t]):
                raise ValueError('STL_SUBSET_ONLINE_PARITY')
        checks.append(int(t))
    cfg=copy.deepcopy(original);cfg.pop('identity')
    cfg.update(schema='causal_stl_available_subset.v1',columns=selected,
        availability_binding=binding,source_column_receipts=receipts,
        selection_sha256=digest(selection),source_repair_spliced=False,
        source_original_identity=original['identity'])
    cfg['sources'].update({n:file_sha(lane.ROOT/n) for n in
        ('tools/run_available_stl_subset.py','app/stl_availability_subset.py')})
    cfg['identity']=digest(cfg)
    if (a.root/'CONFIG.json').exists() and read(a.root/'CONFIG.json')!=cfg:
        raise ValueError('STL_SUBSET_OUTPUT_COLLISION')
    write(a.root/'CONFIG.json',cfg);write(a.root/'SOURCE_RECIPE.json',recipe)
    write(a.root/'SELECTION.json',selection)
    np.savez_compressed(a.root/'RAW.npz',values=raw,times=times,parent_indices=indices)
    write(a.root/'PREPARED.json',dict(identity=cfg['identity'],files={n:file_sha(a.root/n) for n in
        ('CONFIG.json','SOURCE_RECIPE.json','RAW.npz','SELECTION.json')}))
    out=a.root/'prepared';out.mkdir(exist_ok=True);report={}
    for k,v in variants.items():
        rec=copy.deepcopy(recipe)
        if k!='raw':
            rec['scaler']['mean']=encoding['parameters']['mean'] if k=='continuous' else [0.]*len(selected)
            rec['scaler']['std']=encoding['parameters']['std'] if k=='continuous' else [1.]*len(selected)
        rec['store']['input_adapter']=dict(kind='causal_stl_seasonal_v1',variant=k,
            source_recipe_identity=parent_recipe['identity'],parameters=parameters,
            encoding=encoding['parameters'],identity=cfg['identity'],clock=cfg['period_clock'],
            common_history_gate=True,availability_selection=digest(selection))
        rec=seal_recipe(rec);inputs=lane.pack_inputs(market.inputs,rec,v,common,indices)
        if not np.array_equal(inputs.valid,common):raise ValueError('STL_SUBSET_SUPPORT')
        write(out/f'{k}_RECIPE.json',rec)
        report[k]=dict(support=support_hash(inputs),shape=list(inputs.windows.shape),
            periods={p:dict(common=int((inputs.valid & (inputs.period==p)).sum())) for p in recipe['split']['periods']})
    if len({r['support'] for r in report.values()})!=1:raise ValueError('STL_SUBSET_UNPAIRED')
    np.savez_compressed(out/'VALUES.npz',**variants,common=common,parent_indices=indices)
    result=dict(identity=cfg['identity'],source_identity=original['identity'],variants=report,
        input_features=len(original['columns']),features=len(selected),groups=len(recipe['representation']['resolved']['names']),
        selected=selected,removed=selection['removed'],monthly_coverage=coverage,
        training_eligible=True,activity_excluded=True,all_features_replaced=True,
        no_imputation=True,source_repair_spliced=False,source_model_unchanged=True,
        original_support_preserved=True,
        live_parity_rows=checks,test_read=False,model_promoted=False)
    write(out/'RESULT.json',result);seal(out,cfg['identity'])
    print({k:result[k] for k in ('identity','features','groups','training_eligible')},flush=True)


def validate_training(root):
    cfg=lane.authenticated(root);verify_done(root/'prepared',cfg['identity'])
    result=read(root/'prepared/RESULT.json');selection=read(root/'SELECTION.json')
    if (cfg['schema']!='causal_stl_available_subset.v1' or not result['training_eligible'] or
            not result['activity_excluded'] or 'act.p_hit' in cfg['columns'] or
            not result['original_support_preserved'] or not selection['preserve_full_reference_train'] or
            cfg['columns']!=selection['selected'] or digest(selection)!=cfg['selection_sha256']):
        raise ValueError('STL_SUBSET_TRAIN_GATE')
    lane.require_coverage(result['monthly_coverage'])
    if len({r['support'] for r in result['variants'].values()})!=1:
        raise ValueError('STL_SUBSET_TRAIN_UNPAIRED')
    if any(r['support']!=cfg['original_support'] for r in result['variants'].values()):
        raise ValueError('STL_SUBSET_TRAIN_REFERENCE_SUPPORT')
    for k in lane.VARIANTS:
        rec=read(root/'prepared'/f'{k}_RECIPE.json')
        if rec['list']['columns']!=cfg['columns']:raise ValueError('STL_SUBSET_RECIPE_COLUMNS')
    return cfg


def queue(a):
    cfg=validate_training(a.root)
    if not a.variants or len(set(a.variants))!=len(a.variants):raise ValueError('STL_SUBSET_QUEUE_VARIANTS')
    queue_root=a.root/('queue_'+'_'.join(a.variants));queue_root.mkdir(exist_ok=True)
    with (queue_root/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for k in a.variants:
            validate_training(a.root)
            write(queue_root/'PROGRESS.json',dict(stage='TRAIN',variant=k,at=now()))
            a.variant=k;lane.train(a)
            result=read(a.root/'training'/k/'RESULT.json')
            verify_done(a.root/'training'/k,result['identity'])
        write(queue_root/'RESULT.json',dict(identity=cfg['identity'],variants=a.variants,
            status='COMPLETE',model_promoted=False,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','queue'))
    for n in ('root','source-root','parent','bundle'):p.add_argument('--'+n,type=Path,required=n in ('root','parent','bundle'))
    p.add_argument('--variants',nargs='+',choices=lane.VARIANTS)
    p.add_argument('--expected-device-name')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    try:
        if a.command=='prepare':
            with (a.root/'prepare.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);prepare(a)
        else:queue(a)
    except Exception as exc:
        name='FAILED_PREPARE.json' if a.command=='prepare' else 'FAILED_'+'_'.join(a.variants or [])+'.json'
        write(a.root/name,dict(error=repr(exc),at=now()));raise
