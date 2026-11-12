"""Causal seasonal-input ablation: frozen tuning, resumable columns and SAC fits."""
import argparse
import copy
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import statsmodels
import torch

from app.causal_stl_inputs import choose_parameters,CausalSTLColumn,encode_seasonal,OnlineSTLInputs
from app.frozen_recipe_inputs import FrozenRecipeInputs,file_sha
from app.rl_subset import digest,seal_recipe,support_hash
from app.selected_input_market import market_for
from app.structural_refit import build_model,verify_files,load_bundle,export_bundle
from app.three_signal_learning import environment
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_structural_sac_refit import fit_epochs,assess_full

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=('raw','continuous','ordinal','binary')


def source(args):
    spec=read(args.parent/'SPEC.json');verify_files(args.parent,spec['files'])
    manifest=read(args.bundle/'MANIFEST.json');verify_files(args.bundle,manifest['files'])
    rec=read(args.bundle/'RECIPE.json');market=market_for(args.parent,spec,rec)
    return spec,rec,market


def monthly_coverage(times, valid, periods):
    times=pd.to_datetime(times,utc=True);valid=np.asarray(valid,dtype=bool)
    if valid.shape!=(len(times),):raise ValueError('STL_COVERAGE_SHAPE')
    report={}
    for name,(start,end) in periods.items():
        start=pd.Timestamp(start,tz='UTC');end=pd.Timestamp(end,tz='UTC')
        months=pd.date_range(start.normalize().replace(day=1),end,freq='MS',inclusive='left')
        report[name]={m.strftime('%Y-%m'):int((valid & (times>=max(start,m)) &
            (times<min(end,m+pd.offsets.MonthBegin(1)))).sum()) for m in months}
    return report


def require_coverage(report):
    missing=[f'{p}/{m}' for p,months in report.items() for m,n in months.items() if not n]
    if not report or any(not months for months in report.values()):
        raise ValueError('STL_EMPTY_COVERAGE')
    if missing:raise ValueError('STL_EMPTY_MONTH:'+','.join(missing))


def prepare(args):
    spec,rec,market=source(args)
    frame=pd.read_parquet(args.parent/spec['inputs']['store'],columns=['timestamp']+rec['list']['columns'],
        filters=[('timestamp','<',pd.Timestamp(rec['split']['test_closed_from'],tz='UTC'))])
    times=pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8+3600*10**9
    x=frame[rec['list']['columns']].to_numpy(dtype=float)
    for j,c in enumerate(rec['tensor']['columns']):
        if c['consumer_ops']==['DIFF']:x[:,j]=np.r_[np.nan,np.diff(x[:,j])]
    indices=np.searchsorted(times,market.inputs.row_ts_ns)
    if not np.array_equal(times[indices],market.inputs.row_ts_ns):raise ValueError('STL_PARENT_TIME_ALIGNMENT')
    expected=(x[indices]-np.array(rec['scaler']['mean']))/np.array(rec['scaler']['std'])
    if not np.allclose(expected,market.inputs.windows[:,-1,:],rtol=1e-5,atol=1e-5,equal_nan=True):
        raise ValueError('STL_ENCODED_INPUT_PARITY')
    cfg=dict(schema='causal_stl_lane.v2',parent=spec['identity'],source_sha256=file_sha(args.bundle/'model.pt'),
        recipe_sha256=file_sha(args.bundle/'RECIPE.json'),columns=rec['list']['columns'],
        original_support=support_hash(market.inputs),statsmodels=statsmodels.__version__,
        train_end=int(np.searchsorted(times,pd.Timestamp('2024-01-01',tz='UTC').value)),
        original_window=rec['tensor']['window'],period_clock='admitted observation rows, not wall-clock hours',
        input_context='full producer grid; parent_indices select evaluated endpoints without a second warmup trim',
        grid=dict(period=[6,12,24,48,120],seasonal=[7,13],robust=[False,True],context='max(64,4*period)'),
        origins=16,forecast_horizon=24,variants=list(VARIANTS),seed=7,
        sources={n:file_sha(ROOT/n) for n in ('app/causal_stl_inputs.py','tools/run_causal_stl_lane.py',
            'app/frozen_recipe_inputs.py','app/selected_input_market.py','app/structural_refit.py','tools/run_structural_sac_refit.py')})
    cfg['identity']=digest(cfg)
    if (args.root/'PREPARED.json').exists():
        prior=read(args.root/'CONFIG.json')
        if prior!=cfg:raise ValueError('STL_PREPARE_CHANGED')
        verify_files(args.root,read(args.root/'PREPARED.json')['files']);return
    write(args.root/'CONFIG.json',cfg);write(args.root/'SOURCE_RECIPE.json',rec)
    np.savez_compressed(args.root/'RAW.npz',values=x,times=times,parent_indices=indices)
    write(args.root/'PREPARED.json',dict(identity=cfg['identity'],files={n:file_sha(args.root/n) for n in
        ('CONFIG.json','SOURCE_RECIPE.json','RAW.npz')}))
    print(dict(identity=cfg['identity'],shape=x.shape,train_end=cfg['train_end']),flush=True)


def authenticated(root):
    ready=read(root/'PREPARED.json');verify_files(root,ready['files']);cfg=read(root/'CONFIG.json')
    if cfg['identity']!=digest({k:v for k,v in cfg.items() if k!='identity'}):raise ValueError('STL_IDENTITY')
    if cfg['statsmodels']!=statsmodels.__version__:raise ValueError('STL_LIBRARY_VERSION')
    for n,h in cfg['sources'].items():
        if file_sha(ROOT/n)!=h:raise ValueError('STL_RUNTIME_CHANGED:'+n)
    return cfg


def column(args):
    cfg=authenticated(args.root);j=args.column
    if not 0<=j<len(cfg['columns']):raise ValueError('STL_COLUMN_INDEX')
    out=args.root/'columns'/f'{j:03d}';out.mkdir(parents=True,exist_ok=True)
    with (out/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'DONE.json').exists():verify_done(out,cfg['identity']);return
        with np.load(args.root/'RAW.npz') as f:x=f['values'][:,j]
        started=time.monotonic()
        def progress(stage,**kw):write(out/'PROGRESS.json',dict(column=j,name=cfg['columns'][j],stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
        progress('TRAIN_PARAMETER_SEARCH')
        if (out/'PARAMETERS.json').exists():tuning=read(out/'PARAMETERS.json')
        else:tuning=choose_parameters(x,cfg['train_end']);write(out/'PARAMETERS.json',tuning)
        params=tuning['selected']['parameters'];stream=CausalSTLColumn(params)
        result=np.full(len(x),np.nan);start=0
        if (out/'PARTIAL.npz').exists():
            with np.load(out/'PARTIAL.npz') as f:
                result=f['seasonal'];start=int(f['completed']);stream=CausalSTLColumn(params,f['history'])
        for i in range(start,len(x)):
            result[i]=stream.update(x[i])
            if (i+1)%256==0:
                # Atomic generated array checkpoint, never a revision of emitted past values.
                with (out/'PARTIAL.building').open('wb') as f:
                    np.savez_compressed(f,seasonal=result,completed=i+1,history=stream.history)
                (out/'PARTIAL.building').replace(out/'PARTIAL.npz')
                progress('ENDPOINTS',completed=i+1,total=len(x),parameters=params)
        np.savez_compressed(out/'SEASONAL.npz',values=result)
        eligible=np.flatnonzero(np.isfinite(result));checks=[]
        if not len(eligible):raise ValueError('STL_NO_FINITE_OUTPUT:'+cfg['columns'][j])
        for t in eligible[np.linspace(0,len(eligible)-1,5,dtype=int)]:
            prefix=CausalSTLColumn(params,x[t-params['history']+1:t])
            replay=prefix.update(x[t])
            if replay!=result[t]:raise ValueError('STL_STREAM_PARITY')
            checks.append(dict(row=int(t),difference=0.))
        write(out/'RESULT.json',dict(identity=cfg['identity'],column=j,name=cfg['columns'][j],
            parameters=params,tuning=tuning['selected'],finite_rows=int(len(eligible)),
            seasonal_std=float(np.nanstd(result[:cfg['train_end']])),stream_checks=checks,
            seconds=time.monotonic()-started,finished_at=now(),no_imputation=True))
        # Checkpoints are resumable intermediates; terminal receipt does not need their duplicate arrays.
        (out/'PARTIAL.npz').unlink(missing_ok=True)
        seal(out,cfg['identity']);progress('COMPLETE',completed=len(x),total=len(x))


def pack_inputs(base,recipe,values,common,source_indices=None):
    indices=np.arange(len(values)) if source_indices is None else np.asarray(source_indices)
    if (indices.shape!=(len(base.valid),) or not np.issubdtype(indices.dtype,np.integer)
            or (indices<0).any() or (indices>=len(values)).any()
            or (np.diff(indices)<=0).any()):raise ValueError('STL_SOURCE_INDICES')
    w=base.recipe['tensor']['window'];slots=indices[:,None]-w['stride_rows']*np.arange(w['observations']-1,-1,-1)
    windows=values[slots.clip(0)].astype(np.float32)
    valid=base.valid & (slots[:,0]>=0) & np.isfinite(windows).all(axis=(1,2)) & common
    for a in (windows,valid):a.flags.writeable=False
    return FrozenRecipeInputs(recipe,base.row_ts_ns,base.slot_ts_ns,windows,valid,base.period)


def assemble(args):
    cfg=authenticated(args.root);spec,rec,market=source(args);results=[];cols=[]
    for j in range(len(cfg['columns'])):
        folder=args.root/'columns'/f'{j:03d}';verify_done(folder,cfg['identity'])
        results.append(read(folder/'RESULT.json'))
        with np.load(folder/'SEASONAL.npz') as f:cols.append(f['values'])
    values=np.column_stack(cols)
    with np.load(args.root/'RAW.npz') as f:indices=f['parent_indices'];times=f['times'];raw=f['values']
    mask=times<pd.Timestamp('2024-01-01',tz='UTC').value
    enc=encode_seasonal(values,mask);parameters=[r['parameters'] for r in results]
    original=(raw-np.array(rec['scaler']['mean']))/np.array(rec['scaler']['std'])
    variants={'raw':original,**{k:enc[k] for k in ('continuous','ordinal','binary')}}
    common=market.inputs.valid.copy()
    for k,v in variants.items():common &= pack_inputs(market.inputs,rec,v,common,indices).valid
    if not common.any():raise ValueError('STL_NO_COMMON_SUPPORT')
    live_checks=[]
    for t in np.flatnonzero(np.isfinite(values).all(1))[np.linspace(0,np.isfinite(values).all(1).sum()-1,5,dtype=int)]:
        history=[raw[t-p['history']+1:t,j].tolist() for j,p in enumerate(parameters)]
        for k in ('continuous','ordinal','binary'):
            online=OnlineSTLInputs(parameters,enc['parameters'],k,history).update(raw[t])
            if not np.array_equal(online,enc[k][t]):raise ValueError('STL_LIVE_ENCODING_PARITY')
        live_checks.append(int(t))
    report={};out=args.root/'prepared';out.mkdir(exist_ok=True)
    for k,v in variants.items():
        recipe=copy.deepcopy(rec)
        if k!='raw':
            recipe['scaler']['mean']=enc['parameters']['mean'] if k=='continuous' else [0.]*len(cfg['columns'])
            recipe['scaler']['std']=enc['parameters']['std'] if k=='continuous' else [1.]*len(cfg['columns'])
        recipe['store']['input_adapter']=dict(kind='causal_stl_seasonal_v1',variant=k,
            source_recipe_identity=rec['identity'],parameters=parameters,encoding=enc['parameters'],
            identity=cfg['identity'],clock=cfg['period_clock'],common_history_gate=True,
            note='requires explicit adapter; raw store loader must refuse')
        recipe=seal_recipe(recipe);inputs=pack_inputs(market.inputs,recipe,v,common,indices)
        write(out/f'{k}_RECIPE.json',recipe)
        report[k]=dict(support=support_hash(inputs),shape=list(inputs.windows.shape),
            periods={p:dict(original=int((market.inputs.valid & (market.inputs.period==p)).sum()),
                common=int((inputs.valid & (inputs.period==p)).sum()),
                unavailable=int((market.inputs.valid & ~inputs.valid & (inputs.period==p)).sum())) for p in ('TRAIN','VALIDATION')})
    if len({r['support'] for r in report.values()})!=1:raise ValueError('STL_SUPPORT_UNPAIRED')
    coverage=monthly_coverage(market.inputs.row_ts_ns,common,rec['split']['periods'])
    np.savez_compressed(out/'VALUES.npz',**variants,common=common,parent_indices=indices)
    write(out/'RESULT.json',dict(identity=cfg['identity'],variants=report,columns=results,
        live_parity_rows=live_checks,all_features_replaced=True,no_imputation=True,monthly_coverage=coverage,
        training_eligible=all(n>0 for months in coverage.values() for n in months.values()),
        diagnostic='seasonal-only may discard predictive trend/remainder; no automatic admission'))
    seal(out,cfg['identity']);print(report,flush=True)


def train(args):
    cfg=authenticated(args.root);verify_done(args.root/'prepared',cfg['identity'])
    if not torch.cuda.is_available() or torch.cuda.get_device_name()!=args.expected_device_name:raise ValueError('STL_GPU_IDENTITY')
    spec,_,market=source(args);k=args.variant;recipe=read(args.root/'prepared'/f'{k}_RECIPE.json')
    with np.load(args.root/'prepared/VALUES.npz') as f:
        market.inputs=pack_inputs(market.inputs,recipe,f[k],f['common'],f['parent_indices'])
    market.recipe=recipe
    require_coverage(monthly_coverage(market.inputs.row_ts_ns,market.inputs.valid,recipe['split']['periods']))
    if support_hash(market.inputs)!=read(args.root/'prepared/RESULT.json')['variants'][k]['support']:raise ValueError('STL_TRAIN_SUPPORT')
    design=copy.deepcopy(spec['design']);design.update(head_only=False,initialize_epoch_zero=False,warmup_policy=False,
        max_epochs=2000,patience=20,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=20,
        select_during_warmup=True,require_activity=True,diagnostics=True,replay_backend='memmap')
    effective=read(args.bundle/'MODEL_CONFIG.json');configure(7);torch.backends.cudnn.allow_tf32=False
    run=args.root/'training'/k;run.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(side_lane=cfg['identity'],variant=k,design=design,effective=effective,seed=7))
    with (run/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (run/'DONE.json').exists():verify_done(run,identity);return
        model=build_model(recipe,effective,7,'cuda');risk=read(args.parent/'RISK.json')
        probe=market.inputs.windows[np.flatnonzero(market.inputs.valid)[np.linspace(0,market.inputs.valid.sum()-1,32,dtype=int)]]
        write(run/'EFFECTIVE.json',dict(identity=identity,design=design,effective=effective,variant=k,
            recipe_identity=recipe['identity'],common_support=support_hash(market.inputs),fresh_weights=True,
            features=recipe['list']['n'],gpu=torch.cuda.get_device_name()))
        # Self-contained transformed-input bundle must cold-reload before long training.
        if not (run/'initial_bundle').exists():export_bundle(run/'initial_bundle',model,probe,dict(epoch=0,diagnostic=True,identity=identity))
        state=fit_epochs(model,lambda:environment(market,risk,design['horizon'],
            recipe['split']['periods']['TRAIN'],effective['gamma']),lambda m:assess_full(m,market,risk,design),
            design,7,run,identity,probe)
        metrics=assess_full(model,market,risk,design,save=run/'final',diagnose=True)
        eligible=state['best_bundle'] is not None and all(v['orders_per_week']>2 for v in metrics.values())
        write(run/'RESULT.json',dict(identity=identity,variant=k,metrics=metrics,best_bundle=state['best_bundle'],
            activity_eligible=eligible,status='COMPLETE' if eligible else 'COMPLETE_NO_ELIGIBLE_CHECKPOINT',
            epochs=state['epoch'],finished_at=now(),model_promoted=False))
        # Do not hash many gigabytes of mutable replay as final inference evidence.
        files={str(p.relative_to(run)):file_sha(p) for p in run.rglob('*') if p.is_file()
               and p.name not in ('worker.lock','DONE.json','PROGRESS.json')
               and not any(part in ('replay_live','recovery') for part in p.relative_to(run).parts)}
        write(run/'DONE.json',dict(identity=identity,at=now(),files=files))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','column','assemble','train'))
    p.add_argument('--root',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--bundle',type=Path)
    p.add_argument('--column',type=int);p.add_argument('--variant',choices=VARIANTS);p.add_argument('--expected-device-name')
    a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True);globals()[a.command](a)


if __name__=='__main__':main()
