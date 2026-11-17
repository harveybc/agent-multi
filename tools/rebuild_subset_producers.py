"""Finite, resumable CPU rebuilding of all selected inputs for four TRAIN sizes."""
import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from app.frozen_recipe_inputs import FrozenRecipeInputs,IDENTITY_KEYS,file_sha,validate_recipe
from app.subset_producer_contract import window_mask,assert_fit_rows,fixed_scaler
from app.structural_refit import verify_files
from tools.run_prioritized_training_diagnosis import read,write
from tools.recover_stl_price_warmup import load_module


def plain(x):
    if isinstance(x,dict):return {k:plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return x


def identity(x):
    return hashlib.sha256(json.dumps(plain(x),sort_keys=True,allow_nan=False).encode()).hexdigest()


def activity_fit(AP,X,hit,res,allowed,lookback):
    models=[];rows=[];original=AP._fit
    def capture(values,labels,indices):
        j=len(models);local=allowed.copy()
        if j:
            alw=np.flatnonzero(allowed);edges=np.linspace(0,len(alw),6).astype(int)
            low,high=alw[edges[j-1]],alw[edges[j]-1]
            local[max(0,low-96):min(len(local),high+97)]=False
        assert_fit_rows(indices,local,labels,res,lookback)
        model=original(values,labels,indices);models.append(model);rows.append(indices.copy())
        return model
    AP._fit=capture
    try:p,meta=AP.scope_predictions(X,hit,res,allowed,k=5,embargo=96,lookback=lookback,mode='purged_kfold')
    finally:AP._fit=original
    if len(models)!=6:raise ValueError('ACTIVITY_MODEL_COUNT')
    return p,models,rows,meta


def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('FORCE_CPU')
    a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(a.config);g=read(a.producer_spec);act=read(a.activity_spec)
    parent=Path(cfg['parent']);bundle=Path(cfg['bundle']);recipe=read(bundle/'RECIPE.json')
    verify_files(bundle,read(bundle/'MANIFEST.json')['files'])
    if file_sha(bundle/'model.pt')!=cfg['source']:raise ValueError('SOURCE_MODEL')
    parent_spec=read(parent/'SPEC.json')
    verify_files(parent,{k:v for k,v in parent_spec['files'].items()
        if k.startswith(('data/','engine/')) or k in ('SUPPORT_RECIPE.json','RISK.json')})
    pins={str(a.producer_spec):file_sha(a.producer_spec),str(a.activity_spec):file_sha(a.activity_spec),
          str(bundle/'RECIPE.json'):file_sha(bundle/'RECIPE.json'),str(Path(__file__)):file_sha(__file__),
          str(Path(__file__).parents[1]/'app/subset_producer_contract.py'):file_sha(Path(__file__).parents[1]/'app/subset_producer_contract.py')}
    pins.update({v['path']:v['sha256'] for v in act['files'].values()})
    predictor=Path(act['predictor_root'])
    for p in (predictor/'tools').rglob('*.py'):
        if str(p) not in g['pins']:raise ValueError('UNPINNED_PRODUCER')
        pins[str(p)]=g['pins'][str(p)]
    cross_path=g['producer_settings']['cross_module'];pins[cross_path]=g['pins'][cross_path]
    for pair,path in g['external']['paths']['fx'].items():pins[path]=g['external']['config']['fx'][pair]['sha256']
    def check():
        for path,expected in pins.items():
            if file_sha(path)!=expected:raise ValueError('SOURCE_PIN:'+Path(path).name)
    check()
    binding=dict(schema='subset_producers372.v1',pins=pins,months=[6,12,24,48],
        source=cfg['source'],feature_selection='fixed71; upstream encodings fixed; no global re-selection',
        activity='purged_kfold5 embargo96 full input and label support; not online expanding fit')
    run_id=identity(binding)
    if (a.output/'SPEC.json').exists() and read(a.output/'SPEC.json')!=binding:raise ValueError('RESUME_IDENTITY')
    write(a.output/'SPEC.json',binding)
    import tools as package
    package.__path__=[str(predictor/'tools'),*package.__path__]
    AP=importlib.import_module('tools.activity_producer');F=importlib.import_module('tools.fold_local_producers')
    L2=importlib.import_module('tools.layer2_pool_h1');cross=load_module('subset_cross372',cross_path)
    for mod in (AP,F,L2):
        if not Path(mod.__file__).resolve().is_relative_to(predictor.resolve()):raise ValueError('IMPORT_IDENTITY')
    paths={k:Path(v['path']) for k,v in act['files'].items()}
    bars,X,hit,res,features=AP.load_inputs(paths['snapshot'].parent,paths['bars'].parent,
        paths['zigzag'].parent,paths['pool'],paths['decision'])
    stamps=bars.index.as_unit('ns');closes=stamps+pd.Timedelta(hours=1)
    if (closes>=pd.Timestamp('2025-01-01',tz='UTC')).any():raise ValueError('TEST_FORBIDDEN')
    base=pd.read_parquet(parent/'data/store.parquet').set_index('timestamp')
    if not pd.DatetimeIndex(base.index).equals(bars.index):raise ValueError('BAR_STORE_GRID')
    raw=bars.rename(columns={'OPEN':'BidOpen','HIGH':'BidHigh','LOW':'BidLow','CLOSE':'BidClose'}).copy()
    raw.index.name='timestamp';P=F.exact_parents(raw.reset_index())
    returns={}
    for pair,path in g['external']['paths']['fx'].items():
        prices=pd.read_parquet(path,filters=[('timestamp','<',pd.Timestamp('2025-01-01',tz='UTC'))]).set_index('timestamp').close
        returns[pair]=cross.hourly_features(prices.reindex(stamps),'fx.'+pair)['fx.'+pair+'.ret_1']
    R=pd.DataFrame(returns,index=stamps)[g['external']['config']['params']['factor_pairs']]
    columns=recipe['list']['columns'];lookback=AP.max_lag(features)+1;results=[]
    if any(c['consumer_ops'] for c in recipe['tensor']['columns']):raise ValueError('UNHANDLED_CONSUMER_OP')
    for months in (6,12,24,48):
        out=a.output/f'months_{months}';out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():
            old=read(out/'DONE.json')
            if old['identity']!=run_id:raise ValueError('ARM_IDENTITY')
            verify_files(out,old['files']);results.append(old['summary']);continue
        began=time.monotonic();end=pd.Timestamp('2024-01-01',tz='UTC');start=end-pd.DateOffset(months=months)
        allowed=np.asarray((closes>=start)&(closes<end));interval=[str(start.date()),str(end.date())]
        def progress(stage):
            write(a.output/'PROGRESS.json',dict(months=months,stage=stage,completed=len(results),at=time.time()))
        progress('FITTING_PRODUCERS')
        masks={'kalman':window_mask(allowed,1),'gmm':window_mask(allowed,25),
               'factor':window_mask(allowed,2)&np.isfinite(R.to_numpy()).all(axis=1)}
        for mask in masks.values():assert_fit_rows(np.flatnonzero(mask),allowed)
        kp=L2.fit_kalman(P,masks['kalman']);gp=L2.fit_gmm(P,masks['gmm']);fp=cross.pca_fit(R.loc[masks['factor']],3)
        if not kp['converged'] or not gp['converged']:raise ValueError('PRODUCER_NOT_CONVERGED')
        probability,models,fit_rows,ameta=activity_fit(AP,X,hit,res,allowed,lookback)
        progress('EXCLUDED_DATA_NEGATIVE_CONTROLS')
        changed=P.copy();changed.loc[~allowed]=changed.loc[~allowed]*1.13+0.03
        if identity(kp)!=identity(L2.fit_kalman(changed,masks['kalman'])):raise ValueError('KALMAN_FIT_SCOPE')
        if identity(gp)!=identity(L2.fit_gmm(changed,masks['gmm'])):raise ValueError('GMM_FIT_SCOPE')
        changedR=R.copy();changedR.loc[~allowed]*=99
        if identity(fp)!=identity(cross.pca_fit(changedR.loc[masks['factor']],3)):raise ValueError('FACTOR_FIT_SCOPE')
        xx=X.copy();xx.loc[~allowed]=xx.loc[~allowed]*11+7
        yy=hit.copy();yy[~allowed]=np.nan;rr=res.copy();rr[~allowed]=-1
        _,other,other_rows,_=activity_fit(AP,xx,yy,rr,allowed,lookback)
        for model,alt,rows,arows in zip(models,other,fit_rows,other_rows,strict=True):
            if not np.array_equal(rows,arows) or not np.array_equal(model.predict_proba(X.to_numpy()),alt.predict_proba(X.to_numpy())):
                raise ValueError('ACTIVITY_FIT_SCOPE')
        progress('BUILD_AND_RELOAD')
        state=dict(kalman=kp,gmm=gp,factor=fp,activity=models,activity_features=features,activity_fit_rows=fit_rows,interval=interval)
        joblib.dump(state,out/'PRODUCERS.joblib');loaded=joblib.load(out/'PRODUCERS.joblib')
        for original,restored in zip(models,loaded['activity'],strict=True):
            if not np.array_equal(original.predict_proba(X.to_numpy()),restored.predict_proba(X.to_numpy())):
                raise ValueError('ACTIVITY_RELOAD')
        def apply(parents,ret,params):
            factor=cross.pca_apply(ret,params['factor']).f1
            return pd.concat([L2.apply_kalman(parents,params['kalman'])[0][['kf.lc.level_minus_close']],
                L2.apply_gmm(parents,params['gmm'])[0],pd.DataFrame({'fac.fx_f1.ret_4':factor.rolling(4).sum(),
                    'fac.fx_f1.ret_24':factor.rolling(24).sum()},index=parents.index)],axis=1)
        derived=apply(P,R,state)
        if not np.array_equal(derived.to_numpy(),apply(P,R,loaded).to_numpy(),equal_nan=True):raise ValueError('PRODUCER_RELOAD')
        cut=int(np.flatnonzero(closes>=end)[0])+500
        pc=P.copy();rc=R.copy();pc.iloc[cut:]=pc.iloc[cut:]*1.02+0.1;rc.iloc[cut:]*=2
        altered=apply(pc,rc,state)
        if not np.array_equal(derived.iloc[:cut].to_numpy(),altered.iloc[:cut].to_numpy(),equal_nan=True):raise ValueError('APPLY_FUTURE_LEAK')
        derived['act.p_hit']=probability;rebuilt=base[columns].copy()
        for column in derived:rebuilt[column]=derived[column].to_numpy()
        unchanged=[c for c in columns if c not in derived]
        if len(derived.columns)!=8 or not np.array_equal(rebuilt[unchanged],base[unchanged],equal_nan=True):raise ValueError('UNCHANGED_COLUMNS')
        values=rebuilt.to_numpy();mean,std,constant,scale_rows=fixed_scaler(values,allowed)
        altered_values=values.copy();altered_values[~allowed]*=100
        repeated=fixed_scaler(altered_values,allowed)
        if not all(np.array_equal(x,y) for x,y in zip((mean,std,constant,scale_rows),repeated)):raise ValueError('SCALER_FIT_SCOPE')
        export=rebuilt.loc[closes>=start].reset_index();export.to_parquet(out/'store.parquet',index=False)
        write(out/'manifest.json',dict(schema='selected_subset372.v1',features=columns,columns=['timestamp']+columns,
            file_sha256=file_sha(out/'store.parquet'),fit_window=interval,parent_sha256=recipe['store']['sha256'],
            inherited63='fixed selected transforms and causal warmup; not refitted feature selection'))
        rec=copy.deepcopy(recipe);rec['store']=dict(file='store.parquet',sha256=file_sha(out/'store.parquet'),manifest_sha256=file_sha(out/'manifest.json'))
        rec['split']['train']=interval;rec['split']['periods']['TRAIN']=interval;rec['split']['support_status']='subset372_rebuilt'
        rec['scaler'].update(mean=mean.tolist(),std=std.tolist(),constant_columns=[columns[i] for i in constant],
            fit_period=interval,derivation='subset-only finite complete rows; zero std recorded and set to1')
        rec['provenance']=dict(parent_recipe_identity=recipe['identity'],derivation='subset372 producer refit; fresh policy required')
        rec['identity']='recipe_'+identity({k:rec[k] for k in IDENTITY_KEYS})[:16];validate_recipe(rec)
        write(out/'RECIPE.json',rec)
        inp=FrozenRecipeInputs.load(rec,out/'store.parquet',out/'manifest.json')
        fit=inp.valid&(inp.period=='TRAIN');val=inp.valid&(inp.period=='VALIDATION')
        if not fit.any() or not val.any() or np.any(inp.slot_ts_ns[fit,0]<start.value):raise ValueError('REBUILT_SUPPORT')
        np.savez_compressed(out/'FIT_ROWS.npz',**{k:np.flatnonzero(m) for k,m in masks.items()},scaler=scale_rows,
            **{f'activity_{i}':r for i,r in enumerate(fit_rows)},row_close_ns=closes.asi8)
        summary=dict(months=months,fit_window=interval,train_windows=int(fit.sum()),validation_windows=int(val.sum()),
            validation_support_sha256=hashlib.sha256(inp.row_ts_ns[val].astype('<i8').tobytes()).hexdigest(),
            columns=len(columns),groups=len(rec['representation']['resolved']['indices']),
            changed_columns=list(derived.columns),unchanged_columns=len(unchanged),activity=ameta,
            fit_counts={k:int(m.sum()) for k,m in masks.items()},constant_columns=rec['scaler']['constant_columns'],
            producer_converged=True,negative_controls_passed=True,reload_exact=True,
            seconds=time.monotonic()-began,neural_training_started=False,recipe_identity=rec['identity'])
        write(out/'RESULT.json',summary);check()
        files={p.name:file_sha(p) for p in out.iterdir() if p.is_file() and p.name!='DONE.json'}
        write(out/'DONE.json',dict(identity=run_id,files=files,summary=summary));results.append(summary)
        print(json.dumps(summary),flush=True)
    if len({r['validation_support_sha256'] for r in results})!=1:raise ValueError('UNPAIRED_VALIDATION_SUPPORT')
    write(a.output/'DONE.json',dict(identity=run_id,arms=results,training_launched=False,
        next_action='Fresh neural initialization and matched-update learning-curve runner; stores ready, do not load champion encoder weights'))
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',completed=4,at=time.time()))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('config','producer-spec','activity-spec','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.output/'FAILURE.json',dict(type=type(exc).__name__,message=str(exc),at=time.time()));raise
