"""Prepare all71 paired inputs with repaired activity and identical warm starts."""
import argparse
import copy
import fcntl
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from app.causal_stl_inputs import CausalSTLColumn
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.stl_full_pair import paired_coverage,assert_declared_changes
from app.structural_refit import verify_files
from app.transformed_feature_scope import context_fit_overlap
from tools import run_causal_stl_lane as lane
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now

FX_COLUMNS={'fac.fx_f1.ret_4','fac.fx_f1.ret_24','wv.usd.A4','hb.eur.inst_freq'}


def receipt(root):
    done=read(root/'DONE.json');verify_files(root,done['files']);return done


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=True)
    with (a.root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old=read(a.source_root/'CONFIG.json');ready=read(a.source_root/'PREPARED.json')
        verify_files(a.source_root,ready['files'])
        if old['identity']!=digest({k:v for k,v in old.items() if k!='identity'}):raise ValueError('FULL71_SOURCE_IDENTITY')
        for name,h in old['sources'].items():
            if name!='tools/run_causal_stl_lane.py' and file_sha(lane.ROOT/name)!=h:
                raise ValueError('FULL71_REUSE_CODE:'+name)
        spec,rec,market=lane.source(a)
        if (len(old['columns'])!=71 or rec['list']['columns']!=old['columns']
                or len(rec['representation']['resolved']['names'])!=28
                or file_sha(a.bundle/'model.pt')!=old['source_sha256']):raise ValueError('FULL71_RECIPE')
        receipt(a.activity);receipt(a.fx);receipt(a.propagation)
        act=read(a.activity/'RESULT.json');prop=read(a.propagation/'RESULT.json')
        if act['overlap_after']!=0 or not act['validation_exact']:raise ValueError('FULL71_ACTIVITY_REPAIR')
        if prop['pins']['parent_prepared']!=file_sha(a.source_root/'PREPARED.json') or prop['pins']['repair_done']!=file_sha(a.fx/'DONE.json'):
            raise ValueError('FULL71_FX_BINDING')
        with np.load(a.source_root/'RAW.npz') as f:
            before=f['values'].copy();times=f['times'].copy();indices=f['parent_indices'].copy()
        frame=pd.read_parquet(a.fx/'STORE_CANDIDATE.parquet',columns=['timestamp']+old['columns'],
            filters=[('timestamp','<',pd.Timestamp('2025-01-01',tz='UTC'))])
        if not np.array_equal(pd.DatetimeIndex(frame.timestamp).as_unit('ns').asi8+3600*10**9,times):raise ValueError('FULL71_FX_CLOCK')
        raw=frame[old['columns']].to_numpy(dtype=float,copy=True)
        for j,c in enumerate(rec['tensor']['columns']):
            if c['consumer_ops']==['DIFF']:raw[:,j]=np.r_[np.nan,np.diff(raw[:,j])]
            elif c['consumer_ops']:raise ValueError('FULL71_CONSUMER_OP')
        assert_declared_changes(before,raw,old['columns'],FX_COLUMNS)
        af=pd.read_parquet(a.activity/'ACTIVITY_CANDIDATE.parquet')
        if not np.array_equal(pd.DatetimeIndex(af.timestamp).as_unit('ns').asi8+3600*10**9,times):raise ValueError('FULL71_ACTIVITY_CLOCK')
        jact=old['columns'].index('act.p_hit');raw[:,jact]=af['act.p_hit'].to_numpy(float)
        assert_declared_changes(before,raw,old['columns'],FX_COLUMNS|{'act.p_hit'})
        with np.load(a.propagation/'REPAIRED_SEASONAL.npz') as f:
            seasonal=f['values'].copy()
            if not np.array_equal(times,f['times']):raise ValueError('FULL71_SEASONAL_CLOCK')
        bindings={name:file_sha(path/'DONE.json') for name,path in [('activity',a.activity),('fx',a.fx),('propagation',a.propagation)]}
        cfg=copy.deepcopy(old);cfg.pop('identity')
        cfg.update(schema='full71_stl_pair.v1',coverage_policy='full71_matched_available_v1',
            initialization='selected299_exact_weights',source_bindings=bindings,
            original_prepared=file_sha(a.source_root/'PREPARED.json'),historical_mixed_cot=True,
            live_qualified=False,minimum_support_fraction=0.8,activity_repaired=True,
            activity_max_context=act['context_rows'],source_changes=sorted(FX_COLUMNS|{'act.p_hit'}))
        cfg['sources'].update({n:file_sha(lane.ROOT/n) for n in (
            'tools/run_causal_stl_lane.py','tools/run_full_stl_pair.py','app/stl_full_pair.py')})
        cfg['identity']=digest(cfg)
        if (a.root/'CONFIG.json').exists() and read(a.root/'CONFIG.json')!=cfg:raise ValueError('FULL71_RESUME_BINDING')
        write(a.root/'CONFIG.json',cfg);write(a.root/'SOURCE_RECIPE.json',rec)
        if not (a.root/'RAW.npz').exists():np.savez_compressed(a.root/'RAW.npz',values=raw,times=times,parent_indices=indices)
        else:
            with np.load(a.root/'RAW.npz') as f:
                if not np.array_equal(raw,f['values'],equal_nan=True):raise ValueError('FULL71_RESUME_RAW')
        write(a.root/'PREPARED.json',dict(identity=cfg['identity'],files={n:file_sha(a.root/n) for n in ('CONFIG.json','SOURCE_RECIPE.json','RAW.npz')}))
        for j,name in enumerate(old['columns']):
            if j==jact:continue
            src=a.source_root/'columns'/f'{j:03d}';verify_done(src,old['identity'])
            if prop['pins']['column_receipts'][name]!=file_sha(src/'DONE.json'):raise ValueError('FULL71_COLUMN_BINDING')
            out=a.root/'columns'/f'{j:03d}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():verify_done(out,cfg['identity']);continue
            result=read(src/'RESULT.json');params=result['parameters']
            finite=np.flatnonzero(np.isfinite(seasonal[:,j]))
            probes=finite[np.linspace(0,len(finite)-1,5,dtype=int)]
            for t in probes:
                if CausalSTLColumn(params,raw[t-params['history']+1:t,j]).update(raw[t,j])!=seasonal[t,j]:
                    raise ValueError('FULL71_REUSED_PREFIX:'+name)
            result.update(identity=cfg['identity'],reused_from=old['identity'],prefix_rows=probes.tolist(),
                fx_parameters_frozen=name in FX_COLUMNS)
            np.savez_compressed(out/'SEASONAL.npz',values=seasonal[:,j]);write(out/'RESULT.json',result);seal(out,cfg['identity'])
        write(a.root/'PREPARATION_PROGRESS.json',dict(stage='ACTIVITY_STL_RETUNE',at=now(),reused_columns=70))
        a.column=jact;lane.column(a)
        lane.assemble(a)
        # Audit the repaired fitted producer over the ACTUAL resulting context.
        bundle=joblib.load(a.activity/'CONSUMER_ACTIVITY.joblib')
        params=read(a.root/'columns'/f'{jact:03d}'/'RESULT.json')['parameters']
        context=params['history']+24
        if context>act['context_rows']:raise ValueError('FULL71_ACTIVITY_CONTEXT')
        overlap,_,_=context_fit_overlap(bundle['fit_rows'],bundle['prediction_rows'],np.isfinite(raw[:,jact]),np.arange(context))
        with np.load(a.root/'prepared/VALUES.npz') as f:common=f['common']
        if overlap[indices][common].any():raise ValueError('FULL71_ACTIVITY_OVERLAP')
        coverage=paired_coverage(market.inputs.row_ts_ns,market.inputs.valid,common,rec['split']['periods'])
        report=dict(identity=cfg['identity'],features=71,groups=28,periods=coverage,
            monthly=read(a.root/'prepared/RESULT.json')['monthly_coverage'],
            activity_context=context,activity_overlap=0,source_bindings=bindings,
            source_weights=cfg['source_sha256'],historical_mixed_cot=True,
            initialization=cfg['initialization'],model_promoted=False,live_qualified=False,at=now())
        out=a.root/'qualification';write(out/'RESULT.json',report);seal(out,cfg['identity'])
        shutil.copy2(a.activity/'CONSUMER_ACTIVITY.joblib',a.root/'ACTIVITY_PRODUCER.joblib')
        print(report,flush=True)


def validate_pair(a,market):
    cfg=lane.authenticated(a.root);verify_done(a.root/'qualification',cfg['identity'])
    report=read(a.root/'qualification/RESULT.json')
    if (cfg['schema']!='full71_stl_pair.v1' or not cfg['activity_repaired']
            or report['activity_overlap']!=0 or report['features']!=71):raise ValueError('FULL71_TRAIN_GATE')
    source=read(a.bundle/'RECIPE.json')
    if market.recipe['list']['columns']!=source['list']['columns']:raise ValueError('FULL71_TRAIN_COLUMNS')
    if market.recipe['representation']!=source['representation']:raise ValueError('FULL71_TRAIN_GROUPS')
    prepared=read(a.root/'prepared/RESULT.json')
    if len({prepared['variants'][k]['support'] for k in ('raw','continuous')})!=1:raise ValueError('FULL71_UNPAIRED')
    if support_hash(market.inputs)!=prepared['variants']['raw']['support']:raise ValueError('FULL71_TRAIN_SUPPORT')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','train'))
    for name in ('root','source-root','activity','fx','propagation','parent','bundle'):
        p.add_argument('--'+name,type=Path,required=name in ('root','parent','bundle'))
    p.add_argument('--variant',choices=('raw','continuous'));p.add_argument('--expected-device-name')
    a=p.parse_args()
    try:
        if a.command=='prepare':prepare(a)
        else:lane.train(a)
    except Exception as exc:
        write(a.root/('FAILED_'+(a.variant or 'prepare')+'.json'),dict(error=repr(exc),at=now()));raise
