"""Reuse authenticated STL columns only with identical raw input and tuning scope."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from app.causal_stl_inputs import CausalSTLColumn
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.structural_refit import verify_files
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_causal_stl_lane import authenticated, column, assemble
from tools.run_discrete_curve_probe import read, write


def reusable_columns(old, new, before, after):
    for field in ('train_end','statsmodels','grid','origins','forecast_horizon','period_clock'):
        if old[field]!=new[field]:raise ValueError('STL_REUSE_SCOPE:'+field)
    if old['sources']['app/causal_stl_inputs.py']!=new['sources']['app/causal_stl_inputs.py']:
        raise ValueError('STL_REUSE_ALGORITHM')
    if not np.array_equal(before['times'],after['times']):raise ValueError('STL_REUSE_CLOCK')
    if before['values'].shape!=(len(before['times']),len(old['columns'])):
        raise ValueError('STL_REUSE_OLD_SHAPE')
    if after['values'].shape!=(len(after['times']),len(new['columns'])):
        raise ValueError('STL_REUSE_NEW_SHAPE')
    if len(set(old['columns']))!=len(old['columns']) or len(set(new['columns']))!=len(new['columns']):
        raise ValueError('STL_REUSE_DUPLICATE_COLUMN')
    matches=[]
    for j,name in enumerate(new['columns']):
        if name in old['columns']:
            i=old['columns'].index(name)
            if np.array_equal(before['values'][:,i],after['values'][:,j],equal_nan=True):
                matches.append((i,j))
    return matches


def run(a):
    cfg=authenticated(a.root);old=read(a.source/'CONFIG.json')
    if a.complete and (a.root/'prepared/DONE.json').exists():
        verify_done(a.root/'prepared',cfg['identity'])
        print(dict(identity=cfg['identity'],stage='ALREADY_ASSEMBLED',training_released=False),flush=True)
        return
    verify_files(a.source,read(a.source/'PREPARED.json')['files'])
    if old['identity']!=digest({k:v for k,v in old.items() if k!='identity'}):
        raise ValueError('STL_REUSE_SOURCE_IDENTITY')
    with np.load(a.source/'RAW.npz',allow_pickle=False) as old_arrays, np.load(a.root/'RAW.npz',allow_pickle=False) as new_arrays:
        # NPZ indexing decompresses on every access. Materialize once for all columns.
        before={k:old_arrays[k] for k in ('times','values')}
        after={k:new_arrays[k] for k in ('times','values')}
        pairs=reusable_columns(old,cfg,before,after)
        for i,j in pairs:
            src=a.source/'columns'/f'{i:03d}';dest=a.root/'columns'/f'{j:03d}'
            verify_done(src,old['identity']);prior=read(src/'RESULT.json')
            if prior['column']!=i or prior['name']!=cfg['columns'][j]:
                raise ValueError('STL_REUSE_RECEIPT_COLUMN')
            if (dest/'DONE.json').exists():verify_done(dest,cfg['identity']);continue
            if dest.exists() and any(dest.iterdir()):raise ValueError('STL_REUSE_BUSY_OR_PARTIAL')
            with np.load(src/'SEASONAL.npz',allow_pickle=False) as f:seasonal=f['values']
            raw=after['values'][:,j];params=prior['parameters']
            if seasonal.shape!=raw.shape:raise ValueError('STL_REUSE_RESULT_SHAPE')
            finite=np.flatnonzero(np.isfinite(seasonal))
            if not len(finite):raise ValueError('STL_REUSE_EMPTY')
            for t in finite[np.linspace(0,len(finite)-1,5,dtype=int)]:
                stream=CausalSTLColumn(params,raw[t-params['history']+1:t])
                if stream.update(raw[t])!=seasonal[t]:raise ValueError('STL_REUSE_ENDPOINT_PARITY')
            dest.mkdir(parents=True,exist_ok=True)
            for name in ('PARAMETERS.json','SEASONAL.npz'):shutil.copyfile(src/name,dest/name)
            write(dest/'RESULT.json',dict(prior,identity=cfg['identity'],column=j,
                reused=dict(source_identity=old['identity'],source_column=i,
                    receipt_sha256=file_sha(src/'DONE.json'),raw_equal=True,clock_equal=True,
                    tuning_scope_equal=True,endpoint_replays=5)))
            seal(dest,cfg['identity'])
    reused={j for _,j in pairs}
    report=dict(identity=cfg['identity'],source_identity=old['identity'],reused=len(pairs),
        compute=[dict(column=j,name=n) for j,n in enumerate(cfg['columns']) if j not in reused],
        training_released=False,code_sha256=file_sha(Path(__file__)))
    write(a.root/'REUSE.json',report);print(report,flush=True)
    if a.complete:
        if a.parent is None or a.bundle is None:raise ValueError('STL_COMPLETE_SOURCE_PATHS')
        for item in report['compute']:
            a.column=item['column'];column(a)
        assemble(a)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--complete',action='store_true')
    p.add_argument('--parent',type=Path);p.add_argument('--bundle',type=Path)
    args=p.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.root/'REUSE_FAILED.json',dict(error=repr(exc)));raise
