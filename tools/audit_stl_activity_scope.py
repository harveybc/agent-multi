"""Audit learned-feature cross-fit lineage before extending its temporal context."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest
from app.transformed_feature_scope import context_fit_overlap
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_causal_stl_lane import authenticated
from tools.run_discrete_curve_probe import read, write


def run(a):
    cfg=authenticated(a.root)
    receipt=read(a.activity.parent/'RESULT.json')
    if file_sha(a.activity)!=receipt['weights_sha256']:raise ValueError('ACTIVITY_WEIGHTS')
    bundle=joblib.load(a.activity)
    j=cfg['columns'].index('act.p_hit');folder=a.root/'columns'/f'{j:03d}'
    verify_done(folder,cfg['identity']);parameters=read(folder/'RESULT.json')['parameters']
    prepared=read(a.root/'prepared/RESULT.json');verify_done(a.root/'prepared',cfg['identity'])
    with np.load(a.root/'RAW.npz',allow_pickle=False) as raw:
        x=raw['values'][:,j];times=raw['times'];indices=raw['parent_indices']
    if not np.array_equal(times,bundle['times_ns']+3600*10**9):raise ValueError('ACTIVITY_CLOCK')
    with np.load(a.root/'prepared/VALUES.npz',allow_pickle=False) as packed:common=packed['common']
    w=cfg['original_window'];raw_offsets=w['stride_rows']*np.arange(w['observations'])
    seasonal_offsets=np.unique((raw_offsets[:,None]+np.arange(parameters['history'])[None,:]).ravel())
    reports={}
    for name,offsets in [('raw',raw_offsets),('seasonal',seasonal_offsets)]:
        overlap,lags,owners=context_fit_overlap(bundle['fit_rows'],bundle['prediction_rows'],np.isfinite(x),offsets)
        selected=overlap[indices]&common
        samples=np.flatnonzero(selected)[:12]
        reports[name]=dict(source_rows_with_overlap=int(overlap.sum()),
            paired_evaluation_rows_with_overlap=int(selected.sum()),
            validation_rows_with_overlap=int((overlap[cfg['train_end']:]).sum()),
            maximum_lag=int(offsets.max()),examples=[dict(row=int(indices[t]),
                target_time_ns=int(times[indices[t]]),lag=int(lags[indices[t]]),
                context_estimator=int(owners[indices[t]-lags[indices[t]]])) for t in samples])
    result=dict(schema='stl_activity_context_scope.v1',lane_identity=cfg['identity'],
        activity_sha256=file_sha(a.activity),raw_sha256=file_sha(a.root/'RAW.npz'),
        prepared_receipt_sha256=file_sha(a.root/'prepared/DONE.json'),reports=reports,
        passed=reports['seasonal']['paired_evaluation_rows_with_overlap']==0,
        limitation='Direct target-row membership witness, not a complete causal certification of every upstream dependency',
        training_released=False,source_changed=False,
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'app/transformed_feature_scope.py','tools/audit_stl_activity_scope.py')})
    result['identity']=digest(result)
    write(a.output/'RESULT.json',result);seal(a.output,result['identity']);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','activity','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
