"""CPU review of frozen critic/actor geometry and sparse native target coverage."""
import argparse
from collections import Counter
import csv
import fcntl
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from app.action_ranking_contrast import ranking_metrics
from app.critic_coverage_geometry import geometry,distances,nearest_values,label_energy,encoder_identity
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest,support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle,verify_files
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_role_routed_critic import cache_embeddings

ROOT=Path(__file__).resolve().parents[1]


def describe(values):
    v=np.asarray(values,float)
    return dict(n=len(v),minimum=float(v.min()),median=float(np.median(v)),
        p90=float(np.quantile(v,.9)),maximum=float(v.max()))


def run(a):
    start=time.monotonic();configure(319)
    cfg318=read(a.previous/'CONFIG.json');verify_done(a.previous,cfg318['identity'])
    ac=read(a.actor_cache/'CONFIG.json');verify_done(a.actor_cache,ac['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    sha=file_sha(a.bundle/'model.pt')
    if sha!=cfg318['source_sha256'] or sha!=ac['source_sha256']:
        raise ValueError('GEOMETRY_SOURCE_IDENTITY')
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe)
    if model.recipe['list']['n']!=71 or support_hash(market.inputs)!=ac['support_sha256']:
        raise ValueError('GEOMETRY_SUPPORT')
    cfg=dict(schema='critic_coverage_geometry.v1',previous=cfg318['identity'],actor_cache=ac['identity'],
        source_sha256=sha,support=ac['support_sha256'],per_year=512,permutations=99,seed=319,
        neighbors=[1,5],geometry=['native','fit_standardized'],device='cpu',optimizer_steps=0,
        code={n:file_sha(ROOT/n) for n in ('tools/diagnose_critic_coverage_geometry.py',
            'app/critic_coverage_geometry.py','app/head_spillover.py','tools/run_role_routed_critic.py')})
    cfg['identity']=digest(cfg)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('GEOMETRY_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-start,**kw))
    records=read(a.previous/'panel/RESULT.json')['records']
    train=np.flatnonzero(market.inputs.valid&(market.inputs.period=='TRAIN'))
    years=pd.DatetimeIndex(market.inputs.row_ts_ns,tz='UTC').year.to_numpy()
    cohort=[]
    for year in range(2020,2024):
        indices=train[years[train]==year]
        cohort.extend(indices[np.linspace(0,len(indices)-1,min(512,len(indices)),dtype=int)].tolist())
    cohort=np.array(sorted(set(cohort)|{r['row'] for r in records}))
    if not set(cohort)<=set(train):raise ValueError('GEOMETRY_NONTRAIN')
    lookup={int(row):i for i,row in enumerate(cohort)}
    ix=np.array([lookup[r['row']] for r in records]);record_rows=np.array([r['row'] for r in records])
    for r in records:
        if int(market.inputs.row_ts_ns[r['row']])!=r['timestamp']:raise ValueError('GEOMETRY_ROW_IDENTITY')
    saved={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    diagnostic_state=torch.load(a.previous/'A_19_0.01/SELECTED.pt',map_location='cpu',weights_only=True)['state']
    counts=dict(actor_head=sum(v.numel() for k,v in saved.items() if k.startswith('actor_network_params.') and '.head.' in k),
        native_twin_heads=sum(v.numel() for k,v in saved.items() if k.startswith('qvalue_network_params.') and '.head.' in k),
        diagnostic_six_heads=sum(v.numel() for v in diagnostic_state.values()))
    progress('CRITIC_EMBEDDINGS',rows=len(cohort))
    z,_,errors=cache_embeddings(model,torch.tensor(market.inputs.windows[cohort]))
    with np.load(a.actor_cache/'TRAIN_ARRAYS.npz',allow_pickle=False) as old:
        if not np.array_equal(old['rows'],train):raise ValueError('GEOMETRY_ACTOR_ROWS')
        at=np.searchsorted(train,cohort)
        actor=old['core'][at].copy();hidden=old['hidden_source'][at].copy()
        logits=old['source_logits'][np.searchsorted(train,record_rows)].copy()
    representations=dict(raw_window=market.inputs.windows[cohort].reshape(len(cohort),-1),
        actor_core=actor,actor_hidden=hidden,critic0=z[0].numpy(),critic1=z[1].numpy())
    np.savez_compressed(a.output/'EMBEDDINGS.npz',rows=cohort,**representations)
    targets=np.array([[v['reward_return'] for v in r['outcomes']] for r in records])
    results={};flat=[];permutations={}
    for fold in ('A','B'):
        groups={k:np.array(v,int) for k,v in read(a.previous/f'FOLD_{fold}.json').items()}
        fit=groups['fit'];audit=groups['audit'];fy={records[i]['year'] for i in fit}
        if any(records[i]['original_role']!='fit' for i in fit):raise ValueError('GEOMETRY_FIT_ROLE')
        rng=np.random.default_rng(319)
        perms=np.array([rng.permutation(len(fit)) for _ in range(99)]);permutations[fold]=perms.tolist()
        labels=dict(fit=label_energy(targets[fit]),audit=label_energy(targets[audit]),
            fit_quarter_counts=dict(Counter(f"{records[i]['year']}Q{pd.Timestamp(records[i]['timestamp']).quarter}" for i in fit)),
            fit_kind_counts=dict(Counter(records[i]['kind'] for i in fit)),
            fit_feature_window_overlap_pairs=sum(records[i]['window_start']<=records[j]['timestamp']
                and records[j]['window_start']<=records[i]['timestamp'] for n,i in enumerate(fit) for j in fit[n+1:]),
            fit_origin_coverage={str(y):dict(origins=sum(records[i]['year']==y for i in fit),
                valid_train_windows=int((years[train]==y).sum())) for y in sorted(fy)})
        baseline=dict(actor=ranking_metrics(logits[audit],targets[audit]),
            fit_mean=ranking_metrics(np.tile(targets[fit].mean(0),(len(audit),1)),targets[audit]))
        result=dict(labels=labels,baselines=baseline,representations={})
        for name,values in representations.items():
            for standardized in (False,True):
                key=name+('/standardized' if standardized else '/native')
                progress('GEOMETRY',fold=fold,representation=key)
                f,q,info=geometry(values[ix[fit]],values,standardized)
                d=distances(f,q);within=distances(f,f);np.fill_diagonal(within,np.inf)
                ref=within.min(1);threshold=float(np.quantile(ref,.95))
                same=np.array([i for i,row in enumerate(cohort) if years[row] in fy and row not in set(record_rows[fit])])
                cover=dict(fit_leave_one_out=describe(ref),same_year_nonfit=describe(d[same].min(1)),
                    stop=describe(d[ix[groups['stop']]].min(1)),audit=describe(d[ix[audit]].min(1)),
                    audit_fraction_beyond_fit_loo_p95=float((d[ix[audit]].min(1)>threshold).mean()),
                    same_year_fraction_beyond_fit_loo_p95=float((d[same].min(1)>threshold).mean()))
                probes={}
                for k in (1,5):
                    pred,neighbors=nearest_values(d[ix[audit]],targets[fit],k)
                    metrics=ranking_metrics(pred,targets[audit])
                    shuffled=[ranking_metrics(targets[fit][perm][neighbors].mean(1),targets[audit])['mean_regret'] for perm in perms]
                    probes[str(k)]=dict(metrics=metrics,negative_control_regret=describe(shuffled),
                        shuffled_fraction_at_or_below_observed=float((np.array(shuffled)<=metrics['mean_regret']).mean()),
                        neighbors_source_indices=[[records[int(fit[j])]['source_index'] for j in row] for row in neighbors])
                    flat.append(dict(fold=fold,representation=name,geometry='standardized' if standardized else 'native',
                        k=k,fit_rows=len(fit),width=values.shape[1],effective_rank=info['spectrum']['effective_rank'],
                        dims99=info['spectrum']['dimensions_99pct'],audit_regret=metrics['mean_regret'],
                        audit_rmse=metrics['rmse'],permutation_median=float(np.median(shuffled)),
                        audit_outside=cover['audit_fraction_beyond_fit_loo_p95']))
                result['representations'][key]=dict(geometry=info,coverage=cover,neighbors=probes)
        results[fold]=result
    if (any(not torch.equal(v,saved[k]) for k,v in model.loss.state_dict().items())
            or file_sha(a.bundle/'model.pt')!=sha):raise ValueError('GEOMETRY_SOURCE_MUTATED')
    write(a.output/'PERMUTATIONS.json',permutations)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],cohort_rows=len(cohort),
        train_rows=len(train),critic_parity_errors=errors,encoder_identity=encoder_identity(model.loss.state_dict()),
        activation_difference_actor_critics=[float(np.max(np.abs(actor-z[i].numpy()))) for i in (0,1)],
        head_parameter_counts=counts,folds=results,source_exact=True,
        optimizer_steps=0,model_promoted=False,validation_read=False,test_read=False,historical_mixed_cot=True))
    with (a.output/'SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# Frozen representation review','',
        'Descriptive nearest-neighbor probes, no training or promotion. All variants reported.','',
        '| Fold | Representation | Geometry | K | Effective rank | Audit regret | Shuffle median |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for r in flat:
        lines.append(f"| {r['fold']} | {r['representation']} | {r['geometry']} | {r['k']} | {r['effective_rank']:.3f} | {r['audit_regret']:.6f} | {r['permutation_median']:.6f} |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    seal(a.output,cfg['identity']);progress('COMPLETE')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','bundle','previous','actor-cache','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise
