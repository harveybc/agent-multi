"""Replay prequalified FIT witnesses on full accounts without fitting or promotion."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.decision_boundary_reachability import temporary_head_step
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, support_hash
from app.selected_input_market import market_for
from app.structural_refit import load_bundle, export_bundle, verify_files
from app.three_signal_learning import emit, environment, evaluate
from app.frozen_signal_masks import prepared_signals
from tools.attribute_ranking_improvement import attribute_pair
from tools.diagnose_selected_refinement import head
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import configure, read, write, now


def fixed_candidates(probe):
    if not probe['source_restored_exactly'] or probe['account_evaluated']:
        raise ValueError('ACCOUNT_REQUIRES_FIT_ONLY_PROBE')
    result=[(r['panel_index'],r['target']) for r in probe['results'] if r['qualified']]
    if len(result)!=len(set(result)) or not result:
        raise ValueError('ACCOUNT_WITNESS_LIST')
    return result


def run(a):
    started=time.monotonic();configure(304)
    probe=read(a.probe/'REPORT.json');verify_done(a.probe,probe['identity'])
    cfg=read(a.diagnostic/'CONFIG.json');verify_done(a.diagnostic,cfg['identity'])
    spec=read(a.parent/'SPEC.json');verify_files(a.parent,spec['files'])
    floor=read(a.floor/'CONFIG.json');verify_done(a.floor,floor['identity'])
    source_sha=file_sha(a.bundle/'model.pt')
    if (probe['source']!=cfg['identity'] or probe['source_sha256']!=source_sha
            or cfg['floor']!=floor['identity'] or spec['identity']!=floor['parent_identity']):
        raise ValueError('ACCOUNT_WITNESS_LINEAGE')
    exported=read(a.bundle.parent/'RESULT.json');verify_done(a.bundle.parent,exported['identity'])
    if cfg['export']!=exported['identity']:raise ValueError('ACCOUNT_EXPORT_LINEAGE')
    candidates=fixed_candidates(probe)
    model,_=load_bundle(a.bundle,'cpu');market=market_for(a.parent,spec,model.recipe)
    if support_hash(market.inputs)!=floor['support_sha256']:raise ValueError('ACCOUNT_SUPPORT')
    periods=market.recipe['split']['periods']
    if set(periods)!={'TRAIN','VALIDATION'}:raise ValueError('ACCOUNT_PERIODS')
    risk=read(a.parent/'RISK.json')
    config=dict(probe=probe['identity'],diagnostic=cfg['identity'],source_sha256=source_sha,
        candidates=candidates,support_sha256=support_hash(market.inputs),periods=periods,
        code={p:file_sha(Path(__file__).resolve().parents[1]/p) for p in (
            'tools/replay_boundary_account.py','app/decision_boundary_reachability.py',
            'app/three_signal_learning.py','tools/attribute_ranking_improvement.py',
            'app/selected_input_market.py')})
    identity=digest(config);config['identity']=identity
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=config:
        raise ValueError('ACCOUNT_OUTPUT_COLLISION')
    write(a.output/'CONFIG.json',config)
    if (a.output/'DONE.json').exists():verify_done(a.output,identity);return
    def progress(stage,**kw):
        write(a.output/'PROGRESS.json',dict(stage=stage,at=now(),seconds=time.monotonic()-started,**kw))
    def make(period):
        return environment(market,risk,spec['design']['horizon'],periods[period],model.effective['gamma'])
    def measure(logits,path):
        prepared=prepared_signals(logits)
        return {p:evaluate(make(p),logits,save=path/p,prepared=prepared) for p in periods}
    progress('REFERENCE_INFERENCE')
    baseline,_=emit(model,market.inputs);reference=measure(baseline,a.output/'reference')
    params=[p for _,p in head(model)]
    saved={k:v.detach().clone() for k,v in model.loss.state_dict().items()}
    valid=market.inputs.valid
    probe_rows=np.flatnonzero(valid)[np.linspace(0,valid.sum()-1,32,dtype=int)]
    results=[]
    with np.load(a.diagnostic/'LOCAL_DIRECTIONS.npz',allow_pickle=False) as directions:
        for i,target in candidates:
            name=f'{i}_{target}_guarded';out=a.output/name
            progress('CANDIDATE_INFERENCE',candidate=name)
            with temporary_head_step(params,directions[name]):
                logits,_=emit(model,market.inputs)
                export_bundle(out/'bundle',model,market.inputs.windows[probe_rows],
                    dict(source=source_sha,witness=name,diagnostic_only=True))
                cold,_=load_bundle(out/'bundle','cpu');cold_logits,_=emit(cold,market.inputs)
                if not np.array_equal(logits[valid],cold_logits[valid]):
                    raise ValueError('ACCOUNT_COLD_LOGITS')
                metrics=measure(logits,out/'account');cold_metrics=measure(cold_logits,out/'cold')
                if metrics!=cold_metrics:raise ValueError('ACCOUNT_COLD_METRICS')
                attribution={}
                for period in periods:
                    progress('ATTRIBUTION',candidate=name,period=period)
                    value=attribute_pair(lambda p=period:make(p),market.inputs.row_ts_ns,
                        a.output/'reference'/period,out/'account'/period,
                        lambda **kw:progress('ATTRIBUTION',candidate=name,period=period,**kw))
                    write(out/f'{period}_ATTRIBUTION.json',value)
                    attribution[period]={k:v for k,v in value.items() if k!='events'}
                results.append(dict(witness=name,metrics=metrics,attribution=attribution,
                    cold_logits_exact=True,cold_metrics_exact=True,
                    changed_actions={p:int((logits[m].argmax(-1)!=baseline[m].argmax(-1)).sum())
                        for p in periods for m in [valid & (market.inputs.period==p)]}))
            if any(not torch.equal(v,model.loss.state_dict()[k]) for k,v in saved.items()):
                raise ValueError('ACCOUNT_SOURCE_NOT_RESTORED')
    if file_sha(a.bundle/'model.pt')!=source_sha:raise ValueError('ACCOUNT_SOURCE_CHANGED')
    result=dict(identity=identity,reference=reference,candidates=results,source_restored_exactly=True,
        model_promoted=False,optimizer_steps=0,test_read=False,seconds=time.monotonic()-started)
    write(a.output/'RESULT.json',result);seal(a.output,identity);progress('COMPLETE')
    print(result,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('probe','diagnostic','floor','bundle','parent','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:run(args)
    except Exception as exc:
        write(args.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
