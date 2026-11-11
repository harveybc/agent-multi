"""Authenticate completed all-state contrast and attribute every fixed endpoint."""
import argparse
import fcntl
from pathlib import Path
import time

import numpy as np

from app.all_state_refinement import constant_controls, validate_panel
from app.frozen_recipe_inputs import file_sha
from app.rl_subset import digest, project_inputs, support_hash
from app.structural_refit import verify_files
from app.three_signal_learning import environment
from tools.attribute_ranking_improvement import attribute_pair
from tools.run_bayesian_head_contrast import seal, verify_done
from tools.run_discrete_curve_probe import read, write, configure, now
from tools.run_structural_sac_refit import market_for


def run(a):
    configure(281);started=time.monotonic()
    cfg=read(a.source/'CONFIG.json');verify_done(a.source,cfg['identity'])
    result=read(a.source/'RESULT.json')
    if result['identity']!=cfg['identity']:raise ValueError('ALL_STATE_REPORT_RESULT')
    parent=read(a.parent/'SPEC.json');verify_files(a.parent,parent['files'])
    recipe=read(a.bundle/'RECIPE.json')
    if (parent['identity']!=cfg['parent_identity'] or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']
            or recipe['identity']!=cfg['recipe_identity']):raise ValueError('ALL_STATE_REPORT_SOURCE')
    market=market_for(a.parent,parent);market.inputs=project_inputs(market.inputs,recipe);market.recipe=recipe
    if support_hash(market.inputs)!=cfg['support_sha256']:raise ValueError('ALL_STATE_REPORT_SUPPORT')
    config=dict(source=cfg['identity'],receipt_sha256=file_sha(a.source/'DONE.json'),
        code=file_sha(__file__),controls_code=file_sha(Path(__file__).resolve().parents[1]/'app/all_state_refinement.py'),
        endpoint=8,periods=['TRAIN','VALIDATION'],seeds=[19,43],
        scope='post-run symmetric diagnostic; no fitting, checkpoint selection or promotion')
    identity=digest(config)
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=config:raise ValueError('ALL_STATE_REPORT_CONFIG')
    if (a.output/'DONE.json').exists():verify_done(a.output,identity);return
    write(a.output/'CONFIG.json',config)
    records=read(a.source/'panel/RECORDS.json');validate_panel(records)
    controls={}
    for role in ('fit','monitor'):
        for state in ('all','flat_allowed','flat_restricted','long','short'):
            ix=[]
            for i,r in enumerate(records):
                allowed=all(o['first_effect']['entries']>0 for o in r['outcomes'][1:])
                label=('long' if r['position']==1 else 'short') if r['position'] else ('flat_allowed' if allowed else 'flat_restricted')
                if r['role']==role and state in ('all',label):ix.append(i)
            controls[f'{role}:{state}']=dict(n=len(ix),reward=constant_controls(
                [[o['reward_return'] for o in records[i]['outcomes']] for i in ix])) if ix else dict(n=0)
    write(a.output/'CONSTANT_CONTROLS.json',controls)
    risk=read(a.parent/'RISK.json');accounts=[]
    for arm in result['arms']:
        for period in config['periods']:
            key=f"{arm['arm']}_{period}";out=a.output/'attribution'/key
            if (out/'DONE.json').exists():verify_done(out,identity);accounts.append(read(out/'SUMMARY.json'));continue
            def progress(**kw):
                row=dict(stage=key,seconds=time.monotonic()-started,at=now(),**kw)
                write(a.output/'PROGRESS.json',row);print(row,flush=True)
            if arm['seed']==19 and period=='TRAIN':
                prior=a.source/'attribution'/arm['arm']
                events=read(prior/'EVENTS.json');summary=read(prior/'SUMMARY.json')
            else:
                value=attribute_pair(lambda:environment(market,risk,parent['design']['horizon'],
                    recipe['split']['periods'][period],cfg['effective']['gamma']),market.inputs.row_ts_ns,
                    a.source/'reference'/period,a.source/arm['arm']/'actor_0008'/period,progress)
                events=value['events'];summary={k:v for k,v in value.items() if k!='events'}
            # Raw symbol is not an order: BUY also closes a short or holds a long.
            effective=[e for e in events if abs(e['contribution_usd'])>1e-6]
            effects=[dict(timestamp=e['timestamp'],old_signal=e['old_signal'],new_signal=e['new_signal'],
                position=0 if e['state']['position'] is None else e['state']['position']['direction'],
                old_effect=e['prefix_old']['first_effect'],new_effect=e['prefix_new']['first_effect'],
                contribution_usd=e['contribution_usd']) for e in effective]
            summary=dict(summary,arm=arm['arm'],period=period,helpful=sum(e['contribution_usd']>1e-6 for e in events),
                harmful=sum(e['contribution_usd']< -1e-6 for e in events),effective_events=effects)
            write(out/'EVENTS.json',events);write(out/'SUMMARY.json',summary);seal(out,identity);accounts.append(summary)
    summary=dict(identity=identity,source_identity=cfg['identity'],reference=result['reference'],
        endpoints=[dict(arm=r['arm'],seed=r['seed'],critic=r['critic'],entropy=r['entropy'],**r['points'][-1]) for r in result['arms']],
        accounts=accounts,constant_controls=controls,source_unchanged=True,model_promoted=False,
        worker_seconds=result['seconds'],report_seconds=time.monotonic()-started)
    verify_done(a.source,cfg['identity'])
    write(a.output/'SUMMARY.json',summary);seal(a.output,identity)
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',identity=identity,at=now(),accounts=len(accounts)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','parent','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(error=repr(exc),at=now()));raise
