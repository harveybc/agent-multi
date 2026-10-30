#!/usr/bin/env python3
"""Resumable frozen close interventions, with native fixed-position optima."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.suppressed_close_counterfactuals import SuppressedCloseWorld,compare_intervention
from app.zigzag_opportunities import engine,sha
from tools.collect_causal_history_lanes import authenticate
from tools.diagnose_frozen_history_policy import verify_sources
from tools.optimize_dynamic_account import digest
from tools.run_crossed_account_queue import load,save
from tools.zigzag_opportunity_deap import write

EXTRA=('app/suppressed_close_counterfactuals.py','app/frozen_policy_diagnostics.py',
       'app/sequential_account_pareto.py','tools/diagnose_suppressed_closes.py',
       'tools/diagnose_frozen_history_policy.py','tools/collect_causal_history_lanes.py')


def unit(job):
    path=Path(job['output'])/f"WORLD_{job['world']:03d}.json"
    identity=digest(job)
    cached=load(path,identity)
    if cached is not None: return cached
    engine(Path(job['engine_root']))
    w=SuppressedCloseWorld(job['policies'][0]['archived']['spec'],job['config'],job['mode'])
    results={}
    for policy in job['policies']:
        if policy['archived']['spec']!=job['policies'][0]['archived']['spec']:
            raise ValueError('WORLD_ALIGNMENT')
        w.intervention=None;w.collect_suppressed=True;w.diagnostic=Counter()
        baseline=w.run(policy['genes'],details=True)
        if any(value!=policy['archived'][name] for name,value in baseline.items()):
            raise ValueError('SUPPRESSED_BASELINE_REPLAY')
        proposals=w.proposals;counts=dict(w.diagnostic);pairs=[]
        for proposal in proposals:
            target={k:proposal[k] for k in ('bar','entry_bar','direction')}
            w.intervention=target;w.collect_suppressed=False
            alternative=w.run(policy['genes'],details=True,verify=True)
            comparison=compare_intervention(baseline,alternative,target)
            matching=next(t for t in alternative['ledger'] if t['entry_bar']==target['entry_bar'])
            if abs(matching['net']-proposal['immediate'])>1e-9:
                raise ValueError('FIXED_AND_ACCOUNT_CLOSE_DISAGREE')
            pairs.append(dict(proposal=proposal,comparison=comparison,account=alternative))
        results[policy['key']]=dict(counts=counts,pairs=pairs,baseline_orders=baseline['orders'])
    save(path,identity,results)
    return results


def summarize(rows):
    def mean(values): return sum(values)/len(values) if values else None
    def sign(x): return 'helpful' if x>1e-9 else 'harmful' if x< -1e-9 else 'neutral'
    return dict(ticks=len(rows),positions=len({(r['world'],r['proposal']['entry_bar']) for r in rows}),
        local=dict(Counter(sign(r['proposal']['delta_vs_hold']) for r in rows)),
        account=dict(Counter(sign(r['comparison']['account_delta_usd']) for r in rows)),
        local_account_sign_disagreement=sum(sign(r['proposal']['delta_vs_hold'])!=
            sign(r['comparison']['account_delta_usd']) for r in rows),
        preempted=sum(r['proposal']['preempted'] for r in rows),
        mean_immediate_regret_risk=mean([r['proposal']['regret_risk'] for r in rows]),
        mean_passive_regret_risk=mean([r['proposal']['passive_regret_risk'] for r in rows]),
        mean_position_delta_usd=mean([r['comparison']['position_delta_usd'] for r in rows]),
        mean_account_delta_usd=mean([r['comparison']['account_delta_usd'] for r in rows]),
        mean_continuation_delta_usd=mean([r['comparison']['continuation_delta_usd'] for r in rows]),
        by_direction=dict(Counter(str(r['proposal']['direction']) for r in rows)),
        passive_reason=dict(Counter(r['proposal']['hold_reason'] for r in rows)))


def run(source,engine_root,output,workers):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    receipt=authenticate(source);sid=receipt['identity']
    source_manifest=json.loads((source/'MANIFEST.json').read_text())
    verify_sources(source_manifest,ROOT,engine_root)
    config_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(config_path)!=source_manifest['config_sha256']: raise ValueError('CONFIG_CHANGED')
    config=json.loads(config_path.read_text());front=load(source/'FRONTIER.json',sid)
    manifest=dict(schema='suppressed_close_diagnostic.v1',source_identity=sid,
        source_complete_sha256=receipt['complete_sha256'],code={p:sha(ROOT/p) for p in EXTRA},
        selection='raw_pass_AND_actual_HOLD',confirmation_used_for_tuning=False,
        scope='POST_HOC_ONE_INTERVENTION_PER_REPLAY_NOT_A_DEPLOYABLE_POLICY')
    identity=digest(manifest);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'LOCK').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=load(output/'MANIFEST.json',identity)
    if old is None: save(output/'MANIFEST.json',identity,manifest)
    elif old!=manifest: raise ValueError('DIAGNOSTIC_MANIFEST_CHANGED')
    archives={k:json.loads(gzip.decompress((source/f'CONFIRM_{k}.json.gz').read_bytes()))['worlds']
              for k in front['representatives']}
    count=len(source_manifest['confirmation'])
    if any(len(a)!=count for a in archives.values()): raise ValueError('WORLD_COUNT')
    jobs=[dict(world=i,output=str(output),engine_root=str(engine_root),config=config,
        mode=receipt['mode'],identity=identity,policies=[dict(key=k,genes=front['genes'][k],archived=archives[k][i])
        for k in front['representatives']]) for i in range(count)]
    accumulated={k:[] for k in front['representatives']};counts={k:Counter() for k in accumulated}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(unit,jobs)):
            for k,v in result.items():
                counts[k].update(v['counts'])
                accumulated[k].extend(dict(world=i,proposal=p['proposal'],comparison=p['comparison']) for p in v['pairs'])
            if (i+1)%12==0: print('worlds',i+1,'/',count,flush=True)
    if any(sha(ROOT/p)!=h for p,h in manifest['code'].items()): raise ValueError('CODE_MUTATED')
    verify_sources(source_manifest,ROOT,engine_root)
    report=dict(**manifest,identity=identity,baseline_replays=count*len(accumulated),
        intervention_replays=sum(map(len,accumulated.values())),
        rows=[dict(key=k,summary=summarize(rows),gate_counts=dict(counts[k]),proposals=rows)
              for k,rows in accumulated.items()],
        units={p.name:sha(p) for p in sorted(output.glob('WORLD_*.json'))})
    write(output/'REPORT.json',report)
    lock.close()
    print(json.dumps([{k:v for k,v in r.items() if k in ('key','summary')} for r in report['rows']]),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','engine-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.source,a.engine_root,a.output,a.workers)
