#!/usr/bin/env python3
"""Diagnose frozen confirmation gates without changing actions or coefficients."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.causal_history_account import HISTORY_GENES
from app.frozen_policy_diagnostics import DiagnosticWorld
from app.zigzag_opportunities import engine,sha
from tools.collect_causal_history_lanes import authenticate
from tools.run_crossed_account_queue import load
from tools.zigzag_opportunity_deap import write


def verify_sources(manifest,root,engine_root):
    groups=[(manifest['sources'],root),
            (manifest['native_sources']['agent_multi'],root),
            (manifest['native_sources']['heuristic_strategy'],engine_root)]
    for files,base in groups:
        for name,digest in files.items():
            if Path(name).is_absolute() or '..' in Path(name).parts or sha(base/name)!=digest:
                raise ValueError('DIAGNOSTIC_SOURCE_CHANGED:'+name)


def world_diagnostic(job):
    engine(Path(job['engine_root']))
    rows=job['policies']
    w=DiagnosticWorld(rows[0]['archived']['spec'],job['config'],job['mode'])
    result={}
    for row in rows:
        if row['archived']['spec']!=rows[0]['archived']['spec']:
            raise ValueError('DIAGNOSTIC_WORLD_ALIGNMENT')
        w.diagnostic=Counter()
        actual=w.run(row['genes'],details=True)
        for field,value in actual.items():
            if value!=row['archived'][field]: raise ValueError('DIAGNOSTIC_REPLAY:'+field)
        result[row['key']]=dict(counts=dict(w.diagnostic),horizons=row['archived']['horizons'])
    return result


def run(source,engine_root,output,workers=2):
    if output.exists(): raise ValueError('NEW_DIAGNOSTIC_ROOT_REQUIRED')
    if workers not in (1,2,3,4): raise ValueError('DIAGNOSTIC_WORKER_BUDGET')
    receipt=authenticate(source);identity=receipt['identity']
    manifest=json.loads((source/'MANIFEST.json').read_text())
    verify_sources(manifest,ROOT,engine_root)
    config_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(config_path)!=manifest['config_sha256']: raise ValueError('DIAGNOSTIC_CONFIG')
    config=json.loads(config_path.read_text());engine(engine_root)
    front=load(source/'FRONTIER.json',identity)
    candidates={json.loads(line)['key']:json.loads(line)['result']
                for line in (source/'CANDIDATES.jsonl').read_text().splitlines()}
    output.mkdir(parents=True)
    archives={}
    for key in front['representatives']:
        archives[key]=json.loads(gzip.decompress((source/f'CONFIRM_{key}.json.gz').read_bytes()))['worlds']
    count=len(manifest['confirmation'])
    if any(len(a)!=count for a in archives.values()): raise ValueError('DIAGNOSTIC_WORLD_COUNT')
    jobs=[dict(engine_root=str(engine_root),config=config,mode=receipt['mode'],
               policies=[dict(key=k,genes=front['genes'][k],archived=archives[k][i])
                         for k in front['representatives']]) for i in range(count)]
    totals={k:dict(counts=Counter(),horizons=Counter()) for k in front['representatives']}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(world_diagnostic,jobs),1):
            for k,r in result.items():
                totals[k]['counts'].update(r['counts']);totals[k]['horizons'].update(r['horizons'])
            if i%12==0: print('worlds',i,'/',count,flush=True)
    rows=[]
    for key in front['representatives']:
        genes=front['genes'][key];counts=totals[key]['counts'];horizons=totals[key]['horizons']
        summary=receipt['confirmation'][key]
        row=dict(key=key,genes=dict(zip(HISTORY_GENES,genes)),calibration=candidates[key],
                 confirmation=summary,counts=dict(counts),horizons=dict(horizons),accounts_replayed=count)
        rows.append(row);print(key[:8],dict(counts),flush=True)
    result=dict(schema='frozen_history_gate_diagnostic.v1',source_identity=identity,
        source_complete_sha256=receipt['complete_sha256'],confirmation_used_for_tuning=False,
        scope='POST_HOC_SAME_VISITED_STATES_NOT_COUNTERFACTUAL_POLICY_PERFORMANCE',
        worlds_per_finalist=len(manifest['confirmation']),workers=workers,rows=rows,
        code_sha256={p:sha(ROOT/p) for p in ('app/frozen_policy_diagnostics.py','tools/diagnose_frozen_history_policy.py')})
    write(output/'REPORT.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','engine-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();run(a.source,a.engine_root,a.output,a.workers)
