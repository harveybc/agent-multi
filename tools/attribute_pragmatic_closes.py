#!/usr/bin/env python3
"""Authenticate frozen accounts, suppress each close once, report account attribution."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.pragmatic_close_attribution import (SuppressOneClose,compare_close,
    compact_account,targets,verify_baseline)
from app.pragmatic_strategy import ControllerConfig,PragmaticController,PragmaticWorld,run_account
from app.zigzag_opportunities import engine,sha
from tools.optimize_dynamic_account import digest,sources
from tools.run_crossed_account_queue import load,save
from tools.run_pragmatic_contrast import TESTS as CONTRAST_TESTS,EXTRA,source_identity
from tools.verify_absolute_early_close import write

TESTS=CONTRAST_TESTS+['test_pragmatic_close_attribution']
NEW=['app/pragmatic_close_attribution.py','tools/attribute_pragmatic_closes.py',
     'tests/unit/test_pragmatic_close_attribution.py']


def authenticated(path,expected_sha):
    if sha(path)!=expected_sha: raise ValueError('ATTRIBUTION_SOURCE_HASH')
    body=json.loads(path.read_text())
    data=load(path,body['identity'])
    if data is None: raise ValueError('ATTRIBUTION_SOURCE_MISSING')
    return data


def own_sources():
    return source_identity()|{p:sha(ROOT/p) for p in EXTRA+NEW}


def check_code(manifest,engine_root):
    if own_sources()!=manifest['sources'] or sources(engine_root)!=manifest['native_sources']:
        raise ValueError('ATTRIBUTION_SOURCE_MUTATED')


def unit(job):
    out=Path(job['output']);identity=digest(job);path=out/(job['name']+'.json')
    check_code(job['manifest'],Path(job['engine_root']))
    cached=load(path,identity)
    if cached is not None:
        for name,h in cached['cases'].items():
            if sha(out/name)!=h: raise ValueError('ATTRIBUTION_CASE_CHANGED')
        return cached
    start=time.monotonic()
    archived=authenticated(Path(job['source'])/job['source_file'],job['source_sha'])['rows'][job['key']]
    engine(Path(job['engine_root']))
    world=PragmaticWorld(archived['spec'],job['risk'],job['reporter'])
    params=dict(archived['config']);params['close_offsets']=tuple(params['close_offsets'])
    config=ControllerConfig(**params)
    baseline=run_account(world,config,diagnose=True)
    verify_baseline(baseline,archived)
    rows=[];receipts={}
    for target in targets(baseline):
        casepath=out/(job['name']+f"_BAR_{target['bar']:03d}.json")
        caseid=digest(dict(unit=identity,target=target))
        result=load(casepath,caseid)
        if result is None:
            wrapper=SuppressOneClose(PragmaticController(config),target)
            alternative=run_account(world,config,controller=wrapper);wrapper.verify()
            comparison=compare_close(baseline,alternative,target)
            local=next(c for c in baseline['close_rows'] if c['bar']==target['bar'])
            result=dict(target=target,passive_delta_usd=local['delta_vs_hold'],
                risk=local['risk'],regret_risk=local['regret_risk'],comparison=comparison,
                alternative=compact_account(alternative),baseline_digest=digest(baseline))
            check_code(job['manifest'],Path(job['engine_root']))
            save(casepath,caseid,result)
        if result['target']!=target or result['baseline_digest']!=digest(baseline):
            raise ValueError('ATTRIBUTION_CASE_BINDING')
        rows.append({k:v for k,v in result.items() if k!='alternative'})
        receipts[casepath.name]=sha(casepath)
    result=dict(key=job['key'],reporter=job['reporter'],world=job['world'],
        baseline_digest=digest(baseline),baseline=compact_account(baseline),
        rows=rows,cases=receipts,seconds=time.monotonic()-start)
    check_code(job['manifest'],Path(job['engine_root']))
    save(path,identity,result)
    return result


def summarize(rows):
    def sign(x): return 'helpful' if x>1e-9 else 'harmful' if x< -1e-9 else 'neutral'
    def mean(v): return sum(v)/len(v) if v else None
    comparisons=[r['comparison'] for r in rows]
    return dict(closes=len(rows),passive=dict(Counter(sign(r['passive_delta_usd']) for r in rows)),
        position=dict(Counter(sign(c['position_delta_usd']) for c in comparisons)),
        account=dict(Counter(sign(c['account_delta_usd']) for c in comparisons)),
        pareto=dict(Counter(c['pareto_relation'] for c in comparisons)),
        locally_harmful_account_helpful=sum(r['passive_delta_usd']< -1e-9 and
            r['comparison']['account_delta_usd']>1e-9 for r in rows),
        locally_harmful_account_harmful=sum(r['passive_delta_usd']< -1e-9 and
            r['comparison']['account_delta_usd']< -1e-9 for r in rows),
        mean_account_delta_usd=mean([c['account_delta_usd'] for c in comparisons]),
        mean_position_delta_usd=mean([c['position_delta_usd'] for c in comparisons]),
        mean_continuation_delta_usd=mean([c['continuation_delta_usd'] for c in comparisons]),
        mean_drawdown_advantage_pp=mean([c['drawdown_advantage_pp'] for c in comparisons]),
        directions=dict(Counter(str(r['target']['direction']) for r in rows)))


def run(source,engine_root,output,workers,smoke=False):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    source_manifest=json.loads((source/'MANIFEST.json').read_text())
    sid=source_manifest['identity'];body={k:v for k,v in source_manifest.items() if k!='identity'}
    if digest(body)!=sid: raise ValueError('ATTRIBUTION_SOURCE_IDENTITY')
    complete=load(source/'COMPLETE.json',sid)
    if complete is None: raise ValueError('ATTRIBUTION_SOURCE_INCOMPLETE')
    for name,h in complete['artifacts'].items():
        if Path(name).name!=name or sha(source/name)!=h: raise ValueError('ATTRIBUTION_ARCHIVE_CHANGED')
    frozen=load(source/'SELECTION.json',sid)
    selected=[r['key'] for r in frozen if r['variant']=='sequential']
    if len(selected)!=2: raise ValueError('ATTRIBUTION_TWO_FROZEN_POLICIES_REQUIRED')
    riskpath=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(riskpath)!=source_manifest['risk_sha256']: raise ValueError('ATTRIBUTION_RISK_CHANGED')
    if sources(engine_root)!=source_manifest['native_sources']: raise ValueError('ATTRIBUTION_NATIVE_CHANGED')
    risk=json.loads(riskpath.read_text())
    reporters=['ideal','noise_1'] if smoke else source_manifest['config']['confirmation_reporters']
    world_count=1 if smoke else len(source_manifest['confirmation'])
    manifest=dict(schema='pragmatic_close_attribution.v1',source_identity=sid,
        source_complete_sha256=sha(source/'COMPLETE.json'),sources=own_sources(),native_sources=sources(engine_root),
        risk_sha256=sha(riskpath),policies=selected,reporters=reporters,world_count=world_count,
        smoke=smoke,intervention='SUPPRESS_ONE_EXECUTED_CLOSE_FOR_ONE_TICK_THEN_FROZEN_POLICY',
        financial_training=False,confirmation_tuning=False)
    identity=digest(manifest);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**values):
        write(output/'STATE.json',dict(identity=identity,status=status,heartbeat_utc=time.time(),**values))
    try:
        old=load(output/'MANIFEST.json',identity)
        if old is None: save(output/'MANIFEST.json',identity,manifest)
        elif old!=manifest: raise ValueError('ATTRIBUTION_MANIFEST_CHANGED')
        done=load(output/'COMPLETE.json',identity)
        if done is not None:
            for name,h in done['artifacts'].items():
                if sha(output/name)!=h: raise ValueError('ATTRIBUTION_COMPLETE_CHANGED')
            check_code(manifest,engine_root)
            return done
        if load(output/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                     STRATEGY_ENGINE_ROOT=str(engine_root))
            with (output/'TESTS.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','pytest','-q',
                    *['tests/unit/'+t+'.py' for t in TESTS],'--junitxml='+str(output/'TESTS.xml')],
                    cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            suites=list(ET.parse(output/'TESTS.xml').iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
            if result.returncode or any(counts[k] for k in ('errors','failures','skipped')):
                raise ValueError(('ATTRIBUTION_TESTS',counts))
            save(output/'TESTS.json',identity,counts)
        accumulated=[];units={};finished=0;total=world_count*len(reporters)*len(selected)
        for reporter in reporters:
            jobs=[]
            for i in range(world_count):
                filename=f'confirmation_{reporter}_{i:03d}.json'
                for key in selected:
                    jobs.append(dict(name=f'WORLD_{reporter}_{i:03d}_{key[:8]}',source=str(source),
                        source_file=filename,source_sha=complete['artifacts'][filename],
                        output=str(output),engine_root=str(engine_root),manifest=manifest,
                        risk=risk,key=key,reporter=reporter,world=i))
            state('RUNNING',phase=reporter,completed=finished,total=total)
            start=time.monotonic();durations=[]
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(unit,j):j for j in jobs}
                for f in as_completed(futures):
                    job=futures[f];result=f.result();finished+=1
                    accumulated.append(dict(key=result['key'],reporter=reporter,world=result['world'],rows=result['rows']))
                    units[job['name']+'.json']=sha(output/(job['name']+'.json'))
                    durations.append(result['seconds']);check_code(manifest,engine_root)
                    state('RUNNING',phase=reporter,completed=finished,total=total,
                        interventions=sum(len(a['rows']) for a in accumulated),
                        stage_elapsed_seconds=time.monotonic()-start,
                        stage_remaining_estimate_seconds=sum(durations)/len(durations)*(len(jobs)-len(durations))/workers)
                    print(reporter,finished,'/',total,'cases',sum(len(a['rows']) for a in accumulated),flush=True)
            accumulated.sort(key=lambda r:(r['reporter'],r['world'],r['key']))
            rows=[dict(reporter=mode,key=k,summary=summarize([r for a in accumulated
                if a['reporter']==mode and a['key']==k for r in a['rows']]))
                for mode in reporters if any(a['reporter']==mode for a in accumulated) for k in selected]
            report=dict(identity=identity,source_identity=sid,baseline_replays=finished,
                intervention_replays=sum(len(a['rows']) for a in accumulated),summaries=rows,units=units,
                scope='POST_HOC_SYNTHETIC_DIAGNOSTIC_NO_TUNING_NOT_ADDITIVE_POLICY_GAINS')
            write(output/'REPORT.json',report)
            lines=['# Frozen close attribution','',f'Identity: `{identity}`.',
                'Positive delta favors the original CLOSE over one-tick HOLD then frozen-policy continuation.',
                'These overlapping interventions are not additive policy gains. Synthetic diagnostic; no training.',
                '', '| Reporter | Policy | Closes | Account helpful | Harmful | Neutral | Close dominates | Hold dominates | Tradeoff | Mean account delta USD |',
                '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
            for row in rows:
                s=row['summary'];a=s['account'];p=s['pareto'];m=s['mean_account_delta_usd']
                lines.append(f"| {row['reporter']} | {row['key'][:8]} | {s['closes']} | {a.get('helpful',0)} | "
                    f"{a.get('harmful',0)} | {a.get('neutral',0)} | {p.get('CLOSE_DOMINATES',0)} | "
                    f"{p.get('HOLD_DOMINATES',0)} | {p.get('TRADEOFF',0)} | {('-' if m is None else f'{m:.6f}')} |")
            (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        if sha(source/'COMPLETE.json')!=manifest['source_complete_sha256']:
            raise ValueError('ATTRIBUTION_SOURCE_COMPLETION_CHANGED')
        check_code(manifest,engine_root)
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('STATE.json','LOCK','COMPLETE.json')}
        done=dict(artifacts=artifacts,baseline_replays=finished,
            intervention_replays=report['intervention_replays'],tests=load(output/'TESTS.json',identity))
        save(output/'COMPLETE.json',identity,done);state('COMPLETE',**{k:v for k,v in done.items() if k!='artifacts'})
        return done
    except BaseException as error:
        state('FAILED',error_type=type(error).__name__,detail=str(error));raise
    finally:
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','engine-root','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();run(args.source,args.engine_root,args.output,args.workers,args.smoke)
