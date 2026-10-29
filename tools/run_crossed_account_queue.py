#!/usr/bin/env python3
"""Pinned CPU queue: witness tests -> quality controls -> DEAP -> frozen confirmation."""
import argparse
import fcntl
import gc
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import deap
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.crossed_account_panel import CrossedWorld, panel_specs, QUALITIES
from app.crossed_account_workers import ShardPool
from app.dynamic_account_arbiter import frontier, GENES, bounds
from app.zigzag_opportunities import engine, sha
from tools.optimize_dynamic_account import Receipts, search, digest, controls, compact, sources
from tools.verify_absolute_early_close import write

TESTS=['test_external_audit_repairs','test_crossed_account_panel','test_dynamic_account_arbiter',
       'test_dynamic_account_panel','test_dynamic_account_queue','test_close_hold_controller',
       'test_close_hold_reports','test_close_hold_trace_audit','test_early_close_metrics',
       'test_absolute_exit_corpus','test_horizon_risk_rules','test_zigzag_risk_profiles',
       'test_protection_reference','test_mechanism_games','test_mechanism_m0_runner',
       'test_crossed_account_queue']


def pins(native):
    p=sources(native)
    for n in ['app/crossed_account_panel.py','app/crossed_account_workers.py','app/close_hold_reports.py','app/close_hold_verification.py',
              'tools/run_crossed_account_queue.py']+['tests/unit/'+t+'.py' for t in TESTS]:
        p['agent_multi'][n]=sha(ROOT/n)
    return p


def save(path,identity,data):
    body=dict(identity=identity,data=data);body['seal']=digest(body)
    write(path,body)


def load(path,identity):
    if not path.exists(): return None
    body=json.loads(path.read_text());seal=body.pop('seal')
    if body['identity']!=identity or digest(body)!=seal: raise ValueError('QUEUE_RESULT_IDENTITY')
    return body['data']


def representatives(vectors,genes):
    selected=frontier(vectors);groups={}
    for k in selected: groups.setdefault(tuple(vectors[k]),[]).append(k)
    ordered=sorted(groups,key=lambda v:(-v[0],-v[1]))
    ranks=sorted({round(x) for x in np.linspace(0,len(ordered)-1,min(5,len(ordered)))})
    reps=[min(groups[ordered[i]],key=lambda k:tuple(genes[k])) for i in ranks]
    return selected,reps,len(groups)


def verify_completed(output,identity,report,seeds):
    receipts=Receipts(output/'CANDIDATES.jsonl',identity)
    if (receipts.chain!=report['receipt_tail'] or len(receipts.rows)!=report['candidates']
            or sha(output/'CANDIDATES.jsonl')!=report['receipt_sha256']):
        raise ValueError('COMPLETED_RECEIPTS_CHANGED')
    for r in report['replay_files']:
        if sha(output/r['file'])!=r['sha256']: raise ValueError('COMPLETED_REPLAY_CHANGED')
    f=load(output/'FRONTIER.json',identity)
    if f is None or f['representatives']!=report['representatives']: raise ValueError('COMPLETED_FRONTIER_CHANGED')
    for seed in seeds:
        done=load(output/f'SEED_DONE_{seed}.json',identity)
        if done is None or sha(output/f'SEED_{seed}.json')!=done['sha256']: raise ValueError('COMPLETED_SEED_CHANGED')


def summary_table(path,rows,identity,label):
    lines=['# '+label,'',f'Identity: `{identity}`.',
        'Synthetic account experiments only; no financial TRAIN/VALIDATION/TEST result.',
        'Returns are mean independent-account returns. Mirrors and cost/envelope transforms are paired.',
        '', '| Quality | Policy | Return % | Worst DD % | Orders | Buy / sell | Win % | Buy win % | Sell win % | Early closes |',
        '| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    def rate(w,n): return f'{100*w/n:.2f}' if n else '-'
    for r in rows:
        lines.append(f"| {r['quality']} | {r['name']} | {r['objectives'][0]:.5f} | {-r['objectives'][1]:.5f} | "
            f"{r['orders']} | {r['buys']} / {r['sells']} | {rate(r['wins'],r['orders'])} | "
            f"{rate(r['buy_wins'],r['buys'])} | {rate(r['sell_wins'],r['sells'])} | {r['early_closes']} |")
    temporary=path.with_suffix('.tmp');temporary.write_text('\n'.join(lines)+'\n');temporary.replace(path)


class MeasuredReceipts(Receipts):
    def __init__(self,path,identity,output):
        super().__init__(path,identity);self.output=output;self.last=time.monotonic();self.seconds=[]
    def append(self,genes,result):
        now=time.monotonic();self.seconds.append(now-self.last);self.last=now
        super().append(genes,result)
        mean=float(np.mean(self.seconds[-20:]))
        write(self.output/'PROGRESS.json',dict(phase='DEAP',identity=self.identity,
            candidates=len(self.rows),last_receipt=self.chain,heartbeat_utc=time.time(),
            recent_seconds_per_candidate=mean,
            conservative_remaining_search_seconds=max(0,18060-len(self.rows))*mean,
            eta_scope='search upper evaluation budget only; confirmation additional'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--workers',type=int,default=4,choices=(1,2,3,4))
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    native=a.engine_root.resolve();engine(native)
    code=pins(native);config=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    cfg=json.loads(config.read_text());generations=2 if a.smoke else 300;seeds=[7] if a.smoke else [7,19,43]
    def specs(bank,q):
        rows=panel_specs(bank,q)
        return rows[:2] if a.smoke else rows
    manifest=dict(schema='crossed_account.v2',smoke=a.smoke,synthetic_only=True,
        sources=code,config_sha256=sha(config),population=20,generations=generations,seeds=seeds,workers=a.workers,
        genes=GENES,bounds=bounds(),calibration=specs('calibration','correlated'),
        confirmation=specs('confirmation','correlated'),qualities=QUALITIES,
        selection_rung='correlated',shared_entry_close_uncertainty=True,
        objectives=['mean_return_pct','negative_worst_mtm_dd_pct'],minimum_mean_orders=3.,
        representative_rule='distinct objective-vector ranks; lexicographic genome within duplicate vector',
        versions=dict(python=sys.version.split()[0],numpy=np.__version__,pandas=pd.__version__,deap=deap.__version__))
    identity=digest(manifest);manifest['identity']=identity
    mp=out/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('RESUME_IDENTITY')
    write(mp,manifest)
    def check():
        if pins(native)!=code or sha(config)!=manifest['config_sha256']: raise ValueError('SOURCE_MUTATION')
    def state(phase,**more):
        write(out/'STATE.json',dict(status='RUNNING',phase=phase,identity=identity,heartbeat_utc=time.time(),**more))
    def worlds(bank,q): return ShardPool(specs(bank,q),cfg,native,a.workers)
    started=time.monotonic()
    ws=None
    try:
        completed=load(out/'COMPLETE.json',identity)
        if completed is not None:
            check();verify_completed(out,identity,completed,seeds)
            print('COMPLETE_ALREADY_AUTHENTICATED',flush=True);return
        state('WITNESS_TESTS')
        if load(out/'TESTS.json',identity) is None:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                     STRATEGY_ENGINE_ROOT=str(native))
            command=[sys.executable,'-m','pytest','-q']+['tests/unit/'+t+'.py' for t in TESTS]
            with (out/'TESTS.log').open('w') as log:
                subprocess.run(command+['--junitxml='+str(out/'TESTS.xml')],cwd=ROOT,env=env,
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            cases=ET.parse(out/'TESTS.xml').getroot().findall('.//testcase')
            if not cases or any(c.find(x) is not None for c in cases for x in ('failure','error','skipped')):
                raise ValueError('WITNESS_SUITE_NOT_COMPLETE')
            save(out/'TESTS.json',identity,dict(passed=len(cases),log_sha256=sha(out/'TESTS.log')))
        summaries=[];control_jobs=0
        for q in QUALITIES:
            ws=worlds('calibration',q)
            for name,g,enabled in controls():
                state('CONTROLS',quality=q,policy=name,completed=control_jobs,total=len(QUALITIES)*7)
                path=out/f'CONTROL_{q}_{name}.json';r=load(path,identity)
                if r is None:
                    check();tick=time.monotonic();result=ws.evaluate(g,enabled,verify=True,details=False)
                    r=dict(name=name,quality=q,**compact(result),worlds=result['worlds'],
                           elapsed_seconds=time.monotonic()-tick)
                    save(path,identity,r)
                summaries.append({k:v for k,v in r.items() if k!='worlds'});control_jobs+=1
            ws.close();ws=None;gc.collect()
        save(out/'CONTROLS.json',identity,summaries)
        summary_table(out/'CONTROLS.md',summaries,identity,'Frozen calibration controls')
        # Negative performance is a result, never a reason to change fixtures or costs.
        ws=worlds('calibration','correlated')
        receipts=MeasuredReceipts(out/'CANDIDATES.jsonl',identity,out)
        for seed in seeds:
            state('DEAP',seed=seed,generations=generations,candidates=len(receipts.rows))
            sealed=load(out/f'SEED_DONE_{seed}.json',identity)
            if sealed is not None:
                if sha(out/f'SEED_{seed}.json')!=sealed['sha256']: raise ValueError('SEED_RESULT_CHANGED')
                continue
            search([],receipts,seed,20,generations,out,identity,check,evaluate_fn=lambda _,g:ws.evaluate(g))
            save(out/f'SEED_DONE_{seed}.json',identity,dict(sha256=sha(out/f'SEED_{seed}.json')))
        check()
        genes={}
        for line in (out/'CANDIDATES.jsonl').read_text().splitlines():
            r=json.loads(line);genes[r['key']]=r['genes']
        vectors={k:r['objectives'] for k,r in receipts.rows.items() if r['eligible']}
        selected,reps,unique=representatives(vectors,genes)
        if not selected: raise ValueError('NO_ACTIVITY_ELIGIBLE_FRONTIER')
        save(out/'FRONTIER.json',identity,dict(keys=selected,representatives=reps,unique_vectors=unique,
            rows=[dict(key=k,genes=genes[k],**receipts.rows[k]) for k in selected]))
        # Verify frozen representatives against their calibration receipts first.
        for key in reps:
            if compact(ws.evaluate(genes[key]))!=receipts.rows[key]: raise ValueError('CALIBRATION_REPLAY')
        ws.close();ws=None;gc.collect()
        final=[];checks=[]
        candidates=controls()+[(k,genes[k],None) for k in reps]
        for key in reps:
            if genes[key][8]!=1.:
                fixed=list(genes[key]);fixed[8]=1.
                candidates.append((key+'_size1',fixed,None))
        for q in QUALITIES:
            ws=worlds('confirmation',q)
            for name,g,enabled in candidates:
                state('FROZEN_CONFIRMATION',quality=q,policy=name,completed=len(final),total=5*len(candidates))
                path=out/f'FINAL_{q}_{name}.json';r=load(path,identity)
                archive=out/f'REPLAY_{q}_{name}.json.gz'
                if r is None:
                    check();result=ws.evaluate(g,enabled,verify=True,details=True)
                    raw=json.dumps(result,sort_keys=True,allow_nan=False).encode()
                    tmp=archive.with_suffix('.tmp');tmp.write_bytes(gzip.compress(raw,mtime=0));tmp.replace(archive)
                    sequences={digest(w['trace']) for w in result['worlds']}
                    r=dict(name=name,quality=q,**compact(result),replay_sha256=sha(archive),
                           distinct_world_traces=len(sequences),
                           action_signature=digest([w['trace'] for w in result['worlds']]))
                    save(path,identity,r)
                if sha(archive)!=r['replay_sha256']: raise ValueError('FINAL_REPLAY_CHANGED')
                final.append(r);checks.append(dict(file=archive.name,sha256=r['replay_sha256']))
            ws.close();ws=None;gc.collect()
        check()
        report=dict(status='SMOKE_ONLY' if a.smoke else 'COMPLETE_CROSSED_SYNTHETIC_FRONTIER',
            identity=identity,controls=summaries,confirmation=final,candidates=len(receipts.rows),
            receipt_tail=receipts.chain,receipt_sha256=sha(out/'CANDIDATES.jsonl'),
            frontier_policies=len(selected),unique_vectors=unique,representatives=reps,
            behavior_classes={q:len({r['action_signature'] for r in final if r['quality']==q}) for q in QUALITIES},
            replay_files=checks,elapsed_this_invocation=time.monotonic()-started,
            learned_reporter_tested=False,global_optimum_certified=False)
        save(out/'COMPLETE.json',identity,report)
        summary_table(out/'RESULTS.md',final,identity,'Frozen confirmation and fixed-size diagnostics')
        write(out/'STATE.json',dict(status='COMPLETE',identity=identity,smoke=a.smoke,
            candidates=len(receipts.rows),frontier_policies=len(selected),heartbeat_utc=time.time()))
        print(json.dumps(dict(status=report['status'],candidates=len(receipts.rows))),flush=True)
    except Exception as exc:
        write(out/'STATE.json',dict(status='FAILED',identity=identity,error_type=type(exc).__name__,error=str(exc),heartbeat_utc=time.time()))
        raise
    finally:
        if ws is not None: ws.close()


if __name__=='__main__': main()
