#!/usr/bin/env python3
"""Unattended causal controls, conditional exploratory DEAP, confirmation and audit packet.

Synthetic engineering study only. No financial training, feature selection,
learned-reporter qualification or automatic acceptance of an audit.
"""
import argparse
import copy
import fcntl
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import deap
from deap import base,creator,tools as D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.causal_history_account import (HistoryWorld,base_genes,history_bounds,
    decode_history,specifications,HISTORY_GENES)
from app.causal_history_workers import HistoryPool
from app.zigzag_opportunities import engine,sha
from tools.optimize_dynamic_account import digest,Receipts,compact,sources
from tools.run_crossed_account_queue import TESTS,save,load,representatives
from tools.run_post_audit_diagnostics import EXTRA_TESTS,counterfactuals,close_metrics
from tools.analyze_crossed_account_results import paired_clusters
from tools.verify_absolute_early_close import write

SUITE=TESTS+EXTRA_TESTS+['test_causal_history_account','test_causal_history_queue']


def source_identity():
    names=['app/causal_history_account.py','app/causal_history_workers.py',
        'app/causal_report_baseline.py','app/report_track_record.py',
        'app/close_hold_controller.py','app/dynamic_account_panel.py',
        'app/crossed_account_panel.py','app/crossed_account_workers.py',
        'tools/run_causal_history_queue.py','tools/run_crossed_account_queue.py',
        'tools/run_post_audit_diagnostics.py','tools/analyze_crossed_account_results.py']
    names+=['tests/unit/'+x+'.py' for x in SUITE]
    return {n:sha(ROOT/n) for n in names}


def freeze_manifest(path,manifest):
    identity=digest(manifest);body=dict(manifest,identity=identity)
    if path.exists() and json.loads(path.read_text())!=body:
        raise ValueError('HISTORY_RESUME_IDENTITY')
    write(path,body)
    return identity


def proposals():
    return [('neither',base_genes(),None,True),('entry',base_genes(1,0),None,True),
            ('close',base_genes(0,1),None,True),('both',base_genes(1,1),None,True),
            ('no_close',base_genes(),None,False),('nop',base_genes(),set(),False)]


def search_gate(controls):
    base=controls['neither'];accepted=[]
    for name in ('entry','close','both'):
        r=controls[name]
        if (r['mean_orders']>=3 and r['eligible'] and
                r['objectives'][0]>base['objectives'][0]+1e-12 and
                r['objectives'][1]>=base['objectives'][1]-1e-12):
            accepted.append(name)
    return dict(eligible=bool(accepted),arms=accepted,
        rule='active history arm improves mean calibration return without increasing worst DD',
        learned_qualification=False)


def search(pool,receipts,seed,generations,out,identity,check,state):
    if not hasattr(creator,'HistoryFitness'):
        creator.create('HistoryFitness',base.Fitness,weights=(1.,1.))
        creator.create('HistoryIndividual',list,fitness=creator.HistoryFitness)
    random.seed(seed)
    free=[i for i,(lo,hi) in enumerate(history_bounds()) if lo!=hi]
    bounds=[history_bounds()[i] for i in free];lo,hi=zip(*bounds)
    def full(x):
        g=base_genes()
        for i,v in zip(free,x): g[i]=v
        return list(decode_history(g))
    def fit(x):
        g=full(x);x[:]=[g[i] for i in free];key=digest(g)
        if key not in receipts.rows:
            start=time.monotonic();result=compact(pool.evaluate(g));receipts.append(g,result)
            state('DEAP',seed=seed,candidates=len(receipts.rows),
                  last_candidate_seconds=time.monotonic()-start)
        r=receipts.rows[key]
        return tuple(r['objectives']) if r['eligible'] else (-1e6,-1e6)
    population=[creator.HistoryIndividual([g[i] for i in free]) for _,g,_,_ in proposals()[:4]]
    population.extend(creator.HistoryIndividual([random.uniform(a,b) for a,b in bounds]) for _ in range(16))
    for x in population: x.fitness.values=fit(x)
    population=D.selNSGA2(population,20);history=[]
    for generation in range(1,generations+1):
        children=[copy.deepcopy(x) for x in D.selTournamentDCD(population,20)]
        for a,b in zip(children[::2],children[1::2]):
            if random.random()<.9: D.cxSimulatedBinaryBounded(a,b,eta=15,low=lo,up=hi)
            for x in (a,b):
                D.mutPolynomialBounded(x,eta=20,low=lo,up=hi,indpb=1/len(free))
                x.fitness.values=fit(x)
        population=D.selNSGA2(population+children,20)
        history.append(dict(generation=generation,genes=[full(x) for x in population],
                            fitness=[list(x.fitness.values) for x in population]))
        check();save(out/f'SEED_{seed}.json',identity,dict(seed=seed,history=history))
        state('DEAP',seed=seed,generation=generation,generations=generations,candidates=len(receipts.rows))
    save(out/f'SEED_DONE_{seed}.json',identity,dict(sha256=sha(out/f'SEED_{seed}.json')))


def report(output,identity,mode,gate,rows,comparisons,smoke):
    lines=['# Causal history controller experiment','',f'Identity: `{identity}`.',
        f'Reporter: {mode}. Smoke: {smoke}. Search calibration gate: {gate["eligible"]}.',
        'Synthetic prefix-only reports; NOT financial validation, a learned RL model or final R6 qualification.',
        'Size fixed at 1; rel_volume=0.1 once; native costs and calendar. No hindsight fitness.',
        '', '| Policy | Return % | Worst DD % | Mean native Sharpe | Orders/week | Buy / sell | Win % | Early closes | Harmful closes | Regret / risk |',
        '| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |']
    def fmt(x): return '-' if x is None else f'{x:.4f}'
    for name,r in rows.items():
        c=r['close_metrics'];n=r['orders']
        lines.append(f"| {name} | {r['objectives'][0]:.4f} | {-r['objectives'][1]:.4f} | "
            f"{fmt(r['mean_native_sharpe'])} | {r['mean_orders']:.2f} | {r['buys']} / {r['sells']} | "
            f"{fmt(100*r['wins']/n if n else None)} | {r['early_closes']} | {c['destructive']} | {fmt(c['mean_regret_risk'])} |")
    lines+=['','Mean native Sharpe is an average of defined account Sharpes, not a pooled portfolio Sharpe.',
        'Paired draw-cluster contrasts are descriptive, not independent tests for each transformed account.',
        'The error penalty is not a calibrated joint-path band. It can suppress beneficial closes.',
        'All confirmed policies have native ledgers, costs, directions and fixed-position close alternatives.',
        'No confirmation result selects or modifies a policy. No synthetic result opens TEST.',
        'Full PC1--PC6 learned-interface qualification and final R6 remain outside this engineering experiment.',
        'No automatic RL training, floating selection or strategy promotion follows this report.']
    write(output/'COMPARISONS.json',comparisons)
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (output/'AUDIT_REQUEST.md').write_text('\n'.join([
        '# Satoshi: causal history controller experiment audit','',f'Identity: `{identity}`.',
        'Review MANIFEST, TESTS, SEARCH_GATE, CANDIDATES, SEED receipts, CONFIRM archives,',
        'COMPARISONS and COMPLETE. This is a diagnostic audit, not a final strategy release.',
        '1. Mutate future OHLC and verify prefix reports, matured error snapshots and decisions.',
        '2. Reproduce native account cash and close counterfactuals including TP sacrifice and SL savings.',
        '3. Verify independent entry/close reliability and close uncertainty; fixed size and business contract.',
        '4. Check conditional search is calibration-only and confirmation never selects a winner.',
        '5. Attack cold resume, source pins, receipt chain, exact ledger replay and two-host independence.',
        '6. Challenge poor activity, mirror asymmetry, horizon concentration, lack of useful causal signal,',
        '   history warmup suppression and the non-calibrated nature of this empirical penalty.',
        '7. Report missing PC1--PC6 qualification explicitly; do not infer a learned reporter from this baseline.',
        'No external audit verdict has been generated or presumed by this script.','']))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=('median','zero'),required=True)
    p.add_argument('--workers',type=int,choices=(1,2,3,4),default=2)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine(a.engine_root)
    cfgpath=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    cfg=json.loads(cfgpath.read_text());code=source_identity();native=sources(a.engine_root)
    generations=2 if a.smoke else 300;seeds=[7] if a.smoke else [7,19,43]
    manifest=dict(schema='causal_history_queue.v1',mode=a.mode,smoke=a.smoke,
        sources=code,native_sources=native,config_sha256=sha(cfgpath),workers=a.workers,
        calibration=specifications('calibration',a.smoke),confirmation=specifications('confirmation',a.smoke),
        population=20,generations=generations,seeds=seeds,genes=HISTORY_GENES,bounds=history_bounds(),
        objectives=['mean_return_pct','negative_worst_mtm_dd_pct'],activity_minimum=3.,
        penalty='matured mean absolute price error / unit risk; minimum20,last64,not joint calibration',
        versions=dict(python=sys.version.split()[0],numpy=np.__version__,deap=deap.__version__),
        synthetic_only=True,financial_training=False,final_r6=False)
    # JSON normalization makes tuple/list representations identical on resume.
    manifest=json.loads(json.dumps(manifest));identity=freeze_manifest(out/'MANIFEST.json',manifest)
    def check():
        if source_identity()!=code or sources(a.engine_root)!=native or sha(cfgpath)!=manifest['config_sha256']:
            raise ValueError('HISTORY_SOURCE_MUTATION')
    def state(phase,**kw):
        write(out/'STATE.json',dict(status='RUNNING',identity=identity,phase=phase,heartbeat_utc=time.time(),**kw))
    artifacts={};pool=None;started=time.monotonic()
    def archive(name,fn):
        path=out/(name+'.json');blob=out/(name+'.json.gz');old=load(path,identity)
        if old is None:
            check();tick=time.monotonic();data=fn()
            tmp=blob.with_suffix('.tmp');tmp.write_bytes(gzip.compress(json.dumps(data,sort_keys=True,allow_nan=False).encode(),mtime=0));tmp.replace(blob)
            old=dict(sha256=sha(blob),seconds=time.monotonic()-tick)
            save(path,identity,old)
        if sha(blob)!=old['sha256']: raise ValueError('HISTORY_ARTIFACT_MUTATION')
        artifacts[blob.name]=old['sha256'];artifacts[path.name]=sha(path)
        return json.loads(gzip.decompress(blob.read_bytes()))
    try:
        done=load(out/'COMPLETE.json',identity)
        if done is not None:
            check()
            for name,value in done['artifacts'].items():
                if sha(out/name)!=value: raise ValueError('HISTORY_COMPLETED_ARTIFACT_CHANGED')
            receipts=Receipts(out/'CANDIDATES.jsonl',identity)
            if receipts.chain!=done['receipt_tail']: raise ValueError('HISTORY_COMPLETED_CHAIN')
            print('COMPLETE_ALREADY_AUTHENTICATED',flush=True);return
        state('TESTS')
        if load(out/'TESTS.json',identity) is None:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(a.engine_root))
            command=[sys.executable,'-m','pytest','-q']+['tests/unit/'+t+'.py' for t in SUITE]
            with (out/'TESTS.log').open('w') as log:
                subprocess.run(command+['--junitxml='+str(out/'TESTS.xml')],cwd=ROOT,env=env,
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            cases=ET.parse(out/'TESTS.xml').getroot().findall('.//testcase')
            if not cases or any(c.find(t) is not None for c in cases for t in ('failure','error','skipped')):
                raise ValueError('HISTORY_TEST_FAILURE_OR_SKIP')
            save(out/'TESTS.json',identity,dict(passed=len(cases),skipped=0))
        artifacts['TESTS.json']=sha(out/'TESTS.json')
        pool=HistoryPool(manifest['calibration'],cfg,a.engine_root,a.mode,a.workers)
        calibration={}
        for name,g,enabled,voluntary in proposals():
            state('CALIBRATION_CONTROLS',unit=name)
            r=archive('CAL_'+name,lambda:pool.evaluate(g,enabled,True,True,voluntary))
            calibration[name]=compact(r)
        gate=search_gate(calibration);save(out/'SEARCH_GATE.json',identity,gate)
        artifacts['SEARCH_GATE.json']=sha(out/'SEARCH_GATE.json')
        receipts=Receipts(out/'CANDIDATES.jsonl',identity)
        if gate['eligible'] or a.smoke:
            for seed in seeds:
                done_seed=load(out/f'SEED_DONE_{seed}.json',identity)
                if done_seed is None:
                    search(pool,receipts,seed,generations,out,identity,check,state)
                elif sha(out/f'SEED_{seed}.json')!=done_seed['sha256']:
                    raise ValueError('HISTORY_SEED_MUTATION')
                artifacts[f'SEED_{seed}.json']=sha(out/f'SEED_{seed}.json')
                artifacts[f'SEED_DONE_{seed}.json']=sha(out/f'SEED_DONE_{seed}.json')
        genes={}
        if (out/'CANDIDATES.jsonl').exists():
            for line in (out/'CANDIDATES.jsonl').read_text().splitlines():
                r=json.loads(line);genes[r['key']]=r['genes']
            artifacts['CANDIDATES.jsonl']=sha(out/'CANDIDATES.jsonl')
        vectors={k:r['objectives'] for k,r in receipts.rows.items() if r['eligible']}
        selected,reps,unique=representatives(vectors,genes) if vectors else ([],[],0)
        save(out/'FRONTIER.json',identity,dict(keys=selected,representatives=reps,unique_vectors=unique,genes=genes))
        artifacts['FRONTIER.json']=sha(out/'FRONTIER.json')
        for key in reps:
            if compact(pool.evaluate(genes[key]))!=receipts.rows[key]:
                raise ValueError('HISTORY_COLD_CALIBRATION_REPLAY')
        pool.close();pool=None
        candidates=proposals()+[(k,genes[k],None,True) for k in reps]
        # Finalists also receive matched no-close accounts, not only fixed-position comparisons.
        candidates += [(k+'_no_close',genes[k],None,False) for k in reps]
        pool=HistoryPool(manifest['confirmation'],cfg,a.engine_root,a.mode,a.workers)
        summaries={};accounts={}
        for i,(name,g,enabled,voluntary) in enumerate(candidates):
            state('CONFIRMATION',unit=name,completed=i,total=len(candidates))
            def evaluate_final():
                result=pool.evaluate(g,enabled,True,True,voluntary);closes=[];preempted=0
                for n,row in enumerate(result['worlds']):
                    w=HistoryWorld(row['spec'],cfg,a.mode)
                    records,count=counterfactuals(w,row,g)
                    closes.extend(dict(world=n,**x) for x in records);preempted+=count
                return dict(**result,close_rows=closes,preempted=preempted,
                    close_metrics=close_metrics(closes,len(result['worlds'])),
                    by_passive={k:close_metrics([r for r in closes if r['hold_reason']==k],len(result['worlds'])) for k in ('SL','TP','FORCED_FLAT')})
            r=archive('CONFIRM_'+name,evaluate_final)
            accounts[name]=r['worlds'];sharpes=[w['native_sharpe'] for w in r['worlds'] if w['native_sharpe'] is not None]
            summaries[name]=dict(**compact(r),close_metrics=r['close_metrics'],by_passive=r['by_passive'],
                mean_native_sharpe=float(np.mean(sharpes)) if sharpes else None)
        comparisons={name:paired_clusters(rows,accounts['neither']) for name,rows in accounts.items() if name!='neither'}
        for key in reps: comparisons[key+'_vs_own_no_close']=paired_clusters(accounts[key],accounts[key+'_no_close'])
        check();report(out,identity,a.mode,gate,summaries,comparisons,a.smoke)
        for n in ('RESULTS.md','AUDIT_REQUEST.md','COMPARISONS.json'): artifacts[n]=sha(out/n)
        completed=dict(status='SMOKE_ONLY' if a.smoke else 'COMPLETE_CAUSAL_HISTORY_EXPERIMENT',
            mode=a.mode,search_gate=gate,candidates=len(receipts.rows),receipt_tail=receipts.chain,
            confirmation=summaries,artifacts=artifacts,elapsed_this_invocation=time.monotonic()-started,
            learned_adapter_qualified=False,automatic_next_training=False,final_r6=False)
        save(out/'COMPLETE.json',identity,completed)
        write(out/'STATE.json',dict(status='COMPLETE',identity=identity,heartbeat_utc=time.time(),
            candidates=len(receipts.rows),search_gate=gate['eligible'],financial_training=False))
        print(completed['status'],flush=True)
    except Exception as exc:
        write(out/'STATE.json',dict(status='FAILED',identity=identity,heartbeat_utc=time.time(),
            error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        if pool is not None: pool.close()


if __name__=='__main__': main()
