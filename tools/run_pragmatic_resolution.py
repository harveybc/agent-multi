#!/usr/bin/env python3
"""Wait for frozen close attribution, then test only the missing 1h/2h plan choices."""
import argparse
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
from app.zigzag_opportunities import sha
from tools.attribute_pragmatic_closes import TESTS as ATTR_TESTS,own_sources
from tools.optimize_dynamic_account import digest,sources
from tools.run_crossed_account_queue import load,save
from tools.run_pragmatic_contrast import worker,summarize
from tools.verify_absolute_early_close import write

TESTS=ATTR_TESTS+['test_pragmatic_resolution']


def paired_configs(source,selected,config):
    if (config['schema']!='pragmatic_resolution_contrast.v1'
            or config['financial_training'] is not False or config['coefficient_selection'] is not False):
        raise ValueError('RESOLUTION_CONFIG')
    keys={r['key'] for r in selected};rows=[]
    for row in source['configurations']:
        if row['key'] not in keys: continue
        c=row['config']
        if c['close_offsets']!=config['old_close_offsets']: raise ValueError('RESOLUTION_OLD_OFFSETS')
        rows.append(dict(row,arm=c['variant'],parent=None))
        if c['variant']=='sequential':
            new=dict(c,close_offsets=config['new_close_offsets'])
            rows.append(dict(key=digest(new),config=new,arm='sequential_1h',parent=row['key']))
    if len(rows)!=6 or len({r['key'] for r in rows})!=6: raise ValueError('RESOLUTION_PAIR_COUNT')
    return rows


def fresh_specs(source,bank,config,smoke):
    seeds={config['calibration_seed'],config['confirmation_seed']}
    old={s['seed'] for b in ('calibration','confirmation') for s in source[b]}
    if len(seeds)!=2 or seeds&old: raise ValueError('RESOLUTION_SEED_OVERLAP')
    rows=[dict(s,seed=config[bank+'_seed'],bank='resolution_'+bank) for s in source[bank]]
    return rows[:1] if smoke else rows


def dependency_ready(root,identity,source_identity,smoke):
    manifest=load(root/'MANIFEST.json',identity)
    if manifest is not None and (manifest['source_identity']!=source_identity or manifest['smoke']!=smoke):
        raise ValueError('RESOLUTION_DEPENDENCY_BINDING')
    completed=load(root/'COMPLETE.json',identity)
    if completed is None:
        state=root/'STATE.json'
        if state.exists() and json.loads(state.read_text()).get('status')=='FAILED':
            raise ValueError('RESOLUTION_DEPENDENCY_FAILED')
        return None
    if manifest is None: raise ValueError('RESOLUTION_DEPENDENCY_MANIFEST')
    for name,h in completed['artifacts'].items():
        if Path(name).name!=name or sha(root/name)!=h: raise ValueError('RESOLUTION_DEPENDENCY_ARTIFACT')
    return sha(root/'COMPLETE.json')


def verify_dependency_hash(observed,expected):
    if observed is None or observed!=expected:
        raise ValueError('RESOLUTION_DEPENDENCY_CHANGED')


def run(source,dependency,dependency_identity,engine_root,config_path,output,workers,smoke=False):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    config=json.loads(config_path.read_text());source_manifest=json.loads((source/'MANIFEST.json').read_text())
    sid=source_manifest['identity']
    if digest({k:v for k,v in source_manifest.items() if k!='identity'})!=sid:
        raise ValueError('RESOLUTION_SOURCE_IDENTITY')
    complete=load(source/'COMPLETE.json',sid)
    if complete is None: raise ValueError('RESOLUTION_SOURCE_INCOMPLETE')
    for name,h in complete['artifacts'].items():
        if Path(name).name!=name or sha(source/name)!=h: raise ValueError('RESOLUTION_SOURCE_ARTIFACT')
    configs=paired_configs(source_manifest,load(source/'SELECTION.json',sid),config)
    riskpath=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(riskpath)!=source_manifest['risk_sha256'] or sources(engine_root)!=source_manifest['native_sources']:
        raise ValueError('RESOLUTION_EXECUTION_CHANGED')
    def pins():
        return own_sources()|{p:sha(ROOT/p) for p in ('tools/run_pragmatic_resolution.py','tests/unit/test_pragmatic_resolution.py')}
    manifest=dict(schema='pragmatic_resolution_manifest.v1',source_identity=sid,
        source_complete_sha256=sha(source/'COMPLETE.json'),dependency_identity=dependency_identity,
        sources=pins(),native_sources=sources(engine_root),config=config,config_sha256=sha(config_path),
        configs=configs,calibration=fresh_specs(source_manifest,'calibration',config,smoke),
        confirmation=fresh_specs(source_manifest,'confirmation',config,smoke),smoke=smoke)
    identity=digest(manifest);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**kw): write(output/'STATE.json',dict(identity=identity,status=status,heartbeat_utc=time.time(),**kw))
    def check():
        if pins()!=manifest['sources'] or sources(engine_root)!=manifest['native_sources'] or sha(config_path)!=manifest['config_sha256']:
            raise ValueError('RESOLUTION_SOURCE_MUTATED')
    try:
        old=load(output/'MANIFEST.json',identity)
        if old is None: save(output/'MANIFEST.json',identity,manifest)
        elif old!=manifest: raise ValueError('RESOLUTION_RESUME_MANIFEST')
        done=load(output/'COMPLETE.json',identity)
        if done is not None:
            for name,h in done['artifacts'].items():
                if sha(output/name)!=h: raise ValueError('RESOLUTION_COMPLETE_CHANGED')
            receipt=load(output/'DEPENDENCY.json',identity)
            if receipt is None: raise ValueError('RESOLUTION_DEPENDENCY_RECEIPT')
            verify_dependency_hash(dependency_ready(dependency,dependency_identity,sid,smoke),receipt['complete_sha256'])
            check();return done
        if load(output/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(engine_root))
            with (output/'TESTS.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','pytest','-q',*['tests/unit/'+t+'.py' for t in TESTS],
                    '--junitxml='+str(output/'TESTS.xml')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            suites=list(ET.parse(output/'TESTS.xml').iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
            if result.returncode or any(counts[k] for k in ('errors','failures','skipped')):
                raise ValueError(('RESOLUTION_TESTS',counts))
            save(output/'TESTS.json',identity,counts)
        while True:
            dep_hash=dependency_ready(dependency,dependency_identity,sid,smoke)
            if dep_hash is not None: break
            state('WAIT_DEPENDENCY',phase='CLOSE_ATTRIBUTION');check();time.sleep(30)
        receipt=load(output/'DEPENDENCY.json',identity)
        if receipt is not None: verify_dependency_hash(dep_hash,receipt['complete_sha256'])
        else: save(output/'DEPENDENCY.json',identity,dict(identity=dependency_identity,complete_sha256=dep_hash))
        risk=json.loads(riskpath.read_text());results={}
        for phase in ('calibration','confirmation'):
            reporters=['noise_1'] if phase=='calibration' else (['ideal','noise_1'] if smoke else config['reporters'])
            jobs=[dict(name=f'{phase}_{r}_{i:03d}',output=str(output),engine_root=str(engine_root),identity=identity,
                spec=s,risk=risk,reporter=r,configs=configs,diagnose=phase=='confirmation')
                for r in reporters for i,s in enumerate(manifest[phase])]
            totals={r:{c['key']:[] for c in configs} for r in reporters};seen={};durations=[]
            state('RUNNING',phase=phase,completed=0,total=len(jobs));start=time.monotonic()
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(worker,j):j for j in jobs}
                collected=[]
                for n,f in enumerate(as_completed(futures),1):
                    job=futures[f];result=f.result();collected.append((job,result));durations.append(result['seconds'])
                    market_id=digest(job['spec'])
                    if market_id in seen and seen[market_id]!=result['market_digest']: raise ValueError('RESOLUTION_UNPAIRED_MARKET')
                    seen[market_id]=result['market_digest'];check()
                    state('RUNNING',phase=phase,completed=n,total=len(jobs),elapsed_seconds=time.monotonic()-start,
                        remaining_estimate_seconds=sum(durations)/n*(len(jobs)-n)/workers)
                    print(phase,n,'/',len(jobs),flush=True)
            for job,result in sorted(collected,key=lambda pair:pair[0]['name']):
                for key,row in result['rows'].items(): totals[job['reporter']][key].append(row)
            results[phase]={r:{k:summarize(v) for k,v in bykey.items()} for r,bykey in totals.items()}
            write(output/'REPORT.json',dict(identity=identity,configurations=configs,results=results,
                selection_performed=False,financial_training=False,scope='FROZEN_ONE_VARIABLE_SYNTHETIC_CONTRAST'))
        lines=['# Frozen temporal resolution contrast','',f'Identity: `{identity}`.',
            'Six fixed policies; new worlds; no coefficient selection; synthetic outputs, not learned RL.',
            '', '| Reporter | Arm | Key | Return % | Worst DD % | Sharpe | Orders/week | Buy/sell | Win % | Early | Locally harmful |',
            '|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|']
        def fmt(x): return '-' if x is None else f'{x:.5f}'
        for mode,bykey in results['confirmation'].items():
            for row in configs:
                r=bykey[row['key']]
                lines.append(f"| {mode} | {row['arm']} | {row['key'][:8]} | {fmt(r['mean_return_pct'])} | {fmt(r['worst_dd_pct'])} | "
                    f"{fmt(r['mean_sharpe'])} | {fmt(r['orders_per_week'])} | {r['buys']}/{r['sells']} | {fmt(r['win_pct'])} | "
                    f"{r['close_reasons'].get('AGENT_CLOSE',0)} | {r['harmful_closes']} |")
        lines+=['','No automatic strategy promotion or floating-selection release. Per-close account attribution is distinct from local passive comparisons.']
        (output/'RESULTS.md').write_text('\n'.join(lines)+'\n');check()
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('STATE.json','LOCK','COMPLETE.json')}
        count=len(configs)*(len(manifest['calibration'])+len(manifest['confirmation'])*len(results['confirmation']))
        done=dict(artifacts=artifacts,accounts=count,tests=load(output/'TESTS.json',identity))
        save(output/'COMPLETE.json',identity,done);state('COMPLETE',accounts=count);return done
    except BaseException as error:
        state('FAILED',error_type=type(error).__name__,detail=str(error));raise
    finally:
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','dependency','engine-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--dependency-identity',required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pragmatic_resolution_contrast.json')
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.source,a.dependency,a.dependency_identity,a.engine_root,a.config,a.output,a.workers,a.smoke)
