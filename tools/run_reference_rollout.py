#!/usr/bin/env python3
"""Tests -> frozen-reference contrast -> actual-policy close attribution -> audit packet."""
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

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.pragmatic_strategy import ControllerConfig,PragmaticController,PragmaticWorld,run_account
from app.reference_rollout import ReferenceRolloutController,CaptureCloses,resume_without_close
from app.pragmatic_close_attribution import targets,compare_close,compact_account
from app.zigzag_opportunities import engine,sha
from tools.diagnose_pragmatic_plan_gap import pins as base_pins,TESTS as BASE_TESTS,checked_archive
from tools.optimize_dynamic_account import digest,sources
from tools.run_crossed_account_queue import save,load
from tools.run_pragmatic_contrast import summarize
from tools.verify_absolute_early_close import write

TESTS=BASE_TESTS+['test_reference_rollout','test_reference_rollout_queue']


def pins():
    return base_pins()|{p:sha(ROOT/p) for p in ('app/reference_rollout.py','tools/run_reference_rollout.py',
        'tests/unit/test_reference_rollout.py','tests/unit/test_reference_rollout_queue.py')}


def configs(source,selected,config):
    if (config['schema']!='reference_rollout_contrast.v1' or config['financial_training'] is not False
            or config['coefficient_selection'] is not False or config['account_risk_weight']!=1.
            or config['reporters']!=['ideal','noise_0.5','noise_1','noise_2','causal']
            or config['automatic_close_attribution']!='joint_ideal_every_executed_close'):
        raise ValueError('REFERENCE_CONFIG')
    chosen={r['key'] for r in selected}
    rows=[dict(row,arm=row['config']['variant']+'_'+row['key'][:8],references=None)
          for row in source['configurations'] if row['key'] in chosen]
    refs={r['config']['variant']:r['config'] for r in rows if r['config']['variant']!='sequential'}
    if len(rows)!=4 or set(refs)!={'simple','predictive_proxy'}: raise ValueError('REFERENCE_FROZEN_CONTROLS')
    for name,names in [('rollout_simple',['simple']),('rollout_proxy',['predictive_proxy']),
                       ('rollout_joint',['simple','predictive_proxy'])]:
        row=dict(arm=name,config=refs['simple'],references={n:refs[n] for n in names},
                 account_risk_weight=config['account_risk_weight'])
        rows.append(dict(row,key=digest(row)))
    return rows


def fresh_specs(source,config,bank,smoke):
    keys=['calibration_seed','confirmation_seed','smoke_calibration_seed','smoke_confirmation_seed']
    seeds=[config[k] for k in keys];old={s['seed'] for b in ('calibration','confirmation') for s in source[b]}
    if len(set(seeds))!=4 or old&set(seeds): raise ValueError('REFERENCE_SEED_OVERLAP')
    seed=config[('smoke_' if smoke else '')+bank+'_seed']
    rows=[dict(s,seed=seed,bank='reference_'+bank) for s in source[bank]]
    return rows[:1] if smoke else rows


def build(row):
    def cfg(params): return ControllerConfig(**dict(params,close_offsets=tuple(params['close_offsets'])))
    config=cfg(row['config'])
    controller=(PragmaticController(config) if row['references'] is None else ReferenceRolloutController(
        {n:cfg(c) for n,c in row['references'].items()},row['account_risk_weight']))
    return config,controller


def check(manifest,native):
    if pins()!=manifest['sources'] or sources(native)!=manifest['native_sources']:
        raise ValueError('REFERENCE_SOURCE_MUTATION')


def verify_case_files(output,result):
    for name,h in result.get('case_files',{}).items():
        if Path(name).name!=name or sha(output/name)!=h:
            raise ValueError('REFERENCE_CASE_MUTATED')


def remaining_seconds(jobs,collected,workers):
    completed={j['name'] for j,_ in collected};byarm={}
    for j,r in collected: byarm.setdefault(j['policy']['arm'],[]).append(r['seconds'])
    left=[j for j in jobs if j['name'] not in completed]
    if any(j['policy']['arm'] not in byarm for j in left): return None
    return sum(float(np.mean(byarm[j['policy']['arm']])) for j in left)/workers


def unit(job):
    out=Path(job['output']);identity=digest(job);path=out/(job['name']+'.json')
    native=Path(job['engine_root']);check(job['manifest'],native)
    cached=load(path,identity)
    if cached is not None:
        verify_case_files(out,cached)
        return cached
    engine(native);start=time.monotonic()
    w=PragmaticWorld(job['spec'],job['risk'],job['reporter']);config,controller=build(job['policy'])
    market=digest(np.column_stack((w.env.o,w.env.h,w.env.l,w.env.c)).tolist())
    if job['kind']=='account':
        account=run_account(w,config,diagnose=True,controller=controller)
        result=dict(account=account,market_digest=market,policy=job['policy']['key'])
    else:
        basepath=out/(job['baseline_name']+'.json')
        baseline=load(basepath,job['baseline_identity'])['account']
        observer=CaptureCloses(controller)
        replay=run_account(w,config,diagnose=True,controller=observer)
        if digest(replay)!=digest(baseline): raise ValueError('REFERENCE_SNAPSHOT_BASELINE_MISMATCH')
        rows=[];case_files={};original=compact_account(baseline)
        for target in targets(baseline):
            case=out/(job['name']+f"_BAR_{target['bar']:03d}.json")
            cid=digest(dict(job=identity,target=target,baseline=digest(baseline)))
            data=load(case,cid)
            if data is None:
                alternative=resume_without_close(w,observer.snapshots[target['bar']])
                alternative['trace']=[r for r in original['trace'] if r['bar']<target['bar']]+alternative['trace']
                comparison=compare_close(original,alternative,target)
                data=dict(target=target,comparison=comparison,alternative=alternative,
                          baseline_digest=digest(baseline))
                save(case,cid,data)
            rows.append(dict(target=target,comparison=data['comparison']))
            case_files[case.name]=sha(case)
        result=dict(cases=rows,case_files=case_files,baseline_digest=digest(baseline),market_digest=market)
    check(job['manifest'],native);result['seconds']=time.monotonic()-start
    save(path,identity,result);return result


def emit_report(output,identity,policies,results,attribution):
    report=dict(identity=identity,policies=policies,results=results,attribution=attribution,
        scope='SYNTHETIC_REFERENCE_POLICY_ROLLOUT_NOT_LEARNED_RL',selection_performed=False,
        uncertainty_calibrated=False,financial_training=False)
    write(output/'REPORT.json',report)
    lines=['# Frozen-reference rollout contrast','',f'Identity: `{identity}`.',
        'Seven frozen policies. Synthetic engineering, not financial validation. All costs retained.',
        'Value is Q of an explicit reference continuation, not exact Q of the adaptive selector.',
        'Carried forecasts preserve the current whole ensemble; no future real reports are read.',
        '', '| Bank / reporter | Policy | Return % | Worst DD % | Mean Sharpe | Orders/week | Buy/sell | Win % | Buy win % | Sell win % | Early | Local harmful |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x): return '-' if x is None else f'{x:.5f}'
    for bank,bykey in results.items():
        for policy in policies:
            r=bykey[policy['key']]
            lines.append(f"| {bank} | {policy['arm']} | {fmt(r['mean_return_pct'])} | {fmt(r['worst_dd_pct'])} | "
                f"{fmt(r['mean_sharpe'])} | {fmt(r['orders_per_week'])} | {r['buys']}/{r['sells']} | "
                f"{fmt(r['win_pct'])} | {fmt(r['buy_win_pct'])} | {fmt(r['sell_win_pct'])} | "
                f"{r['close_reasons'].get('AGENT_CLOSE',0)} | {r['harmful_closes']} |")
    lines+=['','## Actual Adaptive Close Attribution','',json.dumps(attribution,sort_keys=True),
        '', 'Positive account delta means CLOSE outperformed a one-tick HOLD followed by the same adaptive policy.',
        'This is separate from the local passive-position comparison. No automatic strategy promotion.',
        'World transformations are paired; aggregate means do not constitute independent-sample significance.',
        'Account summaries and raw receipts retain costs, horizon mix, order sizes, activity and undefined Sharpe.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def run(source,native,config_path,output,workers,smoke=False):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    sid,_=checked_archive(source);source_manifest=json.loads((source/'MANIFEST.json').read_text())
    config=json.loads(config_path.read_text());policies=configs(source_manifest,load(source/'SELECTION.json',sid),config)
    riskpath=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(riskpath)!=source_manifest['risk_sha256'] or sources(native)!=source_manifest['native_sources']:
        raise ValueError('REFERENCE_NATIVE_CHANGED')
    manifest=dict(schema='reference_rollout_manifest.v1',source_identity=sid,source_complete_sha256=sha(source/'COMPLETE.json'),
        sources=pins(),native_sources=sources(native),config=config,config_sha256=sha(config_path),policies=policies,
        calibration=fresh_specs(source_manifest,config,'calibration',smoke),
        confirmation=fresh_specs(source_manifest,config,'confirmation',smoke),smoke=smoke)
    identity=digest(manifest);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**kw): write(output/'STATE.json',dict(identity=identity,status=status,heartbeat_utc=time.time(),**kw))
    def verify():
        check(manifest,native)
        if sha(config_path)!=manifest['config_sha256']: raise ValueError('REFERENCE_CONFIG_MUTATION')
    try:
        old=load(output/'MANIFEST.json',identity)
        if old is None: save(output/'MANIFEST.json',identity,manifest)
        elif old!=manifest: raise ValueError('REFERENCE_RESUME_MANIFEST')
        done=load(output/'COMPLETE.json',identity)
        if done is not None:
            for name,h in done['artifacts'].items():
                if Path(name).name!=name or sha(output/name)!=h: raise ValueError('REFERENCE_COMPLETE_MUTATED')
            verify();return done
        if load(output/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(native))
            with (output/'TESTS.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','pytest','-q',*['tests/unit/'+t+'.py' for t in TESTS],
                    '--junitxml='+str(output/'TESTS.xml')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            suites=list(ET.parse(output/'TESTS.xml').iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
            if result.returncode or any(counts[k] for k in ('errors','failures','skipped')):
                raise ValueError(('REFERENCE_TESTS',counts))
            save(output/'TESTS.json',identity,counts)
        results={};attribution={};alljobs=[];ideal_jobs=[];risk=json.loads(riskpath.read_text());seen={}
        # Ideal and its actual-policy attribution run first, then noise/causal stress.
        phases=[('confirmation','ideal'),('attribution','ideal'),('calibration','noise_1')]+[
            ('confirmation',r) for r in (['noise_1'] if smoke else config['reporters'][1:])]
        for bank,reporter in phases:
            phase=bank+'_'+reporter
            if bank=='attribution':
                jobs=[dict(j,name='attribution_'+j['name'],kind='attribution',baseline_name=j['name'],
                           baseline_identity=digest(j)) for j in ideal_jobs if j['policy']['arm']=='rollout_joint']
            else:
                jobs=[dict(name=f'{bank}_{reporter}_{i:03d}_{p["key"][:8]}',kind='account',output=str(output),
                    engine_root=str(native),manifest=manifest,risk=risk,policy=p,spec=s,reporter=reporter)
                    for i,s in enumerate(manifest[bank]) for p in policies]
                if phase=='confirmation_ideal': ideal_jobs=jobs
            alljobs+=jobs;collected=[];start=time.monotonic()
            state('RUNNING',phase=phase,completed=0,total=len(jobs))
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures={pool.submit(unit,j):j for j in jobs}
                for n,f in enumerate(as_completed(futures),1):
                    job=futures[f];result=f.result();collected.append((job,result))
                    market_id=digest(job['spec'])
                    if market_id in seen and seen[market_id]!=result['market_digest']: raise ValueError('REFERENCE_UNPAIRED_MARKET')
                    seen[market_id]=result['market_digest'];verify()
                    state('RUNNING',phase=phase,completed=n,total=len(jobs),elapsed_seconds=time.monotonic()-start,
                        remaining_phase_estimate_seconds=remaining_seconds(jobs,collected,workers))
                    print(phase,n,'/',len(jobs),flush=True)
            collected.sort(key=lambda pair:pair[0]['name'])
            if bank=='attribution':
                cases=[c for _,result in collected for c in result['cases']];c=[r['comparison'] for r in cases]
                attribution=dict(accounts=len(jobs),closes=len(c),profit_helpful=sum(x['account_delta_usd']>1e-9 for x in c),
                    profit_harmful=sum(x['account_delta_usd']< -1e-9 for x in c),
                    utility_helpful=sum(x['return_delta_pp']+x['drawdown_advantage_pp']>1e-9 for x in c),
                    utility_harmful=sum(x['return_delta_pp']+x['drawdown_advantage_pp']< -1e-9 for x in c),
                    pareto_relations=dict(Counter(x['pareto_relation'] for x in c)))
            else:
                results[phase]={p['key']:summarize([r['account'] for j,r in collected if j['policy']['key']==p['key']])
                                for p in policies}
            emit_report(output,identity,policies,results,attribution)
        verify()
        (output/'AUDIT_REQUEST.md').write_text('\n'.join([
            '# Audit Request: Frozen-Reference Rollout', '',f'Identity: `{identity}`.',
            'Read MANIFEST.json, TESTS.json, RESULTS.md, REPORT.json, COMPLETE.json and all per-account/case receipts.',
            '1. Attack carried-ensemble information boundaries and controller-memory isolation.',
            '2. Independently replay both frozen controls and new controllers in the pinned native engine.',
            '3. Check common action/reference choice, no recursive selector or sample-wise oracle maximization.',
            '4. Check every joint ideal close against the actual adaptive HOLD continuation, including detrimental cases.',
            '5. Separate reference-value correctness from adaptive-policy quality and financial learnability.',
            '6. Verify costs, SL/TP, sizing, weekly gates, direction balance, activity, horizon mix and all noise/causal losses.',
            'No global Pareto optimum, calibrated uncertainty, learned profitability or floating-selection release claimed.',
            'No coefficients were selected on these banks. Give execution, mechanism and learned-interface verdicts separately.','']))
        done=dict(accounts=sum(j['kind']=='account' for j in alljobs),attribution=attribution,tests=load(output/'TESTS.json',identity),
            artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('LOCK','STATE.json','COMPLETE.json')})
        save(output/'COMPLETE.json',identity,done);state('COMPLETE',accounts=done['accounts'],attribution=attribution);return done
    except BaseException as error:
        state('FAILED',error_type=type(error).__name__,detail=str(error));raise
    finally:
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','engine-root','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reference_rollout_contrast.json')
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();run(a.source,a.engine_root,a.config,a.output,a.workers,a.smoke)
