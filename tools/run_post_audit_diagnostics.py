#!/usr/bin/env python3
"""Durable sequential audit repairs, native exit counterfactuals and causal baseline diagnostics.

No learned policy or optimizer is launched. Incomplete scientific qualification
is an explicit terminal result, never authorization for floating selection.
"""
import argparse
import copy
import fcntl
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app.causal_report_baseline import sampled_prefix_report
from app.crossed_account_panel import CrossedWorld, execution_specs, QUALITIES
from app.dynamic_account_panel import HOUR
from app.report_track_record import ReportTrackRecord
from app.zigzag_opportunities import engine, sha
from tools.analyze_crossed_account_results import paired_clusters
from tools.optimize_dynamic_account import digest
from tools.run_crossed_account_queue import TESTS, load, save, pins
from tools.verify_absolute_early_close import write

EXTRA_TESTS=['test_crossed_account_analysis','test_crossed_audit_boundary_regressions',
             'test_report_cache_binding','test_report_track_record','test_causal_report_baseline',
             'test_post_audit_diagnostics']
EVIDENCE=ROOT/'docs/audits/evidence/crossed_account_final_20261029'


def clone_position(env):
    other=copy.copy(env)
    for name in ('trades','log','decisions'):
        setattr(other,name,list(getattr(env,name)))
    for name in ('entries_day','counts'):
        setattr(other,name,dict(getattr(env,name)))
    other.pos=copy.copy(env.pos)
    return other


def close_metrics(rows,accounts):
    if not rows:
        return dict(count=0,helpful=0,destructive=0,neutral=0,mean_regret_risk=None,
                    delta_vs_hold_pp=0.,by_hold_reason={},by_direction={})
    classes=[('helpful' if r['executed']-r['hold']>.00002*r['units'] else
              'destructive' if r['executed']-r['hold']<-.00002*r['units'] else 'neutral') for r in rows]
    return dict(count=len(rows),**{k:classes.count(k) for k in ('helpful','destructive','neutral')},
        mean_regret_risk=float(np.mean([(r['optimal']-r['executed'])/r['risk'] for r in rows])),
        delta_vs_hold_pp=sum(r['executed']-r['hold'] for r in rows)/accounts/10000*100,
        by_hold_reason={k:sum(r['hold_reason']==k for r in rows) for k in ('SL','TP','FORCED_FLAT')},
        by_direction={str(k):sum(r['direction']==k for r in rows) for k in (-1,1)})


def counterfactuals(world,archived,genes):
    env=copy.copy(world.env);env.reset()
    records=[];risk=None;preempted=0
    def apply(t):
        action=t['action']
        plan=world.entry_plan(t['horizon'],1 if action==1 else -1,genes[8]) if action in (1,2) else None
        return env.step(t['bar'],action,entry_plan=plan)
    for event in archived['trace']:
        if event['action']==3 and env.pos is not None:
            hold=clone_position(env);alternatives=[]
            while hold.pos is not None:
                candidate=clone_position(hold)
                candidate.step(candidate.next_decision,3)
                alternatives.append(candidate.trades[-1]['net_usd'])
                hold.step(hold.next_decision,0)
            passive=hold.trades[-1]
            info=apply(event);trade=env.trades[-1]
            if abs(trade['net_usd']-alternatives[0])>1e-9:
                raise ValueError('CLOSE_CLONE_CASH')
            if info['close']!='AGENT_CLOSE':
                preempted+=1
            else:
                records.append(dict(entry_bar=trade['entry_bar_abs'],close_bar=trade['exit_bar_abs'],
                    direction=trade['direction'],units=trade['units'],risk=risk,
                    executed=trade['net_usd'],hold=passive['net_usd'],hold_reason=passive['close_reason'],
                    optimal=max(alternatives+[passive['net_usd']]),alternatives=len(alternatives)+1))
        else:
            info=apply(event)
            if info.get('entry'):
                risk=info['absolute_order']['modeled_loss']
    expected=archived['ledger']
    actual=env.trades
    if len(expected)!=len(actual):
        raise ValueError('TRACE_LEDGER_COUNT')
    for x,y in zip(expected,actual):
        if (x['entry_bar']!=y['entry_bar_abs'] or x['exit_bar']!=y['exit_bar_abs']
                or x['units']!=y['units'] or x['reason']!=y['close_reason']
                or abs(x['net']-y['net_usd'])>1e-9):
            raise ValueError('TRACE_LEDGER_PARITY')
    return records,preempted


def exit_unit(key,quality,genes,config,smoke=False):
    archive=json.loads(gzip.decompress((EVIDENCE/f'REPLAY_{quality}_{key}.json.gz').read_bytes()))['worlds']
    if smoke:
        archive=archive[:2]
    rows=[];no_close=[];preempted=0
    for index,old in enumerate(archive):
        world=CrossedWorld(old['spec'],config)
        if world.run(genes,verify=True,details=True)!=old:
            raise ValueError('REPAIRED_ACCOUNT_NOT_EXACT')
        found,n=counterfactuals(world,old,genes)
        rows.extend(dict(world=index,**r) for r in found);preempted+=n
        control=world.run(genes,verify=True,details=True,voluntary_close=False)
        if control['closes'].get('AGENT_CLOSE',0):
            raise ValueError('NO_CLOSE_ABLATION_BYPASS')
        no_close.append(control)
    return dict(policy=key,quality=quality,accounts=len(archive),exact_native_replays=len(archive),
        preempted=preempted,close_metrics=close_metrics(rows,len(archive)),
        by_passive={k:close_metrics([r for r in rows if r['hold_reason']==k],len(archive)) for k in ('SL','TP','FORCED_FLAT')},
        by_side={str(k):close_metrics([r for r in rows if r['direction']==k],len(archive)) for k in (-1,1)},
        account_paired_return=paired_clusters(archive,no_close),
        baseline_return=float(np.mean([w['return_pct'] for w in archive])),
        no_close_return=float(np.mean([w['return_pct'] for w in no_close])),
        baseline_worst_dd=max(w['drawdown_pct'] for w in archive),
        no_close_worst_dd=max(w['drawdown_pct'] for w in no_close),
        baseline_orders=sum(w['orders'] for w in archive),no_close_orders=sum(w['orders'] for w in no_close),
        close_rows=rows,no_close_worlds=no_close)


def calibration_unit(mode,config,smoke=False):
    specifications=execution_specs('confirmation','correlated',False)
    if smoke:
        specifications=specifications[:2]
    horizons=(1,3,6,12,24);measurements=[]
    for index,s in enumerate(specifications):
        spec=dict(s,seed=s['seed']+40000+(20000 if smoke else 0),bank='causal_calibration_diagnostic')
        world=CrossedWorld(spec,config);e=world.env
        ledger=ReportTrackRecord('resampled-prefix-v1/'+mode,horizons=horizons)
        assessments=[];radii=None
        for b in range(2,world.end):
            now=int(e.v[b])+HOUR
            ledger.observe(now,float(e.c[b]),now=now)
            for row in assessments:
                if row['target_bar']==b:
                    row['observed']=float(e.c[b])
                    row['raw_covered']=bool(row['raw_band'][0]<=e.c[b]<=row['raw_band'][1])
                    row['calibrated_covered']=bool(row['calibrated_band'][0]<=e.c[b]<=row['calibrated_band'][1])
            if b==64:
                radii={h:ledger.interval(0.,h,now=now) for h in horizons}
                if any(x is None for x in radii.values()):
                    raise ValueError('CALIBRATION_WARMUP_SUPPORT')
            if b+1>=world.end:
                continue
            count=world.end-b-1;f=e.cfg.profile
            times=now+np.arange(count)*HOUR
            spread=np.array([f.spread_at(int(t),int(e.mo[0])) for t in times])
            seed=np.random.SeedSequence([spec['seed'],('trend','reversal','overlap').index(spec['pattern']),
                ('constant','increasing','decreasing').index(spec['volatility']),b,1831])
            report=sampled_prefix_report(np.column_stack((e.o,e.h,e.l,e.c))[:b+1],now=now,
                step=HOUR,deadline=int(e.ff[0]),spreads=spread,
                terminal_spread=f.spread_at(int(e.ff[0]),int(e.mo[0])),seed=seed,drift=mode)
            ledger.issue(report,now=now,expected_deadline=int(e.ff[0]))
            if radii is not None:
                for h in horizons:
                    if h>count:
                        continue
                    values=report.paths[:,h-1,3];center=float(report.weights@values)
                    assessments.append(dict(world=index,horizon=h,target_bar=b+h,
                        raw_band=np.quantile(values,[.05,.95],method='inverted_cdf').tolist(),
                        calibrated_band=[center+x for x in radii[h]]))
        if any('observed' not in row for row in assessments):
            raise ValueError('UNMATURED_ASSESSMENT')
        restored=ReportTrackRecord.restore(ledger.state(),expected_producer='resampled-prefix-v1/'+mode)
        if ledger.state()!=restored.state():
            raise ValueError('TRACK_RECORD_RESTART')
        measurements.extend(assessments)
    rows={}
    for h in horizons:
        subset=[r for r in measurements if r['horizon']==h]
        rows[str(h)]=dict(n=len(subset),raw_coverage=float(np.mean([r['raw_covered'] for r in subset])),
            calibrated_coverage=float(np.mean([r['calibrated_covered'] for r in subset])),
            mean_width=float(np.mean([r['calibrated_band'][1]-r['calibrated_band'][0] for r in subset])))
    return dict(mode=mode,worlds=len(specifications),samples=32,fit_cutoff_bar=64,
        adaptation_after_cutoff=False,horizons=rows,
        nominal_coverage_screen=all(.8<=r['calibrated_coverage']<=.95 and r['mean_width']>0 for r in rows.values()),
        scope='Empirical baseline calibration diagnostic; not learned-model or joint-path qualification',measurements=measurements)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    lock=(out/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    engine(a.engine_root)
    code=pins(a.engine_root)
    extra=['app/report_track_record.py','app/causal_report_baseline.py',
           'tools/run_post_audit_diagnostics.py']+['tests/unit/'+x+'.py' for x in EXTRA_TESTS]
    code['agent_multi'].update({name:sha(ROOT/name) for name in extra})
    cfg=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    config=json.loads(cfg.read_text())
    complete=json.loads((EVIDENCE/'COMPLETE.json').read_text())['data']
    front=json.loads((EVIDENCE/'FRONTIER.json').read_text())['data']
    recipes={r['key']:r['genes'] for r in front['rows']}
    reps=front['representatives'][:1] if a.smoke else front['representatives']
    units=[('exits_'+q+'_'+key,dict(kind='exit',key=key,quality=q)) for q in QUALITIES for key in reps]
    units.extend(('calibration_'+mode,dict(kind='calibration',mode=mode)) for mode in ('median','zero'))
    evidence_hashes={r['file']:r['sha256'] for r in complete['replay_files']}
    for name,expected in evidence_hashes.items():
        if sha(EVIDENCE/name)!=expected:
            raise ValueError('OLD_ARCHIVE_CHANGED')
    manifest=dict(schema='post_audit_diagnostics.v1',smoke=a.smoke,sources=code,
        config_sha256=sha(cfg),archive_identity=complete['identity'],archives=evidence_hashes,
        frontier_sha256=sha(EVIDENCE/'FRONTIER.json'),units=units,
        horizon_list=[1,3,6,12,24],causal_seed_start=120100 if a.smoke else 100100,
        minimum_warmup_errors=20,calibration_cutoff_bar=64,coverage_screen=[.8,.95],
        nominal=.9,samples=32,close_tolerance_sigma=.02,new_optimization=False)
    identity=digest(manifest);manifest['identity']=identity
    path=out/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('DIAGNOSTIC_RESUME_IDENTITY')
    write(path,manifest)
    def check():
        current=pins(a.engine_root)
        if any(current[key]!=code[key] for key in current if key!='agent_multi'):
            raise ValueError('DIAGNOSTIC_NATIVE_SOURCE_MUTATION')
        for name,expected in code['agent_multi'].items():
            if sha(ROOT/name)!=expected:
                raise ValueError('DIAGNOSTIC_SOURCE_MUTATION')
        if sha(cfg)!=manifest['config_sha256']:
            raise ValueError('DIAGNOSTIC_CONFIG_MUTATION')
    def state(status,**more):
        write(out/'STATE.json',dict(status=status,identity=identity,heartbeat_utc=time.time(),**more))
    try:
        if load(out/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(a.engine_root))
            with (out/'TESTS.log').open('w') as log:
                subprocess.run([sys.executable,'-m','pytest','-q']+['tests/unit/'+x+'.py' for x in TESTS+EXTRA_TESTS]+
                    ['--junitxml='+str(out/'TESTS.xml')],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            cases=ET.parse(out/'TESTS.xml').getroot().findall('.//testcase')
            if not cases or any(c.find(t) is not None for c in cases for t in ('failure','error','skipped')):
                raise ValueError('DIAGNOSTIC_TESTS_FAILED')
            save(out/'TESTS.json',identity,dict(passed=len(cases),skipped=0))
        summaries=[];timings=[]
        for index,(name,spec) in enumerate(units):
            check();receipt=load(out/(name+'.json'),identity)
            artifact=out/(name+'.json.gz')
            if receipt is None:
                state('RUNNING',phase=spec['kind'],unit=name,completed=index,total=len(units),
                      elapsed_units=timings)
                started=time.monotonic()
                result=(exit_unit(spec['key'],spec['quality'],recipes[spec['key']],config,a.smoke)
                        if spec['kind']=='exit' else calibration_unit(spec['mode'],config,a.smoke))
                raw=json.dumps(result,sort_keys=True,allow_nan=False).encode()
                temporary=artifact.with_suffix('.tmp');temporary.write_bytes(gzip.compress(raw,mtime=0));temporary.replace(artifact)
                summary={k:v for k,v in result.items() if k not in ('close_rows','no_close_worlds','measurements')}
                receipt=dict(unit=name,artifact=artifact.name,sha256=sha(artifact),seconds=time.monotonic()-started,summary=summary)
                save(out/(name+'.json'),identity,receipt)
            if sha(artifact)!=receipt['sha256']:
                raise ValueError('DIAGNOSTIC_RESULT_CHANGED')
            timings.append(receipt['seconds']);summaries.append(receipt)
            print(json.dumps(dict(unit=name,completed=index+1,total=len(units),seconds=receipt['seconds'])),flush=True)
        check()
        result=dict(identity=identity,status='SMOKE_ONLY' if a.smoke else 'COMPLETE_DIAGNOSTICS_NOT_LEARNED_QUALIFICATION',
            completed=len(units),units=summaries,tests=load(out/'TESTS.json',identity),
            new_optimization=False,learned_adapter_qualified=False,automatic_next_training=False)
        save(out/'COMPLETE.json',identity,result)
        state('COMPLETE',phase='DIAGNOSTICS',completed=len(units),total=len(units),new_optimization=False)
        lines=['# Post-audit diagnostic results','',result['status'],'',
            '| Policy | Quality | Closes | Destructive | Regret / risk | Return % | No-close return % |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
        for receipt in summaries:
            s=receipt['summary']
            if 'close_metrics' in s:
                c=s['close_metrics'];reg='-' if c['mean_regret_risk'] is None else f"{c['mean_regret_risk']:.5f}"
                lines.append(f"| {s['policy'][:8]} | {s['quality']} | {c['count']} | {c['destructive']} | {reg} | {s['baseline_return']:.5f} | {s['no_close_return']:.5f} |")
        lines+=['','Causal calibration is diagnostic only; see calibration receipts for coverage and width.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    except Exception as exc:
        state('FAILED',error_type=type(exc).__name__,error=str(exc))
        raise


if __name__=='__main__':
    main()
