#!/usr/bin/env python3
"""Test-qualified, resumable synthetic controller grid and frozen confirmation."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import fcntl
import itertools
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
from app.crossed_account_panel import panel_specs
from app.dynamic_account_arbiter import frontier
from app.pragmatic_strategy import ControllerConfig,PragmaticWorld,run_account
from app.zigzag_opportunities import engine,sha
from tools.optimize_dynamic_account import digest,sources
from tools.run_causal_history_queue import SUITE,source_identity
from tools.run_crossed_account_queue import save,load
from tools.verify_absolute_early_close import write

TESTS=SUITE+['test_sequential_account_pareto','test_suppressed_close_counterfactuals',
             'test_frozen_policy_diagnostics','test_pragmatic_strategy','test_pragmatic_queue']
EXTRA=['app/pragmatic_strategy.py','app/sequential_account_pareto.py',
       'app/suppressed_close_counterfactuals.py','app/frozen_policy_diagnostics.py',
       'app/close_hold_reports.py','tools/run_pragmatic_contrast.py',
       'tools/diagnose_suppressed_closes.py','tools/diagnose_frozen_history_policy.py',
       'tools/collect_causal_history_lanes.py']+['tests/unit/'+t+'.py' for t in TESTS]


def configurations(config,smoke):
    if config['schema']!='pragmatic_controller_contrast.v1' or config['horizons']!=[3,6,12,24]:
        raise ValueError('PRAGMATIC_RUN_CONFIG')
    rows=[]
    for variant in config['variants']:
        for alpha,uncertainty,persistence in itertools.product(
                config['grid']['alpha'],config['grid']['uncertainty'],config['grid']['persistence']):
            c=ControllerConfig(variant,alpha=alpha,uncertainty=uncertainty,persistence=persistence,
                               **dict(config['fixed'],close_offsets=tuple(config['fixed']['close_offsets'])))
            values=asdict(c);rows.append(dict(key=digest(values),config=values))
            if smoke: break
    if len({r['key'] for r in rows})!=len(rows): raise ValueError('DUPLICATE_GRID_CANDIDATE')
    return rows


def summarize(rows):
    if not rows: raise ValueError('EMPTY_SUMMARY')
    n=len(rows);orders=sum(r['orders'] for r in rows);buys=sum(r['buys'] for r in rows);sells=sum(r['sells'] for r in rows)
    wins=sum(r['wins'] for r in rows);bw=sum(r['buy_wins'] for r in rows);sw=sum(r['sell_wins'] for r in rows)
    sharpes=[r['sharpe'] for r in rows if r['sharpe'] is not None]
    mean_orders=float(np.mean([r['orders_per_week'] for r in rows]))
    costs=Counter();horizons=Counter();reasons=Counter()
    for r in rows: costs.update(r['costs']);horizons.update(r['horizons']);reasons.update(r['close_reasons'])
    measured=all(r['close_diagnostics']['measured'] for r in rows)
    closes=[c for r in rows for c in r['close_rows'] if not c['preempted']]
    return dict(accounts=n,mean_return_pct=float(np.mean([r['return_pct'] for r in rows])),
        mean_net_usd=float(np.mean([r['net_usd'] for r in rows])),worst_dd_pct=max(r['drawdown_pct'] for r in rows),
        mean_sharpe=float(np.mean(sharpes)) if sharpes else None,defined_sharpes=len(sharpes),
        orders=orders,orders_per_week=mean_orders,activity_eligible=mean_orders>2,
        buys=buys,sells=sells,win_pct=100*wins/orders if orders else None,
        buy_share_pct=100*buys/orders if orders else None,buy_win_pct=100*bw/buys if buys else None,
        sell_win_pct=100*sw/sells if sells else None,costs=dict(costs),horizons=dict(horizons),
        close_reasons=dict(reasons),harmful_closes=sum(c['delta_vs_hold']< -1e-9 for c in closes) if measured else None,
        mean_close_regret_risk=float(np.mean([c['regret_risk'] for c in closes])) if closes else None)


def freeze_selection(configs,metrics):
    output=[]
    for variant in sorted({c['config']['variant'] for c in configs}):
        keys=[c['key'] for c in configs if c['config']['variant']==variant]
        eligible=[k for k in keys if metrics[k]['orders_per_week']>2]
        pool=eligible or keys
        front=frontier({k:(metrics[k]['mean_return_pct'],-metrics[k]['worst_dd_pct']) for k in pool})
        chosen=[max(front,key=lambda k:(metrics[k]['mean_return_pct'],-metrics[k]['worst_dd_pct'],k)),
                min(front,key=lambda k:(metrics[k]['worst_dd_pct'],-metrics[k]['mean_return_pct'],k))]
        for key in dict.fromkeys(chosen):
            output.append(dict(key=key,variant=variant,activity_eligible=bool(eligible),
                               selection_bank='calibration_noise_1',fallback=not bool(eligible)))
    return output


def job_receipt(path,job,result=None):
    identity=digest(job)
    if result is None: return load(path,identity)
    save(path,identity,result)


def worker(job):
    path=Path(job['output'])/(job['name']+'.json')
    cached=job_receipt(path,job)
    if cached is not None: return cached
    engine(Path(job['engine_root']));start=time.monotonic()
    world=PragmaticWorld(job['spec'],job['risk'],job['reporter'])
    rows={}
    for row in job['configs']:
        params=dict(row['config']);params['close_offsets']=tuple(params['close_offsets'])
        rows[row['key']]=run_account(world,ControllerConfig(**params),diagnose=job['diagnose'])
    result=dict(rows=rows,seconds=time.monotonic()-start,
                market_digest=digest(np.column_stack((world.env.o,world.env.h,world.env.l,world.env.c)).tolist()))
    job_receipt(path,job,result)
    return result


def specs(config,bank,smoke):
    seed=(930000 if bank=='calibration' else 940000) if smoke else config[bank+'_seed']
    rows=[dict(r,seed=seed,bank='pragmatic_'+bank) for r in panel_specs(bank,draws=1)]
    return rows[:2] if smoke else rows


def report(output,manifest,calibration,selection,confirmation):
    lines=['# Pragmatic controller contrast','',f"Identity: `{manifest['identity']}`.",
        'Synthetic strategy/output engineering. Not financial VALIDATION or learned RL performance.',
        'Predictive proxy is not a reproduction of the historical strategy. Sequential is bounded two-order planning.',
        'All calibration candidates retained. Representatives frozen before confirmation; >2 orders/week is explicit.',
        '', '| Arm / profile | Reporter | Return % | Worst DD % | Mean Sharpe | Orders/week | Buy / sell | Win % | Buy win % | Sell win % | Early closes | Harmful |',
        '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|']
    def fmt(x): return '-' if x is None else f'{x:.4f}'
    variants={r['key']:r['variant'] for r in selection}
    for reporter,rows in confirmation.items():
        for key,r in rows.items():
            lines.append(f"| {variants[key]} / {key[:8]} | {reporter} | {fmt(r['mean_return_pct'])} | {fmt(r['worst_dd_pct'])} | "
                f"{fmt(r['mean_sharpe'])} | {fmt(r['orders_per_week'])} | {r['buys']} / {r['sells']} | {fmt(r['win_pct'])} | "
                f"{fmt(r['buy_win_pct'])} | {fmt(r['sell_win_pct'])} | {r['close_reasons'].get('AGENT_CLOSE',0)} | {r['harmful_closes']} |")
    lines+=['','The uncertainty term is report-sample dispersion, not a calibrated confidence level.',
        'Mean Sharpe averages defined account Sharpes; it is not pooled portfolio Sharpe. Costs and horizon mix are in REPORT.json.',
        'Orders/week uses market sessions, not ISO labels: Sunday and Monday belong to one operational week here. Native ISO metric is preserved in each account.',
        'Harmful close is local fixed-position versus passive; it does not include subsequent-account opportunities.',
        'Worlds share mirrored/envelope transformations; no independent-sample significance claim.',
        'No optimizer parameters, data selection or model release are changed by confirmation.',
        'The finite plan frontier is enumerated; ties and close safeguards can override its score winner. No global or executed-policy optimum is claimed.',
        'NOP earns zero with zero orders and is ineligible. Negative results are preserved.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    write(output/'REPORT.json',dict(identity=manifest['identity'],calibration=calibration,
        selection=selection,confirmation=confirmation,scope='SYNTHETIC_ENGINEERING_NOT_LEARNED_FINANCIAL_RESULT'))


def run(config_path,engine_root,output,workers,smoke=False):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    config=json.loads(config_path.read_text())
    if config['activity_min_exclusive']!=2 or config['financial_training'] is not False:
        raise ValueError('BUSINESS_OR_TRAINING_CONTRACT')
    configurations_=configurations(config,smoke)
    risk_path=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    risk=json.loads(risk_path.read_text());native=sources(engine_root)
    own=source_identity()|{p:sha(ROOT/p) for p in EXTRA}
    body=dict(schema='pragmatic_contrast_manifest.v1',smoke=smoke,config=config,
        config_sha256=sha(config_path),risk_sha256=sha(risk_path),sources=own,native_sources=native,
        configurations=configurations_,calibration=specs(config,'calibration',smoke),
        confirmation=specs(config,'confirmation',smoke))
    identity=digest(body);manifest=dict(body,identity=identity)
    output.mkdir(parents=True,exist_ok=True);lock=(output/'LOCK').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    mp=output/'MANIFEST.json'
    if mp.exists() and json.loads(mp.read_text())!=json.loads(json.dumps(manifest)):
        raise ValueError('CONTRAST_RESUME_IDENTITY')
    write(mp,manifest)
    def check():
        if own!=source_identity()|{p:sha(ROOT/p) for p in EXTRA} or sources(engine_root)!=native:
            raise ValueError('CONTRAST_SOURCE_MUTATION')
        if sha(config_path)!=body['config_sha256'] or sha(risk_path)!=body['risk_sha256']:
            raise ValueError('CONTRAST_CONFIG_MUTATION')
    def state(status,**more): write(output/'STATE.json',dict(identity=identity,status=status,heartbeat_utc=time.time(),**more))
    try:
        complete=load(output/'COMPLETE.json',identity)
        if complete is not None:
            for name,h in complete['artifacts'].items():
                if sha(output/name)!=h: raise ValueError('COMPLETE_ARTIFACT_CHANGED')
            check();return complete
        if load(output/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(engine_root))
            with (output/'TESTS.log').open('w') as stream:
                result=subprocess.run([sys.executable,'-m','pytest','-q',*[f'tests/unit/{t}.py' for t in TESTS],
                    '--junitxml='+str(output/'TESTS.xml')],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            tree=ET.parse(output/'TESTS.xml');suites=list(tree.iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
            if result.returncode or counts['failures'] or counts['errors'] or counts['skipped']:
                raise ValueError(('CONTRAST_TESTS',counts))
            save(output/'TESTS.json',identity,counts)
        def stage(phase,reporters,configs):
            jobs=[dict(name=f'{phase}_{reporter}_{i:03d}',output=str(output),engine_root=str(engine_root),
                identity=identity,spec=spec,risk=risk,reporter=reporter,configs=configs,diagnose=phase=='confirmation')
                for reporter in reporters for i,spec in enumerate(body[phase])]
            totals={reporter:{r['key']:[] for r in configs} for reporter in reporters}
            expected_markets={};durations=[];start=time.monotonic()
            state('RUNNING',phase=phase,completed=0,total=len(jobs))
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                for i,result in enumerate(pool.map(worker,jobs)):
                    job=jobs[i];reporter=job['reporter'];market_id=digest(job['spec'])
                    if market_id in expected_markets and expected_markets[market_id]!=result['market_digest']:
                        raise ValueError('UNPAIRED_MARKET')
                    expected_markets[market_id]=result['market_digest']
                    for key,row in result['rows'].items(): totals[reporter][key].append(row)
                    durations.append(result['seconds']);check()
                    state('RUNNING',phase=phase,completed=i+1,total=len(jobs),
                        elapsed_seconds=time.monotonic()-start,mean_unit_seconds=float(np.mean(durations)),
                        remaining_estimate_seconds=float(np.mean(durations))*(len(jobs)-i-1)/workers)
                    print(phase,i+1,'/',len(jobs),flush=True)
            return {reporter:{key:summarize(rows) for key,rows in bykey.items()} for reporter,bykey in totals.items()}
        cal=load(output/'CALIBRATION.json',identity)
        if cal is None:
            cal=stage('calibration',['noise_1'],configurations_)['noise_1']
            save(output/'CALIBRATION.json',identity,cal)
        selection=freeze_selection(configurations_,cal)
        frozen=load(output/'SELECTION.json',identity)
        if frozen is not None and frozen!=selection: raise ValueError('FROZEN_SELECTION_CHANGED')
        if frozen is None: save(output/'SELECTION.json',identity,selection)
        keys={r['key'] for r in selection};selected=[r for r in configurations_ if r['key'] in keys]
        reporters=['ideal','noise_1','causal'] if smoke else config['confirmation_reporters']
        confirmation=stage('confirmation',reporters,selected)
        report(output,manifest,cal,selection,confirmation);check()
        artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('STATE.json','LOCK','COMPLETE.json')}
        completed=dict(artifacts=artifacts,candidates=len(configurations_),representatives=len(selection),
            tests=load(output/'TESTS.json',identity),financial_training=False)
        save(output/'COMPLETE.json',identity,completed);state('COMPLETE',**{k:v for k,v in completed.items() if k!='artifacts'})
        return completed
    except BaseException as error:
        state('FAILED',error_type=type(error).__name__,detail=str(error))
        raise
    finally:
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/pragmatic_controller_contrast.json')
    p.add_argument('--engine-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.config,a.engine_root,a.output,a.workers,a.smoke)
