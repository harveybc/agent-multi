#!/usr/bin/env python3
"""Attribute ideal-report decision reversals to committed versus replanned continuation."""
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
from app.pragmatic_plan_gap import committed_replay,check_valuation,decompose,first_deviation
from app.pragmatic_close_attribution import targets,verify_baseline
from app.pragmatic_strategy import ControllerConfig,PragmaticController,PragmaticWorld,run_account
from app.zigzag_opportunities import engine,sha
from tools.attribute_pragmatic_closes import authenticated,own_sources,TESTS as BASE_TESTS
from tools.optimize_dynamic_account import digest,sources
from tools.run_crossed_account_queue import save,load
from tools.verify_absolute_early_close import write

TESTS=BASE_TESTS+['test_pragmatic_plan_gap']


def checked_archive(root):
    envelope=json.loads((root/'COMPLETE.json').read_text());identity=envelope['identity']
    complete=load(root/'COMPLETE.json',identity)
    for name,h in complete['artifacts'].items():
        if Path(name).name!=name or sha(root/name)!=h: raise ValueError('PLAN_GAP_ARCHIVE_CHANGED')
    return identity,complete


def pins():
    return own_sources()|{p:sha(ROOT/p) for p in ('app/pragmatic_plan_gap.py',
        'tools/diagnose_pragmatic_plan_gap.py','tests/unit/test_pragmatic_plan_gap.py')}


def check_sources(manifest,native):
    if pins()!=manifest['sources'] or sources(native)!=manifest['native_sources']:
        raise ValueError('PLAN_GAP_SOURCE_MUTATION')


def unit(job):
    out=Path(job['output']);identity=digest(job);path=out/(job['name']+'.json')
    check_sources(job['manifest'],Path(job['engine_root']))
    cached=load(path,identity)
    if cached is not None: return cached
    engine(Path(job['engine_root']))
    archive=authenticated(Path(job['source'])/job['source_file'],job['source_sha'])['rows'][job['key']]
    if archive['reporter']!='ideal': raise ValueError('PLAN_GAP_IDEAL_REQUIRED')
    params=dict(archive['config']);params['close_offsets']=tuple(params['close_offsets'])
    config=ControllerConfig(**params);world=PragmaticWorld(archive['spec'],job['risk'],'ideal')
    expected={r['bar']:r for r in targets(archive)};rows=[];start=time.monotonic()
    class Observer(PragmaticController):
        def decide(self,w,e,equity,horizon):
            decision=super().decide(w,e,equity,horizon);b=e.next_decision
            if b not in expected: return decision
            target=expected[b]
            if (decision['action']!='CLOSE' or e.pos is None or e.pos.entry_bar!=target['entry_bar']
                    or e.pos.direction!=target['direction']): raise ValueError('PLAN_GAP_TARGET')
            pc=decision['plan'];ph=max((p for p in pc['plans'] if p['plan'][0]=='HOLD'),key=lambda p:p['score'])
            if pc['plan'][0]!='CLOSE': raise ValueError('PLAN_GAP_SELECTED_ACTION')
            committed_close=committed_replay(w,e,equity,pc['plan'])
            committed_hold=committed_replay(w,e,equity,ph['plan'])
            errors=[check_valuation(committed_close,pc,config.account_risk_weight),
                    check_valuation(committed_hold,ph,config.account_risk_weight)]
            name=job['case_prefix']+f'_BAR_{b:03d}.json'
            case=authenticated(Path(job['attribution'])/name,job['case_hashes'][name])
            if case['target']!=target or case['baseline_digest']!=digest(archive):
                raise ValueError('PLAN_GAP_CASE_BINDING')
            alternative=case['alternative']
            row=dict(target=target,valuation_error=max(errors),close_plan=pc['plan'],hold_plan=ph['plan'],
                decomposition=decompose(committed_close,committed_hold,archive,alternative,config.account_risk_weight),
                close_first_deviation=first_deviation(committed_close['trace'],archive['trace']),
                hold_first_deviation=first_deviation(committed_hold['trace'],alternative['trace']),
                committed_close=committed_close,committed_hold=committed_hold,
                actual_close_return_pct=archive['return_pct'],actual_hold_return_pct=alternative['return_pct'])
            rows.append(row)
            return decision
    replay=run_account(world,config,diagnose=True,controller=Observer(config))
    verify_baseline(replay,archive)
    if len(rows)!=len(expected): raise ValueError('PLAN_GAP_COVERAGE')
    check_sources(job['manifest'],Path(job['engine_root']))
    result=dict(key=job['key'],world=job['world'],rows=rows,baseline_digest=digest(replay),seconds=time.monotonic()-start)
    save(path,identity,result);return result


def run(source,attribution,engine_root,output,workers,smoke=False):
    if workers not in (1,2,3,4): raise ValueError('WORKER_BUDGET')
    sid,original=checked_archive(source);aid,attributed=checked_archive(attribution)
    source_manifest=json.loads((source/'MANIFEST.json').read_text())
    attr_manifest=load(attribution/'MANIFEST.json',aid)
    if source_manifest['identity']!=sid or attr_manifest['source_identity']!=sid:
        raise ValueError('PLAN_GAP_ARCHIVE_BINDING')
    selected=[r['key'] for r in load(source/'SELECTION.json',sid) if r['variant']=='sequential']
    riskpath=ROOT/'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json'
    if sha(riskpath)!=source_manifest['risk_sha256'] or sources(engine_root)!=source_manifest['native_sources']:
        raise ValueError('PLAN_GAP_NATIVE_CHANGED')
    manifest=dict(schema='pragmatic_plan_gap.v1',source_identity=sid,attribution_identity=aid,
        source_complete_sha256=sha(source/'COMPLETE.json'),attribution_complete_sha256=sha(attribution/'COMPLETE.json'),
        sources=pins(),native_sources=sources(engine_root),smoke=smoke,
        world_count=2 if smoke else len(source_manifest['confirmation']),policies=selected,
        calibration_tuning=False,financial_training=False)
    identity=digest(manifest);output.mkdir(parents=True,exist_ok=True)
    lock=(output/'LOCK').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def state(status,**kw): write(output/'STATE.json',dict(identity=identity,status=status,heartbeat_utc=time.time(),**kw))
    try:
        old=load(output/'MANIFEST.json',identity)
        if old is None: save(output/'MANIFEST.json',identity,manifest)
        elif old!=manifest: raise ValueError('PLAN_GAP_RESUME_MANIFEST')
        done=load(output/'COMPLETE.json',identity)
        if done is not None:
            for name,h in done['artifacts'].items():
                if sha(output/name)!=h: raise ValueError('PLAN_GAP_COMPLETION')
            check_sources(manifest,engine_root);return done
        if load(output/'TESTS.json',identity) is None:
            state('RUNNING',phase='TESTS')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',STRATEGY_ENGINE_ROOT=str(engine_root))
            with (output/'TESTS.log').open('w') as log:
                result=subprocess.run([sys.executable,'-m','pytest','-q',*['tests/unit/'+t+'.py' for t in TESTS],
                    '--junitxml='+str(output/'TESTS.xml')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            suites=list(ET.parse(output/'TESTS.xml').iter('testsuite'))
            counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
            if result.returncode or any(counts[k] for k in ('errors','failures','skipped')):
                raise ValueError(('PLAN_GAP_TESTS',counts))
            save(output/'TESTS.json',identity,counts)
        jobs=[]
        for i in range(manifest['world_count']):
            filename=f'confirmation_ideal_{i:03d}.json'
            for key in selected:
                prefix=f'WORLD_ideal_{i:03d}_{key[:8]}'
                hashes={n:h for n,h in attributed['artifacts'].items() if n.startswith(prefix+'_BAR_')}
                jobs.append(dict(name=f'WORLD_{i:03d}_{key[:8]}',source=str(source),attribution=str(attribution),
                    source_file=filename,source_sha=original['artifacts'][filename],case_prefix=prefix,case_hashes=hashes,
                    output=str(output),engine_root=str(engine_root),manifest=manifest,
                    risk=json.loads(riskpath.read_text()),world=i,key=key))
        results=[]
        with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(unit,j):j for j in jobs}
            for n,f in enumerate(as_completed(futures),1):
                results.append(f.result());check_sources(manifest,engine_root)
                state('RUNNING',phase='PLAN_REPLAY',completed=n,total=len(jobs),closes=sum(len(r['rows']) for r in results))
                print('replays',n,'/',len(jobs),flush=True)
        results.sort(key=lambda r:(r['world'],r['key']))
        summaries=[]
        for key in selected:
            rows=[r for account in results if account['key']==key for r in account['rows']]
            d=[r['decomposition'] for r in rows]
            summaries.append(dict(key=key,closes=len(rows),ranking_reversals=sum(x['ranking_reversal'] for x in d),
                actual_positive_margin=sum(x['actual_margin']>1e-9 for x in d),
                actual_negative_margin=sum(x['actual_margin']< -1e-9 for x in d),
                max_valuation_error=max((r['valuation_error'] for r in rows),default=0),
                max_decomposition_error=max((x['identity_error'] for x in d),default=0),
                close_path_deviates=sum(r['close_first_deviation'] is not None for r in rows),
                hold_path_deviates=sum(r['hold_first_deviation'] is not None for r in rows)))
        report=dict(identity=identity,baseline_replays=len(jobs),committed_replays=2*sum(s['closes'] for s in summaries),
            summaries=summaries,units={j['name']+'.json':sha(output/(j['name']+'.json')) for j in jobs},
            scope='IDEAL_REPORT_CONTINUATION_DIAGNOSTIC_NOT_A_NEW_POLICY')
        write(output/'REPORT.json',report)
        lines=['# Ideal-report plan versus execution gap','',f'Identity: `{identity}`.',
            'Original policies frozen. Positive utility margin favors CLOSE. No new strategy selected.',
            '', '| Policy | Closes | Ranking reversals | Actual positive | Actual negative | Max valuation error | Close path differs | Hold path differs |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for s in summaries:
            lines.append(f"| {s['key'][:8]} | {s['closes']} | {s['ranking_reversals']} | {s['actual_positive_margin']} | "
                f"{s['actual_negative_margin']} | {s['max_valuation_error']:.3g} | {s['close_path_deviates']} | {s['hold_path_deviates']} |")
        lines+=['','Ranking reversals attribute a limited committed-plan valuation to a different replanned continuation. This is not proof that permanent commitment is preferable.']
        (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        check_sources(manifest,engine_root)
        done=dict(artifacts={p.name:sha(p) for p in output.iterdir() if p.is_file() and p.name not in ('LOCK','STATE.json','COMPLETE.json')},
            baseline_replays=len(jobs),committed_replays=report['committed_replays'],tests=load(output/'TESTS.json',identity))
        save(output/'COMPLETE.json',identity,done);state('COMPLETE',baseline_replays=len(jobs),committed_replays=report['committed_replays'])
        return done
    except BaseException as error:
        state('FAILED',error_type=type(error).__name__,detail=str(error));raise
    finally:
        lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','attribution','engine-root','output'): p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.source,a.attribution,a.engine_root,a.output,a.workers,a.smoke)
