"""Fit and freeze all close thresholds before evaluating archived monitor scores."""
import argparse
import csv
import fcntl
from pathlib import Path
import time

import numpy as np

from app.close_threshold_calibration import BASELINE,TIE_TOLERANCE,fit_thresholds,apply_thresholds
from app.frozen_recipe_inputs import file_sha
from app.position_value_surface import query_weights
from app.rl_subset import digest
from tools.audit_fixed_actor_critics import verify_collection
from tools.report_position_decision_loss import conflict_metrics
from tools.run_action_ranking_contrast import ensure_config,seal
from tools.run_discrete_curve_probe import read,write,now
from tools.run_position_decision_loss import decision_description

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    started=time.monotonic();receipts={n:verify_collection(getattr(a,n)) for n in ('primary','constant','decision')}
    dc=read(a.decision/'CONFIG.json');cc=read(a.constant/'CONFIG.json')
    if (dc['primary']!=receipts['primary']['identity'] or dc['constant']!=receipts['constant']['identity']
            or cc['primary']!=receipts['primary']['identity']):raise ValueError('THRESHOLD_CHAIN')
    config=dict(parents={n:r['identity'] for n,r in receipts.items()},fit_roles=['fit'],monitor_roles=['monitor'],
        modes=['baseline','common','directional'],baseline_threshold=BASELINE,tie_tolerance=TIE_TOLERANCE,
        primary_comparison='surface81:market:sign_pair:common_both_seeds',optimizer_steps=0,model_promoted=False,
        source_sha256={n:file_sha(ROOT/n) for n in ('tools/run_close_threshold_calibration.py','app/close_threshold_calibration.py')})
    config['identity']=digest(config);ensure_config(a.output,config)
    if (a.output/'DONE.json').exists():verify_collection(a.output);return
    def progress(stage,**kw):write(a.output/'PROGRESS.json',dict(at=now(),stage=stage,seconds=time.monotonic()-started,**kw))
    panel=read(a.primary/'panel/RESULT.json');records=panel['records'];banks=panel['banks']
    directions=np.array([r['direction'] for r in records]);monitor=np.array([r['role']=='monitor' for r in records])
    altered=[dict(r,gap=-123.*r['gap']+99.) if r['role']=='monitor' else dict(r) for r in records]
    cases={}
    for source,root in (('primary',a.primary),('constant',a.constant),('decision',a.decision)):
        for arm in read(root/'RESULT.json')['arms'].values():
            e=arm['effective'];mode=e['mode']
            if mode not in ('progress9','surface81'):continue
            context=e.get('embedding','constant' if source=='constant' else 'market');objective=e.get('objective','huber')
            key=f"{e['seed']}:{mode}:{context}:{objective}";path=root/f"r{e['seed']}"/mode
            if source=='decision':path=path/context/objective
            twins=np.load(path/'PREDICTIONS.npy')
            if twins.shape!=(2,len(records)) or not np.isfinite(twins).all():raise ValueError('THRESHOLD_SCORE_SHAPE')
            if key in cases:raise ValueError('THRESHOLD_DUPLICATE_ARM')
            cases[key]=dict(scores=twins.mean(0),expected=arm['points'][-1]['metrics'],
                seed=e['seed'],interface=mode,embedding=context,objective=objective,
                prediction_sha256=file_sha(path/'PREDICTIONS.npy'))
    if len(cases)!=24:raise ValueError('THRESHOLD_ARM_COUNT')
    calibrations={};index={};canaries=0
    for key,case in cases.items():
        calibrations[key]={};index[key]={}
        corrupted=case['scores'].copy();corrupted[monitor]=corrupted[monitor]*-99.+123.
        for mode in ('common','directional'):
            c=fit_thresholds(case['scores'],records,banks,mode)
            if fit_thresholds(corrupted,altered,banks,mode)!=c:raise ValueError('THRESHOLD_MONITOR_LEAK')
            canaries+=1;path=a.output/'thresholds'/key/(mode+'.json');write(path,c)
            calibrations[key][mode]=c
            index[key][mode]=dict(mode=c['mode'],prediction_sha256=case['prediction_sha256'],
                artifact_sha256=file_sha(path),thresholds={name:{k:v for k,v in values.items() if k!='candidates'}
                                                        for name,values in c['thresholds'].items()})
        progress('FIT_ONLY',completed=len(index),total=24)
    frozen=dict(identity=config['identity'],fit_completed_at=now(),monitor_evaluations_so_far=0,
        monitor_perturbation_canaries=canaries,calibrations=index)
    write(a.output/'CALIBRATION_INDEX.json',frozen);frozen_sha=file_sha(a.output/'CALIBRATION_INDEX.json')
    axes=read(a.primary/'AXES.json');coords=np.array([r['coords'] for r in records]);dirs=(directions=='short').astype(int)
    witness_ix=[i for i,r in enumerate(records) if (r['origin'],r['lag'],r['direction'])==('2023Q3:71',4,'short')]
    if len(witness_ix)!=3:raise ValueError('THRESHOLD_WITNESS')
    endpoints=[];strata=[];witnesses={};detailed={}
    for key,case in cases.items():
        _,outside=query_weights(coords,dirs,axes,case['interface'])
        variants={'baseline':dict(mode='common',thresholds={'all':dict(threshold=BASELINE)}),**calibrations[key]}
        for mode,c in variants.items():
            mask=apply_thresholds(case['scores'],directions,c);encoding=np.where(mask,1.,-1.)
            report=decision_description(encoding,records,banks,outside)
            if mode=='baseline':
                for role in ('fit','monitor'):
                    for group,metrics in report[role].items():
                        old=case['expected'][role][group]
                        if (metrics is None)!=(old is None):raise ValueError('THRESHOLD_BASELINE_GROUP')
                        if metrics is not None:
                            for name,value in metrics.items():
                                expected=old[name]
                                if (value is None)!=(expected is None) or (value is not None and abs(value-expected)>1e-12):
                                    raise ValueError('THRESHOLD_BASELINE_PARITY:'+name)
            base={n:case[n] for n in ('seed','interface','embedding','objective')};base['calibration']=mode
            for role in ('fit','monitor'):
                endpoints.append(dict(**base,role=role,**report[role]['all']))
                for conflict in (False,True):
                    m=conflict_metrics(encoding,records,banks,role,conflict)
                    if m is not None:strata.append(dict(**base,role=role,conflicting=conflict,**m))
            label=key+':'+mode;detailed[label]=report
            wanted=[records[i]['gap']>BASELINE for i in witness_ix]
            witnesses[label]=dict(scores=case['scores'][witness_ix].tolist(),close=mask[witness_ix].tolist(),
                                  expected=wanted,passed=mask[witness_ix].tolist()==wanted)
    summary=[]
    for interface in ('progress9','surface81'):
        for context in ('market','constant'):
            for objective in ('huber','sign','sign_pair'):
                for mode in ('baseline','common','directional'):
                    rows=[r for r in endpoints if (r['interface'],r['embedding'],r['objective'],r['calibration'],r['role'])
                          ==(interface,context,objective,mode,'monitor')]
                    if len(rows)!=2:raise ValueError('THRESHOLD_SUMMARY_COUNT')
                    closes=sum(r['effective_closes'] for r in rows);harm=sum(r['harmful_effective_closes'] for r in rows)
                    summary.append(dict(interface=interface,embedding=context,objective=objective,calibration=mode,
                        regret_mean=float(np.mean([r['bank_regret'] for r in rows])),
                        accuracy_mean=float(np.mean([r['sign_accuracy'] for r in rows])),effective_closes=closes,
                        harmful_closes=harm,beneficial_closes=sum(r['beneficial_effective_closes'] for r in rows),
                        harmful_fraction=harm/closes if closes else None))
    if file_sha(a.output/'CALIBRATION_INDEX.json')!=frozen_sha:raise ValueError('THRESHOLD_CHANGED_AFTER_MONITOR')
    for n,prior in receipts.items():
        if verify_collection(getattr(a,n))!=prior:raise ValueError('THRESHOLD_PARENT_CHANGED')
    result=dict(identity=config['identity'],calibration_index_sha256=frozen_sha,summary=summary,endpoints=endpoints,
        strata=strata,detailed=detailed,monitor_perturbation_canaries=canaries,baseline_parity=True,
        parents_unchanged=True,optimizer_steps=0,model_promoted=False,independent_new_positions=False)
    write(a.output/'RESULT.json',result);write(a.output/'WITNESS.json',witnesses)
    for name,rows in (('SUMMARY',summary),('ENDPOINTS',endpoints),('STRATA',strata)):
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    seal(a.output,config['identity'],started);progress('COMPLETE',evaluations=72)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('primary','constant','decision','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:write(a.output/'FAILED.json',dict(at=now(),error=repr(exc)));raise


if __name__=='__main__':main()
