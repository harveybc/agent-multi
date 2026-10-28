"""Authenticate closed exit results and apply the missing TRAIN-only clock control."""
import argparse
import copy
from pathlib import Path

import numpy as np

from app.exit_delay_control import fit_delay, evaluate_delay
from app.frozen_recipe_inputs import file_sha, require
from app.optimal_exit import outcome
from app.rl_subset import digest
from tools.exit_learning_repair import load_corpus, qualifies
from tools.optimal_exit_study import summarize
from tools.recipe_sac_queue import read, write, source_pins


def rescore(study, source):
    closed=read(study/'CLOSE.json');identity=read(study/'IDENTITY.json')
    require(digest(identity)==closed['identity'],'DELAY_RESCORE_STUDY_IDENTITY')
    for name,sha in closed['artifacts'].items():
        require(file_sha(study/name)==sha,'DELAY_RESCORE_ARTIFACT')
    require('CONFIRMATION.json' in closed['artifacts'],'DELAY_RESCORE_CONFIRMATION_RECEIPT')
    require(identity['source_close_sha']==file_sha(source/'CLOSE.json'),'DELAY_RESCORE_SOURCE')
    parent=read(source/'CLOSE.json')
    train,_,_=load_corpus(source,'train',parent)
    control=fit_delay(train)
    # Fit the control before reading any confirmation case.
    confirmation,_,_=load_corpus(source,'confirmation',parent)
    confirm_rows={r['unit']:r for r in read(study/'CONFIRMATION.json')}
    def recheck(report,cases):
        require([r['case_id'] for r in report['deterministic']['cases']]==[c['id'] for c in cases],
                'DELAY_RESCORE_SUPPORT')
        rows=[]
        for recorded,case in zip(report['deterministic']['cases'],cases):
            actions=np.zeros(len(case['close_net']))
            index=recorded['exit_index']
            require(index is None or type(index) is int and 0<=index<len(actions),'DELAY_RESCORE_EXIT_INDEX')
            if index is not None:actions[index]=1
            row=outcome(case,actions)
            require(all(recorded[k]==v for k,v in row.items()),'DELAY_RESCORE_OUTCOME')
            rows.append(row)
        rebuilt=copy.deepcopy(report)
        rebuilt['deterministic']['summary']=summarize(rows)
        rebuilt['fixed_delay_control']=control
        return dict(qualified=qualifies(rebuilt,cases),summary=rebuilt['deterministic']['summary'])
    rows=[]
    for name in sorted(n for n in closed['artifacts'] if n.startswith('fits/') and n.endswith('/RESULT.json')):
        fit=read(study/name);unit=fit['unit'];confirmed=confirm_rows[unit]
        tr=recheck(fit['train'],train);co=recheck(confirmed['report'],confirmation)
        rows.append(dict(unit=unit,kind=fit['definition']['kind'],old_qualified=confirmed['qualified'],
            train=tr,confirmation=co,qualified=tr['qualified'] and co['qualified']))
    return dict(schema='exit_fixed_delay_rescore.v1',study_close_sha=file_sha(study/'CLOSE.json'),
        source_close_sha=file_sha(source/'CLOSE.json'),artifacts_verified=len(closed['artifacts']),
        control=control,confirmation_control=evaluate_delay(confirmation,control),rows=rows,
        qualified_pure_rl=[r['unit'] for r in rows if r['qualified'] and r['kind']=='pure_rl'],
        new_fits=0,source_artifacts_modified=False,financial_admission=False,
        code=source_pins(Path(__file__).resolve().parents[1],('app','tools')))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args()
    require(not args.output.exists(),'DELAY_RESCORE_OUTPUT_EXISTS')
    require(not any(args.output.resolve().is_relative_to(root.resolve()) for root in (args.study,args.source)),
            'DELAY_RESCORE_OUTPUT_ISOLATION')
    report=rescore(args.study,args.source);write(args.output,report)
    print(dict(units=len(report['rows']),qualified=report['qualified_pure_rl'],delay=report['control']['delay'],new_fits=0))
