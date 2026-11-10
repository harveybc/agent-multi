"""Read sealed field contrasts and report paired endpoints and conflict strata."""
import argparse
import csv
from pathlib import Path
import time

import numpy as np

from app.frozen_recipe_inputs import file_sha
from app.position_value_surface import decision_metrics
from app.rl_subset import digest
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_action_ranking_contrast import ensure_config,seal
from tools.run_discrete_curve_probe import read,write


def conflict_metrics(prediction,records,banks,role,conflicting):
    selected=[]
    for b in banks:
        if not all(records[i]['role']==role for i in b):continue
        y=np.array([records[i]['gap'] for i in b]); conflict=bool((y>1e-8).any() and (y < -1e-8).any())
        if conflict==conflicting:selected.append(b)
    ids=[i for b in selected for i in b]
    if not ids:return None
    lookup={i:j for j,i in enumerate(ids)};remapped=[[lookup[i] for i in b] for b in selected]
    rows=[records[i] for i in ids]
    result=decision_metrics(np.asarray(prediction)[ids],np.array([r['gap'] for r in rows]),
        np.array([r['local_delta'] for r in rows]),np.array([r['effective_close'] for r in rows]),remapped)
    result.pop('gap_rmse');return dict(banks=len(selected),**result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('primary','constant','decision','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();start=time.monotonic()
    receipts={n:verify_collection(getattr(a,n)) for n in ('primary','constant','decision')}
    cfg=read(a.decision/'CONFIG.json')
    if (cfg['primary']!=receipts['primary']['identity'] or cfg['constant']!=receipts['constant']['identity']):
        raise ValueError('DECISION_REPORT_CHAIN')
    identity=dict(inputs={n:r['identity'] for n,r in receipts.items()},source_sha256=file_sha(Path(__file__)))
    identity['identity']=digest(identity);ensure_config(a.output,identity)
    if (a.output/'DONE.json').exists():verify_collection(a.output);return
    panel=read(a.primary/'panel/RESULT.json');records=panel['records'];banks=panel['banks'];endpoints=[];strata=[]
    for source,root in (('primary',a.primary),('constant',a.constant),('decision',a.decision)):
        for key,arm in read(root/'RESULT.json')['arms'].items():
            e=arm['effective'];mode=e['mode']
            if mode not in ('progress9','surface81'):continue
            context=e.get('embedding','constant' if source=='constant' else 'market')
            objective=e.get('objective','huber')
            base=dict(seed=e['seed'],interface=mode,embedding=context,objective=objective)
            path=root/f"r{e['seed']}"/mode
            if source=='decision':path=path/context/objective
            pred=np.load(path/'PREDICTIONS.npy').mean(0)
            for role in ('fit','monitor'):
                m=dict(arm['points'][-1]['metrics'][role]['all']);m.pop('gap_rmse',None)
                endpoints.append(dict(**base,role=role,**m))
                for conflict in (False,True):
                    metrics=conflict_metrics(pred,records,banks,role,conflict)
                    if metrics is not None:strata.append(dict(**base,role=role,conflicting=conflict,**metrics))
    summaries=[]
    for mode in ('progress9','surface81'):
        for context in ('market','constant'):
            for objective in ('huber','sign','sign_pair'):
                rows=[r for r in endpoints if (r['interface'],r['embedding'],r['objective'],r['role'])==(mode,context,objective,'monitor')]
                if len(rows)!=2:raise ValueError('DECISION_REPORT_ARMS')
                harmful=sum(r['harmful_effective_closes'] for r in rows);closes=sum(r['effective_closes'] for r in rows)
                summaries.append(dict(interface=mode,embedding=context,objective=objective,
                    monitor_regret_mean=float(np.mean([r['bank_regret'] for r in rows])),
                    accuracy_mean=float(np.mean([r['sign_accuracy'] for r in rows])),
                    harmful_total=harmful,closes_total=closes,harmful_fraction=harmful/closes if closes else None,
                    beneficial_total=sum(r['beneficial_effective_closes'] for r in rows)))
    result=dict(identity=identity['identity'],summary=summaries,endpoints=endpoints,strata=strata,
        seed_counts_not_independent_positions=True,post_endpoint_analysis=True,model_promoted=False)
    write(a.output/'RESULT.json',result)
    for name,rows in (('SUMMARY',summaries),('ENDPOINTS',endpoints),('STRATA',strata)):
        with (a.output/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    seal(a.output,identity['identity'],start);print(summaries)


if __name__=='__main__':main()
