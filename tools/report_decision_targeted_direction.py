"""Compact authenticated344 candidate ledger; full trial trees stay in sealed storage."""
import argparse
import csv
from pathlib import Path
from app.frozen_recipe_inputs import file_sha
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write


def run(a):
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity']);r=read(a.run/'RESULT.json');rows=[]
    for u in r['units']:
        for t in u['trials']:
            c=t['candidate'];last=t['record']['repairs'][-1];v=t['assessment']
            q=v['metrics']['numpy'] if v else None
            rows.append(dict(unit=u['unit'],row=c['row'],target_action=c['action'],lower_bound=c['lower_bound'],margin=c['margin'],
                accepted=t['record']['accepted'],reason=t['record']['reason'],final_solver_feasible=last['solver']['feasible'],
                final_projected_norm_ratio=last['norm_ratio'],derivative_error=t['derivative_max_error'],
                cost=q['cost'] if q else None,regret=q['regret'] if q else None,
                cold_exact=t['cold_exact'],repeated_exact=t['repeated_exact']))
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'CANDIDATES.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(a.output/'PRODUCER_RECEIPT.json',dict(identity=cfg['identity'],result_sha256=file_sha(a.run/'RESULT.json'),
        seconds=r['seconds'],witnesses=len(r['units']),candidates=len(rows),accepted=sum(t['accepted'] for t in rows),
        maximum_derivative_error=max(t['derivative_error'] for t in rows),
        eligible_wrong_occurrences=sum(u['counts']['eligible_wrong'] for u in r['units']),
        screened_too_far=sum(u['counts']['too_far'] for u in r['units']),actor_updates=0,model_promoted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
