"""Authenticate and export compact retrospective action attribution evidence."""
import argparse
import csv
from pathlib import Path

import numpy as np

from app.frozen_recipe_inputs import file_sha
from tools.audit_fixed_actor_critics import verify_collection
from tools.run_discrete_curve_probe import read, write


def event_row(cell,event):
    old,new=event['old_signal'],event['new_signal']
    samples=[x['soft_delta'] for x in event['isolated']['stochastic']]
    a,b=event['prefix_old']['first_effect'],event['prefix_new']['first_effect']
    return dict(period=cell['period'],arm=cell['arm'],updates=cell['updates'],utc=event['utc'],
        old_signal=old,new_signal=new,old_command=a['command'],new_command=b['command'],
        old_entries=a['entries'],new_entries=b['entries'],old_closes=a['closes'],new_closes=b['closes'],
        chronological_delta_usd=event['chronological_contribution_usd'],isolated_delta_usd=event['isolated']['delta_usd'],
        q_gap=event['q_gap'],soft_gap_mean=float(np.mean(samples)),
        soft_gap_min=min(samples),soft_gap_max=max(samples),
        reference_margin=event['reference_logits'][old]-event['reference_logits'][new],
        candidate_margin=event['candidate_logits'][new]-event['candidate_logits'][old],
        positive_q_negative_all_three=event['q_gap']>0 and max(samples)<0,
        three_draws_not_exact_expectation=True)


def summarize(source,output):
    receipt=verify_collection(source);result=read(source/'RESULT.json')
    rows=[];cells=[]
    for summary in result['accounts']:
        key=f"{summary['arm']}_{summary['updates']}"
        cell=read(source/summary['period']/(key+'.json'))
        events=[event_row(cell,e) for e in cell['events']];rows.extend(events)
        cells.append(dict(**summary,effective_changes=sum(abs(e['chronological_delta_usd'])>1e-6 for e in events)))
    output.mkdir(parents=True,exist_ok=True)
    with (output/'DECISIONS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    payload=dict(identity=result['identity'],finished_at=receipt['finished_at'],seconds=receipt['seconds'],
        verified_files=len(receipt['files']),source_done_sha256=file_sha(source/'DONE.json'),
        accounts=cells,event_occurrences=len(rows),
        unique_events=len({(r['period'],r['utc'],r['old_signal'],r['new_signal']) for r in rows}),
        no_training=True,no_test=True,model_promoted=False,
        chronological_order_dependent=True,isolated_effects_not_additive=True,
        stochastic_draws=3,three_draws_not_exact_expectation=True)
    write(output/'SUMMARY.json',payload)
    write(output/'EXPORT.json',dict(source_identity=result['identity'],source_done_sha256=file_sha(source/'DONE.json'),
        producer_sha256=file_sha(Path(__file__)),
        files={n:file_sha(output/n) for n in ('DECISIONS.csv','SUMMARY.json')}))
    return payload


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();summarize(a.source,a.output)


if __name__=='__main__':main()
