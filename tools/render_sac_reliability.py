"""Plot saved357 performance and weight motion without claiming training epochs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from app.frozen_recipe_inputs import file_sha
from tools.run_discrete_curve_probe import read, write


ARMS=('control','actor_lr01','critic_lr01','alpha_lr01','all_lr01')
COLORS=('#333333','#167c80','#cb4335','#7b4b9a','#b18400')


def termination(arm):
    if arm['updates']!=512 or arm['history'][-1]['step']!=512:
        raise ValueError('INCOMPLETE_357_BUDGET')
    return dict(stop_reason='FIXED_UPDATE_BUDGET',optimizer_updates=512,
        online_training_epochs=0,early_stopping_enabled=False,patience_used=None,
        patience_limit=None,batch_size=arm['samples']//arm['updates'],
        replay_rows=arm['replay_rows'],sampled_rows=arm['distinct_rows_sampled'],
        account_evaluation_updates=[r['step'] for r in arm['history'] if 'metrics' in r])


def role_drift(row,role):
    values=[v for k,v in row['diagnostics']['modules'].items() if k.startswith(role+'.')]
    return (sum(v['delta2'] for v in values)/max(sum(v['initial2'] for v in values),1e-30))**.5


def render(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    records=[read(source/f'{seed}.json') for seed in (7,19,43)]
    if any(not r['verified'] or not r['qualification']['passed'] for r in records):
        raise ValueError('UNVERIFIED_357_INPUT')
    runs={name:[next(a for a in r['result']['arms'] if a['name']==name) for r in records] for name in ARMS}
    stops=[dict(seed=r['spec']['seed'],arm=a['name'],**termination(a)) for r in records for a in r['result']['arms']]
    write(output/'STOPPING.json',dict(runs=stops,source_hashes={p.name:file_sha(p) for p in
        [source/'7.json',source/'19.json',source/'43.json']},training_repaired=False))
    rows=[]
    for r in records:
        for a in r['result']['arms']:
            for h in a['history']:
                if 'metrics' not in h:continue
                rows.append(dict(seed=r['spec']['seed'],arm=a['name'],update=h['step'],
                    train_return=h['metrics']['TRAIN']['return_pct'],validation_return=h['metrics']['VALIDATION']['return_pct'],
                    actor_drift=role_drift(h,'actor'),critic_drift=role_drift(h,'critic'),
                    action_flips=h['diagnostics']['action_flips'],entropy=h['diagnostics']['entropy_mean']))
    with (output/'ALIGNED.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

    def curve(ax,name,color,field,account=False):
        histories=[[h for h in a['history'] if not account or 'metrics' in h] for a in runs[name]]
        x=[h['step'] for h in histories[0]]
        if any([h['step'] for h in hs]!=x for hs in histories):raise ValueError('UNPAIRED_PLOT_STEPS')
        y=np.array([[field(h) for h in hs] for hs in histories])
        ax.plot(x,y.mean(0),marker='o',markersize=3,color=color,label=name)
        ax.fill_between(x,y.min(0),y.max(0),color=color,alpha=.12)
        ax.set_xlabel('Optimizer updates (not epochs)');ax.grid(alpha=.2)

    fig=Figure(figsize=(12,7),layout='constrained');axes=fig.subplots(2,1)
    for ax,period in zip(axes,('TRAIN','VALIDATION')):
        for name,color in zip(ARMS,COLORS):curve(ax,name,color,lambda h,p=period:h['metrics'][p]['return_pct'],True)
        ax.set_ylabel('Net return (%)');ax.set_title(period+' full real account');ax.axhline(0,color='#aaaaaa',linewidth=.7)
    axes[0].legend(ncol=3,fontsize=9)
    fig.suptitle('357: fixed 512-update budget; no early stopping\nMean of 3 seeds; shaded min-max, not confidence intervals')
    FigureCanvasAgg(fig);fig.savefig(output/'TRAIN_VALIDATION_UPDATES.png',dpi=140)

    fig=Figure(figsize=(12,8),layout='constrained');axes=list(fig.subplots(2,2).flat)
    fields=[('Actor relative weight drift',lambda h:role_drift(h,'actor')),
            ('Critic relative weight drift',lambda h:role_drift(h,'critic')),
            ('Changed greedy actions / 128 TRAIN probes',lambda h:h['diagnostics']['action_flips']),
            ('Policy entropy on TRAIN probes',lambda h:h['diagnostics']['entropy_mean'])]
    for ax,(title,field) in zip(axes,fields):
        for name,color in zip(ARMS,COLORS):curve(ax,name,color,field)
        ax.set_title(title)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=5,fontsize=9)
    fig.suptitle('357: weight and function diagnostics aligned by update\nAssociations alone do not identify the cause; no late-convergence claim')
    FigureCanvasAgg(fig);fig.savefig(output/'WEIGHT_FUNCTION_UPDATES.png',dpi=140)
    return stops


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();print(json.dumps(dict(runs=len(render(a.source,a.output)),output=str(a.output))))
