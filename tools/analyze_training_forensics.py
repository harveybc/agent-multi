"""Read sealed322 outputs and independently check stopping/prefix comparisons."""
import argparse
from pathlib import Path

import torch

from app.training_forensics import tree_digest
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import read,write


def frozen_selection(curve):
    best=float('inf');bad=0;selected=None;stopped=None
    for row in curve:
        if not row['step'] or row['step']%64 or stopped is not None:continue
        score=row['metrics']['stop']['mean_regret']
        if best-score>1e-8:best=score;bad=0;selected=row['step']
        else:bad+=1
        if bad>=20:stopped=row['step']
    return selected,stopped


def analyze(root,previous):
    result=read(root/'RESULT.json');verify_done(root,result['identity']);units={};prefixes=[]
    for r in result['units']:
        task=r['task'];name=task['name'];d=root/'units'/name;curve=read(d/'CURVE.json')
        if frozen_selection(curve)!=(r['selected_step'],r['stopped_at']):raise ValueError('FORENSIC_SELECTION_REPLAY')
        chosen=next(v for v in curve if v['step']==r['selected_step'])
        later=[v for v in curve if r['stopped_at'] is not None and v['step']>r['stopped_at']]
        later_best=min(later,key=lambda v:v['metrics']['stop']['mean_regret']) if later else None
        best_all=min(curve,key=lambda v:v['metrics']['stop']['mean_regret'])
        initial=curve[0]['geometry'];geometries=[v['geometry'] for v in curve]
        units[name]=dict(task=task,selected_step=r['selected_step'],stopped_at=r['stopped_at'],
            initial=r['initial'],selected=chosen['metrics'],final=r['final'],
            initial_beats_selected=curve[0]['metrics']['stop']['mean_regret']<chosen['metrics']['stop']['mean_regret'],
            best_observed_step=best_all['step'],best_observed_stop_regret=best_all['metrics']['stop']['mean_regret'],
            post_stop_best_step=None if later_best is None else later_best['step'],
            post_stop_beats_selected=later_best is not None and later_best['metrics']['stop']['mean_regret']<chosen['metrics']['stop']['mean_regret']-1e-8,
            initial_level_to_gap_gradient_ratio=initial['level_gradient_norm']/max(initial['gap_gradient_norm'],1e-30),
            negative_gradient_cosine_checks=sum(v['gradient_cosine']<0 for v in geometries),
            geometry_checks=len(geometries),clipped_updates=r['clipped_updates'],
            max_measured_update_weight_ratio=max((max(v['update_weight_ratio'].values(),default=0) for v in curve)),
            teacher_residual=r['teacher_residual'],audit_constant_rmse=r['audit_constant_rmse'])
        if task['target']=='real' and not task.get('trace'):
            units[name]['layer_interventions']=read(d/'LAYER_INTERVENTIONS.json')
            if name.endswith('base'):
                original=torch.load(previous/f"{task['fold']}_{task['seed']}_0.01/FIXED.pt",map_location='cpu',weights_only=True)['state']
                newer=torch.load(d/'checkpoints/004096.pt',map_location='cpu',weights_only=False)['model']
                exact=tree_digest(original)==tree_digest(newer)
                if not exact:raise ValueError('FORENSIC_EXTENDED_PREFIX')
                prefixes.append(dict(unit=name,matches320_at4096=True))
    paired=[]
    for fold in ('A','B'):
        for seed in (19,43):
            b=root/f'units/{fold}_{seed}_base';l=root/f'units/{fold}_{seed}_low'
            bc=read(b/'CURVE.json');lc=read(l/'CURVE.json')
            if bc[0]['model_sha256']!=lc[0]['model_sha256']:raise ValueError('FORENSIC_LR_INITIALIZATION')
            low_equal=next(v for v in lc if v['step']==8192)
            paired.append(dict(fold=fold,seed=seed,initial_exact=True,
                base8192=bc[-1]['metrics'],low8192=low_equal['metrics'],low81920=lc[-1]['metrics']))
    return dict(identity=result['identity'],reproducibility=result['reproducibility'],
        original_prefix_checks=prefixes,stopping_decisions_verified=len(units),units=units,paired=paired,
        limits='Frozen-head TRAIN-block diagnostic; neither full SAC reproducibility nor trading-profit evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.resolve().is_relative_to(a.root.resolve()):raise ValueError('DO_NOT_MUTATE_SEALED_RUN')
    write(a.output,analyze(a.root,a.previous))
