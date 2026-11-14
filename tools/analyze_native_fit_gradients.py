"""FIT-only frozen-checkpoint gradient conflicts and numerical witnesses."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.fit_gradient_conflict import inspect_gradients,partition
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import template

ROOT=Path(__file__).resolve().parents[1]


def run(a):
    started=time.monotonic();configure(318)
    pc,cc,wc=[read(p/'CONFIG.json') for p in (a.previous,a.coverage,a.warm)]
    for p,c in ((a.previous,pc),(a.coverage,cc),(a.warm,wc)):verify_done(p,c['identity'])
    if cc['previous']!=pc['identity'] or wc['previous']!=pc['identity'] or wc['coverage']!=cc['identity']:
        raise ValueError('FIT_GRADIENT_LINEAGE')
    cfg=dict(schema='native_fit_gradient_conflict.v1',previous=pc['identity'],coverage=cc['identity'],
        warm=wc['identity'],source_sha256=wc['source_sha256'],endpoints=[0,8192,32768],
        precision='cpu_float64_copy_of_saved_float32',finite_difference_eps=[1e-4,1e-5],
        optimizer_steps=0,held_rows_evaluated=0,
        code={k:file_sha(ROOT/k) for k in ('app/fit_gradient_conflict.py','tools/analyze_native_fit_gradients.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('FIT_GRADIENT_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('FIT_GRADIENT_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    support=read(a.coverage/'prepared/SUPPORT.json');records=read(a.coverage/'PANEL.json')['records']
    data=torch.load(a.coverage/'prepared/DATA.pt',map_location='cpu',weights_only=True)
    if support['rows']!=[r['row'] for r in records] or support['old_count']!=239:raise ValueError('FIT_GRADIENT_DATA')
    results=[];vectors={}
    for arm in read(a.warm/'RESULT.json')['arms']:
        fold,seed,scope=arm['fold'],arm['seed'],arm['scope'];info=support['folds'][fold]
        old,added,expanded=partition(info['masks']['fit'],info['expanded'],len(records))
        if set(expanded)&(set(info['masks']['stop'])|set(info['masks']['audit'])):raise ValueError('FIT_GRADIENT_HELD_ROWS')
        if len(expanded)!=4*len(old) or any(records[i]['year'] not in (2020,2021) for i in expanded):
            raise ValueError('FIT_GRADIENT_PERIOD')
        # Restrict both tensors and targets before any forward or differentiation.
        z=data['z'][:,expanded].double()
        y=torch.from_numpy(np.stack([validate_record(records[i]) for i in expanded]))
        oldlocal=np.array([i for i,r in enumerate(expanded) if r in set(old)])
        newlocal=np.array([i for i,r in enumerate(expanded) if r in set(added)])
        for step in cfg['endpoints']:
            path=(a.previous/f'units/{fold}_{seed}_base/LATEST.pt' if step==0 else
                  a.warm/f'units/{fold}_{seed}_{scope}/ENDPOINT_{step}.pt')
            sha=file_sha(path);saved=torch.load(path,map_location='cpu',weights_only=False)
            state=saved['model'] if step==0 else saved['state']
            expected=arm['initialization_sha256'] if step==0 else arm['endpoints'][str(step)]['state_sha256']
            if tree_digest(state)!=expected:raise ValueError('FIT_GRADIENT_STATE')
            head=RoleHeads(template('cpu'),1,seed);head.load_state_dict(state)
            names=[]
            for name,p in head.named_parameters():
                active=scope=='full_head' or name.split('.')[3]=='2';p.requires_grad_(active)
                if active:names.append(name)
            if names!=arm['trainable_names'] or sum(p.numel() for p in head.parameters() if p.requires_grad)!=arm['trainable_parameters']:
                raise ValueError('FIT_GRADIENT_SCOPE')
            head.double().eval();before=tree_digest(head.state_dict())
            analysis,gradient=inspect_gradients(head,z,y,oldlocal,newlocal)
            if tree_digest(head.state_dict())!=before or file_sha(path)!=sha:raise ValueError('FIT_GRADIENT_MUTATION')
            key=f'{fold}_{seed}_{scope}_{step}'
            vectors.update({key+'__'+k:v for k,v in gradient.items()})
            results.append(dict(fold=fold,seed=seed,scope=scope,step=step,checkpoint_sha256=sha,
                state_sha256=expected,original_rows=len(old),added_rows=len(added),expanded_rows=len(expanded),
                fit_rows_sha256=tree_digest(expanded),original_rows_sha256=tree_digest(old),
                added_rows_sha256=tree_digest(added),analysis=analysis))
            write(a.output/'PROGRESS.json',dict(stage='ANALYZE',completed=len(results),total=24,at=now()))
    if len(results)!=24 or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('FIT_GRADIENT_COMPLETENESS')
    summary=[]
    for scope in wc['scopes']:
        for step in cfg['endpoints']:
            rows=[r for r in results if r['scope']==scope and r['step']==step]
            summary.append(dict(scope=scope,step=step,observations=len(rows),
                original_added_total_conflicts=sum(r['analysis']['groups']['all']['original_total:added_total']['conflict'] for r in rows),
                original_added_gap_conflicts=sum(r['analysis']['groups']['all']['original_gap:added_gap']['conflict'] for r in rows),
                expanded_descent_harms_original_mse=sum(r['analysis']['groups']['all']['original_total:expanded_total']['conflict'] for r in rows),
                expanded_descent_harms_original_gap=sum(r['analysis']['groups']['all']['original_gap:expanded_total']['conflict'] for r in rows),
                level_descent_harms_original_gap=sum(r['analysis']['groups']['all']['original_gap:expanded_level']['conflict'] for r in rows)))
    np.savez_compressed(a.output/'GRADIENTS.npz',**vectors)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],observations=results,summary=summary,
        optimizer_steps=0,held_rows_evaluated=0,source_preserved=True,model_promoted=False,
        max_finite_difference_error=max(r['analysis']['finite_difference_max_error'] for r in results),
        max_gradient_union_error=max(r['analysis']['gradient_union_error'] for r in results)))
    lines=['# Native FIT Gradient Conflict','','Counts over four fold/seed pairs, not independent market samples.',
           'Raw full-batch gradient geometry, not actual AdamW trajectory attribution.','',
           '| Scope | Updates | Old/new MSE conflict | Old/new gap conflict | Expanded descent harms old MSE | Expanded descent harms old gap | Level descent harms old gap |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in summary:
        lines.append(f"| {r['scope']} | {r['step']} | {r['original_added_total_conflicts']}/4 | {r['original_added_gap_conflicts']}/4 | {r['expanded_descent_harms_original_mse']}/4 | {r['expanded_descent_harms_original_gap']}/4 | {r['level_descent_harms_original_gap']}/4 |")
    (a.output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    seal(a.output,cfg['identity'])
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',observations=24,at=now(),seconds=time.monotonic()-started))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','coverage','warm','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a)
