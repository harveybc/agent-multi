"""Native FIT row attribution and numerical gradient witnesses after331."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from app.fit_gradient_conflict import inspect_gradients
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from tools.probe_robust_adamw_direction import fit_quality
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import configure,read,write,now
from tools.run_training_forensics import template

ROOT=Path(__file__).resolve().parents[1]


def decompose(q,y):
    if q.ndim!=3 or q.shape[0]!=2 or q.shape[1:]!=y.shape or y.shape[-1]!=3 or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError('ATTRIBUTION_SHAPE_NONFINITE')
    err=q-y[None];level=err.mean(-1,keepdims=True)
    values=dict(total=float(np.mean(err**2)),level=float(np.mean(level**2)),gap=float(np.mean((err-level)**2)))
    if abs(values['total']-values['level']-values['gap'])>1e-12:raise ValueError('ATTRIBUTION_MSE_DECOMPOSITION')
    return values


def run(a):
    started=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('ATTRIBUTION_GPU')
    cfgs=[read(p/'CONFIG.json') for p in (a.chain,a.probe,a.coverage)]
    for p,c in zip((a.chain,a.probe,a.coverage),cfgs):verify_done(p,c['identity'])
    if cfgs[0]['probe']!=cfgs[1]['identity'] or cfgs[0]['coverage']!=cfgs[2]['identity']:raise ValueError('ATTRIBUTION_LINEAGE')
    cfg=dict(schema='robust_refinement_attribution.v1',parents=[c['identity'] for c in cfgs],
        source_sha256=cfgs[0]['source_sha256'],optimizer_steps=0,scope='expanded_FIT_only',
        code={k:file_sha(ROOT/k) for k in ('tools/analyze_robust_refinement_decisions.py','app/fit_gradient_conflict.py')})
    cfg['identity']=digest(cfg)
    if file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('ATTRIBUTION_SOURCE')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=cfg:raise ValueError('ATTRIBUTION_IDENTITY')
    if (a.output/'DONE.json').exists():verify_done(a.output,cfg['identity']);return
    write(a.output/'CONFIG.json',cfg)
    records=read(a.coverage/'PANEL.json')['records'];support=read(a.coverage/'prepared/SUPPORT.json')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')
    result=read(a.chain/'RESULT.json');outcomes=[];vectors={}
    for row in result['sequential']:
        if row['step']!=256:continue
        fold,seed,kind=row['fold'],row['seed'],row['kind'];name=f'{fold}_{seed}_{kind}'
        info=support['folds'][fold];fit=np.array(info['expanded']);oldset=set(info['masks']['fit'])
        if set(fit)&(set(info['masks']['stop'])|set(info['masks']['audit'])):raise ValueError('ATTRIBUTION_SCOPE')
        old=np.array([i for i,r in enumerate(fit) if r in oldset]);added=np.array([i for i,r in enumerate(fit) if r not in oldset])
        y=np.stack([validate_record(records[i]) for i in fit]);z=data['z'][:,fit]
        before=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')['initial']
        path=a.chain/'sequential'/name/'ENDPOINT_256.pt';sha=file_sha(path)
        after=torch.load(path,weights_only=False,map_location='cpu')['state']
        head=RoleHeads(template('cuda'),1,seed);predictions=[]
        for state in (before,after):
            head.load_state_dict(state)
            with torch.no_grad():predictions.append(head(z.to('cuda')).mean(2).cpu().double().numpy())
        start,end=predictions
        if fit_quality(end,y)!=row['assessment']['observation']['fit']['gpu']:raise ValueError('ATTRIBUTION_REPLAY')
        a0=start.min(0).argmax(1);a1=end.min(0).argmax(1);ix=np.arange(len(fit))
        regrets0=y.max(1)-y[ix,a0];regrets1=y.max(1)-y[ix,a1];changes=regrets1-regrets0
        if abs(float(changes.mean())-(fit_quality(end,y)['regret']-fit_quality(start,y)['regret']))>=1e-12:
            raise ValueError('ATTRIBUTION_REGRET_RECONCILIATION')
        startloss=decompose(start,y);endloss=decompose(end,y)
        byrole={}
        for role in sorted({records[i]['kind'] for i in fit}):
            mask=np.array([records[i]['kind']==role for i in fit]);v=changes[mask]
            byrole[role]=dict(rows=int(mask.sum()),helpful=int((v< -1e-8).sum()),harmful=int((v>1e-8).sum()),
                regret_change_sum=float(v.sum()),regret_change_mean=float(v.mean()))
        cpu=RoleHeads(template('cpu'),1,seed).double();cpu.load_state_dict(after)
        analysis,gradient=inspect_gradients(cpu,z.double(),torch.from_numpy(y),old,added)
        vectors.update({name+'__'+k:v for k,v in gradient.items()})
        changed=[dict(row=int(fit[i]),original_fit=int(fit[i]) in oldset,kind=records[fit[i]]['kind'],
            before_action=int(a0[i]),after_action=int(a1[i]),before_regret=float(regrets0[i]),
            after_regret=float(regrets1[i]),regret_change=float(changes[i]),
            before_q=start.min(0)[i].tolist(),after_q=end.min(0)[i].tolist(),target=y[i].tolist())
            for i in range(len(fit)) if a0[i]!=a1[i]]
        displacement={}
        for group,layer in (('hidden','0'),('output','2')):
            d=torch.cat([(after[k].double()-before[k].double()).flatten() for k in before if k.split('.')[3]==layer])
            displacement[group]=float(d.norm())
        outcomes.append(dict(fold=fold,seed=seed,kind=kind,before=fit_quality(start,y),after=fit_quality(end,y),
            mse_before=startloss,mse_after=endloss,roles=byrole,changed=changed,gradient=analysis,
            displacement_norms=displacement,checkpoint_sha256=sha,state_sha256=tree_digest(after)))
        if file_sha(path)!=sha:raise ValueError('ATTRIBUTION_PARENT_MUTATION')
        write(a.output/'PROGRESS.json',dict(stage='ANALYSIS',completed=len(outcomes),total=8,at=now()))
    if len(outcomes)!=8 or file_sha(a.bundle/'model.pt')!=cfg['source_sha256']:raise ValueError('ATTRIBUTION_COMPLETENESS')
    np.savez_compressed(a.output/'GRADIENTS.npz',**vectors)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],seconds=time.monotonic()-started,arms=outcomes,
        actor_updates=0,optimizer_steps=0,model_promoted=False))
    seal(a.output,cfg['identity']);write(a.output/'PROGRESS.json',dict(stage='COMPLETE',arms=8,at=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('chain','probe','coverage','bundle','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);run(a)
