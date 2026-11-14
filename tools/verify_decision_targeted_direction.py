"""Independent cold finite verification of344 decisions and matched controls."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from tools.probe_robust_adamw_direction import vector
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write
from tools.verify_optimal_protection_continuation import score


def run(a):
    configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('TARGET_VERIFY_DEVICE')
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity']);result=read(a.run/'RESULT.json')
    for k in ('crossings','control','probe','coverage'):
        c=read(getattr(a,k)/'CONFIG.json');verify_done(getattr(a,k),c['identity'])
        if c['identity']!=cfg['parents'][k]:raise ValueError('TARGET_VERIFY_LINEAGE')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);rows=[];accepted=0;controls=0;successes=0
    for u in result['units']:
        fold,seed,offset=u['fold'],u['seed'],u['offset'];unit=a.run/'units'/u['unit'];verify_done(unit,cfg['identity'])
        path=a.crossings/'cases'/f'{fold}_{seed}'/'witnesses'/f'{offset:03d}.pt'
        if file_sha(path)!=u['witness_sha256']:raise ValueError('TARGET_VERIFY_WITNESS')
        w=torch.load(path,weights_only=False,map_location='cpu');norm=float(np.linalg.norm(vector(w['proposed'])-vector(w['before'])))
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);before=views(ctx,w['before'],ctx.fit);target=y[ctx.fit]
        optimal=target==target.max(1,keepdims=True);eligible=optimal.sum(1)<3
        for q in before.values():eligible&=optimal[np.arange(len(target)),q.min(0).argmax(1)]
        baseline={k:score(q,target,ctx.temperature) for k,q in before.items()}
        control=torch.load(a.control/'units'/u['unit']/'ENDPOINT.pt',weights_only=False,map_location='cpu')
        if not ctx.observe(control['state'],single=True)['passed']:raise ValueError('TARGET_VERIFY_CONTROL_GUARD')
        control_views=views(ctx,control['state'],ctx.fit);controls+=1
        for k,q in control_views.items():
            actual=score(q,target,ctx.temperature)
            for metric in ('cost','regret'):
                if abs(actual[metric]-u['control']['metrics'][k][metric])>1e-12:raise ValueError('TARGET_VERIFY_CONTROL_METRIC')
                if abs(baseline[k][metric]-u['baseline'][k][metric])>1e-12:raise ValueError('TARGET_VERIFY_BASELINE')
        checked=[]
        for trial in u['trials']:
            if not trial['record']['accepted']:continue
            c=trial['candidate'];row=c['row'];pos=c['position']
            if int(ctx.fit[pos])!=row or row in set(ctx.old.tolist()):raise ValueError('TARGET_VERIFY_ROW')
            initial=[int(q.min(0).argmax(1)[pos]) for q in before.values()]
            if len(set(initial))!=1 or optimal[pos,initial[0]]:raise ValueError('TARGET_VERIFY_WRONG')
            cp=torch.load(unit/f'TARGET_{row}.pt',weights_only=False,map_location='cpu')
            if cp['identity']!=cfg['identity'] or cp['witness_sha256']!=file_sha(path) or cp['continuation_allowed'] or 'optimizer' in cp:raise ValueError('TARGET_VERIFY_CHECKPOINT')
            motion=float(np.linalg.norm(vector(cp['state'])-vector(w['before'])))
            if not 0<motion<=4*norm or not ctx.observe(cp['state'],single=True)['passed']:raise ValueError('TARGET_VERIFY_GUARD')
            qs=views(ctx,cp['state'],ctx.fit)
            for k,q in qs.items():
                actions=q.min(0).argmax(1);actual=score(q,target,ctx.temperature)
                if not optimal[pos,actions[pos]] or not optimal[np.flatnonzero(eligible),actions[eligible]].all():raise ValueError('TARGET_VERIFY_OPTIMAL')
                if not actual['cost']<baseline[k]['cost'] or not actual['regret']<baseline[k]['regret']:raise ValueError('TARGET_VERIFY_QUALITY')
                for metric in ('cost','regret'):
                    if abs(actual[metric]-trial['assessment']['metrics'][k][metric])>1e-12:raise ValueError('TARGET_VERIFY_METRIC')
            actions=qs['numpy'].min(0).argmax(1);initial=before['numpy'].min(0).argmax(1)
            delta=target[np.arange(len(target)),actions]-target[np.arange(len(target)),initial]
            checked.append(dict(unit=u['unit'],target=row,cost=score(qs['numpy'],target,ctx.temperature)['cost'],
                regret=score(qs['numpy'],target,ctx.temperature)['regret'],improved=int(np.sum(delta>0)),worsened=int(np.sum(delta<0)),
                changes=int(np.sum(actions!=initial)),motion_ratio=motion/norm));accepted+=1
        best=min(checked,key=lambda r:(r['regret'],r['cost'],r['target'])) if checked else None
        if (best['target'] if best else None)!=u['best_row']:raise ValueError('TARGET_VERIFY_BEST')
        successes+=int(best is not None)
        rows.append(dict(unit=u['unit'],examined=u['counts']['examined'],accepted=len(checked),
            source_regret=baseline['numpy']['regret'],control_regret=u['control']['metrics']['numpy']['regret'],
            best_target=best['target'] if best else None,best_regret=best['regret'] if best else None,
            best_cost=best['cost'] if best else None,improved=best['improved'] if best else None,
            worsened=best['worsened'] if best else None,motion_ratio=best['motion_ratio'] if best else None))
    if len(rows)!=10:raise ValueError('TARGET_VERIFY_COUNT')
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'CONFIG.json',cfg)
    report=dict(identity=cfg['identity'],units=10,controls=controls,accepted_candidates=accepted,successful_units=successes,rows=rows,
        actor_updates=0,optimizer_steps=0,model_promoted=False)
    write(a.output/'VERIFICATION.json',report)
    with (a.output/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(rows))
    ax.bar(x-.18,[r['control_regret'] for r in rows],width=.36,label='Protected AdamW control')
    ax.bar(x+.18,[r['best_regret'] if r['best_regret'] is not None else np.nan for r in rows],width=.36,label='Feasible targeted correction')
    ax.set_xticks(x,[r['unit'] for r in rows],rotation=35,ha='right');ax.set_ylabel('FIT decision regret');ax.legend()
    fig.tight_layout();fig.savefig(a.output/'PAIRED_REGRET.png',dpi=140);plt.close(fig)
    verify_done(a.run,cfg['identity']);seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','crossings','control','probe','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);run(p.parse_args())
