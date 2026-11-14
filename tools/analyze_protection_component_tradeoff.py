"""Enumerate saved critic block exchanges after342; no optimizer or actor update."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.optimal_fit_protection import optimal_support
from app.rl_subset import digest
from app.weight_decision_forensics import BLOCKS,hybrid_state,exact_shapley,cost_parts
from tools.replay_first_harmful_crossings import require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write


def run(a):
    configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('TRADEOFF_DEVICE')
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('run','verified','control','probe','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if parents['run']!=parents['verified']:raise ValueError('TRADEOFF_VERIFICATION')
    for k in ('control','probe','coverage'):
        if parents['run']['parents'][k]!=parents[k]['identity']:raise ValueError('TRADEOFF_LINEAGE')
    cfg=dict(schema='protection_component_tradeoff.v1',parents={k:c['identity'] for k,c in parents.items()},
        masks=list(range(16)),blocks=BLOCKS,actor_updates=0,optimizer_steps=0,
        code_sha256=file_sha(Path(__file__)))
    cfg['identity']=digest(cfg)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);rows=[];cases=[]
    for u in read(a.run/'RESULT.json')['units']:
        fold,seed=u['fold'],u['seed'];name=f'{fold}_{seed}'
        protected_path=a.run/'units'/name/'ENDPOINT_64.pt';control_path=a.control/'units'/f'{name}_complete'/'ENDPOINT_64.pt'
        if file_sha(control_path)!=u['control_sha256']:raise ValueError('TRADEOFF_CONTROL')
        p=torch.load(protected_path,weights_only=False,map_location='cpu')['state'];c=torch.load(control_path,weights_only=False,map_location='cpu')['state']
        snap=torch.load(a.probe/'units'/f'{name}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);baseline=views(ctx,p,ctx.fit);target=y[ctx.fit]
        allowed,eligible=optimal_support(target,baseline);values=[];details=[]
        for mask in range(16):
            state=hybrid_state(p,c,mask)
            if mask in (0,15):require_equal(state,p if mask==0 else c,'TRADEOFF_ENDPOINT')
            original=ctx.observe(state,single=True)['passed'];qs=views(ctx,state,ctx.fit);metrics={}
            for k,q in qs.items():
                action=q.min(0).argmax(1);v=cost_parts(q,target,ctx.temperature);b=cost_parts(baseline[k],target,ctx.temperature)
                cost=float(v['row_cost'].mean());regret=float(v['row_regret'].mean());lost=int(np.sum(~allowed[eligible,action[eligible]]))
                metrics[k]=dict(cost=cost,regret=regret,lost_optimal=lost,
                    recovery=bool(original and lost==0 and cost<float(b['row_cost'].mean()) and regret<float(b['row_regret'].mean())))
            if mask in (0,15):
                expected=u['curve'][-1]['metrics']['fit']['numpy'] if mask==0 else u['control']['fit']['numpy']
                for k in ('cost','regret'):
                    if metrics['numpy'][k]!=expected[k]:raise ValueError('TRADEOFF_ENDPOINT_METRIC')
            good=all(v['recovery'] for v in metrics.values());details.append(dict(mask=mask,original_passed=original,metrics=metrics,recovery=good))
            values.append([metrics['numpy']['cost'],metrics['numpy']['regret']])
            rows.append(dict(case=name,mask=mask,original_passed=original,recovery=good,
                cost=metrics['numpy']['cost'],regret=metrics['numpy']['regret'],lost_optimal=max(v['lost_optimal'] for v in metrics.values())))
        phi=exact_shapley(values)
        cases.append(dict(case=name,protected_sha256=file_sha(protected_path),control_sha256=file_sha(control_path),
            interventions=details,shapley={b:dict(cost=float(phi[i,0]),regret=float(phi[i,1])) for i,b in enumerate(BLOCKS)}))
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'CONFIG.json',cfg)
    write(a.output/'RESULT.json',dict(identity=cfg['identity'],cases=cases,interventions=len(rows),recoveries=sum(r['recovery'] for r in rows),actor_updates=0,model_promoted=False))
    with (a.output/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','verified','control','probe','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);run(p.parse_args())
