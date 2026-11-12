"""Read-only first-proposal replay: record every original backtracking gate."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from app.bounded_functional_actor import BUDGET, admissible, budget_metrics
from app.fit_decision_preservation import margins, preservation, project_step
from app.selected_input_market import market_for
from app.selected_warm_start import reset_learning_rng
from app.structural_refit import load_bundle
from app.targeted_fit_errors import correction_loss
from tools.diagnose_warmstart_drift import set_arm
from tools.run_bayesian_head_contrast import verify_done
from tools.run_discrete_curve_probe import configure
from tools.run_fit_decision_preservation import flat

parser=argparse.ArgumentParser()
for name in ('run','source','bundle','parent','output'):parser.add_argument('--'+name,type=Path,required=True)
a=parser.parse_args();read=lambda p:json.loads(p.read_text())
cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity'])
source=read(a.source/'CONFIG.json');verify_done(a.source,source['identity'])
configure(292);torch.backends.cudnn.allow_tf32=False
model,_=load_bundle(a.bundle,'cuda');set_arm(model,'actor_only')
market=market_for(a.parent,read(a.parent/'SPEC.json'),model.recipe)
records=read(a.source/'panel/RECORDS.json');rows=[r['row'] for r in records]
fit=np.array([i for i,r in enumerate(records) if r['role']=='fit'])
x=torch.tensor(market.inputs.windows[rows],device='cuda')
with torch.no_grad():initial=model.surfaces(x).detach()
guard=cfg['guard'];ids=torch.tensor(guard['indices'],device='cuda');actions=torch.tensor(guard['actions'],device='cuda')
floor=torch.tensor(guard['source_margins'],device='cuda',dtype=torch.double)
labels=torch.tensor(cfg['error_targets'],device='cuda');weights=torch.tensor(cfg['error_weights'],device='cuda',dtype=torch.double)
all_results=[]
for seed in cfg['seeds']:
    model,_=load_bundle(a.bundle,'cuda');model.load(a.source/f'critic_{seed}/MODEL.pt');set_arm(model,'actor_only')
    model.optimizer.state.clear();reset_learning_rng(seed)
    params=[p for p in model.loss.parameters() if p.requires_grad]
    theta=flat(params);before=[p.detach().clone() for p in params]
    effective=read(a.source/f'calibrated_adamw_{seed}/EFFECTIVE.json')
    lookup={r:i for i,r in enumerate(rows)}
    ix=torch.tensor([lookup[r] for r in effective['sampled_rows'][0]],device='cuda')
    model.optimizer.zero_grad(set_to_none=True)
    loss=correction_loss(model.surfaces(x).double(),labels,weights,ix);loss.backward()
    gradient=flat([p.grad for p in params])
    norm=torch.nn.utils.clip_grad_norm_(params,10.,error_if_nonfinite=True)
    model.optimizer.step();direction=flat(params)-theta
    with torch.no_grad():
        for p,old in zip(params,before):p.copy_(old)
    z=model.surfaces(x);current=margins(z.double(),ids,actions)
    jac=torch.stack([flat(torch.autograd.grad(v,params,retain_graph=True)) for v in current.flatten()])
    matrix=torch.cat([jac,-gradient[None]],dim=0)
    bounds=torch.cat([(floor-current.detach()).flatten(),torch.zeros(1,device='cuda',dtype=torch.double)])
    projected,info=project_step(direction.cpu().numpy(),matrix.cpu().numpy(),bounds.cpu().numpy())
    delta=torch.tensor(projected,device='cuda',dtype=torch.double)
    saved=read(a.run/f'error_project_guard_{seed}/STEPS.json')[0]
    assert info==saved['projection'] and float(norm)==saved['gradient_norm']
    trials=[]
    for trial in range(BUDGET['backtracks']):
        fraction=2.**(-trial);offset=0
        with torch.no_grad():
            for p,old in zip(params,before):
                part=delta[offset:offset+p.numel()].reshape(p.shape)
                p.copy_((old.double()+fraction*part).to(p));offset+=p.numel()
            after=model.surfaces(x).detach()
        metrics=budget_metrics(initial[fit],initial[fit],after[fit],theta,flat(params))
        value=float(correction_loss(after.double(),labels,weights,ix))
        check=preservation(after,floor,ids,actions)
        moved=any(not torch.equal(p,old) for p,old in zip(params,before))
        failures=[]
        if not moved:failures.append('no_parameter_movement')
        if metrics['step_margin_rms']<cfg['minimum_step_margin_rms']:failures.append('below_movement_floor')
        if not check['passed']:failures.append('protected_margin')
        if value>float(loss)+1e-10:failures.append('corrective_loss')
        failures.extend(k for k in metrics if metrics[k]>BUDGET[k]+1e-12)
        accepted=moved and metrics['step_margin_rms']>=cfg['minimum_step_margin_rms'] and check['passed'] and admissible(metrics,float(loss),value)
        assert not accepted
        trials.append(dict(fraction=fraction,metrics=metrics,guard=check,loss_before=float(loss),
            loss_after=value,failures=failures,linear_min_guard_change=float((jac@delta*fraction).min()),
            linear_loss_change=float(gradient@delta*fraction)))
    with torch.no_grad():
        for p,old in zip(params,before):p.copy_(old)
    assert all(torch.equal(p,old) for p,old in zip(params,before))
    all_results.append(dict(seed=seed,projection_reproduced_exactly=True,projection=info,trials=trials))
verify_done(a.run,cfg['identity'])
output=dict(identity=cfg['identity'],scope='first proposal per schedule, no saved candidate or new account',
    proposals=all_results,new_training_candidate=False,test_read=False)
a.output.write_text(json.dumps(output,indent=2)+'\n')
for r in all_results:
    print(r['seed'],[(v['fraction'],v['failures'],v['guard']['minimum_margin_change']) for v in r['trials']],flush=True)
