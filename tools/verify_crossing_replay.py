"""Cold-replay every crossing witness and independently check derivatives with Torch64."""
import argparse
from copy import deepcopy
import csv
from pathlib import Path
import time
import numpy as np
import torch
from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.interior_adamw_repair import transactional_step
from app.native_fit_coverage import validate_record
from app.native_head_warm_start import configure_scope
from app.primal_direction_projection import project_primal
from app.training_forensics import restore_rng,rng_state
from tools.replay_first_harmful_crossings import cpu_state,require_equal
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import read,write,configure


def vector_grad(g):return torch.cat([x.flatten() for x in g]).detach().cpu().double().numpy()


def verify(a):
    begin=time.monotonic();configure(318);torch.backends.cudnn.allow_tf32=False
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('CROSS_VERIFY_DEVICE')
    while not (a.run/'DONE.json').exists():
        progress=read(a.run/'PROGRESS.json') if (a.run/'PROGRESS.json').exists() else {}
        if progress.get('stage')=='FAILED' or time.monotonic()-begin>1200:raise ValueError('CROSS_VERIFY_PRODUCER_NOT_COMPLETE')
        time.sleep(5)
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity']);result=read(a.run/'RESULT.json')
    pc=read(a.probe/'CONFIG.json');cc=read(a.coverage/'CONFIG.json')
    for path,c,k in ((a.probe,pc,'probe'),(a.coverage,cc,'coverage')):
        verify_done(path,c['identity'])
        if cfg['parents'][k]!=c['identity']:raise ValueError('CROSS_VERIFY_LINEAGE')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    records=read(a.coverage/'PANEL.json')['records'];y=np.stack([validate_record(r) for r in records])
    rows=[];checked=0;errors=[]
    for case in result['cases']:
        fold,seed=case['fold'],case['seed'];out=a.run/'cases'/f'{fold}_{seed}';verify_done(out,cfg['identity'])
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit)
        for summary in case['witnesses']:
            path=out/'witnesses'/f"{summary['offset']:03d}.pt"
            if file_sha(path)!=summary['file_sha256']:raise ValueError('CROSS_VERIFY_FILE')
            w=torch.load(path,weights_only=False,map_location='cpu');ix=w['batch']
            if w['identity']!=cfg['identity'] or not set(ix)<=set(ctx.fit):raise ValueError('CROSS_VERIFY_WITNESS')
            ctx.head.load_state_dict(w['before']);opt,_=configure_scope(ctx.head,'full_head',pc['lr'],pc['weight_decay'])
            opt.load_state_dict(deepcopy(w['optimizer_before']));restore_rng(w['rng_before']);captured={}
            def repair(before,proposed):
                captured['proposed']=cpu_state(proposed);captured['optimizer']=deepcopy(opt.state_dict())
                captured['gradient']=vector_grad([p.grad for p in ctx.head.parameters()])
                return ctx.repair(before,proposed,'interior',projector=project_primal,twin_complete=True,
                    finite_evaluate=lambda state:ctx.observe(state,single=True))
            record=transactional_step(ctx.head,opt,ctx.z[:,ix],ctx.y[ix],repair,
                loss_fn=lambda q,t:cost_objective(q,t,'cost',ctx.temperature))
            for x,expected,name in [(captured['proposed'],w['proposed'],'PROPOSED'),(captured['optimizer'],w['optimizer_proposed'],'PROPOSED_OPT'),
                    (captured['gradient'],w['actual_gradient'],'GRAD'),(cpu_state(ctx.head.state_dict()),w['accepted'],'ACCEPTED'),
                    (opt.state_dict(),w['optimizer_after'],'OPT'),(rng_state(),w['rng_after'],'RNG'),
                    (dict(offset=w['offset'],**record),w['receipt'],'RECEIPT')]:require_equal(x,expected,name)
            head=ctx.double;head.load_state_dict(w['before']);pars=list(head.parameters())
            q=head(ctx.zcpu.double()).mean(2);minimum=q.min(0).values;targets=torch.tensor(y)
            full=vector_grad(torch.autograd.grad(cost_objective(q[:,ctx.fit],targets[ctx.fit],'cost',ctx.temperature),pars,retain_graph=True))
            batch=vector_grad(torch.autograd.grad(cost_objective(q[:,ix],targets[ix],'cost',ctx.temperature),pars,retain_graph=True))
            for event in summary['events']:
                i=event['row'];b=event['before_action'];c=event['after_action']
                j=vector_grad(torch.autograd.grad(minimum[i,b]-minimum[i,c],pars,retain_graph=True))
                own=vector_grad(torch.autograd.grad(cost_objective(q[:,i:i+1],targets[i:i+1],'cost',ctx.temperature),pars,retain_graph=True))
                expected=[event['full_fit_descent'],event['analytic_batch_descent'],event['own_batch_descent']]
                actual=[-j@full,-j@batch,-j@own*np.sum(ix==i)/len(ix)]
                err=float(np.max(abs(np.array(actual)-expected)));errors.append(err)
                if err>1e-9:raise ValueError('CROSS_VERIFY_DERIVATIVE')
                margins=[];actions=[];views32=[]
                for state in (w['before'],w['proposed'],w['accepted']):
                    head.load_state_dict(state)
                    with torch.no_grad():m=head(ctx.zcpu[:,i:i+1].double()).mean(2).min(0).values[0]
                    margins.append(float(m[b]-m[c]));actions.append(int(m.argmax()))
                    choices=[];position=int(np.flatnonzero(ctx.fit==i)[0])
                    with torch.no_grad():
                        for model,inputs in ((ctx.head,ctx.z),(ctx.cpu,ctx.zcpu)):
                            model.load_state_dict(state)
                            choices.append(int(model(inputs[:,ctx.fit]).mean(2).min(0).values[position].argmax()))
                            choices.append(int(model(inputs[:,i:i+1]).mean(2).min(0).values[0].argmax()))
                    views32.append(choices)
                if actions!=[event['before_action'],event['proposed_action'],event['after_action']]:raise ValueError('CROSS_VERIFY_ACTION')
                if not np.allclose(margins,[event[k] for k in ('margin_before','margin_proposed','margin_after')],atol=1e-11,rtol=1e-11):raise ValueError('CROSS_VERIFY_MARGIN')
                # Restore before weights before using the retained graph for another event.
                # Rebuild to avoid autograd version-counter invalidation after load_state_dict.
                head.load_state_dict(w['before']);q=head(ctx.zcpu.double()).mean(2);minimum=q.min(0).values
                rows.append(dict(case=f'{fold}_{seed}',offset=w['offset'],row=i,kind=event['kind'],
                    before=b,proposed=event['proposed_action'],after=c,regret_delta=event['regret_delta'],
                    batch_multiplicity=event['batch_multiplicity'],own=event['own_batch_descent'],others=event['others_batch_descent'],
                    actual_batch_descent=event['actual_cuda_batch_descent'],full_fit_descent=event['full_fit_descent'],
                    margin_before=margins[0],margin_proposed=margins[1],margin_after=margins[2],
                    finite_adam=event['finite_adam'],finite_repair=event['finite_repair'],
                    first_order_adam=event['first_order_adam'],first_order_repair=event['first_order_repair'],
                    history=event['adam_contributions']['history'],current=event['adam_contributions']['current'],
                    decay=event['adam_contributions']['decay'],rounding=event['adam_contributions']['rounding'],
                    cpu_gpu32_single_batch_actions_agree=all(all(v==want for v in view) for want,view in zip(actions,views32))))
            checked+=1
    report=dict(identity=cfg['identity'],cold_witnesses_exact=checked,events=len(rows),derivative_max_error=max(errors,default=0.),
        replayed_transactions=result['replayed_transactions'],original_endpoints_exact=sum(c['endpoints_exact'] for c in result['cases']),
        raw_proposal_already_crosses=sum(r['margin_proposed']<0 for r in rows),
        projection_introduces_crossing=sum(r['margin_proposed']>=0 for r in rows),
        batch_gradient_opposes=sum(r['actual_batch_descent']<0 for r in rows),full_gradient_opposes=sum(r['full_fit_descent']<0 for r in rows),
        own_row_absent_from_batch=sum(r['batch_multiplicity']==0 for r in rows),
        events_with_all_finite_geometry_actions_matching=sum(r['cpu_gpu32_single_batch_actions_agree'] for r in rows),
        seconds=time.monotonic()-begin,actor_updates=0,model_promoted=False)
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'VERIFICATION.json',report)
    write(a.output/'RESULT.json',result);write(a.output/'CONFIG.json',cfg)
    with (a.output/'EVENTS.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for path,c in ((a.run,cfg),(a.probe,pc),(a.coverage,cc)):verify_done(path,c['identity'])
    seal(a.output,cfg['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','probe','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);verify(p.parse_args())
