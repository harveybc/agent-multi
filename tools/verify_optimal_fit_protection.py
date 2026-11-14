"""Independent finite optimal-set and actual-data Jacobian checks for341."""
import argparse
import csv
from pathlib import Path
import time
import numpy as np
import torch
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.optimal_fit_protection import complete_pair_jacobian
from tools.replay_first_harmful_crossings import require_equal
from tools.run_affine_critic_calibration import views
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_decision_cost_contrast import CostContext
from tools.run_discrete_curve_probe import configure,read,write


def run(a):
    configure(318);torch.backends.cudnn.allow_tf32=False;start=time.monotonic()
    if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('OPTIMAL_VERIFY_DEVICE')
    config=read(a.run/'CONFIG.json');verify_done(a.run,config['identity']);result=read(a.run/'RESULT.json')
    for k in ('coverage','probe','crossings'):
        c=read(getattr(a,k)/'CONFIG.json');verify_done(getattr(a,k),c['identity'])
        if c['identity']!=config['parents'][k]:raise ValueError('OPTIMAL_VERIFY_LINEAGE')
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');support=read(a.coverage/'prepared/SUPPORT.json')
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']]);rows=[];maximum=0.;checked=0
    for entry in result['units']:
        fold,seed,offset=entry['fold'],entry['seed'],entry['offset'];name=f'{fold}_{seed}_{offset:03d}'
        unit=a.run/'units'/name;verify_done(unit,config['identity'])
        path=a.crossings/'cases'/f'{fold}_{seed}'/'witnesses'/f'{offset:03d}.pt'
        if file_sha(path)!=entry['witness_sha256']:raise ValueError('OPTIMAL_VERIFY_WITNESS')
        w=torch.load(path,weights_only=False,map_location='cpu')
        snap=torch.load(a.probe/'units'/f'{fold}_{seed}_8192_0'/'PROPOSAL.pt',weights_only=False,map_location='cpu')
        old=np.array(support['folds'][fold]['masks']['fit']);limit=float(np.mean(y[old].max(1)-y[old][np.arange(len(old)),snap['actions']]))
        ctx=CostContext(data,y,support,snap,fold,seed,limit);baseline=views(ctx,w['before'],ctx.fit)
        target=y[ctx.fit];optimal=target==target.max(1,keepdims=True);eligible=optimal.sum(1)<3
        for q in baseline.values():eligible&=optimal[np.arange(len(target)),q.min(0).argmax(1)]
        if ctx.fit[eligible].tolist()!=entry['protected_indices']:raise ValueError('OPTIMAL_VERIFY_ELIGIBILITY')
        # Torch autograd constructs the new pairs independently; ties create no mutual ordering.
        ctx.double.load_state_dict(w['before']);q=ctx.double(ctx.zcpu.double()).mean(2);parameters=list(ctx.double.parameters())
        pairs=[];gaps=[];gradients=[]
        for pos in np.flatnonzero(eligible):
            row=int(ctx.fit[pos]);allowed=np.flatnonzero(optimal[pos]);chosen=int(allowed[q[:,row].min(0).values[allowed].argmax()])
            for other in np.flatnonzero(~optimal[pos]):
                pairs.append((row,chosen,int(other)))
                for twin in range(2):
                    gap=q[twin,row,chosen]-q[:,row,int(other)].min();gaps.append(gap.item())
                    gradients.append(torch.cat([g.flatten() for g in torch.autograd.grad(gap,parameters,retain_graph=True)]).detach().numpy())
        if pairs:
            actual,jac=complete_pair_jacobian(w['before'],ctx.zn,pairs)
            error=max(float(np.max(abs(actual-gaps))),float(np.max(abs(jac-np.stack(gradients)))))
            if error>1e-11:raise ValueError('OPTIMAL_VERIFY_JACOBIAN')
            maximum=max(maximum,error);checked+=len(gaps)
        if not entry['record']['accepted']:
            rows.append(dict(unit=name,accepted=False,protected=int(eligible.sum()),cost_before=entry['baseline']['numpy']['cost'],cost_after=None,
                regret_before=entry['baseline']['numpy']['regret'],regret_after=None,changed=0,improved=0,worsened=0));continue
        cp=torch.load(unit/'ENDPOINT.pt',weights_only=False,map_location='cpu')
        if cp['identity']!=config['identity']:raise ValueError('OPTIMAL_VERIFY_ENDPOINT')
        require_equal(cp['optimizer'],w['optimizer_proposed'],'OPTIMAL_OPT');require_equal(cp['rng'],w['rng_after'],'OPTIMAL_RNG')
        if not ctx.observe(cp['state'],single=True)['passed']:raise ValueError('OPTIMAL_VERIFY_ORIGINAL_GUARD')
        candidates=views(ctx,cp['state'],ctx.fit)
        for name_view,qafter in candidates.items():
            qbefore=baseline[name_view];a0=qbefore.min(0).argmax(1);a1=qafter.min(0).argmax(1)
            if not optimal[np.flatnonzero(eligible),a1[eligible]].all():raise ValueError('OPTIMAL_VERIFY_LOST_OPTIMAL')
            costs=target.max(1,keepdims=True)-target
            def metrics(q0,action):
                logits=q0.min(0).astype(float)/ctx.temperature;logits-=logits.max(1,keepdims=True)
                p=np.exp(logits);p/=p.sum(1,keepdims=True)
                return float(ctx.temperature*(p*costs).sum(1).mean()),float(costs[np.arange(len(target)),action].mean())
            before_cost,before_regret=metrics(qbefore,a0);cost,regret=metrics(qafter,a1)
            if not cost<before_cost or regret>before_regret:raise ValueError('OPTIMAL_VERIFY_QUALITY')
            if name_view=='numpy':
                delta=target[np.arange(len(target)),a1]-target[np.arange(len(target)),a0]
                rows.append(dict(unit=name,accepted=True,protected=int(eligible.sum()),cost_before=before_cost,cost_after=cost,
                    regret_before=before_regret,regret_after=regret,changed=int(np.sum(a0!=a1)),improved=int(np.sum(delta>0)),worsened=int(np.sum(delta<0))))
    accepted=sum(r['accepted'] for r in rows);gate=accepted>=8 and len({r['unit'].rsplit('_',1)[0] for r in rows if r['accepted']})==4 and len(rows)==10
    if gate!=result['continuation_gate']:raise ValueError('OPTIMAL_VERIFY_GATE')
    report=dict(identity=config['identity'],units=len(rows),accepted=accepted,continuation_gate=gate,
        added_jacobian_constraints_checked=checked,jacobian_max_error=maximum,improved=sum(r['improved'] for r in rows),
        worsened=sum(r['worsened'] for r in rows),rows=rows,seconds=time.monotonic()-start,actor_updates=0,model_promoted=False)
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'VERIFICATION.json',report);write(a.output/'CONFIG.json',config);write(a.output/'RESULT.json',result)
    with (a.output/'SUMMARY.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verify_done(a.run,config['identity']);seal(a.output,config['identity'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','crossings','probe','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--expected-device-name',required=True);run(p.parse_args())
