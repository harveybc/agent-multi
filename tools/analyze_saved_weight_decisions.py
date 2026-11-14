"""Sealed no-training analysis of native critic weights, interventions and gradients."""
import argparse
import fcntl
from pathlib import Path
import time
import numpy as np
import torch

from app.decision_cost_objective import cost_objective
from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.rl_subset import digest
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from app.weight_decision_forensics import (BLOCKS,native_forward,cost_parts,cost_gradients,
    weighted_min_gradient,block_indices,hybrid_state,exact_shapley,cosine)
from tools.probe_robust_adamw_direction import vector
from tools.run_bayesian_head_contrast import seal,verify_done
from tools.run_discrete_curve_probe import read,write,now
from tools.run_training_forensics import template

ROOT=Path(__file__).resolve().parents[1]


def flatten(grad):return torch.cat([v.reshape(-1) for v in grad]).detach().cpu().double().numpy()


def point(state,optimizer,z,y,fit,temperature,seed,device,kinds):
    q,cache=native_forward(state,z);parts=cost_parts(q,y,temperature)
    gf,_=cost_gradients(state,z[:,fit],y[fit],temperature)
    grad=np.zeros((len(y),gf.shape[1]));grad[fit]=gf
    head=RoleHeads(template('cpu'),1,seed).double();head.load_state_dict(state)
    td=head(torch.tensor(z)).mean(2)
    forward_error=float(np.max(abs(td.detach().numpy()-q)))
    loss=cost_objective(td[:,fit],torch.tensor(y[fit]),'cost',temperature)
    g=flatten(torch.autograd.grad(loss,list(head.parameters())))
    gradient_error=float(np.max(abs(g-grad[fit].mean(0))))
    if forward_error>1e-11 or gradient_error>1e-10:raise ValueError('WEIGHT_ANALYTIC_PARITY')
    d=np.random.default_rng(339).normal(size=g.shape);d/=np.linalg.norm(d)
    def offset(sign,eps):
        result={};i=0
        for k,v in state.items():
            result[k]=v.double()+torch.tensor(sign*eps*d[i:i+v.numel()].reshape(v.shape));i+=v.numel()
        return result
    fd_error=None;fd_epsilon=None
    for eps in (1e-6,1e-7,1e-8,1e-9,1e-10):
        qp=native_forward(offset(1,eps),z)[0];qm=native_forward(offset(-1,eps),z)[0]
        if not (np.array_equal(qp[:,fit].argmin(0),q[:,fit].argmin(0)) and np.array_equal(qm[:,fit].argmin(0),q[:,fit].argmin(0))):continue
        fd=(cost_parts(qp,y,temperature)['row_cost'][fit].mean()-cost_parts(qm,y,temperature)['row_cost'][fit].mean())/(2*eps)
        fd_error=abs(float(fd-g@d));fd_epsilon=eps;break
    if fd_error is not None and fd_error>1e-7:raise ValueError('WEIGHT_FINITE_DIFFERENCE')
    cpu=head.float()
    with torch.no_grad():qc=cpu(torch.tensor(z,dtype=torch.float32)).mean(2).double().numpy()
    # Full-FIT CUDA witness uses no optimizer step and excludes held labels from gradients.
    witness=None;gpu_info=None
    if device=='cuda':
        gpu=RoleHeads(template('cuda'),1,seed);gpu.load_state_dict(state)
        qg=gpu(torch.tensor(z[:,fit],device='cuda',dtype=torch.float32)).mean(2)
        yt=torch.tensor(y[fit],device='cuda',dtype=torch.float32);costs=yt.max(1,keepdim=True).values-yt
        rowloss=temperature*(torch.softmax(qg.min(0).values/temperature,dim=1)*costs).sum(1)
        pars=list(gpu.parameters());gg=flatten(torch.autograd.grad(rowloss.mean(),pars,retain_graph=True))
        rolegrads={}
        for kind in sorted(set(kinds[i] for i in fit)):
            positions=[j for j,i in enumerate(fit) if kinds[i]==kind]
            rolegrads[kind]=flatten(torch.autograd.grad(rowloss[positions].sum()/len(fit),pars,retain_graph=True))
        if not np.allclose(sum(rolegrads.values()),gg,atol=1e-6,rtol=1e-5):raise ValueError('WEIGHT_GPU_ROLE_SUM')
        qgn=qg.detach().cpu().double().numpy()
        gpu_info=dict(full_fit_gradient_cosine=cosine(gg,g),gradient_max_difference=float(np.max(abs(gg-g))),
            full_fit_action_disagreements=int(np.sum(qgn.min(0).argmax(1)!=q[:,fit].min(0).argmax(1))),
            minimum_twin_route_disagreements=int(np.sum(qgn.argmin(0)!=q[:,fit].argmin(0))))
        witness=dict(head=gpu,q=qg,rowloss=rowloss,gradient=gg,roles=rolegrads,positions={int(i):j for j,i in enumerate(fit)})
    arrays={k:v.detach().cpu().double().numpy() for k,v in state.items()};spec={}
    for k,v in arrays.items():
        item=dict(norm=float(np.linalg.norm(v)),max_abs=float(np.max(abs(v))))
        if v.ndim==2:
            s=np.linalg.svd(v,compute_uv=False);p=s/s.sum() if s.sum() else np.zeros_like(s)
            item.update(singular_values=s.tolist(),effective_rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))))
        spec[k]=item
    groups=optimizer['param_groups']
    if len(groups)!=1:raise ValueError('WEIGHT_OPT_GROUPS')
    group=groups[0];moments=[];effective=[]
    for pid,(name,v) in zip(group['params'],state.items()):
        saved=optimizer['state'][pid];step=int(saved['step']);beta1,beta2=group['betas']
        m=saved['exp_avg'].cpu().double().numpy();vv=saved['exp_avg_sq'].cpu().double().numpy()
        if m.shape!=v.shape or vv.shape!=v.shape or step<1:raise ValueError('WEIGHT_OPT_STATE')
        moments.append(m.ravel());effective.append((m/(1-beta1**step)/(np.sqrt(vv/(1-beta2**step))+group['eps'])).ravel())
    if len(group['params'])!=len(state):raise ValueError('WEIGHT_OPT_MAPPING')
    moment=np.concatenate(moments);direction=-np.concatenate(effective)
    summary=dict(forward_max_error=forward_error,gradient_max_error=gradient_error,finite_difference_error=fd_error,
        finite_difference_epsilon=fd_epsilon,finite_difference_route_stable=fd_epsilon is not None,
        cpu32_action_disagreements=int(np.sum(qc.min(0).argmax(1)!=q.min(0).argmax(1))),
        cpu32_route_disagreements=int(np.sum(qc.argmin(0)!=q.argmin(0))),cuda=gpu_info,
        cost=float(parts['row_cost'][fit].mean()),regret=float(parts['row_regret'][fit].mean()),
        full_fit_gradient_norm=float(np.linalg.norm(g)),moment_gradient_cosine=cosine(moment,g),
        moment_direction_vs_full_descent=cosine(direction,-g),spectra=spec,
        activations=[dict(hidden_mean=c['h'][fit].mean(0).tolist(),hidden_std=c['h'][fit].std(0).tolist(),
            silu_derivative_mean=c['derivative'][fit].mean(0).tolist()) for c in cache])
    return dict(state=state,q=q,qcpu=qc,grad=grad,parts=parts,moment_direction=direction,summary=summary,witness=witness)


def transition(before,after,z,y,fit,scopes,kinds,temperature):
    q0=before['q'].min(0);q1=after['q'].min(0);a=q0.argmax(1);b=q1.argmax(1)
    changed=np.intersect1d(np.flatnonzero(a!=b),np.concatenate(list(scopes.values())))
    rows=np.arange(len(y));delta=vector(after['state'])-vector(before['state'])
    columns=block_indices(before['state']);coeff=np.zeros_like(y);coeff[rows,a]=1;coeff[rows,b]-=1
    mg=weighted_min_gradient(before['state'],z,coeff);g=before['grad'];role_keys=sorted(set(kinds[i] for i in fit))
    role_g={k:g[[i for i in fit if kinds[i]==k]].sum(0)/len(fit) for k in role_keys}
    coalition=[]
    for mask in range(16):
        qs=native_forward(hybrid_state(before['state'],after['state'],mask),z)[0]
        parts=cost_parts(qs,y,temperature);m=qs.min(0)
        coalition.append(np.concatenate(([parts['row_cost'][fit].mean(),parts['row_regret'][fit].mean()],m[changed,a[changed]]-m[changed,b[changed]])))
    coalition=np.stack(coalition);phi=exact_shapley(coalition)
    events=[];fitset=set(fit)
    for j,i in enumerate(changed):
        scope=next((k for k,v in scopes.items() if i in set(v)), 'outside_scopes')
        target_delta=float(y[i,a[i]]-y[i,b[i]])
        stable=bool(before['qcpu'].min(0).argmax(1)[i]==a[i] and after['qcpu'].min(0).argmax(1)[i]==b[i])
        item=dict(row=int(i),kind=kinds[i],scope=scope,before=int(a[i]),after=int(b[i]),
            regret_delta=target_delta,harmful=target_delta>0,beneficial=target_delta<0,cpu_geometry_agrees=stable,
            margin_before=float(q0[i,a[i]]-q0[i,b[i]]),margin_after=float(q1[i,a[i]]-q1[i,b[i]]),
            row_cost_before=float(before['parts']['row_cost'][i]),row_cost_after=float(after['parts']['row_cost'][i]),
            probabilities_before=before['parts']['probabilities'][i].tolist(),probabilities_after=after['parts']['probabilities'][i].tolist(),
            target_returns=y[i].tolist(),min_twins_before=before['q'].argmin(0)[i].tolist(),min_twins_after=after['q'].argmin(0)[i].tolist(),
            finite_margin_delta=float((q1[i,a[i]]-q1[i,b[i]])-(q0[i,a[i]]-q0[i,b[i]])),
            first_order_margin_delta=float(mg[i]@delta),
            first_order_by_block={k:float(mg[i,ix]@delta[ix]) for k,ix in columns.items()},
            exact_shapley_margin={k:float(phi[n,j+2]) for n,k in enumerate(BLOCKS)})
        # Held rows get finite descriptive attribution only, never target gradients.
        if i in fitset:
            local=before['parts']['logit_gradient'][i];effects=-(g[fit]@mg[i])/len(fit)
            own=-float(mg[i]@g[i])/len(fit);total=float(effects.sum())
            roles={k:-float(mg[i]@v) for k,v in role_g.items()}
            if abs(sum(roles.values())-total)>1e-8*max(1,abs(total)):raise ValueError('WEIGHT_EVENT_ROLE_SUM')
            top=np.argsort(effects)[:5]
            item['fit_gradients']=dict(independent_logit_descent=float(-(local[a[i]]-local[b[i]])),
                own_descent=own,others_descent=total-own,total_descent=total,roles=roles,
                saved_moment_margin_direction=float(mg[i]@before['moment_direction']),
                most_negative_contributors=[dict(row=int(fit[k]),kind=kinds[fit[k]],effect=float(effects[k])) for k in top])
            witness=before['witness']
            if witness is not None:
                pos=witness['positions'][int(i)];qm=witness['q'].min(0).values;pars=list(witness['head'].parameters())
                gm=flatten(torch.autograd.grad(qm[pos,int(a[i])]-qm[pos,int(b[i])],pars,retain_graph=True))
                go=flatten(torch.autograd.grad(witness['rowloss'][pos]/len(fit),pars,retain_graph=True))
                own32=-float(gm@go);total32=-float(gm@witness['gradient'])
                item['fit_gradients']['cuda32']=dict(own_descent=own32,total_descent=total32,others_descent=total32-own32,
                    roles={k:-float(gm@v) for k,v in witness['roles'].items()},margin_gradient_cosine=cosine(gm,mg[i]))
        events.append(item)
    drift={k:dict(delta_norm=float(np.linalg.norm(delta[ix])),before_norm=float(np.linalg.norm(vector(before['state'])[ix]))) for k,ix in columns.items()}
    return dict(events=events,weight_drift=drift,displacement_vs_full_descent=cosine(delta,-g[fit].mean(0)),
        displacement_vs_saved_moment=cosine(delta,before['moment_direction']),
        exact_shapley_fit_cost={k:float(phi[i,0]) for i,k in enumerate(BLOCKS)},
        exact_shapley_fit_regret={k:float(phi[i,1]) for i,k in enumerate(BLOCKS)},
        coalition_fit_cost=coalition[:,0].tolist(),coalition_fit_regret=coalition[:,1].tolist(),
        shapley_max_efficiency_error=float(np.max(abs(phi.sum(0)-(coalition[-1]-coalition[0])))))


def run(a):
    start=time.monotonic();torch.set_num_threads(1)
    if a.device=='cuda':
        if torch.cuda.get_device_name()!=a.expected_device_name:raise ValueError('WEIGHT_DEVICE')
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    parents={k:read(getattr(a,k)/'CONFIG.json') for k in ('previous','source','coverage')}
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    if any(parents['previous']['parents'][k]!=parents[k]['identity'] for k in ('source','coverage')):raise ValueError('WEIGHT_LINEAGE')
    config=dict(schema='saved_weight_forensics.v1',parents={k:c['identity'] for k,c in parents.items()},
        source_actor_sha256=parents['previous']['source_actor_sha256'],device=a.device,
        components=list(BLOCKS),transitions=[[0,64],[64,256],[0,256]],
        code={k:file_sha(ROOT/k) for k in ('tools/analyze_saved_weight_decisions.py','app/weight_decision_forensics.py','app/decision_cost_objective.py')})
    config['identity']=digest(config)
    if file_sha(a.bundle/'model.pt')!=config['source_actor_sha256']:raise ValueError('WEIGHT_ACTOR')
    if (a.output/'CONFIG.json').exists() and read(a.output/'CONFIG.json')!=config:raise ValueError('WEIGHT_RESUME')
    if (a.output/'DONE.json').exists():verify_done(a.output,config['identity']);return
    write(a.output/'CONFIG.json',config)
    data=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu');z=data['z'].double().numpy()
    records=read(a.coverage/'PANEL.json')['records'];y=np.stack([validate_record(r) for r in records]);kinds=[r['kind'] for r in records]
    support=read(a.coverage/'prepared/SUPPORT.json');results=[]
    for fold in ('A','B'):
        fit=np.array(support['folds'][fold]['expanded']);info=support['folds'][fold]
        scopes=dict(fit=fit,stop=np.array(info['masks']['stop']),audit=np.array(info['masks']['audit']))
        if set(fit)&(set(scopes['stop'])|set(scopes['audit'])):raise ValueError('WEIGHT_LABEL_OVERLAP')
        duplicates={}
        for i in fit:duplicates.setdefault(z[:,i].tobytes(),[]).append(int(i))
        duplicate_groups=[v for v in duplicates.values() if len(v)>1]
        for seed in (19,43):
            out=a.output/'cases'/f'{fold}_{seed}'
            if (out/'DONE.json').exists():verify_done(out,config['identity']);results.append(read(out/'RESULT.json'));continue
            path=a.source/'units'/f'{fold}_{seed}_cost'/'LATEST.pt';base=torch.load(path,weights_only=False,map_location='cpu')
            temperature=base['support']['temperature'];p0=point(base['state'],base['optimizer'],z,y,fit,temperature,seed,a.device,kinds)
            case=dict(fold=fold,seed=seed,source_sha256=file_sha(path),temperature=temperature,fit_rows=len(fit),
                exact_duplicate_fit_embeddings=duplicate_groups,source=p0['summary'],arms=[])
            for method in ('active','complete'):
                directory=a.previous/'units'/f'{fold}_{seed}_{method}';verify_done(directory,parents['previous']['identity'])
                points={0:p0};summaries={}
                for offset in (64,256):
                    cp=torch.load(directory/f'ENDPOINT_{offset}.pt',weights_only=False,map_location='cpu')
                    if cp['source_sha256']!=case['source_sha256']:raise ValueError('WEIGHT_CHECKPOINT_SOURCE')
                    points[offset]=point(cp['state'],cp['optimizer'],z,y,fit,temperature,seed,a.device,kinds)
                    expected=cp['curve'][-1]['assessment']['observation']['fit']['numpy']
                    if abs(points[offset]['summary']['regret']-expected['regret'])>1e-12:raise ValueError('WEIGHT_ENDPOINT_REPLAY')
                    summaries[str(offset)]=dict(points[offset]['summary'],checkpoint_sha256=file_sha(directory/f'ENDPOINT_{offset}.pt'),
                        accepted=cp['accepted'],clipped_steps=sum(bool(t['clipped']) for t in cp['trace']))
                transitions=[]
                for before,after in config['transitions']:
                    value=transition(points[before],points[after],z,y,fit,scopes,kinds,temperature)
                    transitions.append(dict(before=before,after=after,**value))
                case['arms'].append(dict(method=method,points=summaries,transitions=transitions))
                write(a.output/'PROGRESS.json',dict(stage='ANALYZING',fold=fold,seed=seed,method=method,seconds=time.monotonic()-start,at=now()))
                del points
            write(out/'RESULT.json',case);seal(out,config['identity']);results.append(case)
            del p0
    if file_sha(a.bundle/'model.pt')!=config['source_actor_sha256']:raise ValueError('WEIGHT_ACTOR_CHANGED')
    for k,c in parents.items():verify_done(getattr(a,k),c['identity'])
    result=dict(identity=config['identity'],seconds=time.monotonic()-start,cases=results,
        actor_updates=0,optimizer_steps=0,model_promoted=False,finite_hybrids=384)
    write(a.output/'RESULT.json',result);seal(a.output,config['identity'])
    write(a.output/'PROGRESS.json',dict(stage='COMPLETE',at=now(),seconds=time.monotonic()-start))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('previous','source','coverage','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--expected-device-name')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(a)
        except Exception as exc:
            write(a.output/'PROGRESS.json',dict(stage='FAILED',error=repr(exc),at=now()));raise
