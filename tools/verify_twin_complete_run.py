"""Independent NumPy derivatives, finite guards and matched endpoint verification."""
import argparse
from pathlib import Path
import numpy as np
import torch

from app.frozen_recipe_inputs import file_sha
from app.native_fit_coverage import validate_record
from app.robust_adamw_direction import numpy_heads
from app.training_forensics import tree_digest
from tools.probe_robust_adamw_direction import moved,vector
from tools.run_bayesian_head_contrast import verify_done,seal
from tools.run_discrete_curve_probe import read,write


def manual_complete_system(state,z,actions):
    arrays={k:v.detach().cpu().double().numpy() for k,v in state.items()}
    q=[];hidden=[];derivatives=[]
    for twin in range(2):
        p=f'twins.{twin}.0.';u=z[twin]@arrays[p+'0.weight'].T+arrays[p+'0.bias']
        sig=np.exp(-np.logaddexp(0.,-u));h=u*sig
        hidden.append(h);derivatives.append(sig*(1+u*(1-sig)))
        q.append(h@arrays[p+'2.weight'].T+arrays[p+'2.bias'])
    q=np.stack(q);route=q.argmin(0);gaps=[];jac=[]
    for row,action in enumerate(actions):
        for alternative in range(3):
            if alternative==action:continue
            other=int(route[row,alternative])
            for chosen in range(2):
                grad={k:np.zeros_like(v) for k,v in arrays.items()}
                for twin,act,sign in ((chosen,action,1.),(other,alternative,-1.)):
                    p=f'twins.{twin}.0.';du=arrays[p+'2.weight'][act]*derivatives[twin][row]
                    grad[p+'2.weight'][act]+=sign*hidden[twin][row]
                    grad[p+'2.bias'][act]+=sign
                    grad[p+'0.weight']+=sign*du[:,None]*z[twin,row][None,:]
                    grad[p+'0.bias']+=sign*du
                gaps.append(q[chosen,row,action]-q[other,row,alternative])
                jac.append(np.concatenate([v.ravel() for v in grad.values()]))
    return np.array(gaps),np.stack(jac)


def quality(q,y):
    action=q.min(0).argmax(1);rows=np.arange(len(y))
    return dict(mse=float(np.mean((q-y[None])**2)),regret=float(np.mean(y.max(1)-y[rows,action])))


def run(a):
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity'])
    prior=read(a.previous/'CONFIG.json');verify_done(a.previous,prior['identity'])
    if cfg['previous_solver']!=prior['identity']:raise ValueError('VERIFY_PARENT')
    if file_sha(a.bundle/'model.pt')!=cfg['source_actor_sha256']:raise ValueError('VERIFY_ACTOR')
    source=torch.load(a.source/'units/B_43_cost/LATEST.pt',weights_only=False,map_location='cpu')
    if file_sha(a.source/'units/B_43_cost/LATEST.pt')!=cfg['source_file_sha256']:raise ValueError('VERIFY_SOURCE')
    snap=torch.load(a.probe/'units/B_43_8192_0/PROPOSAL.pt',weights_only=False,map_location='cpu')
    z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')['z'].double().numpy()
    support=read(a.coverage/'prepared/SUPPORT.json');info=support['folds']['B']
    old=np.array(info['masks']['fit']);fit=np.array(info['expanded']);actions=snap['actions']
    y=np.stack([validate_record(r) for r in read(a.coverage/'PANEL.json')['records']])
    result=read(a.run/'RESULT.json');stable=np.array(result['stable_added_indices']);checks=[];systems=[];transfer=[]
    def check_state(state,assessment,label):
        q=numpy_heads(state,z);qo=q.min(0)[old];rows=np.arange(len(old))
        gaps=qo[rows,actions,None]-qo;gaps[rows,actions]=np.inf
        surplus=float(np.min(gaps-snap['margin'][:,None]));same=np.array_equal(qo.argmax(1),actions)
        if not same or surplus<0:raise ValueError('VERIFY_FINITE_GUARD')
        for indices,expected in ((fit,assessment['observation']['fit']['numpy']),(stable,assessment['stable_added']['numpy'])):
            scores=quality(q[:,indices],y[indices])
            if any(abs(scores[k]-expected[k])>1e-12 for k in scores):raise ValueError('VERIFY_METRIC')
        checks.append(dict(label=label,min_margin_surplus=surplus))
        return q
    for r in result['probes']:
        attempt=r['attempt'];method=r['method'];out=a.run/'probes'/f'{attempt}_{method}'
        verify_done(out,cfg['identity']);cp=torch.load(out/'ENDPOINT.pt',weights_only=False,map_location='cpu')
        prev=torch.load(a.previous/'probes'/f'{attempt}_{method}'/'ENDPOINT.pt',weights_only=False,map_location='cpu')
        for k in ('proposal','proposed_optimizer'):
            if tree_digest(cp[k])!=tree_digest(prev[k]):raise ValueError('VERIFY_PAIRED_PROPOSAL')
        if cp['record']!=r['record'] or cp['assessment']!=r['assessment']:raise ValueError('VERIFY_RECEIPT')
        if method=='legacy' and cp['record']!=prev['record']:raise ValueError('VERIFY_LEGACY')
        if cp['record']['accepted']:
            trial=cp['record']['repairs'][-1]['trials'][-1]
            want=moved(source['state'],cp['systems'][-1]['p'],trial['scale'])
            opt=cp['proposed_optimizer']
        else:want=source['state'];opt=source['optimizer']
        if tree_digest(want)!=tree_digest(cp['state']) or tree_digest(opt)!=tree_digest(cp['optimizer']):raise ValueError('VERIFY_TRANSACTION')
        check_state(cp['state'],cp['assessment'],f'{attempt}_{method}')
        if method=='legacy':continue
        center=np.zeros_like(vector(source['state']))
        for i,w in enumerate(cp['systems']):
            current=moved(source['state'],center,1.)
            gaps,j=manual_complete_system(current,z[:,old],actions)
            rhs=2*np.repeat(snap['margin'],4)-gaps+j@(vector(current)-vector(source['state']))
            je=float(np.max(abs(j-w['j'])));be=float(np.max(abs(rhs-w['b'])))
            if je>1e-10 or be>1e-10:raise ValueError('VERIFY_MANUAL_DERIVATIVE')
            n=np.linalg.norm(w['j'],axis=1);active=n>1e-14
            violation=max(0.,float(np.max((w['b'][active]-w['j'][active]@w['p'])/n[active]/np.linalg.norm(w['d']))))
            if w['receipt']['feasible'] and violation>1e-7:raise ValueError('VERIFY_PROJECTED_SYSTEM')
            systems.append(dict(attempt=attempt,iteration=i,jacobian_error=je,rhs_error=be,violation=violation))
            center=w['p']
    scopes=dict(fit=fit,stop=np.array(info['masks']['stop']),audit=np.array(info['masks']['audit']))
    if set(fit)&(set(scopes['stop'])|set(scopes['audit'])):raise ValueError('VERIFY_SPLIT')
    states=[('source',source['state'])]
    for arm in result['continuation']:
        out=a.run/'continuation'/arm['method'];verify_done(out,cfg['identity'])
        for step in (64,256):
            cp=torch.load(out/f'ENDPOINT_{step}.pt',weights_only=False,map_location='cpu')
            check_state(cp['state'],cp['curve'][-1]['assessment'],f"{arm['method']}_{step}")
            states.append((f"{arm['method']}_{step}",cp['state']))
    for label,state in states:
        q=numpy_heads(state,z)
        transfer.append(dict(label=label,scopes={k:quality(q[:,ix],y[ix]) for k,ix in scopes.items()}))
    evidence=dict(identity=cfg['identity'],states_verified=len(checks),systems_verified=len(systems),
        max_jacobian_error=max(s['jacobian_error'] for s in systems),
        max_rhs_error=max(s['rhs_error'] for s in systems),
        checks=checks,systems=systems,transfer=transfer,actor_unchanged=True,
        transfer_scope='Internal TRAIN; encoder pretrained on whole TRAIN. Descriptive only, no held-label fitting.')
    write(a.output/'RESULT.json',evidence);seal(a.output,cfg['identity'])
    print({k:v for k,v in evidence.items() if k not in ('checks','systems')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','previous','source','probe','coverage','bundle','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
