"""Read-only CPU reconstruction of sealed projection and finite-network witnesses."""
import argparse
from pathlib import Path
import numpy as np
import torch

from app.robust_adamw_direction import numpy_heads
from app.role_routed_critic import RoleHeads
from app.training_forensics import tree_digest
from tools.probe_robust_adamw_direction import moved,vector,gap_jacobian
from tools.run_bayesian_head_contrast import verify_done,seal
from tools.run_discrete_curve_probe import read,write
from tools.run_training_forensics import template


def run(a):
    cfg=read(a.run/'CONFIG.json');verify_done(a.run,cfg['identity'])
    source=torch.load(a.source/'units/B_43_cost/LATEST.pt',weights_only=False,map_location='cpu')
    snap=torch.load(a.probe/'units/B_43_8192_0/PROPOSAL.pt',weights_only=False,map_location='cpu')
    z=torch.load(a.coverage/'prepared/DATA.pt',weights_only=True,map_location='cpu')['z'].double()
    old=np.array(read(a.coverage/'prepared/SUPPORT.json')['folds']['B']['masks']['fit'])
    state=source['state'];actions=snap['actions'];margin=np.repeat(snap['margin'],2)
    alternatives=np.array([[k for k in range(3) if k!=v] for v in actions]);rows=np.arange(len(old))
    head=RoleHeads(template('cpu'),1,43).double()
    if list(state)!=list(dict(head.named_parameters())):raise ValueError('AUDIT_PARAMETER_ORDER')
    systems=[];trials=[];pairs=0;restored=0
    for attempt in cfg['selected_attempts']:
        pair=[]
        for method in ('legacy','primal'):
            out=a.run/'probes'/f'{attempt}_{method}';verify_done(out,cfg['identity'])
            cp=torch.load(out/'ENDPOINT.pt',weights_only=False,map_location='cpu');pair.append(cp)
            if tree_digest(cp['state'])!=tree_digest(state) or tree_digest(cp['optimizer'])!=tree_digest(source['optimizer']):
                raise ValueError('AUDIT_REJECTED_STATE_CHANGED')
            restored+=1
        if (tree_digest(pair[0]['proposal'])!=tree_digest(pair[1]['proposal']) or
                tree_digest(pair[0]['proposed_optimizer'])!=tree_digest(pair[1]['proposed_optimizer'])):
            raise ValueError('AUDIT_UNPAIRED_PROPOSAL')
        pairs+=1;center=np.zeros_like(vector(state))
        for iteration,w in enumerate(pair[1]['systems']):
            current=moved(state,center,1.);head.load_state_dict(current)
            gaps,jac=gap_jacobian(head,z,old,actions)
            required=2*margin-gaps+jac@(vector(current)-vector(state))
            if not np.array_equal(jac,w['j']) or not np.array_equal(required,w['b']):raise ValueError('AUDIT_JACOBIAN')
            norms=np.linalg.norm(jac,axis=1);active=norms>1e-14
            violation=max(0.,float(np.max((required[active]-jac[active]@w['p'])/norms[active]/np.linalg.norm(w['d']))))
            if violation>1e-7:raise ValueError('AUDIT_PROJECTION')
            systems.append(dict(attempt=attempt,iteration=iteration,violation=violation))
            qc=numpy_heads(current,z.numpy()[:,old]);routes=qc.argmin(0)
            for recorded in pair[1]['record']['repairs'][iteration]['trials']:
                scale=recorded['scale'];candidate=moved(state,w['p'],scale)
                qtw=numpy_heads(candidate,z.numpy()[:,old]);q=qtw.min(0)
                actual=(q[rows,actions,None]-q[rows[:,None],alternatives]).ravel()
                linear=gaps+jac@(vector(candidate)-vector(current))
                bad=np.flatnonzero(actual<margin);route=qtw.argmin(0)
                # Holding the center's branch choices separates branch switches from smooth curvature.
                fixed=qtw[routes,rows[:,None],np.arange(3)[None,:]]
                fixed_gaps=(fixed[rows,actions,None]-fixed[rows[:,None],alternatives]).ravel()
                observed=float(np.min(actual-margin))
                if abs(observed-recorded['guards']['numpy64_guard']['min_margin_surplus'])>1e-12:
                    raise ValueError('AUDIT_FINITE_REPLAY')
                trials.append(dict(attempt=attempt,iteration=iteration,scale=scale,
                    min_actual_surplus=observed,min_linear_surplus=float(np.min(linear-margin)),
                    max_smooth_error=float(np.max(abs(fixed_gaps-linear))),
                    bad_rows=old[bad//2].tolist(),
                    bad_chosen_switches=int(sum(routes[k//2,actions[k//2]]!=route[k//2,actions[k//2]] for k in bad)),
                    bad_alternative_switches=int(sum(routes[k//2,alternatives[k//2,k%2]]!=route[k//2,alternatives[k//2,k%2]] for k in bad))))
            center=w['p']
    result=dict(identity=cfg['identity'],paired_proposals=pairs,restored_states=restored,systems=systems,trials=trials,
        summary=dict(systems=len(systems),trials=len(trials),bad_rows=sorted({r for t in trials for r in t['bad_rows']}),
            all_failed_trials_have_chosen_switch=all(t['bad_chosen_switches']>0 for t in trials),
            max_projection_violation=max(s['violation'] for s in systems),
            min_linear_surplus=min(t['min_linear_surplus'] for t in trials),
            max_smooth_error=max(t['max_smooth_error'] for t in trials)))
    write(a.output/'RESULT.json',result);seal(a.output,cfg['identity']);print(result['summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('run','source','probe','coverage','output'):p.add_argument('--'+k,type=Path,required=True)
    run(p.parse_args())
