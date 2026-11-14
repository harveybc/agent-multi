"""Fixed-step attribution: shared gradients, AdamW algebra and finite projection."""
import numpy as np
from app.weight_decision_forensics import native_forward,cost_parts,cost_gradients,weighted_min_gradient


def flat(state):
    return np.concatenate([v.detach().cpu().double().numpy().ravel() for v in state.values()])


def adam_parts(before,proposed,old_opt,new_opt,gradient):
    old_group,=old_opt['param_groups'];group,=new_opt['param_groups']
    if old_group!=group or len(group['params'])!=len(before):raise ValueError('CROSS_OPT_GROUP')
    beta1,beta2=group['betas'];lr=group['lr'];history=[];current=[];updated=[];position=0
    if group.get('amsgrad') or group.get('maximize'):raise ValueError('CROSS_OPT_MODE')
    for pid,value in zip(group['params'],before.values()):
        old=old_opt['state'][pid];new=new_opt['state'][pid];step=int(new['step'])
        if step!=int(old['step'])+1:raise ValueError('CROSS_OPT_STEP')
        m=old['exp_avg'].detach().cpu().double().numpy().ravel()
        n=new['exp_avg'].detach().cpu().double().numpy().ravel()
        v=new['exp_avg_sq'].detach().cpu().double().numpy().ravel()
        g=gradient[position:position+value.numel()];position+=value.numel()
        den=np.sqrt(v/(1-beta2**step))+group['eps'];scale=-lr/(1-beta1**step)
        history.append(scale*beta1*m/den);current.append(scale*(1-beta1)*g/den);updated.append(scale*n/den)
    if position!=len(gradient):raise ValueError('CROSS_GRAD_SIZE')
    history=np.concatenate(history);current=np.concatenate(current);updated=np.concatenate(updated)
    decay=-lr*group['weight_decay']*flat(before);actual=flat(proposed)-flat(before)
    residual=actual-updated-decay;moment_residual=updated-history-current
    # Float32 AdamW applies several rounded operations, unlike this float64 reconstruction.
    if np.max(abs(residual))>5e-7 or np.max(abs(moment_residual))>5e-7:raise ValueError('CROSS_ADAM_RECONSTRUCTION')
    return dict(history=history,current=current,decay=decay,rounding=residual+moment_residual,
        reconstruction_max_error=float(np.max(abs(residual))),moment_max_error=float(np.max(abs(moment_residual))))


def harmful_rows(before,after,y,fit):
    a=before.min(0).argmax(1);b=after.min(0).argmax(1)
    return [int(i) for i in fit if a[i]!=b[i] and y[i,a[i]]>y[i,b[i]]]


def attribute(before,proposed,accepted,old_opt,new_opt,actual_gradient,z,y,fit,batch,rows,temperature):
    states=(before,proposed,accepted);qs=[native_forward(s,z)[0] for s in states]
    actions=[q.min(0).argmax(1) for q in qs];pre,post=actions[0],actions[2]
    indices={int(r):j for j,r in enumerate(fit)}
    if not set(batch)<=set(fit) or not set(rows)<=set(fit):raise ValueError('CROSS_HELD_GRADIENT')
    gradients,_=cost_gradients(before,z[:,fit],y[fit],temperature)
    weights=np.array([np.sum(batch==i)/len(batch) for i in fit])
    batch_g=weights@gradients;full_g=gradients.mean(0)
    coeff=np.zeros_like(y);coeff[rows,pre[rows]]=1;coeff[rows,post[rows]]-=1
    jac=weighted_min_gradient(before,z,coeff)
    adam=adam_parts(before,proposed,old_opt,new_opt,actual_gradient)
    d=flat(proposed)-flat(before);repair=flat(accepted)-flat(proposed)
    costs=[cost_parts(q,y,temperature) for q in qs];events=[]
    for i in rows:
        j=jac[i];own=-float(j@gradients[indices[i]])*weights[indices[i]]
        total=-float(j@batch_g);actual=-float(j@actual_gradient)
        margins=[float(q.min(0)[i,pre[i]]-q.min(0)[i,post[i]]) for q in qs]
        parts={k:float(j@adam[k]) for k in ('history','current','decay','rounding')}
        if not np.isclose(sum(parts.values()),j@d,atol=1e-10,rtol=1e-10):raise ValueError('CROSS_ADAM_SUM')
        events.append(dict(row=int(i),before_action=int(pre[i]),proposed_action=int(actions[1][i]),after_action=int(post[i]),
            regret_delta=float(y[i,pre[i]]-y[i,post[i]]),batch_multiplicity=int(np.sum(batch==i)),
            own_batch_descent=own,others_batch_descent=total-own,analytic_batch_descent=total,
            actual_cuda_batch_descent=actual,full_fit_descent=-float(j@full_g),
            margin_before=margins[0],margin_proposed=margins[1],margin_after=margins[2],
            row_costs=[float(c['row_cost'][i]) for c in costs],
            first_order_adam=float(j@d),first_order_repair=float(j@repair),
            finite_adam=margins[1]-margins[0],finite_repair=margins[2]-margins[1],
            adam_contributions=parts,min_twins=[q.argmin(0)[i].tolist() for q in qs]))
    return dict(events=events,batch_gradient_max_error=float(np.max(abs(batch_g-actual_gradient))),
        adam_reconstruction_max_error=adam['reconstruction_max_error'],adam_moment_max_error=adam['moment_max_error'])
