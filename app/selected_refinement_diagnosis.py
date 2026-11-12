"""Read-only action ranking and first-step shared-head AdamW diagnostics."""
import torch


def validate_diagnostic_source(schema, calibration_only):
    expected = ('selected_input_critic_calibration.v1' if calibration_only
                else 'selected_input_bounded_refinement.v1')
    if schema != expected:
        raise ValueError('SELECTED_DIAG_SOURCE')


def optimal_mass_derivative(native, derivative, tolerance=1e-8):
    """All tied native optima count, not just the first argmax class."""
    if (native.shape != derivative.shape or native.ndim != 2 or native.shape[1] != 3
            or not torch.isfinite(native).all() or not torch.isfinite(derivative).all()):
        raise ValueError('OPTIMAL_MASS_SHAPE_OR_FINITE')
    optimal = native >= native.max(-1, keepdim=True).values - tolerance
    return (derivative * optimal).sum(-1)


def adaptive_parts(part, weights, margin_jacobian, theta, lr, decay, eps):
    g = part['batch_gradient']
    scale = lr / (g.abs() + eps)
    own = -torch.einsum('ndp,np,p->nd', margin_jacobian,
                       part['per_row_gradient'], scale) * weights[:, None]
    total = -torch.einsum('ndp,p,p->nd', margin_jacobian, g, scale)
    wd = margin_jacobian @ (-lr * decay * theta)
    return dict(self=own, cross=total-own, adaptive=total, decay=wd, total=total+wd)


def action_quality(native, q, tolerance=1e-8):
    chosen = q.argmax(-1)
    regret = native.max(-1).values - native.gather(1, chosen[:, None]).squeeze(1)
    gap = native[:, 1:] - native[:, :1]
    qgap = q[:, 1:] - q[:, :1]
    positive = gap > tolerance
    top = native.topk(2, dim=-1).values
    return dict(rows=len(native), optimal_count=int((regret <= tolerance).sum()),
                unique_native_optimum=int(((top[:, 0]-top[:, 1]) > tolerance).sum()),
                mean_regret=float(regret.mean()) if len(regret) else None,
                beneficial_gaps=int(positive.sum()),
                beneficial_gap_sign_errors=int((positive & (qgap <= 0)).sum()),
                predicted_actions=torch.bincount(chosen, minlength=3).tolist(),
                native_actions=torch.bincount(native.argmax(-1), minlength=3).tolist())


def decision_strata(records):
    """Disjoint state groups; blocked flat signals are not executable openings."""
    if any(r['role'] not in ('fit','monitor') or r['position'] not in (-1,0,1)
           or type(r['allowed']) is not bool for r in records):
        raise ValueError('DECISION_STRATA_RECORD')
    kinds=[('flat_allowed' if r['allowed'] else 'flat_blocked') if r['position']==0
           else 'long_open' if r['position']==1 else 'short_open' for r in records]
    return {f'{role}:{kind}':torch.tensor([r['role']==role and k==kind for r,k in zip(records,kinds)],dtype=torch.bool)
            for role in ('fit','monitor') for kind in ('flat_allowed','flat_blocked','long_open','short_open')}


def decision_behavior(records, scores):
    """Native continuation quality, not realized trading accuracy or oracle labels."""
    scores=torch.as_tensor(scores,dtype=torch.double)
    if scores.shape!=(len(records),3) or not torch.isfinite(scores).all():
        raise ValueError('DECISION_SCORES')
    if not records:return dict(rows=0)
    decision_strata(records)
    reward=torch.tensor([[o['reward_return'] for o in r['outcomes']] for r in records],dtype=torch.double)
    cash=torch.tensor([[o['net_usd'] for o in r['outcomes']] for r in records],dtype=torch.double)
    if not torch.isfinite(reward).all() or not torch.isfinite(cash).all():
        raise ValueError('DECISION_TARGETS')
    chosen=scores.argmax(-1);ix=torch.arange(len(records))
    ropt=reward>=reward.max(-1,keepdim=True).values-1e-8
    copt=cash>=cash.max(-1,keepdim=True).values-1e-6
    opening=dict(executable=0,selected_nop=0,missed_beneficial_entry=0,entry_when_nop_strictly_best=0)
    closing=dict(positions=0,executable=0,close_signals=0,nonexecuted_close_signals=0,
                 helpful_closes=0,harmful_closes=0,tie_closes=0,missed_beneficial_closes=0)
    for i,r in enumerate(records):
        action=int(chosen[i]);position=r['position']
        if position==0:
            if r['allowed']:
                opening['executable']+=1;opening['selected_nop']+=int(action==0)
                opening['missed_beneficial_entry']+=int(action==0 and reward[i,1:].max()>reward[i,0]+1e-8)
                opening['entry_when_nop_strictly_best']+=int(action!=0 and reward[i,0]>reward[i,1:].max()+1e-8)
            continue
        closing['positions']+=1;close=2 if position==1 else 1
        executable=r['outcomes'][close]['first_effect']['count_changes'].get('close_AGENT_CLOSE',0)>0
        closing['executable']+=int(executable);closing['close_signals']+=int(action==close)
        if not executable:
            closing['nonexecuted_close_signals']+=int(action==close);continue
        delta=float(reward[i,close]-reward[i,0])
        closing['helpful_closes']+=int(action==close and delta>1e-8)
        closing['harmful_closes']+=int(action==close and delta < -1e-8)
        closing['tie_closes']+=int(action==close and abs(delta)<=1e-8)
        closing['missed_beneficial_closes']+=int(action!=close and delta>1e-8)
    return dict(rows=len(records),reward_quality=action_quality(reward,scores),cash_quality=action_quality(cash,scores,tolerance=1e-6),
        reward_cash_optimum_set_disagreements=int((ropt!=copt).any(-1).sum()),
        reward_optimal_cash_suboptimal=int((ropt[ix,chosen]&~copt[ix,chosen]).sum()),
        cash_optimal_reward_suboptimal=int((copt[ix,chosen]&~ropt[ix,chosen]).sum()),
        constant_symbol_reward_regret={name:float((reward.max(-1).values-reward[:,a]).mean())
                                      for a,name in enumerate(('NOP','BUY','SELL'))},
        opening=opening,closing=closing,
        scope='fixed first action then frozen-policy account continuation; not isolated trade payoff')
