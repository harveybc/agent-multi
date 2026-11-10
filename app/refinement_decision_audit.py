"""Matched target semantics and actor last-layer decision-boundary decomposition."""
import numpy as np

from app.anchored_action_attribution import event_metrics


def boundary_terms(old, new, hidden, parent_weight, parent_bias, weight, bias,
                   parent_logits, candidate_logits):
    h,pw,pb,w,b,pl,cl=[np.asarray(x,dtype=float) for x in
        (hidden,parent_weight,parent_bias,weight,bias,parent_logits,candidate_logits)]
    if (pw.shape!=w.shape or pw.shape!=(3,len(h)) or pb.shape!=b.shape or pb.shape!=(3,)
            or pl.shape!=(3,) or cl.shape!=(3,) or old==new
            or not all(np.isfinite(v).all() for v in (h,pw,pb,w,b,pl,cl))):
        raise ValueError('BOUNDARY_INPUT')
    if not np.allclose(pw@h+pb,pl,atol=2e-5,rtol=2e-5):
        raise ValueError('BOUNDARY_PARENT_RECONSTRUCTION')
    weight_term=float(((w[new]-pw[new])-(w[old]-pw[old]))@h)
    bias_term=float((b[new]-pb[new])-(b[old]-pb[old]))
    mean=w@h+b
    posterior_term=float((cl[new]-cl[old])-(mean[new]-mean[old]))
    change=float((cl[new]-cl[old])-(pl[new]-pl[old]))
    residual=change-weight_term-bias_term-posterior_term
    if abs(residual)>5e-5:raise ValueError('BOUNDARY_TELESCOPE')
    return dict(weight_term=weight_term,bias_term=bias_term,posterior_term=posterior_term,
        log_odds_change=change,reconstruction_error=residual)


def classify_event(event,parent,candidate,twins,alpha,calibration_position,terms):
    result=event_metrics(event,parent,candidate,twins,alpha)
    samples=event['soft_continuations']
    if len(samples)!=3 or {s['seed'] for s in samples}!={7,19,43}:
        raise ValueError('AUDIT_THREE_MATCHED_CONTINUATIONS')
    soft=np.array([r['new']['soft_return']-r['old']['soft_return'] for r in samples])
    reward=np.array([r['new']['reward_return']-r['old']['reward_return'] for r in samples])
    if not np.isfinite([soft,reward]).all():raise ValueError('AUDIT_CONTINUATION_FINITE')
    q=result['q_gap'];tiny=1e-8
    soft_conflict=bool((q>tiny and (soft < -tiny).all()) or (q< -tiny and (soft>tiny).all()))
    pos=event['isolated_state']['position']
    position=0 if pos is None else pos['direction']
    return dict(result,soft_gap_mean=float(soft.mean()),soft_gap_min=float(soft.min()),
        soft_gap_max=float(soft.max()),soft_gap_samples=soft.tolist(),
        stochastic_reward_gap_mean=float(reward.mean()),
        q_soft_all_draws_conflict=soft_conflict,
        actor_change_against_q=bool(q < -tiny),
        entropy_score_favors_change=bool(result['entropy_adjusted_parent_score_gap']>tiny),
        all_soft_draws_favor_change=bool((soft>tiny).all()),
        all_soft_draws_oppose_change=bool((soft< -tiny).all()),
        calibration_position=int(calibration_position),isolated_position=position,
        calibration_position_mismatch=position!=calibration_position,
        boundary=terms,three_draws_not_exact_expectation=True)


def mechanism_summary(rows):
    bad=[r for r in rows if r['outcome']=='harmful']
    good=[r for r in rows if r['outcome']=='helpful']
    keys=('q_soft_all_draws_conflict','actor_change_against_q','entropy_score_favors_change',
          'all_soft_draws_favor_change','calibration_position_mismatch')
    return dict(events=len(rows),helpful=len(good),harmful=len(bad),
        neutral=sum(r['outcome']=='neutral' for r in rows),
        delta_usd=float(sum(r['contribution_usd'] for r in rows)),
        harmful_loss_usd=float(-sum(r['contribution_usd'] for r in bad)),
        harmful_counts={k:sum(r[k] for r in bad) for k in keys},
        harmful_loss_by_flag={k:float(-sum(r['contribution_usd'] for r in bad if r[k])) for k in keys},
        flags_overlap=True,occurrences_not_independent=True)
