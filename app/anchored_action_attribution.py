"""Descriptive frozen-policy margins and matched native action outcomes."""
import numpy as np
from app.frozen_signal_masks import prepared_signals


def event_metrics(event,parent,candidate,twins,alpha):
    parent,candidate,twins=(np.asarray(x,dtype=float) for x in (parent,candidate,twins))
    if (parent.shape!=(3,) or candidate.shape!=(3,) or twins.shape!=(2,3)
            or not all(np.isfinite(x).all() for x in (parent,candidate,twins))
            or not np.isfinite(alpha) or alpha<0):raise ValueError('ANCHORED_EVENT_SURFACE')
    old,new=event['old_signal'],event['new_signal']
    expected=prepared_signals(np.stack((parent,candidate)))[0]
    if old==new or old!=int(expected[0]) or new!=int(expected[1]):
        raise ValueError('ANCHORED_EVENT_ACTION')
    def logprob(x):
        z=x-x.max();return z-np.log(np.exp(z).sum())
    lp,lc=logprob(parent),logprob(candidate);p,c=np.exp(lp),np.exp(lc)
    q=twins.min(0);qgap=float(q[new]-q[old])
    reward=event['isolated_new']['reward_return']-event['isolated_old']['reward_return']
    cash=event['isolated_new']['net_usd']-event['isolated_old']['net_usd']
    delta=event['contribution_usd'];a,b=(event[k]['first_effect'] for k in ('prefix_old','prefix_new'))
    if a['command']==3 and b['command']!=3:effect='suppressed_close_command'
    elif a['command']!=3 and b['command']==3:effect='added_close_command'
    elif a['entries']>b['entries']:effect='suppressed_entry'
    elif a['entries']<b['entries']:effect='added_entry'
    elif a['entries'] and b['entries'] and a['command']!=b['command']:effect='changed_entry_direction'
    else:effect='other_or_no_effect'
    return dict(old_signal=old,new_signal=new,position=0 if event['state']['position'] is None else event['state']['position']['direction'],
        effect=effect,outcome='helpful' if delta>1e-6 else 'harmful' if delta< -1e-6 else 'neutral',
        contribution_usd=delta,isolated_cash_gap=cash,isolated_reward_gap=reward,
        prefix_reward_gap=event['prefix_new']['reward_return']-event['prefix_old']['reward_return'],
        parent_logits=parent.tolist(),candidate_logits=candidate.tolist(),parent_probabilities=p.tolist(),
        candidate_probabilities=c.tolist(),twins=twins.tolist(),q_gap=qgap,
        twin_gaps=(twins[:,new]-twins[:,old]).tolist(),
        parent_pair_margin=float(parent[old]-parent[new]),candidate_pair_margin=float(candidate[new]-candidate[old]),
        parent_probability_margin=float(p[old]-p[new]),candidate_probability_margin=float(c[new]-c[old]),
        kl=float((p*(lp-lc)).sum()),alpha=float(alpha),
        entropy_adjusted_parent_score_gap=qgap-float(alpha*(lp[new]-lp[old])),
        q_reward_sign_conflict=bool(qgap*reward<0 and abs(qgap)>1e-8 and abs(reward)>1e-8),
        cash_reward_sign_conflict=bool(cash*reward<0 and abs(cash)>1e-6 and abs(reward)>1e-8))


def summarize_events(rows):
    active=[r for r in rows if r['outcome']!='neutral']
    return dict(changed_signals=len(rows),effective_changes=len(active),
        helpful=sum(r['outcome']=='helpful' for r in rows),harmful=sum(r['outcome']=='harmful' for r in rows),
        cash_delta_usd=float(sum(r['contribution_usd'] for r in rows)),
        q_reward_conflicts=sum(r['q_reward_sign_conflict'] for r in active),
        cash_reward_conflicts=sum(r['cash_reward_sign_conflict'] for r in active),
        effective_parent_probability_margin_max=max((r['parent_probability_margin'] for r in active),default=None),
        effects={k:sum(r['effect']==k for r in active) for k in sorted({r['effect'] for r in active})})
