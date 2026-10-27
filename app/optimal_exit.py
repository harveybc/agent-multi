"""Finite native exit-menu targets. Future-aware labels, never observations."""
import numpy as np


def targets(close_net, terminal_net, risk_usd):
    close=np.asarray(close_net,dtype=float)
    if close.ndim!=1 or not len(close) or not np.isfinite(close).all():
        raise ValueError('FINITE_NONEMPTY_EXECUTABLE_CLOSE_MENU')
    if not np.isfinite(terminal_net) or not np.isfinite(risk_usd) or risk_usd<=0:
        raise ValueError('FINITE_TERMINAL_AND_POSITIVE_INITIAL_RISK')
    hold=np.maximum.accumulate(np.r_[close[1:],terminal_net][::-1])[::-1]
    value=np.maximum(close,hold)
    best=float(value[0])
    return dict(close_net=close,hold_value=hold,value=value,best_net=best,
        close_advantage=close-hold,regret_close=(best-close)/risk_usd,
        regret_terminal=(best-terminal_net)/risk_usd,
        reward_close=(close-value)/risk_usd,reward_hold=(hold-value)/risk_usd,
        # Earliest executable exit wins a true equality. No 72% quota.
        oracle_close=close>=hold)


def outcome(case, actions):
    """First CLOSE wins; otherwise native automatic terminal. Report global regret."""
    t=targets(case['close_net'],case['terminal_net'],case['risk_usd'])
    actions=np.asarray(actions)
    if actions.shape!=t['close_net'].shape or not np.isin(actions,[0,1]).all():
        raise ValueError('BINARY_CLOSE_MENU_ACTIONS')
    ix=np.flatnonzero(actions)
    at=int(ix[0]) if len(ix) else None
    net=float(t['close_net'][at]) if at is not None else float(case['terminal_net'])
    return dict(exit_index=at,net_usd=net,optimal_net_usd=t['best_net'],
        regret_usd=t['best_net']-net,regret_risk=(t['best_net']-net)/case['risk_usd'],
        early_close=at is not None,direction=case['direction'])
