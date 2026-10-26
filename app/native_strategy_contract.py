"""Isolated execution-parameter contrast; never changes a frozen campaign."""
from dataclasses import dataclass, replace, asdict
import math
from numbers import Integral, Real

from env_plugins.hybrid_orders import HybridOrderEnv


@dataclass(frozen=True)
class RiskParameters:
    tp_fraction: float = 0.5
    sl_fraction: float = 0.5
    order_fraction: float = 1.0

    def __post_init__(self):
        if not (all(math.isfinite(x) for x in asdict(self).values())
                and 0 < self.tp_fraction < 1 and 0 < self.sl_fraction < 1
                and 0 < self.order_fraction <= 1):
            raise ValueError('EXECUTION_PARAMETER_DOMAIN')

    def levels(self, bounds):
        return tuple(a + f * (b-a) for (a,b),f in
                     zip(bounds,(self.tp_fraction,self.sl_fraction)))


class CalibratedOrders(HybridOrderEnv):
    def __init__(self, engine, inputs, parameters=RiskParameters(), gamma=.99):
        super().__init__(engine,inputs,gamma=gamma,learned_risk=False,
                         reward_mode='equity',penalty=0.)
        self.parameters=parameters
        self.contract.update(execution_parameters=asdict(parameters),
                             scope='ISOLATED_PARAMETER_CONTRAST')

    def decode(self, action):
        cmd,tp,sl,label=super().decode(action)
        if cmd in (1,2):
            # Retain the original arithmetic for the exact midpoint control.
            if (self.parameters.tp_fraction,self.parameters.sl_fraction)!=(.5,.5):
                tp,sl=self.parameters.levels(self.feasible_bounds())
            self.engine.cfg=replace(self.engine.cfg,percentual_order_size=self.parameters.order_fraction)
        return cmd,tp,sl,label


def executed_behavior(trades,counts,report):
    activity = report.get('orders_per_week')
    if (isinstance(activity, bool) or not isinstance(activity, Real)
            or not math.isfinite(activity) or activity < 0):
        raise ValueError('INVALID_ACTIVITY_MEASUREMENT')
    if type(report.get('valid')) is not bool:
        raise ValueError('INVALID_NATIVE_VALIDITY')
    for name, count in counts.items():
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ValueError('INVALID_ACTION_COUNT:' + name)
    reasons=[]
    for side,name in ((1,'LONG'),(-1,'SHORT')):
        if not any(t['direction']==side for t in trades): reasons.append('NO_'+name+'_TRADES')
        if not any(t['direction']==side and t['close_reason']=='AGENT_CLOSE' for t in trades):
            reasons.append('NO_EXECUTED_CLOSE_'+name)
    for name in ('NOP','HOLD_LONG','HOLD_SHORT'):
        if not counts.get(name,0): reasons.append('NO_'+name)
    if activity<2: reasons.append('ACTIVITY_BELOW_TWO')
    if not report['valid']: reasons.append('NATIVE_INVALID')
    return dict(eligible=not reasons,reasons=reasons)
