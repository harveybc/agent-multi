"""Isolated per-direction risk calibration; never modifies trained policies."""
from dataclasses import dataclass, replace
import math

from app.frozen_recipe_inputs import require
from env_plugins.hybrid_orders import HybridOrderEnv


@dataclass(frozen=True)
class DirectionalRisk:
    genes: tuple

    def __post_init__(self):
        require(len(self.genes)==6 and all(math.isfinite(x) and 0<=x<=1 for x in self.genes),
                'DIRECTIONAL_RISK_DOMAIN')

    def for_command(self,cmd):
        require(cmd in (1,2),'OPEN_COMMAND_REQUIRED')
        start=0 if cmd==1 else 3
        return tuple(self.genes[start:start+3])


class DirectionalOrders(HybridOrderEnv):
    def __init__(self,engine,inputs,parameters):
        super().__init__(engine,inputs,learned_risk=False,reward_mode='equity')
        self.parameters=parameters
        self.contract.update(scope='SYNTHETIC_DEAP_ONLY',directional_genes=list(parameters.genes))

    def decode(self,action):
        cmd,tp,sl,label=super().decode(action)
        if cmd in (1,2):
            ft,fs,q=self.parameters.for_command(cmd)
            if q==0:
                return 0,tp,sl,'NOP'
            # Preserve exact midpoint arithmetic, including floating-point rounding.
            if (ft,fs)!=(.5,.5):
                tp,sl=tuple(a+f*(b-a) for (a,b),f in zip(self.feasible_bounds(),(ft,fs)))
            self.engine.cfg=replace(self.engine.cfg,percentual_order_size=q)
        return cmd,tp,sl,label


def fitness_components(waves,capital):
    require(capital>0,'POSITIVE_INITIAL_BALANCE')
    ideals=[r for r in waves if r['mode']=='ideal']
    require(len(ideals)==2,'TWO_CALIBRATION_MARKETS')
    return dict(sharpe=sum(r['metrics']['sharpe_daily_sqrt365'] for r in ideals)/2,
        return_fraction=sum(r['metrics']['net_profit_usd']/capital for r in ideals)/2,
        orders_per_day=sum(r['metrics']['trades']/(5*r['weeks']) for r in ideals)/2)


def objective(name,components,eligible,reasons):
    require(all(math.isfinite(x) for x in components.values()),'FINITE_FITNESS')
    require(name in ('sharpe','sharpe_return','sharpe_return_activity'),'FITNESS_NAME')
    if not eligible:return -1e9-len(reasons)
    result=components['sharpe']
    if name!='sharpe':result+=components['return_fraction']
    if name=='sharpe_return_activity':result+=components['orders_per_day']
    return result
