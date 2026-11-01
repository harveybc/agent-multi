"""Independent stopping decisions over a shared, unchanged learning trajectory."""
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ShadowRule:
    name: str
    patience: int
    minimum: int
    delta: float
    guarded: bool
    best: float | None = None
    validation: float | None = None
    selected_step: int | None = None
    bad: int = 0
    inactive: int = 0
    stopped: bool = False
    stop_step: int | None = None
    reason: str | None = None

    def state(self):
        return asdict(self)

    def observe(self, report, step):
        if self.stopped:
            return dict(improved=False,state=self.state())
        score=report['score']
        validation=float(np.mean([r['return_per_week'] for r in report['internal_validation']]))
        if not np.isfinite([score,validation]).all(): raise ValueError('SHADOW_NONFINITE')
        acceptable=(not self.guarded or self.validation is None or validation>=self.validation-self.delta)
        improved=bool(report['eligible'] and acceptable and (self.best is None or score>self.best+self.delta))
        if improved:
            self.best,self.validation,self.selected_step,self.bad=score,validation,step,0
        elif step>=self.minimum and self.best is not None:
            self.bad+=1
        if step>=self.minimum and self.best is None: self.inactive+=1
        else: self.inactive=0
        if step>=self.minimum and max(self.bad,self.inactive)>=self.patience:
            self.stopped,self.stop_step=True,step
            self.reason='no_eligible_checkpoint' if self.best is None else 'patience'
        return dict(improved=improved,state=self.state())


def make_rules(design):
    return [ShadowRule(f'{mode}_p{p}',p,design['minimum_steps'],design['min_delta'],mode=='guarded')
            for mode in ('original','guarded') for p in (3,5,10)]


def observe_rules(rules, report, step):
    decisions={r.name:r.observe(report,step) for r in rules}
    return dict(stop=all(r.stopped for r in rules),stop_reason='all_shadow_rules_stopped',rules=decisions)
