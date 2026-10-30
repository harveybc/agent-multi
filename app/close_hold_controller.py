"""Position-conditioned close/hold decisions on declared market reports.

No simulator object, observed future, model state vector or learning is used.
Synthetic future-aware reports require an explicit test-only permission.
"""
from dataclasses import dataclass, asdict
import math
import hashlib

import numpy as np

from app.protection_reference import protection_exit


@dataclass(frozen=True)
class PositionView:
    identity: str
    direction: int
    entry: float
    units: float
    sl: float
    tp: float
    risk: float
    entry_slip: float
    market_slip: float
    stop_slip: float
    commission_roundtrip: float
    limit_through: float


@dataclass(frozen=True)
class PathReport:
    issued_at: int
    observed_through: int
    first_fill: int
    step: int
    deadline: int
    paths: np.ndarray  # [sample, bar, OPEN/HIGH/LOW/CLOSE], market mid prices
    spreads: np.ndarray  # [sample, bar], full bid/ask spread
    weights: np.ndarray
    provenance: str
    terminal_spreads: np.ndarray  # [sample], spread at deadline liquidation


def report_identity(r):
    h=hashlib.sha256(repr((r.issued_at,r.observed_through,r.first_fill,r.step,
                           r.deadline,r.provenance)).encode())
    for a in (r.paths,r.spreads,r.weights,r.terminal_spreads):
        a=np.asarray(a,dtype=float)
        h.update(repr(a.shape).encode());h.update(a.tobytes())
    return h.digest()


@dataclass(frozen=True)
class ValueBinding:
    position_identity: str
    risk: float
    report_digest: bytes
    values_digest: bytes


def bind_values(position,report,values):
    a=np.asarray(values,dtype=float)
    return ValueBinding(position.identity,position.risk,report_identity(report),
                        hashlib.sha256(repr(a.shape).encode()+a.tobytes()).digest())


@dataclass(frozen=True)
class CloseConfig:
    uncertainty_penalty: float = 0.
    margin_risk: float = 0.
    persistence: int = 1
    urgent_risk: float | None = None

    def __post_init__(self):
        if (not all(math.isfinite(v) and v>=0 for v in (self.uncertainty_penalty,self.margin_risk))
                or type(self.persistence) is not int or self.persistence<1
                or (self.urgent_risk is not None and
                    (not math.isfinite(self.urgent_risk) or self.urgent_risk<self.margin_risk))):
            raise ValueError('CLOSE_CONFIG')


def validate(p,r,opening_funding,ending_funding):
    x=np.asarray(r.paths,float); s=np.asarray(r.spreads,float); w=np.asarray(r.weights,float)
    numeric=[p.entry,p.units,p.sl,p.tp,p.risk,p.entry_slip,p.market_slip,
             p.stop_slip,p.commission_roundtrip,p.limit_through]
    if (not p.identity or p.direction not in (-1,1) or not np.isfinite(numeric).all()
            or min(numeric[:5])<=0 or min(numeric[5:])<0 or p.direction*(p.tp-p.sl)<=0):
        raise ValueError('POSITION_CONTRACT')
    if (x.ndim!=3 or x.shape[2]!=4 or min(x.shape[:2])<1 or s.shape!=x.shape[:2]
            or w.shape!=(len(x),) or not np.isfinite(x).all() or not np.isfinite(s).all()
            or not np.isfinite(w).all() or (w<0).any() or not np.isclose(w.sum(),1,atol=1e-12,rtol=0)
            or (s<0).any() or (x-s[:,:,None]/2<=0).any()
            or (x[:,:,1]<np.max(x,axis=2)).any() or (x[:,:,2]>np.min(x,axis=2)).any()):
        raise ValueError('PATH_CONTRACT')
    terminal=np.asarray(r.terminal_spreads,float)
    if (terminal.shape!=(len(x),) or not np.isfinite(terminal).all()
            or (terminal<0).any() or (x[:,-1,3]-terminal/2<=0).any()):
        raise ValueError('TERMINAL_SPREAD_CONTRACT')
    for f in (opening_funding,ending_funding):
        if np.asarray(f).shape!=(x.shape[1],) or not np.isfinite(f).all():
            raise ValueError('KNOWN_FUNDING_SCHEDULE')
    times=(r.issued_at,r.observed_through,r.first_fill,r.step,r.deadline)
    if (not all(isinstance(t,(int,np.integer)) and not isinstance(t,bool) for t in times)
            or r.step<=0 or r.first_fill!=r.issued_at or r.observed_through>r.issued_at
            or r.deadline!=r.first_fill+x.shape[1]*r.step):
        raise ValueError('REPORT_TIME_SUPPORT')


def validate_context(r,decision_time,expected_deadline):
    """Bind report support to the caller's independent clock and calendar."""
    if (not all(isinstance(t,(int,np.integer)) and not isinstance(t,bool)
                for t in (decision_time,expected_deadline)) or expected_deadline<=decision_time):
        raise ValueError('EXTERNAL_CLOCK_CONTRACT')
    if r.issued_at!=decision_time or r.first_fill!=decision_time or r.observed_through>decision_time:
        raise ValueError('STALE_OR_FUTURE_REPORT')
    if r.deadline!=expected_deadline:
        raise ValueError('EXTERNAL_DEADLINE_MISMATCH')
    if (not isinstance(r.step,(int,np.integer)) or isinstance(r.step,bool) or r.step<=0
            or np.ndim(r.paths)!=3 or r.paths.shape[1]<1
            or r.first_fill+r.paths.shape[1]*r.step!=expected_deadline):
        raise ValueError('REPORT_TIME_SUPPORT')


def value_paths(p,r,opening_funding,ending_funding):
    """Sample x common exit choices: each future open, plus passive exit.

Apply first passage separately per sample; the caller takes expectations only
after computing executable outcomes. Funding arrays are per-unit cumulative
charges since original entry, computed from a known calendar, not price paths.
"""
    validate(p,r,opening_funding,ending_funding)
    x=np.asarray(r.paths,float)-p.direction*np.asarray(r.spreads,float)[:,:,None]/2
    H=x.shape[1]
    result=np.empty((len(x),H+1))
    costs=p.entry_slip+p.market_slip+p.commission_roundtrip
    for i,path in enumerate(x):
        at,quote,reason=protection_exit(p.direction,p.tp,p.sl,*path.T,p.limit_through)
        if reason=='DEADLINE':
            quote=float(r.paths[i,-1,3]-p.direction*r.terminal_spreads[i]/2)
        at_open=(reason!='DEADLINE' and
            (p.direction*(path[at,0]-p.sl)<=0 or p.direction*(path[at,0]-p.tp)>=0))
        funding=opening_funding[at] if at_open else ending_funding[at]
        exit_slip=p.stop_slip if reason=='SL' else 0. if reason=='TP' else p.market_slip
        passive=p.units*(p.direction*(quote-p.entry)-p.entry_slip-exit_slip-p.commission_roundtrip-funding)
        result[i,:]=passive
        count=at if at_open else at+1
        result[i,:count]=p.units*(p.direction*(path[:count,0]-p.entry)-costs-np.asarray(opening_funding)[:count])
    return result


class CloseController:
    def __init__(self,config=None,*,allow_synthetic=False):
        self.config=config or CloseConfig()
        self.allow_synthetic=allow_synthetic
        self.position_id=None
        self.last_tick=None
        self.streak=0

    def state(self):
        return dict(config=asdict(self.config),allow_synthetic=self.allow_synthetic,
                    position_id=self.position_id,last_tick=self.last_tick,streak=self.streak)

    def restore(self,state):
        if (state['config']!=asdict(self.config) or state['allow_synthetic']!=self.allow_synthetic
                or type(state['streak']) is not int or state['streak']<0
                or (state['last_tick'] is not None and type(state['last_tick']) is not int)
                or (state['position_id'] is not None and not isinstance(state['position_id'],str))):
            raise ValueError('CONTROLLER_STATE_IDENTITY')
        self.position_id=state['position_id'];self.last_tick=state['last_tick'];self.streak=state['streak']

    def decide(self,p,r,decision_time,opening_funding,ending_funding,*,expected_deadline):
        validate_context(r,decision_time,expected_deadline)
        if r.provenance!='causal' and not (self.allow_synthetic and r.provenance in ('synthetic_ideal','synthetic_noisy')):
            raise ValueError('ORACLE_PLANE_FORBIDDEN')
        if self.position_id==p.identity and self.last_tick is not None and decision_time<=self.last_tick:
            raise ValueError('DUPLICATE_OR_REVERSED_TICK')
        values=value_paths(p,r,opening_funding,ending_funding)
        return self._decide_values(p.identity,decision_time,r.step,values,r.weights,p.risk,r.provenance)

    def decide_synthetic_values(self,identity,decision_time,step,values,weights,risk,*,report,expected_deadline,binding=None):
        """Test-plane cache adapter; never available to a causal/live consumer."""
        if not self.allow_synthetic:
            raise ValueError('SYNTHETIC_CACHE_FORBIDDEN')
        validate_context(report,decision_time,expected_deadline)
        if report.provenance not in ('synthetic_ideal','synthetic_noisy'):
            raise ValueError('SYNTHETIC_CACHE_PROVENANCE')
        values=np.asarray(values,float); weights=np.asarray(weights,float)
        if (not identity or type(step) is not int or step<=0 or type(decision_time) is not int
                or values.ndim!=2 or values.shape[1]<2 or len(values)<1
                or weights.shape!=(len(values),) or not np.isfinite(values).all()
                or not np.isfinite(weights).all() or (weights<0).any()
                or not np.isclose(weights.sum(),1,rtol=0,atol=1e-12)
                or not math.isfinite(risk) or risk<=0):
            raise ValueError('SYNTHETIC_CACHE_CONTRACT')
        if (step!=report.step or values.shape!=(len(report.paths),report.paths.shape[1]+1)
                or not np.array_equal(weights,report.weights)):
            raise ValueError('SYNTHETIC_CACHE_REPORT_SUPPORT')
        digest=hashlib.sha256(repr(values.shape).encode()+values.tobytes()).digest()
        if (not isinstance(binding,ValueBinding) or binding.position_identity!=identity
                or binding.risk!=risk or binding.report_digest!=report_identity(report)
                or binding.values_digest!=digest):
            raise ValueError('SYNTHETIC_VALUE_BINDING')
        if self.position_id==identity and self.last_tick is not None and decision_time<=self.last_tick:
            raise ValueError('DUPLICATE_OR_REVERSED_TICK')
        return self._decide_values(identity,decision_time,step,values,weights,risk,'synthetic_noisy')

    def _decide_values(self,identity,decision_time,step,values,weights,risk,provenance):
        # Each column is the SAME waiting action across all samples, not E(max).
        delta=values[:,[0]]-values[:,1:]
        w=np.asarray(weights,float)
        mean=w@delta
        sd=np.sqrt(w@((delta-mean)**2))
        advantage=float(np.min(mean-self.config.uncertainty_penalty*sd))/risk
        contiguous=self.position_id==identity and self.last_tick is not None and decision_time==self.last_tick+step
        streak=self.streak if contiguous else 0
        qualifies=advantage>max(self.config.margin_risk,1e-12)
        self.streak=streak+1 if qualifies else 0
        self.position_id=identity;self.last_tick=int(decision_time)
        urgent=qualifies and self.config.urgent_risk is not None and advantage>=self.config.urgent_risk
        close=qualifies and (urgent or self.streak>=self.config.persistence)
        return dict(action='CLOSE' if close else 'HOLD',advantage_risk=advantage,
                    expected_close_net=float(w@values[:,0]),
                    best_expected_wait_net=float(np.max(w@values[:,1:])),
                    qualifies=qualifies,streak=self.streak,urgent=urgent,
                    provenance=provenance)
