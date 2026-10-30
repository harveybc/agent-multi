"""Controller-side forecast ledger. Only timestamp-matured errors are observable.

Intervals are empirical rolling residual diagnostics, not a claim of conformal
coverage under dependent markets. This never normalizes or clips model inputs.
"""
import copy
import hashlib
import json
import math

import numpy as np

from app.close_hold_controller import report_identity, validate_context


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class ReportTrackRecord:
    def __init__(self,producer,*,horizons=(1,3,6,12,24),window=64,minimum=20,
                 nominal=.9,scale_floor=.00001):
        if (not producer or not horizons or any(type(h) is not int or h<1 for h in horizons)
                or len(set(horizons))!=len(horizons) or not 1<=minimum<=window
                or not 0<nominal<1 or not math.isfinite(scale_floor) or scale_floor<=0):
            raise ValueError('TRACK_RECORD_CONFIG')
        self.config=dict(producer=producer,horizons=list(horizons),window=window,
                         minimum=minimum,nominal=nominal,scale_floor=scale_floor)
        self.clock=None;self.last_issue=None;self.last_observation=None
        self.pending=[];self.errors=[];self.issues=[];self.missing_targets=0

    def _clock(self,now):
        if type(now) is not int or (self.clock is not None and now<self.clock):
            raise ValueError('TRACK_RECORD_CLOCK')

    def issue(self,report,*,now,expected_deadline):
        self._clock(now)
        validate_context(report,now,expected_deadline)
        if report.provenance!='causal':
            raise ValueError('CAUSAL_REPORT_REQUIRED')
        if self.last_issue==now:
            raise ValueError('DUPLICATE_REPORT_ISSUE')
        paths=np.asarray(report.paths,float);weights=np.asarray(report.weights,float)
        if (paths.ndim!=3 or paths.shape[2]!=4 or not np.isfinite(paths).all()
                or weights.shape!=(len(paths),) or not np.isfinite(weights).all()
                or (weights<0).any() or not np.isclose(weights.sum(),1,atol=1e-12,rtol=0)):
            raise ValueError('TRACK_RECORD_PATHS')
        rid=digest([self.config['producer'],report_identity(report).hex()])
        points=[]
        for h in self.config['horizons']:
            if h>paths.shape[1]:
                continue
            values=paths[:,h-1,3];mean=float(weights@values)
            sd=float(np.sqrt(weights@((values-mean)**2)))
            order=np.argsort(values);cdf=np.cumsum(weights[order])
            alpha=(1-self.config['nominal'])/2
            limits=[float(values[order[min(len(values)-1,int(np.searchsorted(cdf,q)))]])
                    for q in (alpha,1-alpha)]
            points.append(dict(report=rid,issued=now,available=int(report.first_fill+h*report.step),
                               horizon=h,mean=mean,sd=sd,band=limits))
        self.pending.extend(points)
        self.issues.append(dict(time=now,report=rid))
        self.clock=now;self.last_issue=now
        return rid

    def observe(self,available_at,value,*,now):
        self._clock(now)
        if (type(available_at) is not int or available_at!=now or not math.isfinite(value)
                or value<=0 or (self.last_observation is not None and available_at<=self.last_observation)):
            raise ValueError('OBSERVATION_CLOCK_OR_VALUE')
        remaining=[]
        for point in self.pending:
            if point['available']<available_at:
                self.missing_targets+=1
            elif point['available']==available_at:
                error=float(value-point['mean'])
                self.errors.append(dict(**point,observed=float(value),error=error,
                    z=abs(error)/max(point['sd'],self.config['scale_floor']),
                    covered=bool(point['band'][0]<=value<=point['band'][1])))
            else:
                remaining.append(point)
        self.pending=remaining;self.clock=now;self.last_observation=available_at

    def _history(self,horizon):
        return [r for r in self.errors if r['horizon']==horizon][-self.config['window']:]

    def diagnostic(self,horizon):
        rows=self._history(horizon)
        return dict(count=len(rows),ready=len(rows)>=self.config['minimum'],
            mae=float(np.mean([abs(r['error']) for r in rows])) if rows else None,
            median_z=float(np.median([r['z'] for r in rows])) if rows else None,
            coverage=float(np.mean([r['covered'] for r in rows])) if rows else None)

    def interval(self,center,horizon,*,now):
        self._clock(now)
        if not math.isfinite(center):
            raise ValueError('INTERVAL_CENTER')
        rows=self._history(horizon)
        if len(rows)<self.config['minimum']:
            return None
        errors=sorted(abs(r['error']) for r in rows)
        rank=min(len(errors),math.ceil((len(errors)+1)*self.config['nominal']))
        width=errors[rank-1]
        return [float(center-width),float(center+width)]

    def state(self):
        data=copy.deepcopy(dict(config=self.config,clock=self.clock,last_issue=self.last_issue,
            last_observation=self.last_observation,pending=self.pending,errors=self.errors,
            issues=self.issues,missing_targets=self.missing_targets))
        return dict(data=data,seal=digest(data))

    @classmethod
    def restore(cls,state,*,expected_producer):
        data=copy.deepcopy(state['data'])
        if digest(data)!=state['seal']:
            raise ValueError('TRACK_RECORD_SEAL')
        if data['config']['producer']!=expected_producer:
            raise ValueError('TRACK_RECORD_PRODUCER')
        cfg=data.pop('config');producer=cfg.pop('producer')
        obj=cls(producer,**cfg)
        for key,value in data.items():
            setattr(obj,key,value)
        return obj
