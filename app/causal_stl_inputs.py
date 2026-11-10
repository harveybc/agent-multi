"""Trailing-endpoint STL with frozen TRAIN parameters and online parity."""
from collections import deque
import numpy as np
from statsmodels.tsa.seasonal import STL


def candidates():
    return [dict(period=p,seasonal=s,robust=r,history=max(64,4*p))
            for p in (6,12,24,48,120) for s in (7,13) for r in (False,True)]


def fit_history(x,config):
    return STL(np.asarray(x,dtype=float),period=config['period'],seasonal=config['seasonal'],
               robust=config['robust'],seasonal_deg=1,trend_deg=1,low_pass_deg=1,
               seasonal_jump=1,trend_jump=1,low_pass_jump=1).fit()


class CausalSTLColumn:
    def __init__(self,config,history=()):
        self.config=dict(config)
        if (config['period']<2 or config['seasonal']<3 or config['seasonal']%2!=1
                or config['history']<2*config['period']):raise ValueError('STL_PARAMETERS')
        self._history=deque((float(x) for x in history),maxlen=config['history'])

    @property
    def history(self):return list(self._history)

    def update(self,value):
        self._history.append(float(value))
        if len(self._history)<self.config['history']:return float('nan')
        x=np.asarray(self._history)
        if not np.isfinite(x).all():return float('nan')
        return float(fit_history(x,self.config).seasonal[-1])


def endpoint_series(values,config):
    stream=CausalSTLColumn(config)
    return np.array([stream.update(v) for v in values])


def choose_parameters(values,train_end,candidates=None,origins_count=16):
    grid=globals()['candidates']() if candidates is None else candidates
    x=np.asarray(values,dtype=float)[:train_end];context=max(c['history'] for c in grid);horizon=24
    finite=np.isfinite(x);invalid=np.r_[0,np.cumsum(~finite)]
    eligible=[t for t in range(context-1,len(x)-horizon)
              if invalid[t+horizon+1]==invalid[t-context+1]]
    if len(eligible)<origins_count:raise ValueError('STL_NO_ELIGIBLE_TRAIN_ORIGINS')
    origins=np.array(eligible)[np.linspace(0,len(eligible)-1,origins_count,dtype=int)]
    scores=[]
    for config in grid:
        losses=[];base=[];naive=[];strength=[];periodicity=[]
        for t in origins:
            history=x[t-config['history']+1:t+1];fit=fit_history(history,config);p=config['period']
            slope=np.polyfit(np.arange(p),fit.trend[-p:],1)[0]
            trend=fit.trend[-1]+slope*np.arange(1,horizon+1)
            forecast=trend+np.resize(fit.seasonal[-p:],horizon)
            target=x[t+1:t+horizon+1]
            common=x[t-context+1:t+1]
            scale=max(float(np.abs(np.diff(common)).mean()),np.finfo(float).eps*max(1.,float(np.abs(common).max())))
            losses.append(float(np.abs(forecast-target).mean()/scale))
            base.append(float(np.abs(trend-target).mean()/scale))
            naive.append(float(np.abs(x[t]-target).mean()/scale))
            v=float(np.var(fit.seasonal+fit.resid))
            strength.append(max(0.,1-float(np.var(fit.resid))/v) if v>0 else 0.)
            a,b=fit.resid[p:],fit.resid[:-p]
            periodicity.append(float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-15 and np.std(b)>1e-15 else 0.)
        scores.append(dict(parameters=config,score=float(np.mean(losses)),trend_only=float(np.mean(base)),
            persistence=float(np.mean(naive)),seasonal_strength=float(np.mean(strength)),
            residual_period_correlation=float(np.mean(periodicity)),origin_losses=losses))
    best=min(scores,key=lambda s:(s['score'],s['parameters']['period'],s['parameters']['seasonal'],s['parameters']['robust']))
    return dict(selected=best,scored_candidates=len(scores),origins=origins.tolist(),
                scores=scores,training_rows=len(x),forecast_horizon=horizon)


def encode_seasonal(values,train_mask,parameters=None):
    x=np.asarray(values,dtype=float);mask=np.asarray(train_mask,dtype=bool)
    if x.ndim!=2 or mask.shape!=(len(x),):raise ValueError('STL_ENCODING_SHAPE')
    if parameters is None:
        mean=[];std=[];cuts=[]
        for j in range(x.shape[1]):
            v=x[mask & np.isfinite(x[:,j]),j]
            if not len(v):raise ValueError('STL_NO_TRAIN_SCALER')
            mean.append(float(v.mean()));s=float(v.std());std.append(s if s>0 else 1.)
            cuts.append(np.quantile(v,[.2,.4,.6,.8]).tolist())
        parameters=dict(mean=mean,std=std,cuts=cuts,binary_threshold=0.,fit='TRAIN only')
    continuous=(x-np.array(parameters['mean']))/np.array(parameters['std'])
    ordinal=np.column_stack([np.searchsorted(parameters['cuts'][j],x[:,j],side='right')*.5-1
                            for j in range(x.shape[1])])
    binary=(x>parameters['binary_threshold']).astype(float)
    ordinal[~np.isfinite(x)]=np.nan;binary[~np.isfinite(x)]=np.nan
    return dict(continuous=continuous,ordinal=ordinal,binary=binary,parameters=parameters)


class OnlineSTLInputs:
    """Consumes already encoded real features, emits one transformed closed bar."""
    def __init__(self,parameters,encoding,variant,history=None):
        if variant not in ('continuous','ordinal','binary'):raise ValueError('STL_VARIANT')
        self.streams=[CausalSTLColumn(p,() if history is None else history[j]) for j,p in enumerate(parameters)]
        self.encoding=encoding;self.variant=variant

    def update(self,encoded_row):
        if len(encoded_row)!=len(self.streams):raise ValueError('STL_LIVE_COLUMNS')
        x=np.array([[s.update(v) for s,v in zip(self.streams,encoded_row)]])
        return encode_seasonal(x,np.zeros(1,dtype=bool),self.encoding)[self.variant][0]

    def state(self):return [s.history for s in self.streams]
