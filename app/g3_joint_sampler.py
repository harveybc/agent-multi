"""TRAIN-only conditional joint FX innovations; no independent feature noise."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.neighbors import NearestNeighbors


def clock_keys(index):
    elapsed = np.r_[1., np.diff(index.as_unit('ns').asi8)/3.6e12]
    return np.asarray(index.dayofweek)*48 + np.asarray(index.hour)*2 + (elapsed > 1.01)


def innovations(bars, crosses):
    c = bars.CLOSE.to_numpy(float)
    prev = np.r_[np.nan,c[:-1]]
    o,h,l = (bars[n].to_numpy(float) for n in ('OPEN','HIGH','LOW'))
    return np.column_stack([np.log(c/prev),np.log(o/prev),
        np.log(h/np.maximum(o,c)),np.log(np.minimum(o,c)/l),
        np.log(crosses.to_numpy(float)/crosses.shift(1).to_numpy(float))])


def contexts(macro, eur_returns):
    past = pd.Series(eur_returns,index=macro.index).shift(1)
    return np.column_stack([macro.shift(1).to_numpy(float),past.rolling(24).sum(),
                           np.sqrt(past.pow(2).rolling(24).sum())])


class ConditionalJointSampler:
    def __init__(self, neighbors=10, block_rows=4, warmup=256):
        if any(type(v) is not int or v < 1 for v in (neighbors,block_rows,warmup)) or warmup < 25:
            raise ValueError('G3_SAMPLER_CONFIG')
        self.neighbors,self.block_rows,self.warmup = neighbors,block_rows,warmup

    def fit(self,bars,crosses,macro,train_end):
        if not bars.index.equals(crosses.index) or not bars.index.equals(macro.index):
            raise ValueError('G3_SAMPLER_GRID')
        if bars.index.tz is None or not bars.index.is_unique or not bars.index.is_monotonic_increasing:
            raise ValueError('G3_SAMPLER_GRID')
        # Exclude held periods before computing even the sampler's scaling.
        mask = bars.index < train_end
        b,x,m = bars.loc[mask],crosses.loc[mask],macro.loc[mask]
        if len(b) <= self.warmup or x.shape[1] != 6:
            raise ValueError('G3_SAMPLER_TRAIN_SIZE')
        self.y = innovations(b,x)
        self.context = contexts(m,self.y[:,0])
        valid = np.isfinite(self.y).all(axis=1) & np.isfinite(self.context).all(axis=1)
        valid[:self.warmup] = False
        if not valid.any():
            raise ValueError('G3_SAMPLER_NO_DONORS')
        self.mean = self.context[valid].mean(axis=0)
        self.std = self.context[valid].std(axis=0)
        self.std = np.where(self.std > 0,self.std,1.)
        self.keys = clock_keys(b.index)
        self.groups = {}
        for key in np.unique(self.keys[valid]):
            rows = np.flatnonzero(valid & (self.keys == key))
            nn = NearestNeighbors(n_neighbors=min(self.neighbors,len(rows)),algorithm='kd_tree',n_jobs=1)
            nn.fit((self.context[rows]-self.mean)/self.std)
            self.groups[int(key)] = (rows,nn)
        self.valid = valid
        self.fit_index = b.index.copy()
        return self

    def generate(self,bars,crosses,macro,seed):
        if not bars.index.equals(crosses.index) or not bars.index.equals(macro.index):
            raise ValueError('G3_SAMPLER_GRID')
        n = len(bars)
        if n < self.warmup or not bars.index.equals(self.fit_index[:n]):
            raise ValueError('G3_SAMPLER_PREFIX')
        out = bars.copy()
        fx = crosses.copy()
        cross_masks = np.isfinite(crosses.to_numpy(float))
        # The initial joint price level must be observed, never zero-filled.
        if not cross_masks[self.warmup-1].all():
            raise ValueError('G3_SAMPLER_INITIAL_CROSS')
        raw = innovations(bars,crosses)
        sampled = raw.copy()
        donor = np.full(n,-1,dtype=np.int64)
        keys = clock_keys(bars.index)
        exog = macro.shift(1).to_numpy(float)
        rng = np.random.default_rng(seed)
        prices = crosses.to_numpy(float).copy()
        values = out[['OPEN','HIGH','LOW','CLOSE']].to_numpy(float).copy()
        previous_source,run = -1,0
        for t in range(self.warmup,n):
            past = sampled[t-24:t,0]
            context = np.r_[exog[t],past.sum(),np.sqrt(np.square(past).sum())]
            if not np.isfinite(context).all() or int(keys[t]) not in self.groups:
                raise ValueError('G3_SAMPLER_UNSUPPORTED_CONTEXT:'+str(t))
            candidate = previous_source+1
            continuation = (0 < run < self.block_rows and candidate < len(self.y)
                            and self.valid[candidate] and self.keys[candidate] == keys[t])
            if continuation:
                source = candidate
                run += 1
            else:
                rows,nn = self.groups[int(keys[t])]
                near = nn.kneighbors(((context-self.mean)/self.std)[None],return_distance=False)[0]
                source = int(rows[near[int(rng.integers(len(near)))]])
                run = 1
            donor[t] = source
            sampled[t] = self.y[source]
            y = sampled[t]
            op,cl = values[t-1,3]*np.exp(y[1]),values[t-1,3]*np.exp(y[0])
            values[t] = [op,max(op,cl)*np.exp(y[2]),min(op,cl)*np.exp(-y[3]),cl]
            prices[t] = prices[t-1]*np.exp(y[4:])
            previous_source = source
        out.loc[:,['OPEN','HIGH','LOW','CLOSE']] = values
        prices[~cross_masks] = np.nan
        fx.loc[:,:] = prices
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('G3_SAMPLER_PRICE_DOMAIN')
        return out,fx,donor


def distribution_gate(real,synthetic,index):
    """Coarse frozen envelopes from real TRAIN quarters; never inspect validation."""
    if real.shape != synthetic.shape or len(index) != len(real):
        raise ValueError('G3_DISTRIBUTION_SHAPE')
    periods = index.tz_localize(None).to_period('Q')
    quarters = [real[periods == q] for q in periods.unique()]
    quarters = [v[np.isfinite(v).all(axis=1)] for v in quarters]
    quarters = [v for v in quarters if len(v) >= 100]
    if len(quarters) < 4:
        raise ValueError('G3_TOO_FEW_REFERENCE_QUARTERS')
    a = real[np.isfinite(real).all(axis=1)]
    b = synthetic[np.isfinite(synthetic).all(axis=1)]
    out = []
    for j in range(real.shape[1]):
        sd = a[:,j].std()
        distances = [wasserstein_distance(x[:,j],y[:,j])/sd
                     for i,x in enumerate(quarters) for y in quarters[i+1:]]
        reference_std = [x[:,j].std() for x in quarters]
        reference_tail = [np.quantile(abs(x[:,j]),.99) for x in quarters]
        distance = wasserstein_distance(a[:,j],b[:,j])/sd
        std,tail = b[:,j].std(),np.quantile(abs(b[:,j]),.99)
        bounds = dict(wasserstein=1.25*max(distances),std=[min(reference_std)/1.25,max(reference_std)*1.25],
                      tail=[min(reference_tail)/1.25,max(reference_tail)*1.25])
        passed = distance <= bounds['wasserstein'] and bounds['std'][0] <= std <= bounds['std'][1] and bounds['tail'][0] <= tail <= bounds['tail'][1]
        out.append(dict(column=j,passed=bool(passed),wasserstein=float(distance),std=float(std),
                        tail=float(tail),bounds=bounds))
    correlations = [np.corrcoef(x,rowvar=False) for x in quarters]
    bound = 1.25*max(float(np.max(abs(x-y))) for i,x in enumerate(correlations) for y in correlations[i+1:])
    delta = float(np.max(abs(np.corrcoef(a,rowvar=False)-np.corrcoef(b,rowvar=False))))
    return dict(passed=all(r['passed'] for r in out) and delta <= bound,
                columns=out,correlation_max_delta=delta,correlation_bound=bound,
                reference_quarters=len(quarters))
