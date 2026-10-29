"""Descriptive TRAIN oracle risk references, not executable order levels."""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.zigzag_opportunities import HOUR, PIP


def describe(values):
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError('NONFINITE_OR_NONVECTOR_SUMMARY')
    names = ('min', 'p10', 'p25', 'median', 'p75', 'p90', 'p95', 'max')
    if not len(x):
        return dict(n=0, mean=None, std=None, **dict.fromkeys(names))
    return dict(n=len(x), mean=float(x.mean()), std=float(x.std(ddof=0)),
                **dict(zip(names, map(float, np.quantile(x, [0, .1, .25, .5, .75, .9, .95, 1])))))


def verify_paths(tape, records):
    """Independently reconstruct every complete selected directional path."""
    r, t = records, tape
    origins = r['origin']
    if not np.array_equal(origins, np.flatnonzero(t.allowed)):
        raise ValueError('ORIGIN_SUPPORT')
    n = len(origins)
    if r['pair_valid'].shape != (n,) or r['pair_valid'].dtype != np.bool_:
        raise ValueError('PAIR_MASK')
    valid = r['pair_valid']
    keys = ('score', 'net', 'duration', 'target', 'mae', 'drawdown')
    for key in keys:
        if (r[key].shape != (2, n) or not np.isfinite(r[key][:, valid]).all()
                or not np.isnan(r[key][:, ~valid]).all()):
            raise ValueError('RECORD_VALUES:' + key)
    for key in ('end', 'release'):
        if (r[key].shape != (2, n) or not np.issubdtype(r[key].dtype, np.integer)
                or not np.all(r[key][:, ~valid] == -1)):
            raise ValueError('RECORD_INDICES:' + key)
    if r['candidate_end'].ndim != 2 or r['candidate_end'].shape[0] != n:
        raise ValueError('EVENT_SUPPORT')
    worst = 0.
    paths = 0
    for j in np.flatnonzero(valid):
        i = int(origins[j])
        for side, sign in enumerate((1, -1)):
            end = int(r['end'][side, j])
            if (not i < end < len(t.open) or end not in r['candidate_end'][j]
                    or r['release'][side, j] != end + 1
                    or not np.all(np.diff(t.times[i:end+1]) == HOUR)
                    or not np.all(t.session[i:end+1] == t.session[i])):
                raise ValueError('PATH_SUPPORT')
            target = r['target'][side, j]
            if target not in (t.high[end], t.low[end]):
                raise ValueError('PIVOT_PRICE')
            favorable = sign * (t.high[i:end+1] if side == 0 else t.low[i:end+1])
            adverse = sign * (t.low[i:end+1] if side == 0 else t.high[i:end+1])
            entry = sign * t.open[i]
            peak = np.maximum.accumulate(np.maximum(entry, favorable))
            dd = max(0., float((peak-adverse).max()), peak[-1]-sign*target)
            mae = max(0., entry-float(adverse.min()), entry-sign*target)
            cost = t.fill_cost[i]+t.fill_cost[end]
            net = sign*(target-t.open[i])-cost-(t.financing[side, end]-t.financing[side, i])
            duration = (t.times[end]-t.times[i])/HOUR + 1
            expected = dict(mae=mae, drawdown=dd, net=net, duration=duration,
                            score=net/(max(dd, cost, PIP)*duration))
            for key, value in expected.items():
                worst = max(worst, abs(float(r[key][side, j])-value))
                if not np.isclose(r[key][side, j], value, rtol=1e-12, atol=1e-12):
                    raise ValueError('PATH_VALUE:' + key)
            paths += 1
    return dict(paths=paths, max_abs_error=float(worst))


def profile(tape, records):
    r = records
    origins, valid = r['origin'], r['pair_valid']
    scores = np.where(valid[None, :], r['score'], -np.inf)
    winner = np.argmax(scores, axis=0)  # Same tie rule as the authenticated source.
    years = pd.DatetimeIndex(tape.times[origins], tz='UTC').year.to_numpy()
    ties = valid & (scores[0] > 0) & (scores[0] == scores[1])
    out = dict(support=dict(origins=len(origins), complete=int(valid.sum()),
                           censored=int((~valid).sum()), positive_exact_ties=int(ties.sum()),
                           complete_nop=int((valid & (scores.max(axis=0) <= 0)).sum())),
               directions={})
    for side, (name, sign) in enumerate((('buy', 1), ('sell', -1))):
        calendar = valid & (r['release'][side] <= tape.limit[origins])
        positive = valid & (scores[side] > 0)
        chosen = positive & (winner == side)
        gross = sign*(r['target'][side]-tape.open[origins])/PIP
        metrics = dict(tp_pips=gross, mae_pips=r['mae'][side]/PIP,
                       drawdown_pips=r['drawdown'][side]/PIP,
                       net_pips=r['net'][side]/PIP, duration_hours=r['duration'][side])

        def summarize(mask):
            a, b = gross[mask], metrics['mae_pips'][mask]
            corr = float(np.corrcoef(a, b)[0, 1]) if len(a)>1 and a.std()>0 and b.std()>0 else None
            return dict(n=int(mask.sum()), zero_mae=int((b == 0).sum()),
                        tp_mae_correlation=corr,
                        **{key: describe(x[mask]) for key, x in metrics.items()})

        masks = dict(oracle_all=valid, oracle_positive=positive, oracle_winners=chosen,
                     calendar_all=calendar, positive_calendar=positive & calendar,
                     winners_calendar=chosen & calendar)
        out['directions'][name] = {key: summarize(mask) for key, mask in masks.items()}
        out['directions'][name]['positive_excluded_by_calendar'] = int((positive & ~calendar).sum())
        out['directions'][name]['positive_calendar_by_year'] = {
            str(year): summarize(positive & calendar & (years == year)) for year in sorted(set(years))}
    return out


def size_from_risk(balance, rel_volume, risk_budget, loss_at_q1, quality, *, minimum_q=0.):
    """Linear monetary-loss budget; callers must supply broker-consistent inputs.

    Not a broker sizing adapter. Fixed ticket fees, discrete lots and nonlinear
    loss functions require a separate native execution calculation.
    """
    values = (balance, rel_volume, risk_budget, loss_at_q1, quality, minimum_q)
    if (not np.isfinite(values).all() or balance <= 0 or not 0 < rel_volume <= 1
            or risk_budget < 0 or loss_at_q1 <= 0 or not 0 <= quality <= 1
            or not 0 <= minimum_q <= 1):
        raise ValueError('INVALID_SIZING_INPUT')
    q = quality * min(1., risk_budget/loss_at_q1)
    reason = 'RISK_BOUNDED' if q > 0 else 'ZERO_BUDGET_OR_QUALITY'
    if 0 < q < minimum_q:
        q, reason = 0., 'BELOW_BROKER_MINIMUM'
    return dict(q=q, margin=q*rel_volume*balance,
                estimated_loss=q*loss_at_q1, reason=reason)
