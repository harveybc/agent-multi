"""Read-only, continuous-account and replay diagnostics; no model selection."""
from contextlib import contextmanager

import numpy as np
import pandas as pd


def summary(values):
    x = np.asarray(values, dtype=float).ravel()
    finite = x[np.isfinite(x)]
    out = dict(n=int(x.size), nonfinite=int(x.size - finite.size))
    if finite.size:
        out.update(dict(zip(('min', 'p05', 'median', 'p95', 'p99', 'max'),
                            map(float, np.quantile(finite, [0, .05, .5, .95, .99, 1])))))
        out.update(mean=float(finite.mean()), std=float(finite.std()))
    return out


def ring_timeline(total, capacity, decisions, cursor):
    if min(capacity, decisions) < 1 or total < 0:
        raise ValueError('REPLAY_DIMENSIONS')
    if cursor != total % capacity:
        raise ValueError('REPLAY_CURSOR')
    g = np.arange(max(0, total - capacity), total, dtype=np.int64)
    return dict(slots=g % capacity, decision=g % decisions,
                epoch=g // decisions + 1, age=total - 1 - g)


def verify_replay_windows(stored, windows, rows, timeline, chunk=256):
    """Fail closed before assigning calendar dates to a legacy replay ring."""
    for first in range(0, len(timeline['slots']), chunk):
        slots = timeline['slots'][first:first + chunk]
        ix = rows[timeline['decision'][first:first + chunk]]
        actual = stored[slots]
        if hasattr(actual, 'numpy'):
            actual = actual.numpy()
        if not np.array_equal(actual, windows[ix]):
            raise ValueError('REPLAY_WINDOW_MAPPING')
    return len(timeline['slots'])


def quarter_metrics(timestamps, equity, trades):
    ts = pd.to_datetime(timestamps, utc=True)
    eq = np.asarray(equity, dtype=float)
    if len(ts) != len(eq) or len(eq) == 0 or not ts.is_monotonic_increasing:
        raise ValueError('ACCOUNT_TIMELINE')
    if not np.isfinite(eq).all() or (eq <= 0).any():
        raise ValueError('ACCOUNT_EQUITY')
    quarters = ts.tz_localize(None).to_period('Q').astype(str)
    exits = [str(pd.to_datetime(t['exit_time'], utc=True).tz_localize(None).to_period('Q')) for t in trades]
    out = []
    for quarter in dict.fromkeys(quarters):
        ix = np.flatnonzero(quarters == quarter)
        start = max(0, ix[0] - 1)
        path = eq[start:ix[-1] + 1]
        peak = np.maximum.accumulate(path)
        closed = [t for t, q in zip(trades, exits) if q == quarter]
        daily = pd.Series(eq[start:ix[-1]+1], index=ts[start:ix[-1]+1]).resample('1D').last().ffill()
        ret = daily.pct_change().dropna().to_numpy()
        sd = ret.std(ddof=1) if len(ret) > 1 else 0
        out.append(dict(quarter=quarter, entering_equity_usd=float(path[0]),
            ending_equity_usd=float(path[-1]), equity_change_usd=float(path[-1] - path[0]),
            return_pct=float(100 * (path[-1] / path[0] - 1)),
            max_drawdown_pct=float(100 * np.max((peak - path) / peak)),
            sharpe=float(np.sqrt(365)*ret.mean()/sd) if sd > 0 else None,
            trades_closed=len(closed), buy=sum(t['direction'] == 1 for t in closed),
            sell=sum(t['direction'] == -1 for t in closed),
            win_rate=sum(t['net_usd'] > 0 for t in closed)/len(closed) if closed else None,
            early_close=sum(t['close_reason'] == 'AGENT_CLOSE' for t in closed),
            realized_pnl_exit_attributed=float(sum(t['net_usd'] for t in closed))))
    return out


def context_coverage(times, values, base, observations, stride):
    times, values, base = np.asarray(times), np.asarray(values), np.asarray(base)
    if observations < 1 or stride < 1 or len(times) != len(values) or len(base) != len(times):
        raise ValueError('CONTEXT_SHAPE')
    rows = np.flatnonzero(base)
    offsets = np.arange(observations - 1, -1, -1) * stride
    enough = rows >= offsets[0]
    ix = rows[enough, None] - offsets
    finite = np.isfinite(values[ix]).all(axis=(1, 2))
    elapsed = (times[ix[:, -1]] - times[ix[:, 0]]) / (3600 * 10**9)
    return dict(baseline_rows=len(rows), available_rows=int(finite.sum()),
        unavailable_rows=int(len(rows) - finite.sum()),
        insufficient_history=int((~enough).sum()), nonfinite_context=int((~finite).sum()),
        elapsed_hours=summary(elapsed[finite]), observations=observations, stride_rows=stride,
        row_unit='observed admitted bars; closures are not interpolated', baseline_unchanged=True)


def soft_td_target(reward, terminated, duration, next_probability, next_q, alpha, gamma):
    import torch
    p = next_probability
    v = (p * (next_q.min(0).values - alpha * p.clamp_min(1e-30).log())).sum(-1)
    return reward.reshape(-1) + torch.pow(gamma, duration.reshape(-1)) * (~terminated.reshape(-1)) * v


@contextmanager
def frozen_probe(module):
    """Restore RNG, parameters, buffers and mixed train/eval modes after a probe."""
    import torch
    state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    modes = [(m, m.training) for m in module.modules()]
    with torch.random.fork_rng(devices=[]):
        try:
            yield
        finally:
            module.load_state_dict(state)
            for item, mode in modes:
                item.training = mode
