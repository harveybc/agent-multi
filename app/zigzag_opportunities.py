"""TRAIN-only hindsight opportunity maps; no account simulation or RL inputs."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HOUR = 3600 * 10**9
PIP = 1e-4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine(root):
    import app
    path = str(Path(root).resolve() / 'app')
    if path not in app.__path__:
        app.__path__.append(path)
    from app.short_horizon import zigzag, calendar, execution
    return zigzag, calendar, execution


def contiguous_bounds(times, session):
    cuts = np.flatnonzero((np.diff(times) != HOUR) | (np.diff(session) != 0)) + 1
    return list(zip(np.r_[0, cuts], np.r_[cuts, len(times)]))


@dataclass
class Tape:
    times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    session: np.ndarray
    allowed: np.ndarray
    limit: np.ndarray
    fill_cost: np.ndarray
    financing: np.ndarray
    chunks: list
    metadata: dict


def load_tape(spec):
    paths = spec['inputs']
    for key in ('bars', 'closures', 'recipe'):
        if sha(paths[key]) != spec['input_sha256'][key]:
            raise ValueError('INPUT_HASH:' + key)
    rec = json.loads(Path(paths['recipe']).read_text())
    calpin = rec['business']['calendar']
    if (sha(paths['bars']) != calpin['admitted_sha256'] or
            sha(paths['closures']) != calpin['closures_sha256']):
        raise ValueError('RECIPE_DATA_IDENTITY')
    _, cal, execution = engine(spec['engine_root'])
    start, end = [pd.Timestamp(x, tz='UTC') for x in spec['train']]
    if spec['train'] != ['2020-01-01', '2024-01-01']:
        raise ValueError('TRAIN_ONLY_CONTRACT')
    f = pd.read_parquet(paths['bars'], filters=[('available_at', '>=', start), ('available_at', '<', end)])
    ts = pd.DatetimeIndex(f.timestamp).as_unit('ns')
    av = pd.DatetimeIndex(f.available_at).as_unit('ns')
    x = f[['BidOpen', 'BidHigh', 'BidLow', 'BidClose']].to_numpy(float)
    if (not len(f) or ts.tz is None or av.tz is None or not np.all(np.diff(ts.asi8) > 0)
            or not np.array_equal(av.asi8, ts.asi8 + HOUR)
            or not ((av >= start) & (av < end)).all()
            or not np.isfinite(x).all() or not (x > 0).all()
            or not (x[:, 1] >= x.max(axis=1)).all() or not (x[:, 2] <= x.min(axis=1)).all()):
        raise ValueError('BAR_CONTRACT')
    closures = json.loads(Path(paths['closures']).read_text())
    segments = [{'segment_id': str(i), 'market_open': closures[i][1], 'market_close': closures[i+1][0]}
                for i in range(len(closures)-1) if pd.Timestamp(closures[i][1]) < pd.Timestamp(closures[i+1][0])]
    c = cal.load_calendar(dict(calendar_id='ADMITTED_12H_12H', version='1', segments=segments,
                              entry_delay_after_open_hours=12, force_flat_before_close_hours=12,
                              entry_cutoff_before_flat_hours=0))
    session = c.assign(ts)
    profile = execution.RealisticProfile(**rec['business']['evaluation_and_policy']['profile'])
    n = len(f)
    allowed = np.zeros(n, bool)
    limit = np.zeros(n, int)
    fill_cost = np.zeros(n)
    financing = np.zeros((2, n))
    chunks = contiguous_bounds(ts.asi8, session)
    for a, b in chunks:
        seg = c.segments[session[a]]
        t = ts.asi8[a:b]
        allowed[a:b] = (t >= seg.operational_start.value) & (t < seg.force_flat.value)
        # Last observed quote at/before flat is used conservatively when flat has no quote.
        last = min(b-1, int(np.searchsorted(ts.asi8, seg.force_flat.value, side='right'))-1)
        limit[a:b] = last
        allowed[a:b] &= np.arange(a, b) < last
        for i in range(a, b):
            fill_cost[i] = (profile.spread_at(int(ts.asi8[i]), seg.market_open.value)/2
                            + profile.slippage_market + profile.commission_per_lot_side/profile.lot_units)
        for side, rate in enumerate((profile.swap_long_per_lot, profile.swap_short_per_lot)):
            for k, roll in enumerate(seg.rollovers(), start=1):
                financing[side, a:b] += (t >= roll.value) * (-rate/profile.lot_units) * (3 if k == profile.triple_swap_rollover_index else 1)
    if not allowed.any():
        raise ValueError('NO_ELIGIBLE_ORIGINS')
    metadata = dict(rows=n, origins=int(allowed.sum()), chunks=len(chunks),
                    years={str(y): int((av.year == y).sum()) for y in sorted(set(av.year))},
                    excluded_origins=int((~allowed).sum()), start=str(av.min()), end=str(av.max()),
                    cost_profile=rec['business']['evaluation_and_policy']['profile'],
                    costs_status='ASSUMED_NOT_BROKER_VERIFIED',
                    support_sha256=hashlib.sha256(ts.asi8[allowed].tobytes()).hexdigest())
    if set(metadata['years']) != {'2020', '2021', '2022', '2023'}:
        raise ValueError('TRAIN_YEAR_COVERAGE')
    return Tape(ts.asi8, x[:, 0], x[:, 1], x[:, 2], session, allowed, limit,
                fill_cost, financing, chunks, metadata)


class OpportunityEvaluator:
    def __init__(self, tape, zigzag, horizon):
        self.tape, self.z, self.h = tape, zigzag, horizon
        self.origins = np.flatnonzero(tape.allowed)
        self.max_steps = 4*horizon
        self.precompute()

    def precompute(self):
        t = self.tape
        n, k = len(t.open), self.max_steps
        # Shared per-origin path summaries avoid repeating bar scans for each genome.
        self.mae = np.zeros((2, n, k+1), np.float64)
        self.dd = np.zeros_like(self.mae)
        self.best = np.zeros_like(self.mae)
        best = np.stack([t.open.copy(), -t.open.copy()])
        draw = np.zeros((2, n))
        adverse = np.zeros((2, n))
        for offset in range(k+1):
            idx = np.minimum(np.arange(n) + offset, n-1)
            favorable = np.stack([t.high[idx], -t.low[idx]])
            unfavorable = np.stack([t.low[idx], -t.high[idx]])
            best = np.maximum(best, favorable)
            draw = np.maximum(draw, best-unfavorable)
            adverse = np.maximum(adverse, np.stack([t.open, -t.open])-unfavorable)
            self.mae[:, :, offset] = adverse
            self.dd[:, :, offset] = draw
            self.best[:, :, offset] = best

    def evaluate(self, genes, details=False):
        depth, dev, back = genes
        t, n = self.tape, len(self.tape.open)
        pivots, spans = [], []
        for a, b in t.chunks:
            p = self.z.hindsight_pivots(t.high[a:b], t.low[a:b], self.z.Params(depth, dev, back))
            pivots.extend((a+i, kind, price) for i, kind, price in p)
            spans.extend(np.diff([i for i, _, _ in p]).tolist())
        med = float(np.median(spans)) if spans else None
        eligible = bool(len(spans) >= 100 and self.h/np.sqrt(2) <= med <= self.h*np.sqrt(2))
        o = self.origins
        limit = np.minimum(t.limit[o], o+self.max_steps)
        scores, maes, pns, durations, ends, releases, reasons, targets, draws = [], [], [], [], [], [], [], [], []
        for side, kind in enumerate((1, -1)):
            selected = [(i, price) for i, typ, price in pivots if typ == kind]
            ix = np.array([i for i, _ in selected] + [n], int)
            values = np.array([p for _, p in selected] + [0.0])
            j = np.searchsorted(ix, o, side='right')
            pi = ix[j]
            # Target bar must finish no later than flat; no target after an unobserved gap.
            pivot_exit = pi < limit
            end = np.where(pivot_exit, pi, limit)
            out = np.where(pivot_exit, values[j], t.open[limit])
            dur = (t.times[end]-t.times[o])/HOUR + pivot_exit.astype(float)
            path_end = np.where(pivot_exit, end, end-1)
            offset = path_end-o
            mae = self.mae[side, o, offset].astype(float)
            dd = self.dd[side, o, offset].astype(float)
            # A timed exit at the next open can itself gap beyond the preceding bar.
            adverse_exit = np.maximum(0, kind*(t.open[o]-out))
            mae = np.maximum(mae, adverse_exit)
            # Within contiguous H1 bars next open may gap: conservatively include that
            # drop from the best high/low attained along the complete held interval.
            # Full path summaries are used below to capture that case exactly.
            favorable = self.best[side, o, offset].astype(float)
            dd = np.maximum(dd, favorable-kind*out)
            cost = t.fill_cost[o]+t.fill_cost[end]
            financing = t.financing[side, end]-t.financing[side, o]
            net = kind*(out-t.open[o])-cost-financing
            risk_floor = np.maximum(cost, PIP)
            score = net/(np.maximum(dd, risk_floor)*dur)
            maesc = net/(np.maximum(mae, risk_floor)*dur)
            scores.append(score); maes.append(maesc); pns.append(net); durations.append(dur)
            ends.append(end); releases.append(end+pivot_exit.astype(int)); reasons.append(pivot_exit)
            targets.append(out); draws.append((mae, dd))
        s = np.stack(scores)
        winner = np.argmax(s, axis=0)
        best = np.maximum(0, s.max(axis=0))
        take = s.max(axis=0) > 0
        ix = np.arange(len(o))
        chosen_net = np.stack(pns)[winner, ix]
        years = pd.DatetimeIndex(t.times[o], tz='UTC').year
        result = dict(params=dict(depth=int(depth), deviation_pips=float(dev), backstep=int(back)),
                      eligible=eligible, fitness=float(best.mean()), median_pivot_hours=med,
                      pivot_pairs=len(spans), pivots=len(pivots), origins=len(o),
                      nop_pct=float(100*(~take).mean()), buy_pct=float(100*((winner==0)&take).mean()),
                      sell_pct=float(100*((winner==1)&take).mean()),
                      mean_selected_net_pips=float(chosen_net[take].mean()/PIP) if take.any() else None,
                      mean_selected_duration_hours=float(np.stack(durations)[winner,ix][take].mean()) if take.any() else None,
                      selected_win_pct=float(100*(chosen_net[take]>0).mean()) if take.any() else None,
                      mae_denominator_fitness=float(np.maximum(0,np.stack(maes).max(axis=0)).mean()),
                      yearly_fitness={str(y):float(best[years==y].mean()) for y in sorted(set(years))},
                      scope='HINDSIGHT_OPPORTUNITIES_NOT_ACCOUNT_PERFORMANCE')
        if details:
            result['records'] = dict(origin=o, score=s, net=np.stack(pns), duration=np.stack(durations),
                                     end=np.stack(ends), release=np.stack(releases), pivot_exit=np.stack(reasons),
                                     target=np.stack(targets), mae=np.stack([v[0] for v in draws]),
                                     drawdown=np.stack([v[1] for v in draws]))
        return result


def schedule(origins, release, profits, n):
    """Exact maximum additive fixed-unit profit; tied outcomes prefer NOP."""
    value = np.zeros(n+1)
    chosen = np.full(n, -1, int)
    row_at = {int(o): r for r,o in enumerate(origins)}
    for i in range(n-1, -1, -1):
        value[i] = value[i+1]
        if i not in row_at:
            continue
        r = row_at[i]
        for d in range(2):
            j = int(release[d,r])
            if not i < j <= n:
                raise ValueError('INTERVAL_RELEASE')
            v = float(profits[d,r])+value[j]
            if v > value[i]:
                value[i], chosen[i] = v, d
    trades, i = [], 0
    while i < n:
        if chosen[i] < 0:
            i += 1
        else:
            d, r = int(chosen[i]), row_at[i]
            j = int(release[d,r])
            trades.append(dict(origin=i, release=j, side='buy' if d==0 else 'sell', net_pips=float(profits[d,r]/PIP)))
            i = j
    return dict(net_pips=float(value[0]/PIP), trades=trades,
                scope='EXACT_ADDITIVE_FIXED_UNIT_INTERVAL_LIBRARY_NOT_ACCOUNT')
