"""External economic calibration of frozen market-only SAC reports."""
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.dynamic_account_arbiter import HORIZONS, frontier
from app.sequential_account_pareto import clone
from env_plugins.discrete_curve_native import HOUR


def route(values, direction, z):
    curve = np.asarray(values)[..., 3:].reshape(-1, 2, 9, 2)[:, 0 if direction == 1 else 1]
    t = float(z) * 8
    low = min(7, max(0, int(np.floor(t))))
    return curve[:, low] + (t - low) * (curve[:, low + 1] - curve[:, low])


def choose(vectors, risk_weight=1.):
    passive = 'HOLD' if 'HOLD' in vectors else 'NOP'
    if passive not in vectors or not all(np.shape(v) == (2,) and np.isfinite(v).all() for v in vectors.values()):
        raise ValueError('ECONOMIC_BIDS')
    projected = {k: (float(v[0]), max(0., float(v[1]))) for k, v in vectors.items()}
    efficient = frontier({k: (v[0], -v[1]) for k, v in projected.items()})
    scores = {k: v[0] - risk_weight * v[1] for k, v in projected.items()}
    best = max(scores[k] for k in efficient)
    ties = [k for k in efficient if abs(scores[k] - best) <= 1e-10]
    selected = passive if passive in ties or len(ties) > 1 else ties[0]
    return dict(action=selected, vector=projected[selected], utility=scores[selected],
        frontier=efficient, bids=projected, risk_projection=sum(bool(v[1] < 0) for v in vectors.values()))


def economic_vector(equity):
    x = np.asarray(equity, float)
    if x.ndim != 1 or not len(x) or x[0] <= 0 or not np.isfinite(x).all():
        raise ValueError('ECONOMIC_EQUITY')
    high = np.maximum.accumulate(x)
    return [float(100 * (x[-1] / x[0] - 1)), float(100 * np.max((high - x) / high))]


def calibrate(rows, minimum=16, cutoff='2023-01-01', alpha=1.):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['key']].append(row)
    models, reports = {}, {}
    boundary = pd.Timestamp(cutoff, tz='UTC')
    for key, group in sorted(grouped.items()):
        fit = [r for r in group if pd.to_datetime(r['deadline'], utc=True) < boundary]
        check = [r for r in group if pd.to_datetime(r['origin'], utc=True) >= boundary]
        if len(fit) < minimum:
            raise ValueError('CALIBRATION_SUPPORT:' + key)
        x, y = np.array([r['x'] for r in fit]), np.array([r['y'] for r in fit])
        model = Pipeline([('scale', StandardScaler()), ('ridge', Ridge(alpha=alpha))]).fit(x, y)
        models[key] = model
        report = dict(fit_rows=len(fit), check_rows=len(check), purged_rows=len(group) - len(fit) - len(check),
                      fit_target_mean=y.mean(0).tolist(), fit_rmse=np.sqrt(np.mean((model.predict(x)-y)**2, 0)).tolist())
        if check:
            xc, yc = np.array([r['x'] for r in check]), np.array([r['y'] for r in check])
            error = model.predict(xc) - yc
            report.update(check_rmse=np.sqrt(np.mean(error**2, 0)).tolist(),
                constant_check_rmse=np.sqrt(np.mean((y.mean(0)-yc)**2, 0)).tolist(),
                check_error_mean=error.mean(0).tolist(), check_error_std=error.std(0).tolist())
        reports[key] = report
    return models, reports


class Reports:
    def __init__(self, arrays, adapters):
        self.arrays, self.adapters = arrays, adapters
        self.template = adapters[3]

    def values(self, env, horizon):
        b = env.next_decision
        if b >= len(self.template.valid) or not self.template.valid[b]:
            return None
        row = int(self.template.rows[b])
        return self.arrays[horizon].get(row)

    def features(self, env, horizon):
        values = self.values(env, horizon)
        if values is None:
            return None
        if env.pos is None:
            return values[:, :3].reshape(-1).copy()
        p, b = env.pos, env.next_decision
        quote = env.quote_price(b, env.c[b], -p.direction, int(env.v[b]) + HOUR)
        z = (quote - p.sl) / (p.tp - p.sl)
        return route(values, p.direction, z).reshape(-1)

    def step(self, env, owner, action):
        command, plan = 0, None
        if action == 'CLOSE':
            command = 3
        elif ':' in action:
            side, h = action.split(':')
            h = int(h)
            command = 1 if side == 'BUY' else 2
            adapter = copy.copy(self.adapters[h])
            adapter.engine, adapter.bar = env, env.next_decision
            plan = adapter.entry_plan(1 if command == 1 else -1)
        info = env.step(env.next_decision, command, entry_plan=plan)
        if info['entry']:
            owner = h
        if env.pos is None:
            owner = None
        return owner, info

    def reference(self, env, owner):
        if env.pos is None:
            if not env.entry_gate(env.next_decision)[0]:
                return 'NOP'
            h = HORIZONS[((int(env.v[env.next_decision]) + HOUR) // HOUR) % len(HORIZONS)]
            values = self.values(env, h)
            if values is None:
                return 'NOP'
            logits = values[0, :3]
            if np.count_nonzero(np.isclose(logits, logits.max(), atol=1e-10, rtol=0)) > 1:
                return 'NOP'
            action = int(logits.argmax())
            return 'NOP' if action == 0 else ('BUY:' if action == 1 else 'SELL:') + str(h)
        values = self.features(env, owner)
        return 'CLOSE' if values is not None and values[1] > values[0] else 'HOLD'


def rollout_reference(env, owner, first, reports):
    e = clone(env)
    origin = e.next_decision
    deadline = int(e.ff[int(e.seg_of[origin])])
    equity = [float(e.equity_at(origin))]
    entries = 0
    while e.next_decision < e.last and int(e.v[e.next_decision]) + HOUR < deadline:
        action = first if e.next_decision == origin else reports.reference(e, owner)
        owner, info = reports.step(e, owner, action)
        entries += bool(info['entry'])
        equity.append(float(e.equity_at(e.next_decision)))
    if e.pos is not None:
        raise ValueError('REFERENCE_NOT_FLAT')
    return dict(vector=economic_vector(equity), deadline=pd.Timestamp(deadline, tz='UTC').isoformat(),
                entries_including_root=entries, terminal_equity=equity[-1])


def labelled_state(env, owner, reports):
    origin = pd.Timestamp(int(env.v[env.next_decision]) + HOUR, tz='UTC').isoformat()
    rows = []
    if env.pos is None:
        nop = rollout_reference(env, None, 'NOP', reports)
        for h in HORIZONS:
            x = reports.features(env, h)
            if x is None:
                continue
            for a, action in enumerate(('NOP', 'BUY:' + str(h), 'SELL:' + str(h))):
                outcome = nop if a == 0 else rollout_reference(env, None, action, reports)
                rows.append(dict(key=f'open_{h}_{a}', x=x.tolist(), y=outcome['vector'],
                    origin=origin, deadline=outcome['deadline'], entries_including_root=outcome['entries_including_root']))
    else:
        for action in ('HOLD', 'CLOSE'):
            outcome = rollout_reference(env, owner, action, reports)
            for h in HORIZONS:
                x = reports.features(env, h)
                if x is not None:
                    rows.append(dict(key=f'close_{owner}_{env.pos.direction}_{h}_{action}',
                        x=x.tolist(), y=outcome['vector'], origin=origin,
                        deadline=outcome['deadline'], entries_including_root=outcome['entries_including_root']))
    return rows


class EconomicController:
    def __init__(self, reports, models, risk_weight=1.):
        self.reports, self.models = reports, models
        self.risk_weight = risk_weight

    def predict(self, key, x):
        if key not in self.models:
            raise ValueError('UNFITTED_VALUE:' + key)
        return self.models[key].predict(np.asarray(x)[None])[0]

    def decide(self, env, owner):
        passive = 'NOP' if env.pos is None else 'HOLD'
        features = {h: self.reports.features(env, h) for h in HORIZONS}
        if any(x is None for x in features.values()) or (env.pos is None and not env.entry_gate(env.next_decision)[0]):
            return dict(action=passive, supported=False, reporters=list(features))
        if env.pos is None:
            bids = {'NOP': np.mean([self.predict(f'open_{h}_0', x) for h, x in features.items()], 0)}
            for h, x in features.items():
                bids['BUY:' + str(h)] = self.predict(f'open_{h}_1', x)
                bids['SELL:' + str(h)] = self.predict(f'open_{h}_2', x)
        else:
            bids = {action: np.mean([self.predict(f'close_{owner}_{env.pos.direction}_{h}_{action}', x)
                for h, x in features.items()], 0) for action in ('HOLD', 'CLOSE')}
        return dict(choose(bids, self.risk_weight), supported=True, reporters=list(features))
