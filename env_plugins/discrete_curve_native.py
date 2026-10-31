"""Position evaluation outside the learned network, native fixed-profile orders."""
import numpy as np

from app.frozen_recipe_inputs import require
from app.horizon_risk_rules import propose

HOUR = 3600 * 10**9


class CurveNativeEnv:
    def __init__(self, engine, inputs, risk_config, horizon, gamma=.99):
        self.engine, self.inputs = engine, inputs
        self.risk_config, self.horizon, self.gamma = risk_config, horizon, gamma
        stamps = engine.v + HOUR
        rows = np.searchsorted(inputs.row_ts_ns, stamps)
        self.rows = np.minimum(rows, len(inputs.row_ts_ns) - 1)
        self.valid = (rows < len(inputs.row_ts_ns)) & (inputs.row_ts_ns[self.rows] == stamps)
        self.valid &= inputs.valid[self.rows]
        require(self.valid[engine.first:engine.last].any(), 'NO_DECISION')
        self.refs = {1 if r['direction'] == 'buy' else -1: r['reference']['median']
                     for r in risk_config['rows'] if r['horizon'] == horizon}
        if set(self.refs) != {-1, 1}:
            raise ValueError('HORIZON_PROFILES')
        self.finished = True

    def observation(self):
        e, b = self.engine, self.bar
        feature = self.inputs.observation(int(self.rows[b]))['features']
        p = e.pos
        route, progress, mask = 0, .5, [True, False, False]
        if p is None:
            allowed, _ = e.entry_gate(b)
            mask = [True, allowed, allowed]
        else:
            route = 1 if p.direction == 1 else 2
            liquidation = e.quote_price(b, e.c[b], -p.direction, int(e.v[b]) + HOUR)
            progress = (liquidation - p.sl) / (p.tp - p.sl)
            if not np.isfinite(progress):
                raise ValueError('NONFINITE_POSITION_COORDINATE')
            mask = [True, True, False]
        self.last_observation = dict(features=feature, route=np.array([route], np.int64),
            progress=np.array([progress], np.float32), mask=np.array(mask, bool))
        return self.last_observation

    def reset(self):
        e = self.engine
        self.bar, self.finished, self.auto_holds = e.reset(), False, 0
        self.equity_trace = [(int(e.v[self.bar]) + HOUR, float(e.equity_at(self.bar)))]
        while not self.valid[self.bar]:
            e.step(self.bar, 0)
            self.bar += 1
            self.auto_holds += 1
            self.equity_trace.append((int(e.v[self.bar]) + HOUR, float(e.equity_at(self.bar))))
            if self.bar >= e.last:
                raise ValueError('NO_FINITE_EPISODE')
        return self.observation()

    def entry_plan(self, direction):
        from app.short_horizon.expert_guided.absolute import AbsoluteOrder
        e, ref = self.engine, self.refs[direction]
        profile, b = e.cfg.profile, self.bar
        sigma = float(e.decision_sigma(b))
        si = int(e.seg_of[b + 1])
        deadline = int(e.ff[si])
        rollovers = [(x.value, 3 if i == profile.triple_swap_rollover_index else 1)
                     for i, x in enumerate(e.cal.segments[si].rollovers(), 1)]

        def build(context):
            rate = -(profile.swap_long_per_lot if direction == 1 else profile.swap_short_per_lot)
            reserve = max(0., rate) / profile.lot_units * sum(
                n for t, n in rollovers if context.fill_ns < t <= deadline)
            result = propose(direction=direction, entry_quote=context.entry_quote,
                liquidation_quote=context.liquidation_quote, balance=context.balance,
                tp_reference=ref['tp_pips'] * .0001, mae_reference=ref['mae_pips'] * .0001,
                sigma=sigma, genes=self.risk_config['baseline'], sizing=e.cfg.sizing,
                tick=e.tick, min_distance=e.cfg.rules.min_distance, quality=1.,
                loss_cost_per_unit=profile.slippage_market + profile.slippage_stop
                    + 2 * profile.commission_per_lot_side / profile.lot_units,
                funding_reserve_per_unit=reserve)
            if result['status'] != 'ORDER':
                # A typed refusal is consumed by the native order validator.
                return AbsoluteOrder(result['tp'], result['sl'], 0., 1., reserve)
            return AbsoluteOrder(result['tp'], result['sl'], result['q'],
                                 result['risk_budget'], reserve)
        return build

    def step(self, action):
        if self.finished or type(action) not in (int, np.int64) or not 0 <= action < 3:
            raise ValueError('DISCRETE_ACTION')
        obs = self.last_observation
        if not obs['mask'][action]:
            raise ValueError('MASKED_ACTION')
        occupied = int(obs['route'][0]) != 0
        command = (3 if occupied else action) if action else 0
        plan = self.entry_plan(1 if action == 1 else -1) if command in (1, 2) else None
        e, reward, count = self.engine, 0., 0
        while True:
            previous = e.equity_at(self.bar)
            e.step(self.bar, command, entry_plan=plan)
            self.bar += 1
            delta = 100. * (e.equity_at(self.bar) - previous) / e.cfg.capital
            self.equity_trace.append((int(e.v[self.bar]) + HOUR, float(e.equity_at(self.bar))))
            reward += self.gamma ** count * delta
            count += 1
            if self.bar >= e.last:
                self.finished = True
                # Terminal observation is never bootstrapped or used to open an order.
                return self.last_observation, reward, True, count
            if self.valid[self.bar]:
                return self.observation(), reward, False, count
            command, plan = 0, None
            self.auto_holds += 1


def native_environment(market, risk_config, horizon, period):
    import pandas as pd
    from app.train_native_pareto import bid_environment_class
    a, b = (pd.Timestamp(v, tz='UTC') - pd.Timedelta(hours=1)
            for v in market.recipe['split']['periods'][period])
    engine = bid_environment_class()(market.bars, market.calendar, market.sigma, market.sigma_rule,
                                    a, b, market.config, tick=.00001)
    return CurveNativeEnv(engine, market.inputs, risk_config, horizon)
