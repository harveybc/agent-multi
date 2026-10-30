"""Bounded retrospective exact references, never a causal trading policy."""
import copy
import math

import numpy as np

from app.dynamic_account_arbiter import HORIZONS, objectives
from app.dynamic_account_panel import World


def enumerate_tree(initial, expand, terminal, *, max_nodes=100000):
    """Enumerate every complete history. No pruning of unequal account states."""
    stack = [(initial, ())]
    leaves = []
    nodes = 0
    while stack:
        state, trace = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ValueError('NODE_BUDGET_NO_COMPLETE_FRONTIER')
        vector = terminal(state)
        if vector is not None:
            if len(vector) != 2 or not all(math.isfinite(x) for x in vector):
                raise ValueError('TERMINAL_VECTOR')
            leaves.append(dict(trace=trace, vector=tuple(vector)))
            continue
        children = list(expand(state))
        if not children:
            raise ValueError('NONTERMINAL_DEAD_END')
        if len({a for a, _ in children}) != len(children):
            raise ValueError('DUPLICATE_ACTION')
        stack.extend((child, trace + (action,)) for action, child in reversed(children))
    return leaves, nodes


def nondominated(rows):
    """One deterministic witness per exact vector; both components maximized."""
    ordered = sorted(rows, key=lambda r: (-r['vector'][0], -r['vector'][1], r['trace']))
    result = []
    best = -math.inf
    for row in ordered:
        if row['vector'][1] > best:
            result.append(row)
            best = row['vector'][1]
    return result


def clone(env):
    out = copy.copy(env)
    for name in ('trades', 'log', 'decisions'):
        setattr(out, name, list(getattr(env, name)))
    for name in ('entries_day', 'counts'):
        setattr(out, name, copy.copy(getattr(env, name)))
    out.pos = copy.copy(env.pos)
    return out


def fixture(risk, pattern, mirror, initial, steps):
    """A small terminal subproblem ending at native weekly flat, not a time cap."""
    if not 2 <= steps <= 6 or initial not in (-1, 0, 1) or mirror not in (-1, 1):
        raise ValueError('FINITE_FIXTURE_CONTRACT')
    world = World(dict(seed=7301, pattern='trend', volatility='decreasing', mirror=1), risk)
    e = world.env
    start = world.end - 1 - steps
    templates = {
        'smooth': [0, 7, 14, 22, 30, 38, 46],
        'oscillating': [0, 27, 3, 32, 8, 39, 14],
        'reversal': [0, 5, 12, 9, -5, -20, -31],
    }
    values = np.interp(np.linspace(0, 6, steps + 1), np.arange(7), templates[pattern])
    close = np.full(len(e.c), 1.1)
    close[start:world.end] += mirror * np.asarray(values) * .0001
    close[world.end:] = close[world.end - 1]
    op = np.r_[close[0], close[:-1]]
    high = np.maximum(op, close) + .00001
    low = np.minimum(op, close) - .00001
    e.bars = e.bars.copy()
    for name, vals in zip(('OPEN', 'HIGH', 'LOW', 'CLOSE'), (op, high, low, close)):
        e.bars[name] = vals
    e.o, e.h, e.l, e.c = op, high, low, close
    e.reset()
    eq = [e.cfg.capital]
    while e.next_decision < start:
        b = e.next_decision
        enter = initial != 0 and b == start - 1
        plan = world.entry_plan(12, initial, 1.) if enter else None
        info = e.step(b, (1 if initial == 1 else 2) if enter else 0, entry_plan=plan)
        if info['refused']:
            raise AssertionError('INITIAL_REFUSAL')
        eq.append(float(e.equity_at(b + 1)))
    if initial and e.pos is None:
        raise AssertionError('INITIAL_POSITION_MISSING')
    return world, e, eq


def actions(e):
    if e.pos is not None:
        return ['HOLD', 'CLOSE']
    return ['NOP'] + ([f'{d}:{h}' for h in HORIZONS for d in ('BUY', 'SELL')]
                      if e.entry_gate(e.next_decision)[0] else [])


def advance(world, e, equity, action):
    if action not in actions(e):
        raise ValueError('ILLEGAL_PHASE_ACTION')
    plan = None
    a = 3 if action == 'CLOSE' else 0
    if ':' in action:
        side, h = action.split(':')
        a = 1 if side == 'BUY' else 2
        plan = world.entry_plan(int(h), 1 if a == 1 else -1, 1.)
    b = e.next_decision
    info = e.step(b, a, entry_plan=plan)
    if info['refused']:
        raise AssertionError(('UNEXPECTED_REFUSAL', info))
    return equity + [float(e.equity_at(b + 1))]


def finish(e, equity):
    if e.pos is not None:
        raise AssertionError('NOT_NATIVE_FORCED_FLAT')
    error = float(e.balance - e.cfg.capital - sum(t['net_usd'] for t in e.trades))
    if abs(error) > 1e-7:
        raise AssertionError('CASH_CONSERVATION')
    r, d = objectives(equity, e.cfg.capital)
    return dict(vector=(r, -d), cash_error=error, flat=True,
        trades=[dict(direction=int(t['direction']), entry=int(t['entry_bar_abs']),
                     exit=int(t['exit_bar_abs']), net=float(t['net_usd']),
                     q=float(t['percentual_order_size']), units=int(t['units']),
                     reason=t['close_reason']) for t in e.trades])


def native_case(risk, pattern, mirror, initial, *, steps=5, max_nodes=100000):
    world, env, equity = fixture(risk, pattern, mirror, initial, steps)
    end = world.end - 1

    def terminal(state):
        e, eq = state
        if e.next_decision >= end:
            return finish(e, eq)['vector']
        return None

    def expand(state):
        e, eq = state
        for a in actions(e):
            child = clone(e)
            yield a, (child, advance(world, child, eq, a))

    leaves, nodes = enumerate_tree((env, equity), expand, terminal, max_nodes=max_nodes)
    conditional = {a: nondominated([r for r in leaves if r['trace'][0] == a])
                   for a in actions(env)}
    return dict(pattern=pattern, mirror=mirror, initial_direction=initial, steps=steps,
                complete=True, scope='EXACT_DECLARED_FINITE_NATIVE_TREE_NOT_TRAIN_OR_CAUSAL_POLICY',
                nodes=nodes, terminal_accounts=len(leaves), frontier=nondominated(leaves),
                conditional=conditional), leaves


def replay(risk, pattern, mirror, initial, trace, *, steps=5):
    world, e, equity = fixture(risk, pattern, mirror, initial, steps)
    for action in trace:
        equity = advance(world, e, equity, action)
    if e.next_decision != world.end - 1:
        raise ValueError('INCOMPLETE_TRACE')
    return finish(e, equity)
