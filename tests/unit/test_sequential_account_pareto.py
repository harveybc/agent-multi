import json
import os
from fractions import Fraction
from pathlib import Path

import pytest

from app.sequential_account_pareto import enumerate_tree, nondominated


def test_sequential_opportunity_not_individual_trade_score():
    edges = {'flat': [('long', 'L'), ('short', 'after_short'), ('NOP', 'wait')],
             'after_short': [('short_again', 'SS')], 'wait': [('later', 'W')]}
    payoffs = {'L': (8, -2), 'SS': (12, -1), 'W': (0, 0)}
    leaves, count = enumerate_tree('flat', lambda x: edges.get(x, []), payoffs.get)
    assert count == 6
    front = nondominated(leaves)
    assert {tuple(x['trace']) for x in front} == {('short', 'short_again'), ('NOP', 'later')}
    assert ('long',) not in {tuple(x['trace']) for x in front}


@pytest.mark.parametrize('winner', ['long', 'NOP'])
def test_long_or_wait_can_be_best(winner):
    v = {k: (Fraction(2 if k == winner else 1, 3), -1) for k in ('long', 'short', 'NOP')}
    leaves, _ = enumerate_tree('root', lambda x: [(k, k) for k in v] if x == 'root' else [], v.get)
    assert [x['trace'] for x in nondominated(leaves)] == [(winner,)]


def test_frontier_matches_independent_pairwise_and_keeps_tradeoff():
    leaves = [{'vector': (r, -d), 'trace': (str(r), str(d))}
              for r, d in [(0, 0), (1, 1), (2, 1), (3, 2), (1, 2), (3, 2)]]
    actual = {x['vector'] for x in nondominated(leaves)}
    expected = {x['vector'] for x in leaves if not any(
        all(a >= b for a, b in zip(y['vector'], x['vector'])) and y['vector'] != x['vector']
        for y in leaves)}
    assert actual == expected == {(0, 0), (2, -1), (3, -2)}


def test_exhausted_budget_and_dead_end_are_not_frontiers():
    with pytest.raises(ValueError, match='NODE_BUDGET'):
        enumerate_tree(0, lambda x: [('next', x + 1)], lambda x: None, max_nodes=5)
    with pytest.raises(ValueError, match='NONTERMINAL_DEAD_END'):
        enumerate_tree(0, lambda x: [], lambda x: None)


@pytest.fixture
def risk():
    root = os.environ.get('STRATEGY_ENGINE_ROOT')
    if not root:
        pytest.skip('native engine required')
    from app.zigzag_opportunities import engine
    engine(Path(root))
    return json.loads((Path(__file__).resolve().parents[2] /
        'docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json').read_text())


@pytest.mark.parametrize('initial', [0, 1, -1])
def test_native_phases_and_independent_replay(risk, initial):
    from app.sequential_account_pareto import native_case, replay
    case, leaves = native_case(risk, 'reversal', 1, initial, steps=3)
    assert case['complete'] and case['nodes'] > case['terminal_accounts']
    firsts = {x['trace'][0] for x in leaves}
    if initial == 0:
        assert firsts == {'NOP'} | {f'{side}:{h}' for side in ('BUY', 'SELL') for h in (3,6,12,24)}
    else:
        assert firsts == {'HOLD', 'CLOSE'}
    for row in leaves:
        other = replay(risk, 'reversal', 1, initial, row['trace'], steps=3)
        assert row['vector'] == other['vector']
        assert abs(other['cash_error']) < 1e-7
        assert other['flat']
        for trade in other['trades']:
            assert 0 < trade['q'] <= 1
        assert all(a['exit'] < b['entry'] for a,b in zip(other['trades'], other['trades'][1:]))


def test_mirrored_fixture_has_mirrored_prices(risk):
    from app.sequential_account_pareto import fixture
    a, _, _ = fixture(risk, 'oscillating', 1, 0, 3)
    b, _, _ = fixture(risk, 'oscillating', -1, 0, 3)
    import numpy as np
    assert np.allclose(a.env.c + b.env.c, 2.2)
    assert np.allclose(a.env.h + b.env.l, 2.2)


def test_illegal_cross_phase_actions_and_branch_isolation(risk):
    from app.sequential_account_pareto import fixture, clone, advance
    world, occupied, eq = fixture(risk, 'smooth', 1, 1, 3)
    with pytest.raises(ValueError, match='ILLEGAL_PHASE_ACTION'):
        advance(world, occupied, eq, 'SELL:3')
    branch = clone(occupied)
    advance(world, branch, eq, 'CLOSE')
    assert branch.pos is None and occupied.pos is not None
    assert not occupied.trades and len(branch.trades) == 1
    flatworld, flat, eq = fixture(risk, 'smooth', 1, 0, 3)
    with pytest.raises(ValueError, match='ILLEGAL_PHASE_ACTION'):
        advance(flatworld, flat, eq, 'CLOSE')
