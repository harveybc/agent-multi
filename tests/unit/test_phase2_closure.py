import pytest

from app.phase2_closure import close_phase2, coverage


def evaluator(table):
    return lambda c: dict(score=table.get(tuple(c), -10.), eligible=True)


def test_final_context_catches_removal_missed_earlier_in_order():
    # a was harmful to remove before b; after b leaves, a is neutral.
    table = {('a', 'b', 'c'): 1., ('b', 'c'): 0., ('a', 'c'): 1., ('c',): 1.}
    result = close_phase2(list('abc'), [[c] for c in 'abc'], list('ac'), evaluator(table))
    assert result['status'] == 'LOCAL_FROZEN_NEIGHBORHOOD_STABLE'
    assert result['selected'] == ['c']
    assert len(result['cycles']) == 1


def test_historical_return_and_atomic_blocks():
    table = {('a',): 1., ('a', 's', 'c'): 2.}
    result = close_phase2(list('asc'), [['a'], ['s', 'c']], ['a'], evaluator(table))
    assert result['selected'] == list('asc')
    assert all(r['block'] != ['s'] for check in result['coverage'] for r in check['neighbors'])


def test_delete_ties_qualify_but_return_ties_do_not():
    table = {('a', 'b'): 1., ('a',): 1., ('b',): 0., ('a', 'b', 'c'): 1.}
    check = coverage(list('abc'), [[c] for c in 'abc'], list('ab'), evaluator(table))
    assert [r['qualifies'] for r in check['neighbors']] == [False, True, False]


def test_budget_closure_is_not_convergence():
    result = close_phase2(list('ab'), [['a'], ['b']], ['a'],
                          evaluator({('a',): 1., ('a', 'b'): 2.}), max_cycles=0)
    assert result['status'] == 'BUDGET_CLOSED_NOT_LOCAL_OPTIMUM'
    assert result['coverage'][-1]['qualifying_moves'] == 1


def test_inactive_candidate_cannot_be_selected():
    check = coverage(list('ab'), [['a'], ['b']], ['a'],
                     lambda c: dict(score=len(c), eligible=len(c) == 1))
    assert check['qualifying_moves'] == 0


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_is_not_a_tie(bad):
    with pytest.raises(ValueError, match='SCORE'):
        coverage(list('ab'), [['a'], ['b']], ['a'], lambda c: dict(score=bad, eligible=True))


def test_partial_block_and_budget_refused():
    with pytest.raises(ValueError, match='PARTIAL'):
        coverage(list('abc'), [['a'], ['b', 'c']], list('ab'), evaluator({}))
    with pytest.raises(ValueError, match='BUDGET'):
        close_phase2(['a'], [['a']], ['a'], evaluator({}), max_cycles=3)


def test_cached_restart_has_identical_decisions():
    seen = {}
    def measure(cols):
        key = tuple(cols)
        if key not in seen:
            seen[key] = dict(score=float('c' in cols) - .1 * len(cols), eligible=True)
        return seen[key]
    args = (list('abc'), [[c] for c in 'abc'], list('ab'), measure)
    first = close_phase2(*args)
    n = len(seen)
    assert close_phase2(*args) == first and len(seen) == n
