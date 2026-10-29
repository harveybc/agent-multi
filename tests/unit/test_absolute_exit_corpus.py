from collections import Counter

from app.absolute_exit_corpus import specs
from app.early_close_metrics import evaluate_exit
from tools.verify_absolute_early_close import aggregate


def test_frozen_factorial_has_unique_complete_coverage():
    rows = list(specs())
    assert len(rows) == len({r['id'] for r in rows}) == 1744
    assert Counter(r['mode'] for r in rows) == dict(gain=576, loss=576, tp=576, gap=16)
    assert Counter(r['direction'] for r in rows) == {1: 872, -1: 872}
    assert Counter(r['horizon'] for r in rows) == {3:436, 6:436, 12:436, 24:436}


def test_aggregation_keeps_policy_side_horizon_and_control_separate():
    rows = [dict(spec=dict(horizon=3, direction=1, mode='gain'),
                 policies=dict(oracle=evaluate_exit([5], -10, 10, 0))),
            dict(spec=dict(horizon=6, direction=-1, mode='tp'),
                 policies=dict(oracle=evaluate_exit([5], 10, 10, None)))]
    out = aggregate(rows)
    assert len(out) == 3
    assert out[0]['positions'] == 2 and out[0]['early_closes'] == 1
    assert out[0]['aggregate_capture'] == 1
    assert out[2]['aggregate_capture'] is None
