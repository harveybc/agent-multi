"""Missing economic evidence cannot become a competitive checkpoint."""
import pytest

from pipeline_plugins.rl_pipeline_with_validation import (
    _annotate_risk_adjusted, _early_stop_composite, _selection_value,
)


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('split', ['train', 'validation'])
def test_nonfinite_return_cannot_reset_patience(value, split):
    train = {'total_return': .1, 'trades_total': 0}
    val = {'total_return': .2, 'trades_total': 0}
    (train if split == 'train' else val)['total_return'] = value
    with pytest.raises(ValueError, match='finite'):
        _early_stop_composite(train, val, no_trade_penalty=0, min_trades=0)


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1.0])
def test_missing_or_invalid_risk_cannot_rank_as_risk_free(value):
    summary = {'total_return': .1, 'max_drawdown_pct': value}
    with pytest.raises(ValueError, match='finite'):
        _selection_value(summary, selection_metric='rap', risk_lambda=1)
    _annotate_risk_adjusted(summary, 1)
    assert summary['risk_adjusted_total_return'] is None
    assert summary['max_drawdown_fraction'] is None


def test_measured_no_trade_zero_remains_eligible_without_a_minimum_trade_gate():
    summary = {'total_return': 0., 'max_drawdown_pct': 0., 'trades_total': 0}
    score, raw, gate, *_ = _early_stop_composite(
        summary, summary, no_trade_penalty=0, min_trades=0, selection_metric='rap')
    assert score == raw == 0.
    assert gate
