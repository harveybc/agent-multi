import pytest

from tools.run_post_audit_diagnostics import close_metrics


def test_profitable_exit_can_be_destructive_and_losing_exit_helpful():
    rows=[dict(executed=10.,hold=20.,optimal=25.,risk=10.,units=1000,hold_reason='TP',direction=1),
          dict(executed=-2.,hold=-10.,optimal=0.,risk=10.,units=1000,hold_reason='SL',direction=-1)]
    result=close_metrics(rows,1)
    assert result['helpful']==result['destructive']==1
    assert result['mean_regret_risk']==pytest.approx(.85)
    assert result['delta_vs_hold_pp']==pytest.approx(-.02)


def test_no_closes_has_no_fake_regret():
    result=close_metrics([],1)
    assert result['count']==0 and result['mean_regret_risk'] is None
