import numpy as np
import pytest

from app.early_close_metrics import evaluate_exit, summarize_exits, tick_metrics


def test_owner_example():
    r=evaluate_exit([-30,-15,-5,-25],-60,60,1)
    assert r['improvement_usd']==45
    assert r['regret_usd']==10
    assert r['captured_fraction']==pytest.approx(45/55)
    assert r['regret_risk']==pytest.approx(10/60)
    assert r['beneficial_close'] and r['missed_opportunity']


def test_passive_optimal_does_not_invent_a_capture_percentage():
    r=evaluate_exit([1,4,8],10,20,1)
    assert r['captured_fraction'] is None
    assert r['improvement_usd']==-6 and r['destructive_close']
    passive=evaluate_exit([1,4,8],10,20,None)
    assert passive['regret_risk']==0 and not passive['early_close']


def test_gap_without_close_menu():
    r=evaluate_exit([],-100,20,None)
    assert r['optimal_net_usd']==-100 and r['regret_usd']==0
    assert r['captured_fraction'] is None
    assert tick_metrics([],-100,20)==[]
    with pytest.raises(ValueError): evaluate_exit([],-100,20,0)


def test_remaining_optimum_does_not_erase_earlier_mistake():
    menu=[5,10,3,-1]
    ticks=tick_metrics(menu,-10,20)
    assert ticks[2]['best_remaining_net']==3
    assert ticks[2]['close_regret_remaining_risk']==0
    assert evaluate_exit(menu,-10,20,2)['regret_usd']==7


def test_scaling_preserves_normalized_regret_and_capture():
    a=evaluate_exit([-3,2],-5,10,0)
    b=evaluate_exit([-30,20],-50,100,0)
    assert a['regret_risk']==b['regret_risk']
    assert a['captured_fraction']==b['captured_fraction']


def test_actual_native_value_must_match_the_chosen_alternative():
    with pytest.raises(ValueError,match='NATIVE_EXIT_MISMATCH'):
        evaluate_exit([1,2],-1,10,0,actual_net=99)


@pytest.mark.parametrize('args', [([np.nan],0,1,None), ([1],np.inf,1,None),
    ([1],0,0,None), ([1],0,-1,None), ([1],0,1,3), ([1],0,1,False)])
def test_invalid_metric_input(args):
    with pytest.raises(ValueError): evaluate_exit(*args)


def test_ties_neutrality_and_aggregate_denominator():
    rows=[evaluate_exit([0,0],0,10,0),evaluate_exit([-30,-15,-5],-60,60,1),
          evaluate_exit([1,4],10,20,0)]
    assert rows[0]['neutral_close'] and rows[0]['captured_fraction'] is None
    s=summarize_exits(rows)
    assert s['improvement_usd_sum']==36
    assert s['aggregate_capture']==pytest.approx(36/55)
    assert s['beneficial_closes']==1 and s['destructive_closes']==1
    assert summarize_exits([])['regret_risk_mean'] is None
