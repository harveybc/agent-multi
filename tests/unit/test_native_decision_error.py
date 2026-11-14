import numpy as np
import pytest

from app.native_decision_error import decision_ledger, summarize, fit_gap_bins, gap_labels, layer_delta


def test_regret_switch_accounting_and_equivalent_actions():
    y=np.array([[0.,1.,2.],[0.,2.,1.],[1.,1.,0.]])
    old=np.array([[0.,1.,2.],[0.,1.,2.],[2.,1.,0.]])
    new=np.array([[0.,2.,1.],[0.,2.,1.],[1.,2.,0.]])
    rows=decision_ledger(old,new,y);s=summarize(rows)
    assert s['harmful']==1 and s['helpful']==1 and s['neutral_switches']==1
    assert s['regret_delta']==0 and s['switches']==3
    assert rows[2]['new_optimal'] and rows[2]['old_optimal']
    assert summarize([]) is None


def test_lower_rmse_can_worsen_regret_and_common_shift_does_not_repair_it():
    y=np.array([[0.,1.,2.]])
    old=y-10;new=np.array([[0.,2.,1.]])
    r=decision_ledger(old,new,y)[0]
    assert r['new_mse']<r['old_mse'] and r['regret_delta']==1
    shifted=decision_ledger(old+100,new+100,y)[0]
    assert (r['old_action'],r['new_action'],r['regret_delta'])==(shifted['old_action'],shifted['new_action'],shifted['regret_delta'])
    assert np.isclose(r['old_mse'],r['old_level_mse']+r['old_centered_mse'])


def test_gap_bins_are_fit_only_with_ties_separate():
    fit=np.array([[0.,1.,2.],[0.,1.,4.],[0.,1.,8.],[2.,2.,2.]])
    bins=fit_gap_bins(fit);held=np.array([[0.,2.,2.],[0.,0.,0.],[0.,1.,20.]])
    labels=gap_labels(held,bins)
    assert labels==['low','all_tied','high']
    assert bins==fit_gap_bins(fit)
    assert fit_gap_bins(np.ones((3,3)))==[]


def test_layer_delta_is_exact_and_rejects_nonfinite_or_shape():
    rng=np.random.default_rng(7)
    h0,h1=rng.normal(size=(2,5,8)),rng.normal(size=(2,5,8))
    w0,w1=rng.normal(size=(2,3,8)),rng.normal(size=(2,3,8))
    b0,b1=rng.normal(size=(2,3)),rng.normal(size=(2,3))
    a,b=layer_delta(h0,h1,w0,w1,b0,b1)
    expected=np.einsum('tni,tai->tna',h1,w1)+b1[:,None]-np.einsum('tni,tai->tna',h0,w0)-b0[:,None]
    assert np.allclose(a+b,expected,atol=1e-12)
    with pytest.raises(ValueError):decision_ledger(np.zeros((1,3)),np.zeros((2,3)),np.zeros((1,3)))
    with pytest.raises(ValueError):decision_ledger(np.full((1,3),np.nan),np.zeros((1,3)),np.zeros((1,3)))


def test_twin_minimum_can_choose_neither_twin_winner():
    from tools.analyze_native_decision_error import analyze_scope
    y=np.array([[0.,1.,2.]])
    old=np.stack([y,y]);new=np.array([[[5.,4.,-100.]],[[-100.,4.,5.]]])
    records=[dict(row=12,timestamp=123,kind='flat_both_allowed',position=0)]
    summary,rows=analyze_scope(old,new,y,records,np.array([0]),[],np.zeros_like(new),np.zeros_like(new),np.array([2]))
    assert rows[0]['new_minimum_winner_neither_twin']
    assert rows[0]['new_action']==1 and rows[0]['new_twins']==[0,2]
    assert summary['summary']['harmful']==1 and summary['summary']['actor_regret']==0
    assert summary['by_gap']['unbinned']['n']==1
