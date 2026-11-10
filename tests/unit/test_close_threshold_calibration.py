import json
import numpy as np
import pytest

from app.close_threshold_calibration import BASELINE,fit_threshold,fit_thresholds,apply_thresholds


def brute(scores,target,banks,t):
    return np.mean([np.mean(np.maximum(target[b],0)-(scores[b]>t)*target[b]) for b in banks])


def test_solver_matches_exhaustive_randomized_partitions():
    rng=np.random.default_rng(71)
    for _ in range(40):
        scores=rng.integers(-4,5,17)/10;target=rng.normal(size=17);banks=[list(range(3)),list(range(3,17))]
        got=fit_threshold(scores,target,banks)
        cuts=np.r_[np.nextafter(scores.min(),-np.inf),np.unique(scores),BASELINE]
        best=min(brute(scores,target,banks,t) for t in cuts)
        assert got['fit_regret']==pytest.approx(best,abs=1e-12)
        assert brute(scores,target,banks,got['threshold'])==pytest.approx(best,abs=1e-12)


def test_neutral_targets_and_equal_scores_keep_baseline_when_optimal():
    got=fit_threshold(np.array([-1.,2.]),np.zeros(2),[[0],[1]])
    assert got['threshold']==BASELINE
    conflict=fit_threshold(np.ones(2),np.array([1.,-1.]),[[0,1]])
    assert conflict['threshold']==BASELINE


def test_all_hold_and_all_close_and_json_roundtrip():
    for targets,expected in (([1.,2.],[True,True]),([-1.,-2.],[False,False])):
        s=np.zeros(2);got=fit_threshold(s,np.array(targets),[[0,1]])
        threshold=json.loads(json.dumps(got))['threshold']
        assert (s>threshold).tolist()==expected
    assert fit_threshold(np.array([0.,0.]),np.array([2.,-1.]),[[0,1]])['fit_regret']==pytest.approx(.5)


def test_bank_weighting_not_row_weighting():
    s=np.ones(11);y=np.r_[1.,np.full(10,-.2)]
    assert fit_threshold(s,y,[[0],list(range(1,11))])['threshold']<1
    assert fit_threshold(s,y,[list(range(11))])['threshold']>=1


def test_only_fit_data_changes_calibration_and_directions_are_separate():
    records=[dict(role='fit',direction='long',gap=2.),dict(role='fit',direction='short',gap=-1.),
             dict(role='monitor',direction='long',gap=999.)]
    s=np.zeros(3);banks=[[0],[1],[2]]
    got=fit_thresholds(s,records,banks,'directional')
    assert apply_thresholds(s[:2],['long','short'],got).tolist()==[True,False]
    s[-1]=1e9;records[-1]['gap']=-1e99
    assert fit_thresholds(s,records,banks,'directional')==got
    common=fit_thresholds(s,records,banks,'common')
    assert apply_thresholds(s[:2],['long','short'],common).tolist()==[True,True]


def test_invalid_or_duplicate_bank_support_is_refused():
    for banks in ([[0],[0,1]],[[0]],[]):
        with pytest.raises(ValueError):fit_threshold(np.ones(2),np.ones(2),banks)
    with pytest.raises(ValueError):fit_threshold(np.array([np.nan]),np.ones(1),[[0]])


def test_exact_threshold_tie_is_hold():
    c={'mode':'common','thresholds':{'all':{'threshold':0.1}}}
    assert apply_thresholds(np.array([.1,np.nextafter(.1,np.inf)]),['long','short'],c).tolist()==[False,True]
