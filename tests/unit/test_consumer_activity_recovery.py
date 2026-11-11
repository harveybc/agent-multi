import numpy as np
import pandas as pd
import pytest
from tools.recover_consumer_activity import check_reference, replay


class Constant:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, x):
        return np.tile([1-self.p, self.p], (len(x), 1))


def fixture():
    times = pd.date_range('2020-01-01', periods=6, freq='h', tz='UTC')
    X = pd.DataFrame({'x': np.arange(6.)}, index=times)
    bundle = dict(features=['x'], times_ns=times.asi8,
                  estimators=[Constant(.5), Constant(.2), Constant(.8)],
                  prediction_rows=[np.array([0, 1]), np.array([2, 3])])
    return X, bundle


def test_fold_models_replace_only_assigned_rows():
    X, bundle = fixture()
    assert np.array_equal(replay(bundle, X), [.2, .2, .8, .8, .5, .5])


def test_old_global_producer_is_rejected():
    X, bundle = fixture()
    ref = pd.DataFrame({'timestamp': X.index, 'act.p_hit': replay(bundle, X)})
    with pytest.raises(ValueError, match='PROBABILITY'):
        check_reference(X.index, np.full(len(X), .5), ref)


def test_columns_and_grid_are_bound():
    X, bundle = fixture()
    with pytest.raises(ValueError, match='FEATURE_ORDER'):
        replay(bundle, X.rename(columns={'x': 'other'}))
    X.index += pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match='REPLAY_GRID'):
        replay(bundle, X)


def test_exact_current_consumer_accepted():
    X, bundle = fixture()
    p = replay(bundle, X)
    check_reference(X.index, p, pd.DataFrame({'timestamp': X.index, 'act.p_hit': p}))
