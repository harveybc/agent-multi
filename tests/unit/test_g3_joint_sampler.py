import numpy as np
import pandas as pd
import pytest

from app.g3_joint_sampler import ConditionalJointSampler,clock_keys,innovations


def example(n=1200):
    rng = np.random.default_rng(4)
    index = pd.date_range('2020-01-01',periods=n,freq='h',tz='UTC')
    returns = rng.normal(0,.001,(n,7))
    returns[:,1:] += returns[:,:1]
    close = np.exp(np.cumsum(returns,axis=0))
    op = np.r_[close[0,0],close[:-1,0]]
    bars = pd.DataFrame(dict(OPEN=op,HIGH=np.maximum(op,close[:,0])*1.0001,
        LOW=np.minimum(op,close[:,0])*.9999,CLOSE=close[:,0]),index=index)
    crosses = pd.DataFrame(close[:,1:],index=index)
    macro = pd.DataFrame({'x':np.sin(np.arange(n)/100)},index=index)
    return bars,crosses,macro


def test_deterministic_prefix_joint_donors_and_algebra():
    b,x,m = example()
    fit = ConditionalJointSampler().fit(b,x,m,b.index[-1]+pd.Timedelta(hours=1))
    actual,fx,donor = fit.generate(b,x,m,19)
    small,sfx,sd = fit.generate(b.iloc[:700],x.iloc[:700],m.iloc[:700],19)
    np.testing.assert_array_equal(actual.iloc[:700],small)
    np.testing.assert_array_equal(fx.iloc[:700],sfx)
    np.testing.assert_array_equal(donor[:700],sd)
    assert (actual.HIGH >= actual[['OPEN','CLOSE']].max(axis=1)).all()
    assert (actual.LOW <= actual[['OPEN','CLOSE']].min(axis=1)).all()
    used = donor >= 0
    np.testing.assert_allclose(innovations(actual,fx)[used],fit.y[donor[used]],atol=1e-15)
    np.testing.assert_array_equal(clock_keys(actual.index)[used],fit.keys[donor[used]])
    assert not np.array_equal(actual.iloc[256:],b.iloc[256:])


def test_validation_mutation_does_not_change_fitted_sampler():
    b,x,m = example()
    end = b.index[900]
    first = ConditionalJointSampler().fit(b,x,m,end)
    b.iloc[900:] *= 50
    x.iloc[900:] *= 100
    m.iloc[900:] += 10000
    second = ConditionalJointSampler().fit(b,x,m,end)
    np.testing.assert_array_equal(first.y,second.y)
    np.testing.assert_array_equal(first.context,second.context)
    np.testing.assert_array_equal(first.mean,second.mean)


def test_missing_source_mask_preserved_not_filled():
    b,x,m = example()
    x.iloc[700,2] = np.nan
    fit = ConditionalJointSampler().fit(b,x,m,b.index[-1]+pd.Timedelta(hours=1))
    _,fx,_ = fit.generate(b,x,m,19)
    np.testing.assert_array_equal(np.isfinite(x),np.isfinite(fx))


def test_unavailable_initial_cross_and_config_refuse():
    with pytest.raises(ValueError,match='CONFIG'):
        ConditionalJointSampler(neighbors=0)
    b,x,m = example()
    fit = ConditionalJointSampler().fit(b,x,m,b.index[-1]+pd.Timedelta(hours=1))
    x.iloc[255,0] = np.nan
    with pytest.raises(ValueError,match='INITIAL_CROSS'):
        fit.generate(b,x,m,19)
