import numpy as np
import pytest


def spec(**kwargs):
    return dict(seed=7, drift_before=.0001,drift_after=-.0001,vol_before=.0001,
        vol_after=.0004,switch=60,transition=8,overlay=.00005,
        noise='clustered',shock='gap',mirror=False,**kwargs)


def test_exact_mirror_uses_same_random_realization_and_clock():
    from app.stochastic_strategy_paths import market_path
    a=market_path(spec());b=market_path({**spec(),'mirror':True})
    assert a.index.equals(b.index)
    for x,y in [('OPEN','OPEN'),('CLOSE','CLOSE'),('HIGH','LOW'),('LOW','HIGH')]:
        np.testing.assert_allclose(a[x]+b[y],2.2,rtol=0,atol=1e-15)


@pytest.mark.parametrize('kind',['gaussian','student','clustered'])
def test_seed_replay_real_noise_and_ohlc(kind):
    from app.stochastic_strategy_paths import market_path
    s={**spec(),'noise':kind}
    a=market_path(s);b=market_path(s);c=market_path({**s,'seed':19})
    assert a.equals(b) and not a.equals(c)
    assert (a.HIGH>=a[['OPEN','CLOSE']].max(axis=1)).all()
    assert (a.LOW<=a[['OPEN','CLOSE']].min(axis=1)).all()
    assert (a>0).all().all() and np.isfinite(a.to_numpy()).all()


def test_gaps_are_real_and_prefix_does_not_depend_on_future_length():
    from app.stochastic_strategy_paths import market_path
    a=market_path(spec(),target_bars=240)
    b=market_path(spec(),target_bars=480)
    assert a.equals(b.iloc[:len(a)])
    assert np.count_nonzero(abs(a.OPEN.to_numpy()[1:]-a.CLOSE.to_numpy()[:-1])>1e-8)>0


def test_clock_uses_observed_sessions_and_covers_dst():
    from app.stochastic_strategy_paths import market_path
    a=market_path(spec(),target_bars=480)
    t=a.index.tz_convert('America/New_York')
    opens=t[np.r_[True,np.diff(a.index.asi8)>3600*10**9]]
    assert all(x.hour==17 and x.weekday()==6 for x in opens)
    assert len(set(x.utcoffset().total_seconds() for x in opens))==2
    assert not any(x.weekday()==5 for x in t)


def test_bad_variants_rejected():
    from app.stochastic_strategy_paths import market_path
    with pytest.raises(ValueError):market_path({**spec(),'noise':'unknown'})
    with pytest.raises(ValueError):market_path({**spec(),'vol_before':-1})
