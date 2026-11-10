import numpy as np
import pytest

from app.causal_stl_inputs import CausalSTLColumn, endpoint_series, encode_seasonal, choose_parameters


def config():return dict(period=6,seasonal=7,robust=False,history=64)


def test_prefix_future_poison_and_stream_parity():
    t=np.arange(140);x=.01*t+np.sin(2*np.pi*t/6)
    a=endpoint_series(x,config());b=endpoint_series(x[:100],config())
    assert np.array_equal(a[:100],b,equal_nan=True)
    changed=x.copy();changed[100:]=10000*np.random.default_rng(2).normal(size=40)
    assert np.array_equal(endpoint_series(changed,config())[:100],b,equal_nan=True)
    stream=CausalSTLColumn(config())
    assert np.array_equal([stream.update(v) for v in x],a,equal_nan=True)
    resumed=CausalSTLColumn(config(),history=stream.history)
    assert resumed.update(3.)==stream.update(3.)


def test_no_padding_and_nan_invalidates_history():
    x=np.sin(np.arange(150));x[70]=np.nan
    y=endpoint_series(x,config())
    assert np.isnan(y[:63]).all() and np.isfinite(y[63:70]).all()
    assert np.isnan(y[70:134]).all() and np.isfinite(y[134:]).all()


def test_train_only_encoding_and_missing_preserved():
    x=np.arange(40,dtype=float).reshape(20,2)-10;mask=np.arange(20)<10
    a=encode_seasonal(x,mask)
    changed=x.copy();changed[10:]*=100
    b=encode_seasonal(changed,mask)
    assert a['parameters']==b['parameters']
    assert set(np.unique(a['binary']))<={0.,1.}
    assert set(np.unique(a['ordinal']))<={-1.,-.5,0.,.5,1.}
    x[-1,0]=np.nan
    assert all(np.isnan(encode_seasonal(x,mask)[k][-1,0]) for k in ('continuous','ordinal','binary'))


def test_parameter_selection_never_reads_after_train():
    t=np.arange(300);x=.01*t+np.sin(t*2*np.pi/6)
    candidates=[config(),dict(config(),seasonal=13)]
    a=choose_parameters(x,250,candidates=candidates,origins_count=4)
    x[250:]=1e8
    b=choose_parameters(x,250,candidates=candidates,origins_count=4)
    assert a==b and a['scored_candidates']==2
    with pytest.raises(ValueError,match='ORIGINS'):
        choose_parameters(x,30,candidates=candidates)


def test_live_encoded_history_matches_offline_variants():
    from app.causal_stl_inputs import OnlineSTLInputs
    t=np.arange(130);x=np.column_stack([np.sin(t*2*np.pi/6),.03*t+np.cos(t*2*np.pi/6)])
    s=np.column_stack([endpoint_series(x[:,j],config()) for j in range(2)])
    encoded=encode_seasonal(s,t<100)
    for variant in ('continuous','ordinal','binary'):
        live=OnlineSTLInputs([config(),config()],encoded['parameters'],variant)
        emitted=np.array([live.update(row) for row in x])
        assert np.array_equal(emitted,encoded[variant],equal_nan=True)


def test_transformed_recipe_rejects_raw_loader():
    from tests.unit.test_double_tcn_probe import probe_recipe
    from app.rl_subset import seal_recipe
    from app.frozen_recipe_inputs import FrozenRecipeInputs
    rec=probe_recipe();rec['store']['input_adapter']={'kind':'causal_stl_seasonal_v1'};rec=seal_recipe(rec)
    with pytest.raises(ValueError,match='EXPLICIT_INPUT_ADAPTER'):
        FrozenRecipeInputs.from_frame(rec,None)
