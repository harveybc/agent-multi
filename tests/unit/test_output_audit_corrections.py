from types import SimpleNamespace

import numpy as np
import pytest

from tools.output_audit_corrections import BlindPolicy, trade_economics, require_gene_effects, realized_signature


def trade(reason='TP', net=8.):
    return dict(direction=1, entry_price=1., take_profit=1.002, stop_loss=.998,
                units=10000, gross_usd=10., net_usd=net, spread_usd=1.,
                slippage_usd=.5, commission_usd=.5, financing_usd=0., close_reason=reason)


def test_break_even_is_binary_reference_not_realized_exit_forecast():
    r=trade_economics([trade(),trade('FORCED_FLAT',-1.)])
    assert r['mean_tp_pips']==pytest.approx(20)
    assert r['mean_sl_pips']==pytest.approx(20)
    assert r['mean_cost_pips']==pytest.approx(2)
    assert r['binary_break_even_reference']==pytest.approx(.55)
    assert r['realized_win_rate']==.5
    assert r['non_barrier_exits']==1
    assert r['binary_model_applies_to_all_exits'] is False
    assert r['financing_usd']==0
    assert trade_economics([])['binary_break_even_reference'] is None


def test_blind_rules_ignore_features_preserve_masks_and_seed():
    model=SimpleNamespace(policy=object())
    flat=dict(features=np.full((13,2),np.nan),action_mask=np.array([1,1,1,0,0]))
    for rule,expected in [('LONG_HOLD',1),('SHORT_HOLD',2)]:
        p=BlindPolicy(model,rule,7)
        assert p.predict(flat)[0][0]==expected
        assert p.predict(dict(action_mask=np.array([1,0,0,1,0])))[0][0]==0
    a=BlindPolicy(model,'COIN_HOLD',7);b=BlindPolicy(model,'COIN_HOLD',7)
    commands=[int(a.predict(flat)[0][0]) for _ in range(100)]
    assert commands==[int(b.predict(flat)[0][0]) for _ in range(100)]
    assert set(commands)=={1,2}


def test_unidentified_gene_cannot_be_called_calibrated():
    with pytest.raises(ValueError,match='UNIDENTIFIED_GENES:sl_long,sl_short'):
        require_gene_effects(dict(tp_long=True,sl_long=False,sl_short=False))
    require_gene_effects(dict(tp_long=True,sl_long=True))


def test_unhit_stop_metadata_is_not_an_execution_effect():
    import copy
    original=dict(metrics={'net_profit_usd':8.},commands={'NOP':1},trades=[trade()])
    changed=copy.deepcopy(original);changed['trades'][0]['stop_loss']=.997
    assert realized_signature(original)==realized_signature(changed)
    changed['trades'][0]['net_usd']=7.
    assert realized_signature(original)!=realized_signature(changed)
