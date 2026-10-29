import copy
import math

import pytest

from app.horizon_risk_rules import coefficients, levels, propose


class SizingFixture:
    rel_volume = .1
    leverage = 50
    basis = 'margin'
    min_units = 1

    def units_max(self, balance, price):
        return math.floor(balance*self.rel_volume*self.leverage/price)

    def order_units(self, balance, price, percentual_order_size, risk_accepts_minimum):
        assert risk_accepts_minimum is False
        units = math.floor(self.units_max(balance, price)*percentual_order_size)
        return units if units >= self.min_units else None

    def required_margin(self, units, price):
        return units*price/self.leverage


def proposal(**changes):
    args = dict(direction=1, entry_quote=1.1, liquidation_quote=1.0999,
                balance=10000, tp_reference=.003, mae_reference=.001,
                sigma=.001, genes=coefficients(), sizing=SizingFixture(),
                tick=.00001, min_distance=.00005, quality=1.,
                loss_cost_per_unit=.00014, funding_reserve_per_unit=0.)
    args.update(changes)
    return propose(**args)


def test_default_geometry_and_mirror():
    a = levels(1, 1.1, .003, .001, .001, coefficients(), .00001)
    b = levels(-1, 1.1, .003, .001, .001, coefficients(), .00001)
    assert a['sl'] == pytest.approx(1.0989)
    assert a['tp'] == pytest.approx(1.103)
    assert b['sl'] == pytest.approx(1.1011)
    assert b['tp'] == pytest.approx(1.097)


@pytest.mark.parametrize('name,field,relation', [
    ('a_sl', 'sl', 'less'), ('b_sl', 'sl', 'less'),
    ('buffer_pips', 'sl', 'less'), ('a_tp', 'tp', 'greater'),
    ('b_tp', 'tp', 'less'),
])
def test_each_geometry_gene_changes_its_output(name, field, relation):
    genes = coefficients()
    a = levels(1, 1.1, .003, .001, .001, genes, .00001)
    genes[name] += .5
    b = levels(1, 1.1, .003, .001, .001, genes, .00001)
    assert (b[field] < a[field]) if relation == 'less' else (b[field] > a[field])


def test_wider_sl_reduces_q_and_rho_increases_it():
    g = coefficients(); g['rho'] = .001
    a = proposal(genes=g)
    wide = copy.deepcopy(g); wide['a_sl'] = 2
    b = proposal(genes=wide)
    assert b['q'] < a['q']
    assert a['estimated_loss'] <= 10 and b['estimated_loss'] <= 10
    g['rho'] = .002
    assert proposal(genes=g)['q'] > a['q']


def test_margin_single_factor_costs_quality_and_zero():
    a = proposal(genes=coefficients(rho=.01))
    assert a['q'] == 1
    assert 999 < a['margin'] <= 1000
    assert a['loss_per_unit'] == pytest.approx(.00124)
    b = proposal(quality=.3, genes=coefficients(rho=.01))
    assert b['q'] == pytest.approx(.3)
    assert 299 < b['margin'] <= 300
    assert proposal(quality=0)['status'] == 'NOP'


def test_no_minimum_round_up_and_spread_not_debited_twice():
    sizing = SizingFixture(); sizing.min_units = 100000
    assert proposal(sizing=sizing)['reason'] == 'BELOW_MINIMUM_UNITS'
    a = proposal(liquidation_quote=1.0999)
    b = proposal(liquidation_quote=1.0998)
    assert a['loss_per_unit'] == b['loss_per_unit']
    assert proposal(liquidation_quote=1.098)['reason'] == 'LEVELS_NOT_EXECUTABLE'


def test_zero_mae_needs_cushion_and_tp_is_not_inflated():
    # One pip cushion cannot survive one pip spread plus a minimum stop distance.
    assert proposal(mae_reference=0)['status'] == 'NOP'
    assert proposal(mae_reference=0, liquidation_quote=1.09998)['status'] == 'ORDER'
    assert proposal(mae_reference=0, genes=coefficients(buffer_pips=0))['status'] == 'NOP'
    assert proposal(genes=coefficients(b_tp=4))['status'] == 'NOP'


def test_tick_rounding_is_conservative():
    a = levels(1, 1.100003, .003007, .001001, .001, coefficients(), .00001)
    assert a['sl'] <= 1.100003-.001101
    assert a['tp'] <= 1.100003+.003007


@pytest.mark.parametrize('change', [dict(direction=0), dict(quality=1.1),
    dict(loss_cost_per_unit=-1), dict(sigma=float('nan')), dict(tick=0),
    dict(funding_reserve_per_unit=-1), dict(balance=0), dict(liquidation_quote=1.2)])
def test_invalid_inputs(change):
    with pytest.raises(ValueError):
        proposal(**change)


def test_unknown_gene_and_contract_mismatch():
    with pytest.raises(ValueError):
        coefficients(surprise=1)
    sizing = SizingFixture(); sizing.leverage = 100
    with pytest.raises(ValueError, match='BUSINESS'):
        proposal(sizing=sizing)


def test_funding_reserve_reduces_q_when_budget_binds():
    args = dict(genes=coefficients(rho=.001))
    a = proposal(**args)
    b = proposal(**args, funding_reserve_per_unit=.0005)
    assert b['q'] < a['q']
    assert b['estimated_loss'] <= b['risk_budget']


def test_config_keeps_business_and_reference_values_fixed():
    from tools.verify_horizon_risk_rules import make_config, validate_config
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2]/'docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json'
    refs = make_config(json.loads(path.read_text()), 'digest')
    candidate = copy.deepcopy(refs)
    candidate['axes']['rho'] = [.002, .006]
    validate_config(candidate, refs)
    candidate['rows'][0]['reference']['median']['tp_pips'] += 1
    with pytest.raises(ValueError, match='CONFIG_FIXED'):
        validate_config(candidate, refs)
    candidate = copy.deepcopy(refs); candidate['leverage'] = 100
    with pytest.raises(ValueError, match='CONFIG_FIXED'):
        validate_config(candidate, refs)
    candidate = copy.deepcopy(refs); candidate['axes']['rho'] = []
    with pytest.raises(ValueError, match='CONFIG_EMPTY'):
        validate_config(candidate, refs)
    candidate = copy.deepcopy(refs); candidate['probe_fixed']['quality'] = .5
    with pytest.raises(ValueError, match='CONFIG_PROBE'):
        validate_config(candidate, refs)


def test_json_key_order_does_not_change_grid_order():
    import json
    from tools.verify_horizon_risk_rules import grid_cells
    config = dict(axes=dict(z=[1, 2], a=[3, 4]))
    restored = json.loads(json.dumps(config, sort_keys=True))
    assert list(grid_cells(config)) == list(grid_cells(restored))
