from dataclasses import replace

import numpy as np
import pytest

from app.close_hold_controller import CloseController, bind_values, value_paths
from tests.unit.test_close_hold_controller import position, report
from tests.unit import test_dynamic_account_panel as panel_fixtures

world = panel_fixtures.world


@pytest.mark.parametrize('changed', ['report', 'position', 'risk', 'values'])
def test_cached_value_binding_rejects_same_shape_corruption(changed):
    p = position()
    r = report(tag='synthetic_ideal')
    v = value_paths(p, r, np.zeros(3), np.zeros(3))
    binding = bind_values(p, r, v)
    identity, risk = p.identity, p.risk
    if changed == 'report':
        r = replace(r, paths=r.paths+.01)
    elif changed == 'position':
        identity += '-other'
    elif changed == 'risk':
        risk *= 2
    else:
        v = v+1
    with pytest.raises(ValueError, match='VALUE_BINDING'):
        CloseController(allow_synthetic=True).decide_synthetic_values(
            identity, 10, 1, v, r.weights, risk, report=r, expected_deadline=13, binding=binding)


def test_warm_report_and_bid_cache_refreshes_cost_config(world):
    old = world.report(20)
    bids = world.candidate_bids(20)
    f = world.env.cfg.profile
    world.env.cfg = replace(world.env.cfg, profile=replace(f,
        spread_schedule=((0, 24, .0009),), commission_per_lot_side=f.commission_per_lot_side+3))
    new = world.report(20)
    assert new is not old and not np.array_equal(new.spreads, old.spreads)
    assert world.candidate_bids(20) != bids
