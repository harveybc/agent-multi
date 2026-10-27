import json
from pathlib import Path

import numpy as np
import pytest

from app.strategy_scenario_matrix import cells, market_frame, WARMUP

DESIGN = Path(__file__).resolve().parents[2] / 'configs/validation/strategy_multiscale_matrix.json'


def test_multiscale_cartesian_has_all_combinations():
    design = json.loads(DESIGN.read_text())
    rows = list(cells(design))
    assert len(rows) == 47628
    assert len({json.dumps(r, sort_keys=True) for r in rows}) == len(rows)
    for key, values in design['axes'].items():
        assert {r[key] for r in rows} == set(values)
        for value in values:
            assert sum(r[key] == value for r in rows) == len(rows) // len(values)


def test_generator_preserves_prefix_and_distinguishes_transition_speeds():
    row = next(cells(json.loads(DESIGN.read_text())))
    row.update(drift_before=-.00015, drift_after=.00045, vol_before=0., vol_after=0.)
    a = market_frame({**row, 'transition_bars': 0})
    b = market_frame({**row, 'transition_bars': 12})
    k = WARMUP + row['switch']
    np.testing.assert_array_equal(a.values[:k], b.values[:k])
    assert len(b) == WARMUP + 240
    increments = b.CLOSE.to_numpy() - b.OPEN.to_numpy()
    assert increments[k] == pytest.approx(-.00015)
    assert increments[k+12] == pytest.approx(.00045)
    assert increments[k+6] == pytest.approx(.00015)
    assert (b.LOW > 0).all()


def test_invalid_axis_or_count_fails_before_execution():
    d = json.loads(DESIGN.read_text())
    d['expected_cells'] = 1
    with pytest.raises(ValueError): list(cells(d))
    d = json.loads(DESIGN.read_text())
    d['axes']['horizon'] = [0]
    with pytest.raises(ValueError): list(cells(d))


def test_prescribed_commands_use_literal_expected_answers():
    from tools.systematic_strategy_matrix import prescribed_head_checks
    assert prescribed_head_checks() == 21


def test_original_grid_remains_unchanged():
    assert len(list(cells())) == 5832
