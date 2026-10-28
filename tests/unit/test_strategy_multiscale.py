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


def test_ideal_rank_encoding_preserves_expected_winner_after_float32_cast():
    from tools.systematic_strategy_matrix import rank_scores,action_from_head
    values={0:0.,1:7.276e-12,2:-7.276e-12}
    scores=rank_scores(values)
    for scale in (1e-4,1.,10.):
        action=action_from_head(None,{'action_mask':np.array([1,1,1,0,0],np.float32)},scores,scale)
        assert int(action[0])==1


def test_deterministic_head_does_not_erase_representable_score_difference():
    import torch
    from agent_plugins.hybrid_orders import OrderDistribution
    logits=torch.tensor([[0.,1e-8,0.,0.,0.]])
    mask=torch.tensor([[True,True,True,False,False]])
    dist=OrderDistribution(logits,torch.zeros(1,2,6),mask,False)
    assert int(dist.get_actions(True)[0,0])==1
