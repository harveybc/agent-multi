from copy import deepcopy

import pytest
import torch

from app.actor_plateau import ActorPlateau
from tools.run_branch_capacity_contrast import refinement_design


def test_frozen_refinement_design_and_lr_binding():
    base = dict(patience=20, max_epochs=2000, minimum_epochs=100)
    original = deepcopy(base)
    d = refinement_design(base, 3e-5, [3e-5]*3)
    assert base == original
    assert d['patience'] == 80
    assert d['monitor_mode'] == 'balanced_min'
    assert d['shared_monitor_decision']
    assert d['actor_plateau']['scope'] == 'all'
    assert d['actor_plateau']['factor'] == .2
    assert d['actor_plateau']['exact_patience_divisor'] == 4
    for rates in ([3e-6]*3, [3e-6,3e-5,3e-6]):
        with pytest.raises(ValueError, match='REFINEMENT_START_LR'):
            refinement_design(base, 3e-5, rates)


def scheduler():
    params = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    opt = torch.optim.AdamW([dict(params=[p], lr=3e-5) for p in params])
    config = refinement_design({}, 3e-5, [3e-5]*3)['actor_plateau']
    return opt, ActorPlateau(opt, .1, 80, 0, config)


def test_four_exact_drops_all_roles_floor_and_recovery():
    opt, sch = scheduler()
    events = []
    for epoch in range(1, 102):
        row = sch.observe(.09, False, epoch)
        if row['reduced']:
            events.append(epoch)
        assert row['after'][0] == row['after'][1] == row['after'][2]
        if epoch == 37:
            other_opt, other = scheduler()
            other_opt.load_state_dict(deepcopy(opt.state_dict()))
            other.load_state_dict(sch.state_dict())
        if epoch > 37:
            assert row == other.observe(.09, False, epoch)
    assert events == [20,40,60,80]
    assert row['after'] == pytest.approx([4.8e-8]*3)
    assert sch.earliest_stop() == 100


def test_accepted_improvement_restarts_scheduler_count():
    _, sch = scheduler()
    for epoch in range(1,20):
        assert not sch.observe(.09, False, epoch)['reduced']
    assert sch.observe(.11, True, 20)['bad'] == 0
    for epoch in range(21,40):
        assert not sch.observe(.09, False, epoch)['reduced']
    assert sch.observe(.09, False, 40)['reduced']


def test_invalid_scope_rejected():
    opt, sch = scheduler()
    with pytest.raises(ValueError, match='ACTOR_PLATEAU_SCOPE'):
        ActorPlateau(opt, .1, 80, 0, dict(sch.config, scope='unknown'))
