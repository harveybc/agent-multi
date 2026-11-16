from copy import deepcopy

import pytest
import torch

from app.actor_plateau import ActorPlateau


CONFIG = dict(factor=.5, threshold=1e-8, cooldown=5, min_actor_lr=1e-7, post_reduction_epochs=20)


def fixture():
    ps = [torch.nn.Parameter(torch.tensor(1.)) for _ in range(3)]
    opt = torch.optim.AdamW([{'params':[p], 'lr':lr} for p,lr in zip(ps,[3e-5,3e-5,3e-5])])
    return opt, ActorPlateau(opt, -.1, 20, 40, CONFIG)


def test_eighth_miss_actor_only_cooldown_and_floor():
    opt, sch = fixture()
    for ep in range(41,48):
        assert not sch.observe(-.2,True,ep)['reduced']
    assert sch.observe(-.2,True,48)['after'] == [1.5e-5,3e-5,3e-5]
    assert sch.earliest_stop() == 68
    for ep in range(49,54):
        assert sch.observe(-.2,True,ep)['bad'] == 0
    for ep in range(54,240):
        sch.observe(-.2,True,ep)
    assert [g['lr'] for g in opt.param_groups] == [1e-7,3e-5,3e-5]


def test_signed_absolute_improvement_and_ineligible():
    _, sch = fixture()
    assert sch.observe(-.1+5e-9,True,41)['bad'] == 1
    assert sch.observe(-.1+2e-8,True,42)['bad'] == 0
    best = sch.scheduler.best
    assert sch.observe(10.,False,43)['bad'] == 1
    assert sch.scheduler.best == best
    for x in (float('nan'),float('inf')):
        with pytest.raises(ValueError):sch.observe(x,True,44)


def test_resume_identical_lr_events_and_historical_best():
    opt, sch = fixture()
    for ep in range(41,50):sch.observe(-.2,True,ep)
    saved_opt, saved = deepcopy(opt.state_dict()), sch.state_dict()
    opt2, other = fixture()
    opt2.load_state_dict(saved_opt)
    other.load_state_dict(saved)
    for ep in range(50,80):
        assert sch.observe(-.2,True,ep) == other.observe(-.2,True,ep)
    assert sch.scheduler.best == -.1
    assert sch.earliest_stop() == other.earliest_stop()


def test_role_rate_mutation_and_counter_rejected():
    opt, sch = fixture()
    with pytest.raises(ValueError):sch.observe(-.2,True,42)
    opt.param_groups[1]['lr'] *= .5
    with pytest.raises(ValueError):sch.observe(-.2,True,41)
