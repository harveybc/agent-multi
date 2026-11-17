from copy import deepcopy

import pytest

from app.structural_refit import EpochPatience
from app.actor_plateau import ActorPlateau
from tests.unit.test_actor_plateau import fixture
from tests.unit.test_best_plateau_rollback import CONFIG


def test_signed_min_and_no_mean_compensation():
    m = EpochPatience(mode='balanced_min')
    assert m.observe(-30., 3.)['improved']
    assert m.observe(-10., 3.)['improved']
    assert m.best == -10.
    assert m.observe(100., -2.)['improved']
    assert m.observe(8., 5.)['improved']
    assert not m.observe(1000., 4.)['improved']
    assert m.best == 5.
    assert not m.observe(8., 6., eligible=False)['improved']
    assert not m.observe(5., 9.)['improved']


def test_strict_uses_champion_not_previous_and_allows_negative():
    m = EpochPatience(mode='champion_nonregression')
    assert m.observe(-10., -5.)['improved']
    assert not m.observe(-30., 2.)['improved']
    assert not m.observe(-20., 3.)['improved']
    assert (m.best_train, m.best_validation) == (-10., -5.)
    assert m.observe(-9., -4.)['improved']
    assert (m.best_train, m.best_validation, m.bad) == (-9., -4., 0)
    assert not m.observe(100., -6.)['improved']
    assert not m.observe(-8., -4. + 1e-9)['improved']


@pytest.mark.parametrize('mode', ['balanced_min', 'champion_nonregression'])
def test_shared_scheduler_rejects_unaccepted_validation_record(mode):
    opt, _ = fixture()
    m = EpochPatience(mode=mode)
    m.observe(2., 1.)
    scheduler = ActorPlateau(opt, m.best, 20, 1, CONFIG)
    for epoch in range(2, 12):
        verdict = m.observe(-1., 20.)
        r = scheduler.observe(verdict['score'], verdict['improved'], epoch)
        assert scheduler.scheduler.best == m.best == 1.
        assert r['reduced'] == (epoch == 11)
    assert m.bad == 10


@pytest.mark.parametrize('mode', ['balanced_min', 'champion_nonregression'])
def test_monitor_state_cold_resume(mode):
    m = EpochPatience(mode=mode, start_epoch=3, select_during_warmup=True)
    for t, v in [(1., 1.), (.5, 2.), (2., 2.)]:
        m.observe(t, v)
    cold = EpochPatience(mode=mode, start_epoch=3, select_during_warmup=True)
    cold.__dict__.update(deepcopy(vars(m)))
    for t, v in [(0., 4.), (3., 3.), (20., 2.), (4., 4.)]:
        assert m.observe(t, v) == cold.observe(t, v)


def test_legacy_shadow_modes_remain_stable_for_old_recoveries():
    assert EpochPatience.MODES == ('mean', 'validation', 'mean_no_regression')


@pytest.mark.parametrize('mode', ['balanced_min', 'champion_nonregression'])
@pytest.mark.parametrize('scope', ['actor', 'all'])
def test_native_integrated_rejected_val_record_rollback_restart(tmp_path, request, mode, scope):
    import torch
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from app.g3_replay import JointReplay
    from app.sac_training_reliability import split_optimizer
    from app.three_signal_learning import synthetic_env
    from app.training_forensics import tree_digest
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)
    recipe = probe_recipe()
    base = request.getfixturevalue('curve_env')
    factory = lambda: synthetic_env(recipe, base.risk_config, 101, 1, .99)[0]
    def model():
        m = ThreeSignalSAC(recipe, seed=7)
        split_optimizer(m, dict(actor=1., critic=1., temperature=1.))
        return m
    def assessor(initial):
        def assess(m):
            nonlocal initial
            t, v = (1., 1.) if initial else (.5, 2.)
            initial = False
            return dict(TRAIN=dict(return_per_week=t, orders_per_week=3.),
                        VALIDATION=dict(return_per_week=v, orders_per_week=3.))
        return assess
    design = dict(replay_size=64, batch_size=16, warmup=16, update_every=4, max_epochs=8,
        minimum_epochs=0, patience=4, min_delta=1e-8, monitor_mode=mode, shared_monitor_decision=True,
        monitor_start_epoch=1, initialize_epoch_zero=True, fork_epoch=0,
        actor_plateau=dict(CONFIG, post_reduction_epochs=2, scope=scope), replay_backend='memmap')
    probe = factory().inputs.windows[30:34].copy()
    kwargs = dict(replay_factory=lambda real: JointReplay(real, None, 16, 207))
    a = fit_epochs(model(), factory, assessor(True), design, 7, tmp_path/'a', 'same', probe, **kwargs)
    fit_epochs(model(), factory, assessor(True), dict(design, max_epochs=2), 7, tmp_path/'b', 'same', probe, **kwargs)
    b = fit_epochs(model(), factory, assessor(False), design, 7, tmp_path/'b', 'same', probe, **kwargs)
    assert a['epoch'] == b['epoch'] == 4
    assert a['stopping'] == b['stopping']
    assert a['actor_plateau'] == b['actor_plateau']
    assert a['stopping']['best'] == a['actor_plateau']['scheduler']['best'] == 1.
    assert a['history'][1]['best_rollback']['restored_epoch'] == 0
    rates = a['history'][1]['actor_plateau']['after']
    assert (rates[0] == rates[1] == rates[2]) == (scope == 'all')
    assert [h['mean_update'] for h in a['history']] == [h['mean_update'] for h in b['history']]
    for name in ('model.pt', 'learning.pt'):
        left = torch.load(tmp_path/'a'/'recovery'/'epoch_0004'/name, weights_only=False)
        right = torch.load(tmp_path/'b'/'recovery'/'epoch_0004'/name, weights_only=False)
        assert tree_digest(left) == tree_digest(right)


from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401, E402
