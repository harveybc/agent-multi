from copy import deepcopy

import numpy as np
import pytest
import torch

from app.actor_plateau import ActorPlateau
from app.best_plateau_rollback import restore_validation_best
from app.sac_training_reliability import split_optimizer
from app.structural_refit import export_bundle
from app.training_forensics import rng_state, tree_digest
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_branch_capacity_contrast import batch
from tests.unit.test_modular_temporal_architecture import grouped_recipe
from tests.unit.test_actor_plateau import fixture
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401
from tools.run_discrete_curve_probe import read, write


CONFIG = dict(factor=.5, threshold=1e-8, cooldown=0, min_actor_lr=1e-7,
              post_reduction_epochs=10, exact_patience_divisor=2, restore_best=True,
              stop_before_reduce=True)


def test_exact_tenth_miss_signed_improvement_and_restart():
    opt, _ = fixture()
    sch = ActorPlateau(opt, -.1, 20, 0, CONFIG)
    assert sch.scheduler.patience == 9
    for epoch in range(1, 10):
        assert not sch.observe(-.2, True, epoch)['reduced']
    assert sch.observe(999., False, 10)['reduced']
    assert sch.scheduler.best == -.1 and sch.earliest_stop() == 20
    assert opt.param_groups[0]['lr'] == 1.5e-5
    saved = sch.state_dict()
    opt2, _ = fixture(); opt2.load_state_dict(deepcopy(opt.state_dict()))
    cold = ActorPlateau(opt2, -.1, 20, 0, CONFIG); cold.load_state_dict(saved)
    for epoch in range(11, 22):
        score = -.1+2e-8 if epoch == 19 else -.2
        assert sch.observe(score, True, epoch) == cold.observe(score, True, epoch)
    assert sch.scheduler.best == -.1+2e-8


def source(tmp_path):
    torch.set_num_threads(1)
    m = ThreeSignalSAC(grouped_recipe(), critic_regularization={'kind':'core_dropout','p':.1})
    split_optimizer(m, dict(actor=1.,critic=1.,temperature=1.))
    m.update(batch())
    probe = np.zeros((4, 13, 2), np.float32)
    path = tmp_path/'bundles'/'epoch_0001'
    receipt = export_bundle(path, m, probe, dict(epoch=1, score=.5, run_identity='run'))
    state = dict(best_bundle=dict(relative_path='bundles/epoch_0001', **receipt),
                 stopping=dict(best=.5), epoch=11)
    return m, state, path


def test_complete_restore_cumulative_rates_and_optimizer_binding(tmp_path):
    m, state, path = source(tmp_path)
    original = deepcopy(m.loss.state_dict())
    opt = m.optimizer
    saved = deepcopy(opt.state_dict())
    for factor in (.5, .25):
        m.update(batch())
        rates = [saved['param_groups'][0]['lr']*factor, *[g['lr'] for g in saved['param_groups'][1:]]]
        before = tree_digest(rng_state())
        receipt = restore_validation_best(m, tmp_path, state, rates)
        assert tree_digest(rng_state()) == before
        assert m.optimizer is opt
        assert tree_digest(m.loss.state_dict()) == tree_digest(original)
        assert tree_digest(m.optimizer.state_dict()['state']) == tree_digest(saved['state'])
        assert [g['lr'] for g in opt.param_groups] == rates
        assert receipt['restored_epoch'] == 1 and receipt['global_epoch'] == 11


@pytest.mark.parametrize('tamper', ['bytes', 'score', 'path'])
def test_refuse_invalid_best_before_mutating_model(tmp_path, tamper):
    m, state, path = source(tmp_path)
    m.update(batch()); before = tree_digest(m.loss.state_dict())
    if tamper == 'bytes':
        (path/'model.pt').write_bytes(b'bad')
    elif tamper == 'score':
        state['stopping']['best'] = 1.
    else:
        state['best_bundle']['relative_path'] = '../elsewhere'
    with pytest.raises(ValueError):
        restore_validation_best(m, tmp_path, state, [1e-5,3e-4,3e-4])
    assert tree_digest(m.loss.state_dict()) == before


def test_integrated_rollback_restart_replay_and_early_stop(tmp_path, request):
    from tests.unit.test_double_tcn_probe import probe_recipe
    from app.three_signal_learning import synthetic_env
    from app.g3_replay import JointReplay
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)
    recipe = probe_recipe(); base = request.getfixturevalue('curve_env')
    factory = lambda: synthetic_env(recipe, base.risk_config, 101, 1, .99)[0]
    def make():
        m = ThreeSignalSAC(recipe, seed=7)
        split_optimizer(m, dict(actor=1.,critic=1.,temperature=1.))
        return m
    def assess(m):
        return {p:dict(return_per_week=1.,orders_per_week=3.) for p in ('TRAIN','VALIDATION')}
    # Smaller patience exercises the same half-patience rule and first post-drop stop.
    config = dict(CONFIG, post_reduction_epochs=2)
    d = dict(replay_size=64,batch_size=16,warmup=16,update_every=4,max_epochs=8,
        minimum_epochs=0,patience=4,min_delta=1e-8,monitor_mode='validation',monitor_start_epoch=1,
        initialize_epoch_zero=True,fork_epoch=0,actor_plateau=config,replay_backend='memmap')
    probe = factory().inputs.windows[30:34].copy()
    kwargs = dict(replay_factory=lambda real: JointReplay(real,None,16,207))
    a = fit_epochs(make(),factory,assess,d,7,tmp_path/'a','same',probe,**kwargs)
    fit_epochs(make(),factory,assess,dict(d,max_epochs=2),7,tmp_path/'b','same',probe,**kwargs)
    b = fit_epochs(make(),factory,assess,d,7,tmp_path/'b','same',probe,**kwargs)
    assert a['epoch'] == b['epoch'] == 4
    assert a['stopping'] == b['stopping'] and a['stopping']['bad'] == 4
    assert a['actor_plateau'] == b['actor_plateau']
    assert a['history'][1]['best_rollback']['restored_epoch'] == 0
    assert a['history'][3]['actor_plateau']['skipped_for_early_stop']
    assert a['history'][1]['continuation_metrics'] == read(tmp_path/'a'/'EPOCH_ZERO.json')['metrics']
    assert [x['mean_update'] for x in a['history']] == [x['mean_update'] for x in b['history']]
    for name in ('model.pt','learning.pt'):
        left = torch.load(tmp_path/'a'/'recovery'/'epoch_0004'/name,weights_only=False)
        right = torch.load(tmp_path/'b'/'recovery'/'epoch_0004'/name,weights_only=False)
        assert tree_digest(left) == tree_digest(right)
