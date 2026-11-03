import numpy as np
import pytest
import torch

from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_rl_subset import split_recipe, frame
from app.rl_subset import refit_train_scaler, subset_recipe
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.structural_refit import EpochPatience, export_bundle, load_bundle, capture_rng, restore_rng
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def test_patience_mean_tiny_improvement_reset_and_twenty_failures():
    p = EpochPatience(20, 1e-8)
    assert p.observe(2., -4.)['improved']
    assert p.best == -1.
    assert not p.observe(2., -4.)['stop'] and p.bad == 1
    assert p.observe(2., -4. + 4e-8)['improved'] and p.bad == 0
    for _ in range(19):
        assert not p.observe(2., -4.)['stop']
    assert p.observe(2., -4.)['stop'] and p.bad == 20
    with pytest.raises(ValueError, match='FINITE'):
        p.observe(float('nan'), 0.)


def test_validation_only_cannot_be_compensated_by_train():
    p = EpochPatience(20,1e-8,mode='validation')
    assert p.observe(0.,2.1)['improved']
    assert not p.observe(10000.,-20.)['improved']
    assert p.best == 2.1 and p.bad == 1
    assert p.observe(-100.,2.1+2e-8)['improved'] and p.bad == 0


def test_guarded_mean_vetoes_each_component_and_uses_previous_epoch():
    p = EpochPatience(20,1e-8,mode='mean_no_regression')
    assert p.observe(2.,2.)['improved']
    a = p.observe(20.,1.)
    assert not a['improved'] and 'VALIDATION_DECREASE' in a['rejections']
    b = p.observe(19.,10.)
    assert not b['improved'] and 'TRAIN_DECREASE' in b['rejections']
    assert p.best == 2. and p.bad == 2
    assert p.observe(20.,10.)['improved'] and p.bad == 0
    assert not p.observe(20.,10.)['improved']  # Equality cannot establish a new best.
    assert not p.observe(1.,1.)['improved']
    a = p.observe(25.,9.)
    assert a['improved'] and a['previous_validation'] == 1.
    assert a['best_validation'] == 9.  # Explicitly not a best-checkpoint guard.


def test_guarded_recovery_needs_new_historical_accepted_best():
    p = EpochPatience(20,1e-8,mode='mean_no_regression')
    p.observe(10.,10.); p.observe(1.,1.)
    assert not p.observe(2.,2.)['improved']
    assert p.bad == 2 and p.best == 10.
    clone = EpochPatience(20,1e-8,mode='mean_no_regression')
    clone.__dict__.update(p.__dict__)
    assert clone.observe(11.,11.) == p.observe(11.,11.)
    with pytest.raises(ValueError,match='CONFIG'):
        EpochPatience(mode='unknown')


def test_bundle_reload_is_exact_without_training_and_rejects_tampering(tmp_path):
    model = ThreeSignalSAC(probe_recipe())
    x = np.random.default_rng(7).normal(size=(4, 13, 2)).astype('float32')
    rng = np.random.default_rng(8)
    before = capture_rng(rng)
    export_bundle(tmp_path/'best', model, x, {'epoch': 1, 'score': 0.})
    assert torch.equal(before['torch'], torch.get_rng_state())
    loaded, metadata = load_bundle(tmp_path/'best', 'cpu')
    with torch.no_grad():
        assert torch.equal(model.surfaces(torch.tensor(x)), loaded.surfaces(torch.tensor(x)))
    assert metadata['checkpoint']['epoch'] == 1
    with (tmp_path/'best'/'RECIPE.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='BUNDLE_HASH'):
        load_bundle(tmp_path/'best', 'cpu')


def test_projected_scaler_does_not_fit_validation():
    r = subset_recipe(split_recipe(), ['y']); f = frame()
    a = refit_train_scaler(r, f)
    f.loc[72:, 'y'] = 1e12
    assert refit_train_scaler(r, f)['scaler'] == a['scaler']


def test_rng_restoration_repeats_samples():
    rng = np.random.default_rng(11)
    state = capture_rng(rng)
    a, b = rng.integers(0, 100, 10), torch.rand(10)
    restore_rng(state, rng)
    assert np.array_equal(a, rng.integers(0, 100, 10))
    assert torch.equal(b, torch.rand(10))


@pytest.mark.parametrize('mode',['mean','validation','mean_no_regression'])
def test_epoch_resume_matches_uninterrupted_learning_and_preserves_best(tmp_path, request, mode):
    from app.three_signal_learning import synthetic_env, emit, evaluate
    from tools.run_structural_sac_refit import fit_epochs
    torch.set_num_threads(1)
    r = probe_recipe(); base = request.getfixturevalue('curve_env')
    factory = lambda: synthetic_env(r,base.risk_config,101,1,.99)[0]
    def assess(model):
        out = {}
        for period,seed in [('TRAIN',101),('VALIDATION',102)]:
            env = synthetic_env(r,base.risk_config,seed,1,.99)[0]
            logits,_ = emit(model,env.inputs)
            out[period] = evaluate(env,logits)
        return out
    d = dict(replay_size=64,batch_size=8,warmup=8,update_every=4,
             max_epochs=2,patience=20,min_delta=1e-8,monitor_mode=mode)
    probe = factory().inputs.windows[30:34].copy()
    uninterrupted = ThreeSignalSAC(r,seed=7)
    a = fit_epochs(uninterrupted,factory,assess,d,7,tmp_path/'a','test-id',probe)
    interrupted = ThreeSignalSAC(r,seed=7)
    fit_epochs(interrupted,factory,assess,dict(d,max_epochs=1),7,tmp_path/'b','test-id',probe)
    resumed = ThreeSignalSAC(r,seed=7)
    b = fit_epochs(resumed,factory,assess,d,7,tmp_path/'b','test-id',probe)
    assert a['steps'] == b['steps'] and a['updates'] == b['updates']
    assert a['stopping'] == b['stopping']
    assert [r['metrics'] for r in a['history']] == [r['metrics'] for r in b['history']]
    for name,value in uninterrupted.loss.state_dict().items():
        assert torch.equal(value,resumed.loss.state_dict()[name]), name
    assert (tmp_path/'b'/b['best_bundle']['relative_path']/'RELOAD.json').exists()
    # A completed epoch cannot be trained a second time on resume.
    c = fit_epochs(resumed,lambda:pytest.fail('unexpected training'),assess,d,7,tmp_path/'b','test-id',probe)
    assert c['steps'] == b['steps']
