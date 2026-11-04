import numpy as np
import pytest
import torch

from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_rl_subset import split_recipe, frame
from app.rl_subset import refit_train_scaler, subset_recipe
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.structural_refit import EpochPatience, export_bundle, load_bundle, capture_rng, restore_rng
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def selected_round_fixture():
    from app.rl_subset import digest
    r=split_recipe()
    spec=dict(dataset=dict(features=r['list']['columns'],scaler_hash=digest(r['scaler'])))
    spec['identity']=digest(spec)
    result=dict(identity=spec['identity'],result=dict(selected=['y'],termination='TWO_PASSES_COMPLETE'),
        original_support_preserved=True,unmasked_after_pass_exact=True,test_read=False,final=dict(eligible=True))
    close=dict(identity=spec['identity'],termination='TWO_PASSES_COMPLETE')
    return r,spec,result,close


def test_selected_round_removes_channels_and_empty_branch_without_refitting_scale():
    from tools.run_structural_sac_refit import selected_recipe_for_round
    r,spec,result,close=selected_round_fixture()
    out=selected_recipe_for_round(r,spec,result,close)
    assert out['list']['columns']==['y']
    assert out['representation']['resolved']['names']==['b']
    assert out['scaler']['mean']==[r['scaler']['mean'][1]]
    assert out['scaler']['std']==[r['scaler']['std'][1]]
    assert r['list']['columns']==['x','y']


@pytest.mark.parametrize('field,value',[('test_read',True),('original_support_preserved',False),
                                      ('unmasked_after_pass_exact',False),('identity','wrong')])
def test_selected_round_rejects_untrusted_receipt(field,value):
    from tools.run_structural_sac_refit import selected_recipe_for_round
    r,spec,result,close=selected_round_fixture(); result[field]=value
    with pytest.raises(ValueError,match='RECEIPT'):
        selected_recipe_for_round(r,spec,result,close)


def test_selected_round_rejects_wrong_order_and_partial_encoding():
    from tools.run_structural_sac_refit import selected_recipe_for_round
    from app.rl_subset import digest
    r,spec,result,close=selected_round_fixture()
    result['result']['selected']=['y','x']
    with pytest.raises(ValueError,match='SUBSET_ORDER'):
        selected_recipe_for_round(r,spec,result,close)
    r['list']['columns']=['hour_sin','hour_cos']
    spec['dataset']['features']=r['list']['columns']
    spec['identity']=digest({k:v for k,v in spec.items() if k!='identity'})
    result['identity']=close['identity']=spec['identity']
    result['result']['selected']=['hour_sin']
    with pytest.raises(ValueError,match='PARTIAL_ENCODING'):
        selected_recipe_for_round(r,spec,result,close)


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


@pytest.mark.parametrize('mode',EpochPatience.MODES)
def test_twenty_epoch_warmup_excludes_transient_bests(mode):
    p = EpochPatience(20,1e-8,mode=mode,start_epoch=20)
    for _ in range(19):
        r = p.observe(1000.,1000.)
        assert r['warming_up'] and not r['improved'] and not r['stop']
        assert p.best is None and p.bad == 0
    r = p.observe(-1.,-1.)
    assert r['improved'] and r['epoch'] == 20 and not r['warming_up']
    assert r['previous_train'] is None and p.bad == 0 and p.best == -1.
    for _ in range(19):
        assert not p.observe(-2.,-2.)['stop']
    assert p.observe(-2.,-2.)['stop'] and p.epoch == 40


def test_rebase_keeps_history_and_learning_counters_but_drops_early_best():
    from tools.run_structural_sac_refit import rebase_warmup_state
    old = dict(identity='old',epoch=2,steps=48050,updates=11000,best_bundle={'path':'early'},
        history=[dict(epoch=i,monitor={'score':100.},metrics={
            'TRAIN':{'return_per_week':100.},'VALIDATION':{'return_per_week':100.}}) for i in [1,2]])
    design = dict(monitor_start_epoch=20,max_epochs=2000,patience=20,min_delta=1e-8,monitor_mode='validation')
    new = rebase_warmup_state(old,design,'new')
    assert old['best_bundle'] == {'path':'early'}
    assert new['best_bundle'] is None and new['stopping']['best'] is None
    assert new['steps'] == old['steps'] and new['updates'] == old['updates']
    assert new['stopping']['epoch'] == 2 and new['stopping']['bad'] == 0
    assert all(r['monitor']['warming_up'] for r in new['history'])
    assert [r['metrics'] for r in new['history']] == [r['metrics'] for r in old['history']]
    with pytest.raises(ValueError,match='PREMONITOR'):
        rebase_warmup_state(old,dict(design,monitor_start_epoch=2),'new')


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
@pytest.mark.parametrize('start_epoch',[1,2])
def test_epoch_resume_matches_uninterrupted_learning_and_preserves_best(tmp_path, request, mode, start_epoch):
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
             max_epochs=2,patience=20,min_delta=1e-8,monitor_mode=mode,monitor_start_epoch=start_epoch)
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
