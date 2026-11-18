from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from app.temporal_actor_lr import split_temporal_optimizer, named_moments
from app.training_forensics import tree_digest
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


def model():
    torch.manual_seed(19)
    loss = torch.nn.Module()
    loss.actor_network_params = torch.nn.Module()
    loss.actor_network_params.encoder = torch.nn.Module()
    for name in ('post_fusion', 'core', 'branches'):
        setattr(loss.actor_network_params.encoder, name, torch.nn.Linear(3, 3))
    loss.actor_network_params.head = torch.nn.Linear(3, 3)
    loss.qvalue_network_params = torch.nn.Linear(3, 3)
    loss.log_alpha = torch.nn.Parameter(torch.zeros(()))
    return SimpleNamespace(loss=loss, optimizer=torch.optim.AdamW(loss.parameters(), lr=3e-5))


def update(m):
    m.optimizer.zero_grad()
    sum((p.square().sum() + p.sum()) for p in m.loss.parameters()).backward()
    m.optimizer.step()


def test_target_partition_preserves_weights_moments_and_other_options():
    m=model(); update(m)
    weights=tree_digest(m.loss.state_dict()); moments=tree_digest(named_moments(m))
    receipt=split_temporal_optimizer(m,.2)
    assert tree_digest(m.loss.state_dict()) == weights
    assert tree_digest(named_moments(m)) == moments
    names=[n for g in receipt for n in g['names']]
    assert len(names) == len(set(names)) == len(list(m.loss.parameters()))
    for g in receipt:
        assert g['lr'] == pytest.approx(6e-6 if g['group']=='actor_temporal' else 3e-5)
    assert all(g['weight_decay']==.01 for g in m.optimizer.param_groups)


def test_unit_factor_grouping_is_exact_control():
    a=model(); update(a); b=model()
    b.loss.load_state_dict(a.loss.state_dict())
    b.optimizer.load_state_dict(deepcopy(a.optimizer.state_dict()))
    split_temporal_optimizer(b,1.)
    update(a); update(b)
    assert tree_digest(a.loss.state_dict()) == tree_digest(b.loss.state_dict())
    assert tree_digest(named_moments(a)) == tree_digest(named_moments(b))


@pytest.mark.parametrize('factor',[0,-1,float('nan'),float('inf'),2])
def test_invalid_factors_rejected(factor):
    with pytest.raises(ValueError, match='TEMPORAL_LR_FACTOR'):
        split_temporal_optimizer(model(),factor)


def test_duplicate_partition_refused():
    m=model(); split_temporal_optimizer(m,.2)
    with pytest.raises(ValueError, match='ORIGINAL_SINGLE'):
        split_temporal_optimizer(m,.2)


def test_missing_attention_refused():
    m=model(); del m.loss.actor_network_params.encoder.post_fusion
    m.optimizer=torch.optim.AdamW(m.loss.parameters(),lr=3e-5)
    with pytest.raises(ValueError, match='TEMPORAL_COMPONENTS'):
        split_temporal_optimizer(m,.2)


@pytest.mark.parametrize('factor',[1.,.2])
def test_native_sac_cold_resume_preserves_grouped_optimizer(tmp_path,request,factor):
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from app.nested_history_training import training_design
    from app.three_signal_learning import synthetic_env
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tests.unit.test_causal_patch_attention import ARCH
    from tools.run_structural_sac_refit import fit_epochs
    from tools.run_sac_training_reliability import reseed
    torch.set_num_threads(1)
    recipe=probe_recipe();risk=request.getfixturevalue('curve_env').risk_config
    factory=lambda:synthetic_env(recipe,risk,101,1,.99)[0]
    probe=factory().inputs.windows[30:34].copy()
    def fresh():
        m=ThreeSignalSAC(recipe,seed=7,architecture=ARCH)
        split_temporal_optimizer(m,factor)
        return m
    def assess(m):
        return {p:dict(return_per_week=0.,orders_per_week=0.) for p in ('TRAIN','VALIDATION')}
    d=training_design(dict(replay_size=256,batch_size=8,horizon=6))
    d.update(max_epochs=3,minimum_epochs=0,matched_updates_per_epoch=4,diagnostics=False)
    reseed(7);m=fresh()
    fit_epochs(m,factory,assess,d,7,tmp_path/'full','test',probe)
    expected=tree_digest(m.loss.state_dict()); adam=tree_digest(named_moments(m))
    reseed(7);m=fresh()
    fit_epochs(m,factory,assess,dict(d,max_epochs=2),7,tmp_path/'part','test',probe)
    reseed(123);m=fresh()
    state=fit_epochs(m,factory,assess,d,7,tmp_path/'part','test',probe)
    assert state['updates']==8
    assert tree_digest(m.loss.state_dict())==expected
    assert tree_digest(named_moments(m))==adam
    assert m.optimizer.param_groups[0]['lr']==pytest.approx(m.effective['learning_rate']*factor)


def test_continuation_design_is_fixed_budget_not_validation_stopping():
    from tools.run_temporal_actor_lr import design_for
    from app.nested_history_training import training_design
    source=dict(design=training_design(dict(replay_size=65536,batch_size=128,horizon=6)))
    d=design_for(source)
    assert d['max_epochs']==d['minimum_epochs']==121
    assert d['matched_budget'] and d['matched_updates_per_epoch']==6006
    assert 'actor_plateau' not in d
    assert source['design']['max_epochs']==101


def test_gpu_admission_allows_only_exact_declared_desktop_executable():
    from app.temporal_actor_lr import foreign_gpu_pids
    executables={1:'/desktop/browser',2:'/env/python',3:'/other/browser'}
    assert foreign_gpu_pids([1,2,3],executables.__getitem__,['/desktop/browser'])==[2,3]
    assert foreign_gpu_pids([1],executables.__getitem__,[])==[1]
