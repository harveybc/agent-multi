import copy

import numpy as np
import pytest
import torch

from app.nested_history_training import updates_at_decision, training_design
from tests.unit.test_discrete_curve_native import curve_env  # noqa: F401


@pytest.mark.parametrize('decisions', [2780, 5891, 11841, 24025])
def test_all_histories_have_exact_same_gradient_budget(decisions):
    quota = 6006
    schedule = [updates_at_decision(i, decisions, quota) for i in range(1, decisions+1)]
    assert sum(schedule) == quota
    assert max(schedule)-min(schedule) <= 1
    assert sum(updates_at_decision(i, decisions, 0) for i in range(1, decisions+1)) == 0


@pytest.mark.parametrize('args', [(0,10,2),(11,10,2),(1,0,2),(1,10,-1),(1,10,2.5)])
def test_invalid_gradient_schedules_refused(args):
    with pytest.raises(ValueError, match='UPDATE_SCHEDULE'):
        updates_at_decision(*args)


def test_design_preserves_data_contract_and_disables_early_cutoff():
    parent = dict(batch_size=128, horizon=6, replay_size=65536, max_epochs=2000,
                  head_only=True, actor_plateau={'factor':.5}, warmup=4096)
    original = copy.deepcopy(parent)
    design = training_design(parent)
    assert parent == original
    assert design['max_epochs'] == 101 and design['matched_budget']
    assert design['matched_updates_per_epoch'] == 6006
    assert design['matched_collection_epochs'] == 1
    assert 'actor_plateau' not in design and not design['head_only']
    assert design['batch_size'] == 128 and design['horizon'] == 6


def test_native_budget_and_recovery_exact(tmp_path, request):
    from agent_plugins.three_signal_sac import ThreeSignalSAC
    from app.three_signal_learning import synthetic_env
    from app.training_forensics import tree_digest
    from tests.unit.test_double_tcn_probe import probe_recipe
    from tools.run_structural_sac_refit import fit_epochs
    from tools.run_sac_training_reliability import reseed
    torch.set_num_threads(1)
    recipe=probe_recipe(); risk=request.getfixturevalue('curve_env').risk_config
    factory=lambda: synthetic_env(recipe,risk,101,1,.99)[0]
    probe=factory().inputs.windows[30:34].copy()
    def assess(model):
        return {p:dict(return_per_week=0.,orders_per_week=0.) for p in ('TRAIN','VALIDATION')}
    design=training_design(dict(replay_size=256,batch_size=8,horizon=6))
    design.update(max_epochs=3,minimum_epochs=0,matched_updates_per_epoch=4,diagnostics=False)
    reseed(7);model=ThreeSignalSAC(recipe,seed=7)
    full=fit_epochs(model,factory,assess,design,7,tmp_path/'full','test',probe)
    expected=tree_digest(model.loss.state_dict()); adam=tree_digest(model.optimizer.state_dict())
    assert [r['epoch_updates'] for r in full['history']]==[0,4,4]
    assert full['updates']==8 and full['best_bundle'] is None
    reseed(7);model=ThreeSignalSAC(recipe,seed=7)
    part=fit_epochs(model,factory,assess,dict(design,max_epochs=2),7,tmp_path/'part','test',probe)
    assert part['updates']==4
    reseed(123);model=ThreeSignalSAC(recipe,seed=7)
    resumed=fit_epochs(model,factory,assess,design,7,tmp_path/'part','test',probe)
    assert resumed['updates']==8
    assert tree_digest(model.loss.state_dict())==expected
    assert tree_digest(model.optimizer.state_dict())==adam
    assert np.array_equal([r['decisions'] for r in resumed['history']],
                          [r['decisions'] for r in full['history']])
