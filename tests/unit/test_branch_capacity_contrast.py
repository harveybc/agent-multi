from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from tensordict import TensorDict

from agent_plugins.double_tcn_probe import ResidualTCN
from agent_plugins.temporal_stage_config import stage_config
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.branch_capacity_contrast import reduce_model
from app.structural_refit import load_bundle, export_bundle
from app.training_forensics import tree_digest
from tests.unit.test_modular_temporal_architecture import grouped_recipe


def batch():
    x = torch.randn(32, 13, 2)
    b = TensorDict(dict(features=x, mask=torch.ones(32, 3, dtype=torch.bool),
                        action=torch.arange(32) % 3), [32])
    b['next'] = TensorDict(dict(features=x * .9, mask=torch.ones(32, 3, dtype=torch.bool),
        reward=torch.linspace(-.1, .1, 32)[:, None], done=torch.zeros(32, 1, dtype=torch.bool),
        terminated=torch.zeros(32, 1, dtype=torch.bool)), [32])
    return b


@pytest.mark.parametrize('arm,blocks', [('control', 2), ('A', 1), ('D', 0)])
def test_transfer_layout_causal_dense_and_cold_update(tmp_path, arm, blocks):
    torch.set_num_threads(1)
    source = ThreeSignalSAC(grouped_recipe(), critic_regularization={'kind': 'core_dropout', 'p': .1})
    b = batch()
    source.update(b)
    before = tree_digest(source.loss.state_dict())
    model, receipt = reduce_model(source, arm, 7)
    assert before == tree_digest(source.loss.state_dict())
    assert receipt['all_retained_tensors_exact'] and receipt['optimizer_moments_exact']
    e = model.actor_net.encoder
    assert [sum(isinstance(m, ResidualTCN) for m in branch) for branch in e.branches] == [blocks] * 2
    assert len(e.core) == 2
    source_params = dict(source.loss.named_parameters())
    for n, p in model.loss.named_parameters():
        assert torch.equal(p, source_params[n])
        assert tree_digest(model.optimizer.state.get(p, {})) == tree_digest(source.optimizer.state.get(source_params[n], {}))
    assert all(g['lr'] == 3e-6 for g in model.optimizer.param_groups)
    x = b['features']
    if arm == 'D':
        branch = e.branches[0]
        z = x[:, :, :1]
        expected = F.silu(F.linear(z, branch[0].conv.weight[:, :, 0], branch[0].conv.bias))
        assert torch.allclose(branch(z.transpose(1, 2)).transpose(1, 2), expected, atol=1e-7)
        assert isinstance(branch[1], nn.SiLU)
    e.eval()
    y = x.clone(); y[:, 7:] += 100
    assert torch.equal(e.sequence({'features': x})[:, :, :7], e.sequence({'features': y})[:, :, :7])
    assert model.surfaces(x).shape == (32, 3)
    assert model.q_surfaces(x).shape == (2, 32, 3)
    export_bundle(tmp_path/'bundle', model, x.numpy(), {'epoch': 0})
    cold, _ = load_bundle(tmp_path/'bundle', 'cpu')
    assert tree_digest(model.optimizer.state_dict()) == tree_digest(cold.optimizer.state_dict())
    torch.manual_seed(99); values = model.update(b)
    torch.manual_seed(99); cold.update(b)
    assert np.isfinite(list(values.values())).all()
    assert tree_digest(model.loss.state_dict()) == tree_digest(cold.loss.state_dict())
    assert tree_digest(model.optimizer.state_dict()) == tree_digest(cold.optimizer.state_dict())


@pytest.mark.parametrize('raw', [dict(blocks=1), dict(kernel=3), dict(dilations=[1]), dict(blocks=False)])
def test_dense_rejects_ignored_parameters(raw):
    cfg = dict(kind='dense', blocks=0, kernel=1, dilations=[])
    cfg.update(raw)
    with pytest.raises(ValueError):
        stage_config(cfg, width=8, blocks=2)


def test_unknown_arm_rejected_without_source_mutation():
    m = ThreeSignalSAC(grouped_recipe())
    before = deepcopy(m.effective)
    with pytest.raises(ValueError, match='CAPACITY_ARM'):
        reduce_model(m, 'B', 7)
    assert m.effective == before
