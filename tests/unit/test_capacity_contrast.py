import copy

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from tests.unit.test_double_tcn_probe import probe_recipe
from tests.unit.test_causal_patch_attention import ARCH as ATTENTION
from tools.run_capacity_contrast import widened, choose_width, meta_count, actual_count
from agent_plugins.three_signal_sac import ThreeSignalSAC
from app.structural_refit import export_bundle, load_bundle

ARCH = dict(branch_width=8, core_width=16, head_width=8,
            branch_blocks=2, core_blocks=2, **ATTENTION)


def test_only_widths_change_and_smallest_admissible_count_is_used():
    before = copy.deepcopy(ARCH)
    a, report = choose_width(probe_recipe(), ARCH)
    assert ARCH == before and 70 <= report['ratio'] <= 75
    assert meta_count(probe_recipe(), widened(ARCH, report['width']-2))['trainable'] < 70*report['base']['trainable']
    a.update(branch_width=8, core_width=16, head_width=8)
    a['temporal_attention'].update(embedding=8, feedforward=16)
    assert a == ARCH
    with pytest.raises(ValueError): widened(dict(ARCH, long_context={}), 70)
    with pytest.raises(ValueError): widened(ARCH, 69)
    with pytest.raises(ValueError): widened(dict(ARCH, core_width=32), 70)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_count_causality_real_updates_and_same_device_reload(device, tmp_path):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.set_num_threads(1)
    recipe = probe_recipe()
    arch, audit = choose_width(recipe, ARCH)
    model = ThreeSignalSAC(recipe, architecture=arch, device=device, seed=7)
    assert actual_count(model) == audit['candidate']['trainable']
    x = torch.randn(4, 13, 2, device=device)
    altered = x.clone(); altered[:, 7:] += 100
    model.actor_net.eval()
    with torch.no_grad():
        left = model.actor_net.encoder.sequence({'features':x})
        right = model.actor_net.encoder.sequence({'features':altered})
        torch.testing.assert_close(left[:,:,:7], right[:,:,:7], rtol=0, atol=0)
        assert model.surfaces(x).shape == (4, 3)
        assert model.q_surfaces(x).shape[-1] == 3
    model.actor_net.train()
    td = TensorDict(dict(features=x, mask=torch.ones(4, 3, dtype=torch.bool, device=device)), [4])
    td['action'] = torch.arange(4, device=device)%3
    td['steps_to_next_obs'] = torch.ones(4, 1, dtype=torch.int64, device=device)
    td['next'] = td.select('features', 'mask').clone()
    td['next', 'reward'] = torch.arange(4, device=device, dtype=torch.float32)[:,None]/4
    td['next', 'done'] = torch.zeros(4, 1, dtype=torch.bool, device=device)
    td['next', 'terminated'] = td['next', 'done'].clone()
    before = {k:p.detach().clone() for k,p in model.loss.named_parameters()}
    for _ in range(2):
        assert np.isfinite(list(model.update(td.clone()).values())).all()
    for role in ('actor_network_params', 'qvalue_network_params'):
        for component in ('.branches.', '.fusion.', '.core.', '.post_fusion.', '.head.'):
            selected = [(k,p) for k,p in model.loss.named_parameters() if k.startswith(role) and component in k]
            assert selected, (role, component)
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for _,p in selected)
            assert any(not torch.equal(before[k], p) for k,p in selected)
    export_bundle(tmp_path/'bundle', model, x.cpu().numpy(), dict(epoch=1))
    restored, _ = load_bundle(tmp_path/'bundle', device)
    with torch.no_grad():
        assert torch.equal(model.surfaces(x), restored.surfaces(x))


def test_optimizer_missing_parameter_is_detected():
    model = ThreeSignalSAC(probe_recipe(), architecture=ARCH)
    model.optimizer.param_groups[0]['params'].pop()
    with pytest.raises(ValueError, match='OPTIMIZER_COVERAGE'):
        actual_count(model)
