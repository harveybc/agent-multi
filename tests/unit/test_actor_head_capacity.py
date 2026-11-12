import numpy as np
import pytest
import torch

from app.actor_head_capacity import make_candidate, install_head, verify_fixed, full_schedule, is_actor_head
from app.structural_refit import export_bundle, load_bundle
from agent_plugins.three_signal_sac import ThreeSignalSAC
from tests.unit.test_double_tcn_probe import probe_recipe


@pytest.mark.parametrize('width', [8, 16])
def test_only_actor_head_changes_and_cold_replay(tmp_path, width):
    torch.set_num_threads(1)
    recipe = probe_recipe()
    source = ThreeSignalSAC(recipe, architecture={'head_width':8}, seed=7)
    original = {k:v.clone() for k,v in source.loss.state_dict().items()}
    model, head = make_candidate(source, width, 19)
    verify_fixed(model, original)
    assert model.actor_net.head[0].out_features == width
    assert model.q_net.head[0].out_features == 8
    assert len([n for n,p in model.loss.named_parameters() if is_actor_head(n)]) == 4
    with torch.no_grad():
        head[2].bias.add_(.01)
    install_head(model, head)
    verify_fixed(model, original)
    for k,v in original.items():
        assert torch.equal(v, source.loss.state_dict()[k])
    x = torch.randn(3, recipe['tensor']['window']['observations'], recipe['list']['n'])
    export_bundle(tmp_path/'bundle',model,x.numpy(),{'diagnostic':True})
    cold,_ = load_bundle(tmp_path/'bundle')
    with torch.inference_mode():
        assert torch.equal(model.surfaces(x), cold.surfaces(x))
    captured = []
    hook = source.actor_net.head[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        with torch.inference_mode():source.surfaces(x)
    finally:
        hook.remove()
    cached = captured[0].clone()
    with torch.inference_mode():
        torch.testing.assert_close(head(cached),model.surfaces(x),atol=1e-6,rtol=1e-6)
    opt = torch.optim.AdamW(head.parameters(),lr=3e-5)
    opt.zero_grad();head(cached).square().mean().backward();opt.step()
    install_head(model,head);verify_fixed(model,original)
    with torch.inference_mode():
        torch.testing.assert_close(head(cached),model.surfaces(x),atol=1e-6,rtol=1e-6)


def test_schedule_full_coverage_paired_and_seeded():
    a = full_schedule(17, 4, 8, 19)
    assert np.array_equal(a, full_schedule(17, 4, 8, 19))
    assert set(a.ravel()[:17]) == set(range(17))
    assert not np.array_equal(a, full_schedule(17, 4, 8, 43))


def test_reject_width_and_frozen_tampering():
    source = ThreeSignalSAC(probe_recipe(), architecture={'head_width':8})
    for width in (True, 0, 12, 16.):
        with pytest.raises(ValueError):make_candidate(source,width,7)
    candidate,_ = make_candidate(source,16,19)
    reference = {k:v.clone() for k,v in source.loss.state_dict().items()}
    name = next(k for k in reference if not is_actor_head(k) and reference[k].is_floating_point())
    candidate.loss.state_dict()[name].add_(1)
    with pytest.raises(ValueError,match='FROZEN_CHANGED'):verify_fixed(candidate,reference)
