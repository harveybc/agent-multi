import pytest
import torch

from app.weight_space_diagnostics import (
    center_categorical_head, component, state_delta, tensor_summary, transplant_actor_heads,
)


def test_rank_zero_rank_one_and_nonfinite():
    assert tensor_summary(torch.zeros(3, 4))['stable_rank'] == 0.
    assert tensor_summary(torch.ones(3, 4))['stable_rank'] == pytest.approx(1.)
    with pytest.raises(ValueError):
        tensor_summary(torch.tensor([float('nan')]))


def test_deltas_and_module_roles():
    key = 'actor_network_params.module.0.module.encoder.branches.2.0.conv.weight'
    before = {key: torch.ones(2, 3), 'mask': torch.tensor([1])}
    assert state_delta(before, before)['groups']['actor/branch_2']['changed'] == 0
    after = {key: before[key] + 2, 'mask': before['mask']}
    assert state_delta(before, after)['tensors'][key]['max_abs_delta'] == 2.
    assert component('target_qvalue_network_params.module.head.2.bias') == 'target_critic/head_2'
    with pytest.raises(ValueError):
        state_delta(before, {key: before[key], 'mask': torch.tensor([2])})


def test_centered_head_preserves_distribution_and_source():
    generator = torch.Generator().manual_seed(19)
    w = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    b = torch.randn(3, generator=generator, dtype=torch.float64)
    x = torch.randn(32, 8, generator=generator, dtype=torch.float64)
    old = w.clone()
    nw, nb = center_categorical_head(w, b)
    a, c = x @ w.T + b, x @ nw.T + nb
    torch.testing.assert_close(a.softmax(-1), c.softmax(-1))
    torch.testing.assert_close(a[:, 1:] - a[:, :1], c[:, 1:] - c[:, :1])
    assert torch.equal(a.argmax(-1), c.argmax(-1))
    assert torch.equal(w, old)


def test_transplant_changes_only_requested_head_and_never_aliases():
    prefix = 'actor_network_params.module.0.module.head.0.'
    old = {prefix+'weight': torch.zeros(2, 3), prefix+'bias': torch.zeros(2),
           'qvalue_network_params.module.head.2.bias': torch.ones(3)}
    donor = {k: v+3 for k,v in old.items()}
    new, keys = transplant_actor_heads(old, donor, [0])
    assert len(keys) == 2
    for key in old:
        assert torch.equal(new[key], donor[key] if key in keys else old[key])
        assert new[key].data_ptr() not in (old[key].data_ptr(), donor[key].data_ptr())
    with pytest.raises(ValueError):
        transplant_actor_heads(old, donor, [1])
    with pytest.raises(ValueError):
        transplant_actor_heads(old, donor, [2])
