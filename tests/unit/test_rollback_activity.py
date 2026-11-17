import pytest
import torch

from app.rollback_activity import actor_pressure, component, swap_component


def test_pressure_matches_autograd_and_common_offset():
    z = torch.tensor([[.3, -.2, .7], [-1., 2., .5]], dtype=torch.float64, requires_grad=True)
    q = torch.tensor([[.1, -.4, .2], [.3, .1, -.3]], dtype=torch.float64)
    logp = z.log_softmax(-1)
    loss = (logp.exp() * (.13 * logp - q)).sum()
    expected, = torch.autograd.grad(loss, z)
    actual = actor_pressure(z.detach(), q, .13)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual, actor_pressure(z.detach(), q + 100, .13))
    torch.testing.assert_close(actual.sum(-1), torch.zeros(2, dtype=torch.float64))


def test_nop_pressure_sign():
    z = torch.zeros(1, 3)
    q = torch.tensor([[1., 0., 0.]])
    assert actor_pressure(z, q, .1)[0, 0] < 0


@pytest.mark.parametrize('alpha', [0, -1, float('nan')])
def test_bad_alpha(alpha):
    with pytest.raises(ValueError):
        actor_pressure(torch.zeros(1, 3), torch.zeros(1, 3), alpha)


def test_swap_is_component_only_and_no_mutation():
    keys = ['actor_network_params.module.encoder.branches.0.weight',
            'actor_network_params.module.encoder.post_fusion.gate',
            'actor_network_params.module.encoder.fusion.conv.weight',
            'actor_network_params.module.encoder.core.0.weight',
            'actor_network_params.module.head.0.weight', 'qvalue_network_params.head.weight']
    current = {k: torch.ones(2) for k in keys}
    best = {k: torch.zeros(2) for k in keys}
    assert [component(k) for k in keys] == ['branches', 'attention', 'fusion', 'core', 'head', None]
    for part in ('branches', 'attention', 'fusion', 'core', 'head', 'all'):
        result, replaced = swap_component(current, best, part)
        for key in keys:
            selected = component(key) is not None and (part == 'all' or component(key) == part)
            assert torch.equal(result[key], best[key] if selected else current[key])
            assert torch.equal(current[key], torch.ones(2))
            assert (key in replaced) == selected
    with pytest.raises(ValueError):
        swap_component(current, best, 'unknown')


def test_unknown_actor_tensor_rejected():
    with pytest.raises(ValueError):
        component('actor_network_params.module.new_layer.weight')
