import torch
import pytest

from app.hybrid_learning_diagnostics import gradient_geometry


def test_gradient_geometry_aligned_opposed_zero_and_no_mutation():
    p = torch.nn.Parameter(torch.tensor([2., 3.]))
    p.grad = torch.tensor([8., 9.])
    before = p.detach().clone()
    rng = torch.random.get_rng_state().clone()
    a = p.sum()
    result = gradient_geometry(a, -2*a, [p])
    assert result['cosine'] == pytest.approx(-1.)
    assert result['critic_to_actor_norm'] == pytest.approx(2.)
    result = gradient_geometry(a, 2*a, [p])
    assert result['cosine'] == pytest.approx(1.)
    result = gradient_geometry(a*0, a, [p])
    assert result['cosine'] is None
    assert result['critic_to_actor_norm'] is None
    assert torch.equal(p, before)
    assert torch.equal(p.grad, torch.tensor([8., 9.]))
    assert torch.equal(rng, torch.random.get_rng_state())


def test_disconnected_parameters_and_nonfinite_gradient():
    p = torch.nn.Parameter(torch.tensor([1.]))
    q = torch.nn.Parameter(torch.tensor([2.]))
    assert gradient_geometry(p.sum(), q.sum(), [p, q])['cosine'] == 0.
    with pytest.raises(ValueError, match='NONFINITE_GRADIENT'):
        gradient_geometry((p*float('inf')).sum(), q.sum(), [p, q])
