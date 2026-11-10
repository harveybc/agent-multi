import copy

import pytest
import torch
from torch import nn

from app.bayesian_actor_head import LastLayer, actor_objective, uncertainty


def fixture():
    torch.manual_seed(7)
    return nn.Linear(8, 3), torch.randn(64, 8)


def test_mean_and_zero_noise_parity():
    linear, x = fixture()
    for bayesian in (False, True):
        head = LastLayer(linear, bayesian)
        assert torch.equal(head.mean_logits(x), linear(x))
        if bayesian:
            noise = tuple(torch.zeros_like(v) for v in head.draws(4, 7))
            torch.testing.assert_close(head.logits(x, noise), linear(x).expand(4, -1, -1))


def test_posterior_gradients_and_kl():
    linear, x = fixture()
    head = LastLayer(linear, True)
    q = torch.randn(64, 3, requires_grad=True)
    a = head.logits(x, head.draws(4, 7))
    b = head.logits(x, head.draws(4, 19))
    assert not torch.equal(a, b)
    assert head.kl() > 0
    (actor_objective(a, q, .2) + head.kl()/1000).backward()
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert q.grad is None
    same = LastLayer(linear, True, posterior_std=.1)
    assert abs(float(same.kl().detach())) < 1e-5


def test_objective_formula_and_gradient():
    logits = torch.tensor([[.1, -.2, .5]], requires_grad=True)
    q = torch.tensor([[2., -1., .2]])
    actual = actor_objective(logits[None], q, .3)
    p = torch.softmax(logits, -1)
    expected = (.3 * p * torch.log(p) - p*q).sum(-1).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_inference_partition_and_reload(tmp_path):
    linear, x = fixture()
    head = LastLayer(linear, True)
    noise = head.draws(32, 99)
    full = uncertainty(head.logits(x, noise))
    parts = [uncertainty(head.logits(v, noise)) for v in x.split(8)]
    for name in full:
        torch.testing.assert_close(full[name], torch.cat([v[name] for v in parts]))
    path = tmp_path/'head.pt'
    torch.save(head.state_dict(), path)
    other = LastLayer(linear, True)
    other.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(head.logits(x, noise), other.logits(x, other.draws(32, 99)))
    assert full['mutual_information'].min() >= 0
    assert full['probability_std'].max() > 0
    torch.testing.assert_close(full['probability'].sum(-1), torch.ones(len(x)))


def test_only_last_layer_changes():
    linear, x = fixture()
    before = copy.deepcopy(linear.state_dict())
    head = LastLayer(linear, True)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-5)
    actor_objective(head.logits(x, head.draws(4, 7)), torch.randn(64, 3), .1).backward()
    opt.step()
    assert all(torch.equal(before[k], v) for k, v in linear.state_dict().items())
    assert not torch.equal(head.weight, linear.weight)


def test_invalid_scale_and_implicit_sampling_rejected():
    linear, x = fixture()
    with pytest.raises(ValueError): LastLayer(linear, True, posterior_std=0)
    with pytest.raises(ValueError): LastLayer(linear, True).logits(x)
    with pytest.raises(ValueError): LastLayer(linear, True).draws(3, 7)
