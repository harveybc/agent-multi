import pytest
import torch

from app.actor_gradient_coupling import decompose, first_adamw_delta, logit_terms
from app.actor_learning_diagnostics import actor_terms


def test_logit_formula_and_common_offset():
    torch.manual_seed(3)
    z = torch.randn(7, 3, dtype=torch.double, requires_grad=True)
    q = torch.randn_like(z); alpha = torch.tensor(.12)
    a, b, _ = actor_terms(z, q, alpha)
    terms = logit_terms(z, q, alpha)
    for name, loss in [('value', a), ('entropy', b), ('total', a+b)]:
        g = torch.autograd.grad(loss, z, retain_graph=True)[0]
        torch.testing.assert_close(g*len(z), terms[name])
    shifted = logit_terms(z, q+torch.arange(7)[:, None], alpha)
    torch.testing.assert_close(shifted['total'], terms['total'])


def test_independent_rows_duplicates_and_autograd():
    torch.manual_seed(4)
    z = torch.randn(3, 3, dtype=torch.double, requires_grad=True)
    q = torch.randn_like(z); ix = torch.tensor([0, 0, 1])
    j = torch.eye(9, dtype=torch.double).reshape(3, 3, 9)
    out, weights, _ = decompose(z, q, .1, j, ix)
    torch.testing.assert_close(weights, torch.tensor([2/3, 1/3, 0], dtype=torch.double))
    a, b, _ = actor_terms(z[ix], q[ix], torch.tensor(.1, dtype=torch.double))
    g = torch.autograd.grad(a+b, z)[0].flatten()
    torch.testing.assert_close(out['total']['batch_gradient'], g)
    torch.testing.assert_close(out['total']['cross'], torch.zeros(3, 2, dtype=torch.double))
    torch.testing.assert_close(out['total']['self_alone'], out['total']['local_logit'])
    assert out['total']['total'][2].abs().max() == 0


def test_shared_parameter_cross_row_reversal():
    z = torch.zeros(2, 3, dtype=torch.double)
    q = torch.tensor([[0., 1., 0.], [0., -10., 0.]], dtype=torch.double)
    j = torch.eye(3, dtype=torch.double).expand(2, -1, -1)
    out, _, _ = decompose(z, q, 0., j, torch.tensor([0, 1]))
    d = out['total']
    assert d['local_logit'][0, 0] > 0 and d['self_alone'][0, 0] > 0
    assert d['cross'][0, 0] < 0 and d['total'][0, 0] < 0
    torch.testing.assert_close(d['batch_self']+d['cross'], d['total'])


@pytest.mark.parametrize('decay', [0., .03])
def test_first_adamw_matches_torch(decay):
    p = torch.tensor([1., -2., .3, 4.], dtype=torch.double, requires_grad=True)
    g = torch.tensor([.2, -.7, 0., 1e-12], dtype=torch.double)
    before = p.detach().clone()
    opt = torch.optim.AdamW([p], lr=.002, weight_decay=decay)
    p.grad = g; opt.step()
    a, b = first_adamw_delta(before, g, .002, decay, 1e-8)
    torch.testing.assert_close(p-before, a+b, atol=1e-15, rtol=1e-10)


def test_invalid_batch_and_nonfinite():
    z = torch.zeros(2, 3); j = torch.zeros(2, 3, 3)
    with pytest.raises(ValueError): decompose(z, z, 0., j, torch.tensor([2]))
    with pytest.raises(ValueError): logit_terms(z, z+float('nan'), 0.)
