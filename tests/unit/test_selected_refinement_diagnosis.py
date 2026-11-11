import torch

from app.actor_gradient_coupling import decompose, first_adamw_delta
from app.selected_refinement_diagnosis import adaptive_parts, action_quality


def test_adaptive_self_cross_decay_reconstruct_step():
    torch.manual_seed(91)
    z = torch.randn(4, 3, dtype=torch.double)
    q = torch.randn_like(z)
    j = torch.randn(4, 3, 7, dtype=torch.double)
    parts, weights, jm = decompose(z, q, .1, j, torch.tensor([0, 0, 1, 2]))
    theta = torch.randn(7, dtype=torch.double)
    got = adaptive_parts(parts['total'], weights, jm, theta, .001, .02, 1e-8)
    adaptive, decay = first_adamw_delta(theta, parts['total']['batch_gradient'], .001, .02, 1e-8)
    torch.testing.assert_close(got['self'] + got['cross'], jm @ adaptive)
    torch.testing.assert_close(got['total'], jm @ (adaptive + decay))
    assert torch.equal(got['self'][3], torch.zeros(2, dtype=torch.double))


def test_shared_head_can_reverse_helpful_local_buy():
    z = torch.zeros(2, 3, dtype=torch.double)
    q = torch.tensor([[0., 1., 0.], [0., -10., 0.]], dtype=torch.double)
    j = torch.eye(3, dtype=torch.double).expand(2, -1, -1)
    parts, weights, jm = decompose(z, q, 0., j, torch.tensor([0, 1]))
    got = adaptive_parts(parts['total'], weights, jm, torch.zeros(3, dtype=torch.double), .001, 0., 1e-8)
    assert parts['total']['local_logit'][0, 0] > 0
    assert got['self'][0, 0] > 0 and got['total'][0, 0] < 0


def test_optimal_action_not_just_positive_gap_and_ties():
    native = torch.tensor([[0., 1., 2.], [0., 0., 0.]], dtype=torch.double)
    q = torch.tensor([[0., 3., 1.], [1., 1., 1.]], dtype=torch.double)
    got = action_quality(native, q)
    assert got['optimal_count'] == 1
    assert got['mean_regret'] == .5
    assert got['beneficial_gap_sign_errors'] == 0
    assert got['unique_native_optimum'] == 1
