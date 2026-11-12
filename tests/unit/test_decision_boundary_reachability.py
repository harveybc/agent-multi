import numpy as np
import pytest
import torch

from app.decision_boundary_reachability import target_problem, witness, temporary_head_step


def test_three_actions_not_only_current_winner():
    d, a, b, report = target_problem([2., 0., 1.9], np.eye(3), 1, 10)
    step, result = witness(d, a, b)
    assert result['linear_witness']
    assert np.argmax(np.array([2., 0., 1.9])+step) == 1
    assert report['gap'] == 2
    assert report['head_linear_distance_lower_bound'] == pytest.approx(2/np.sqrt(2))


def test_preservation_can_change_direction_without_freezing():
    d, a, b, _ = target_problem([1., 0., -.1], np.eye(3), 1, 4)
    step, result = witness(d, a, b, np.array([[1., 0., 0.]]), np.array([0.]))
    assert result['linear_witness'] and np.linalg.norm(step) > 0
    assert step[0] >= -1e-8
    assert 1+step[0] < step[1]


def test_zero_sensitivity_is_not_a_successful_correction():
    d, a, b, report = target_problem([1., 0., -.1], np.zeros((3, 2)), 1, 4)
    _, result = witness(d, a, b)
    assert report['zero_pair_sensitivity']
    assert report['head_linear_distance_lower_bound'] is None
    assert not result['linear_witness']


def test_tie_has_positive_buffer_and_can_change_greedy():
    d, a, b, report = target_problem([1., 1., 0.], np.eye(3), 1, 4)
    step, result = witness(d, a, b)
    assert report['gap'] == 0 and result['linear_witness']
    assert np.argmax(np.array([1., 1., 0.])+step) == 1


def test_non_nop_pair_margin_bound():
    _, _, _, report = target_problem([0., 3., 1.], np.eye(3), 2, 4)
    assert report['one_step_fit_margin_rms_lower_bound'] == pytest.approx(.5)


@pytest.mark.parametrize('z,j,t,n', [([1., 0., float('nan')], np.eye(3), 1, 4),
    ([1., 0., 0.], np.eye(3), 4, 4), ([1., 0., 0.], np.eye(3), 1, 0),
    ([1., 0., 0.], np.eye(3), 0, 4)])
def test_bad_inputs_refused(z, j, t, n):
    with pytest.raises(ValueError):
        target_problem(z, j, t, n)


@pytest.mark.parametrize('raise_error', [False, True])
def test_temporary_step_restores_exactly_even_on_error(raise_error):
    p = torch.nn.Parameter(torch.tensor([1., 2.]))
    before = p.detach().clone()
    try:
        with temporary_head_step([p], [.1, -.2]):
            assert not torch.equal(p, before)
            if raise_error:
                raise RuntimeError('probe fails')
    except RuntimeError:
        assert raise_error
    assert torch.equal(p, before)


def test_temporary_step_rejects_nonfinite_before_mutation():
    p = torch.nn.Parameter(torch.tensor([1.]))
    with pytest.raises(ValueError):
        with temporary_head_step([p], [float('nan')]):
            pytest.fail('must not enter')
    assert p.item() == 1.
