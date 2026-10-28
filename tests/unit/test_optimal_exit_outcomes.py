import numpy as np
import pytest

from tools.audit_optimal_exit_outcomes import exit_distribution, expectation


def test_first_close_and_terminal_probability_conserve_mass():
    exits, passive = exit_distribution([.2, .5, .75])
    np.testing.assert_allclose(exits, [.2, .4, .3])
    assert passive == pytest.approx(.1)
    assert exits.sum() + passive == pytest.approx(1.)


def test_expectation_agrees_with_exhaustive_first_exit_outcomes():
    case = dict(close_net=[2., 5., 3.], terminal_net=-4., risk_usd=2.)
    result = expectation(case, [.2, .5, .75])
    assert result['mean_net_usd'] == pytest.approx(2.9)
    assert result['regret_risk'] == pytest.approx(1.05)
    assert result['early_close_probability'] == pytest.approx(.9)


@pytest.mark.parametrize('p', [[], [float('nan')], [-.1], [1.1]])
def test_invalid_probabilities_rejected(p):
    with pytest.raises(ValueError, match='EXIT_PROBABILITY'):
        exit_distribution(p)


def test_passive_and_immediate_limits():
    case = dict(close_net=[2., 5.], terminal_net=-4., risk_usd=2.)
    assert expectation(case, [0., 0.])['mean_net_usd'] == -4.
    assert expectation(case, [1., 0.])['mean_net_usd'] == 2.
