from fractions import Fraction as F
from itertools import product

import pytest

from app.mechanism_games import FinitePolicyGame, pareto_front


def test_two_equilibria_do_not_imply_two_optima():
    values = {(0, 0): 0, (0, 1): -1, (1, 0): -1, (1, 1): 2}
    game = FinitePolicyGame((2, 2), values.__getitem__)
    assert game.equilibria() == ((0, 0), (1, 1))
    assert game.best_response((0, 0))["path"] == [(0, 0)]
    assert pareto_front({str(k): (v,) for k, v in values.items()})["front"] == ["(1, 1)"]


def test_three_options_global_maximum_and_three_local_equilibria():
    game = FinitePolicyGame((3, 3), lambda p: (0, 2, 3)[p[0]] if p[0] == p[1] else -1)
    assert game.equilibria() == ((0, 0), (1, 1), (2, 2))
    assert game.unilateral_certificate((2, 2))["epsilon"] == 0


@pytest.mark.parametrize("scenario", [-2, 0, 3])
def test_difference_identity_on_state_changing_full_policy_replay(scenario):
    def replay(p):
        state = F(10)
        for action in p:
            state = state * (1 + F(action, 10)) + scenario - action
        return state - 10
    game = FinitePolicyGame((2, 3, 2), replay)
    for p in game.profiles():
        for i, size in enumerate(game.sizes):
            for a in range(size):
                q = game.replace(p, i, a)
                assert game.difference_reward(q, i) - game.difference_reward(p, i) == game.value(q) - game.value(p)


def test_twenty_players_certificate_is_linear_and_checks_last_deviation():
    calls = []
    def value(p):
        calls.append(p)
        return int(p[-1] == 26)
    game = FinitePolicyGame((27,) * 20, value)
    cert = game.unilateral_certificate((0,) * 20)
    assert len(calls) == cert["distinct_profiles_required"] == 521
    assert len(cert["deviations"]) == 520
    assert cert["epsilon"] == 1 and not cert["is_nash"]
    assert cert["deviations"][-1] == {"player": 19, "action": 26, "gain": 1}
    game.unilateral_certificate((0,) * 20)
    assert len(calls) == 521
    with pytest.raises(ValueError, match="JOINT_ENUMERATION_BUDGET"):
        game.equilibria()


def test_exact_tiny_gain_not_hidden_by_float_tolerance():
    game = FinitePolicyGame((2,), lambda p: F(p[0], 10**40))
    assert game.unilateral_certificate((0,))["epsilon"] == F(1, 10**40)


def test_strict_response_terminates_and_budget_is_not_certificate():
    game = FinitePolicyGame((3, 3), sum)
    assert game.best_response((0, 0), 0)["status"] == "MOVE_BUDGET_EXHAUSTED"
    result = game.best_response((0, 0))
    assert result["certificate"]["profile"] == (2, 2)
    values = [game.value(p) for p in result["path"]]
    assert all(b > a for a, b in zip(values, values[1:]))


def test_five_redundant_closers_difference_zero_shapley_equal():
    game = FinitePolicyGame((2,) * 5, lambda p: 10 if any(p) else 0)
    p = (1,) * 5
    assert [game.difference_reward(p, i) for i in range(5)] == [0] * 5
    result = game.role_shapley(p, tuple(range(5)))
    assert result["coalitions_evaluated"] == 32
    assert list(result["shares"].values()) == [2] * 5
    assert sum(result["shares"].values()) == result["group_increment"] == 10


def test_shapley_can_incentivize_account_harm():
    table = ((0, 0), (0, 4), (10, 3))
    game = FinitePolicyGame((3, 2), lambda p: table[p[0]][p[1]])
    a, b = (1, 1), (2, 1)
    assert game.value(a) == 4 and game.value(b) == 3
    assert game.role_shapley(a, (0, 1))["shares"][0] == 2
    assert game.role_shapley(b, (0, 1))["shares"][0] == F(13, 2)
    assert game.difference_reward(b, 0) < game.difference_reward(a, 0)


def test_role_shapley_freezes_outside_policies_and_handles_reference_member():
    game = FinitePolicyGame((2, 2, 2), lambda p: 100 * p[2] + 10 * int(any(p[:2])))
    result = game.role_shapley((1, 0, 1), (0, 1))
    assert result["shares"] == {0: 10, 1: 0}
    assert result["group_increment"] == 10


def test_pareto_retains_ties_and_tradeoffs_and_reports_dominators():
    result = pareto_front({"a": (2, 1), "b": (1, 2), "copy": (2, 1), "bad": (1, 1)})
    assert result["front"] == ["a", "b", "copy"]
    assert result["dominance_witnesses"]["bad"] == ["a", "b", "copy"]


@pytest.mark.parametrize("sizes", [(), (0,), (-1,), (True,), (2.0,)])
def test_reject_invalid_sizes(sizes):
    with pytest.raises(ValueError, match="POLICY_LIBRARY_SHAPE"):
        FinitePolicyGame(sizes, sum)


@pytest.mark.parametrize("p", [(), (2,), (-1,), (True,), (1.0,), (0, 0)])
def test_reject_invalid_profiles(p):
    with pytest.raises(ValueError, match="POLICY_PROFILE"):
        FinitePolicyGame((2,), sum).value(p)


@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), True, "1"])
def test_reject_inexact_payoffs(value):
    with pytest.raises(ValueError, match="EXACT_RATIONAL_REQUIRED"):
        FinitePolicyGame((1,), lambda _: value).value((0,))


@pytest.mark.parametrize("vectors", [{}, {"a": ()}, {"a": (1,), "b": (1, 2)}, {"a": (float("nan"),)}])
def test_reject_bad_pareto_panels(vectors):
    with pytest.raises(ValueError):
        pareto_front(vectors)


def test_bad_players_references_groups_and_budgets():
    with pytest.raises(ValueError, match="POLICY_PROFILE"):
        FinitePolicyGame((2,), sum, reference=(2,))
    game = FinitePolicyGame((2, 2), sum)
    for i in (-1, 2, True):
        with pytest.raises(ValueError, match="PLAYER_INDEX"):
            game.difference_reward((0, 0), i)
    for group in ((), (0, 0)):
        with pytest.raises(ValueError, match="SHAPLEY_GROUP"):
            game.role_shapley((1, 1), group)
    with pytest.raises(ValueError, match="MOVE_BUDGET"):
        game.best_response((0, 0), -1)
    with pytest.raises(ValueError, match="SHAPLEY_COALITION_BUDGET"):
        FinitePolicyGame((2,) * 11, sum).role_shapley((1,) * 11, tuple(range(11)))


def test_unilateral_certificates_match_independent_bruteforce():
    game = FinitePolicyGame((2, 3, 2), lambda p: 3*p[0]*p[1] - 4*p[2] + p[1]*p[2])
    profiles = list(product(range(2), range(3), range(2)))
    for p in profiles:
        neighbors = [q for q in profiles if sum(a != b for a, b in zip(p, q)) <= 1]
        expected = max(game.value(q) - game.value(p) for q in neighbors)
        assert game.unilateral_certificate(p)["epsilon"] == expected
