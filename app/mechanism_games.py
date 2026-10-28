"""Exact finite-policy potential games. No market simulator or learning claims."""
from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import factorial, prod
from typing import Callable


Profile = tuple[int, ...]


def exact(value: int | Fraction) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
        raise ValueError("EXACT_RATIONAL_REQUIRED")
    return Fraction(value)


class FinitePolicyGame:
    """A deterministic payoff oracle on a declared, finite policy library."""

    def __init__(self, sizes: tuple[int, ...], payoff: Callable[[Profile], int | Fraction],
                 reference: Profile | None = None):
        if not sizes or any(type(k) is not int or k < 1 for k in sizes):
            raise ValueError("POLICY_LIBRARY_SHAPE")
        self.sizes = tuple(sizes)
        self.payoff = payoff
        self.reference = self.profile(reference if reference is not None else (0,) * len(sizes))
        self.cache: dict[Profile, Fraction] = {}

    def profile(self, profile: Profile) -> Profile:
        p = tuple(profile)
        if len(p) != len(self.sizes) or any(
            type(a) is not int or not 0 <= a < k for a, k in zip(p, self.sizes)
        ):
            raise ValueError("POLICY_PROFILE")
        return p

    def player(self, player: int) -> None:
        if type(player) is not int or not 0 <= player < len(self.sizes):
            raise ValueError("PLAYER_INDEX")

    def value(self, profile: Profile) -> Fraction:
        p = self.profile(profile)
        if p not in self.cache:
            self.cache[p] = exact(self.payoff(p))
        return self.cache[p]

    def replace(self, profile: Profile, player: int, action: int) -> Profile:
        self.player(player)
        p = list(self.profile(profile))
        p[player] = action
        return self.profile(tuple(p))

    def difference_reward(self, profile: Profile, player: int) -> Fraction:
        self.player(player)
        return self.value(profile) - self.value(self.replace(profile, player, self.reference[player]))

    def unilateral_certificate(self, profile: Profile) -> dict:
        p = self.profile(profile)
        base = self.value(p)
        deviations = []
        best = Fraction(0)
        for player, size in enumerate(self.sizes):
            for action in range(size):
                if action == p[player]:
                    continue
                alternate = self.replace(p, player, action)
                gain = self.value(alternate) - base
                deviations.append({"player": player, "action": action, "gain": gain})
                best = max(best, gain)
        return {
            "profile": p, "value": base, "epsilon": best,
            "is_nash": best == 0, "coverage": "EXHAUSTIVE_DECLARED_UNILATERAL_LIBRARY",
            "distinct_profiles_required": 1 + sum(k - 1 for k in self.sizes),
            "deviations": deviations,
        }

    def profiles(self, max_profiles: int = 100000):
        if type(max_profiles) is not int or max_profiles < 1 or prod(self.sizes) > max_profiles:
            raise ValueError("JOINT_ENUMERATION_BUDGET")
        return product(*(range(k) for k in self.sizes))

    def equilibria(self, max_profiles: int = 100000) -> tuple[Profile, ...]:
        return tuple(p for p in self.profiles(max_profiles) if self.unilateral_certificate(p)["is_nash"])

    def best_response(self, initial: Profile, max_moves: int = 1000) -> dict:
        if type(max_moves) is not int or max_moves < 0:
            raise ValueError("MOVE_BUDGET")
        p = self.profile(initial)
        path = [p]
        for step in range(max_moves + 1):
            cert = self.unilateral_certificate(p)
            if cert["is_nash"]:
                return {"status": "NASH_DECLARED_LIBRARY", "path": path, "certificate": cert}
            if step == max_moves:
                return {"status": "MOVE_BUDGET_EXHAUSTED", "path": path, "certificate": cert}
            # Only strict gains move; stable iteration resolves equal positive gains.
            best = max(cert["deviations"], key=lambda item: item["gain"])
            p = self.replace(p, best["player"], best["action"])
            path.append(p)
        raise AssertionError("UNREACHABLE")

    def role_shapley(self, profile: Profile, group: tuple[int, ...]) -> dict:
        p = self.profile(profile)
        if not group or len(set(group)) != len(group):
            raise ValueError("SHAPLEY_GROUP")
        for i in group:
            self.player(i)
        # Exact enumeration is intentionally bounded; no silent approximation.
        if len(group) > 10:
            raise ValueError("SHAPLEY_COALITION_BUDGET")
        n = len(group)
        coalitions = {}
        for mask in range(1 << n):
            q = list(p)
            for j, i in enumerate(group):
                if not mask & (1 << j):
                    q[i] = self.reference[i]
            coalitions[mask] = self.value(tuple(q))
        shares = {}
        for j, i in enumerate(group):
            share = Fraction(0)
            for mask, val in coalitions.items():
                if mask & (1 << j):
                    continue
                k = mask.bit_count()
                weight = Fraction(factorial(k) * factorial(n - k - 1), factorial(n))
                share += weight * (coalitions[mask | (1 << j)] - val)
            shares[i] = share
        return {"shares": shares, "coalitions_evaluated": 1 << n,
                "group_increment": coalitions[(1 << n) - 1] - coalitions[0],
                "scope": "ROLE_CONDITIONAL_ATTRIBUTION_NOT_LEARNING_REWARD"}


def pareto_front(outcomes: dict[str, tuple[int | Fraction, ...]]) -> dict:
    """All objectives maximized; duplicate non-dominated vectors are retained."""
    if not outcomes:
        raise ValueError("EMPTY_PARETO_PANEL")
    vectors = {key: tuple(exact(v) for v in vector) for key, vector in outcomes.items()}
    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1 or 0 in dimensions:
        raise ValueError("PARETO_DIMENSIONS")
    dominated = {}
    for key, vector in vectors.items():
        witnesses = [other for other, alt in vectors.items()
                     if other != key and all(a >= b for a, b in zip(alt, vector))
                     and any(a > b for a, b in zip(alt, vector))]
        if witnesses:
            dominated[key] = witnesses
    return {"front": [key for key in vectors if key not in dominated],
            "dominance_witnesses": dominated, "scope": "EXACT_FINITE_PANEL_NOT_GLOBAL_FRONTIER"}
