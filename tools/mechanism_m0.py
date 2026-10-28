"""Run the exact M0 fixture sequence; stdout report, no training or deployment."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.mechanism_games import FinitePolicyGame, pareto_front  # noqa: E402


def build_report():
    coordination = FinitePolicyGame((3, 3), lambda p: (0, 2, 3)[p[0]] if p[0] == p[1] else -1)
    equilibria = coordination.equilibria()
    assert equilibria == ((0, 0), (1, 1), (2, 2))
    linear = FinitePolicyGame((27,) * 20, lambda p: int(p[-1] == 26))
    cert = linear.unilateral_certificate((0,) * 20)
    assert cert["epsilon"] == 1 and len(linear.cache) == 521
    redundant = FinitePolicyGame((2,) * 5, lambda p: 10 if any(p) else 0)
    shapley = redundant.role_shapley((1,) * 5, tuple(range(5)))
    assert all(v == 2 for v in shapley["shares"].values())
    table = ((0, 0), (0, 4), (10, 3))
    conflicting = FinitePolicyGame((3, 2), lambda p: table[p[0]][p[1]])
    credits = [conflicting.role_shapley(p, (0, 1))["shares"][0] for p in ((1, 1), (2, 1))]
    assert credits == [2, Fraction(13, 2)]
    files = ["app/mechanism_games.py", "tools/mechanism_m0.py",
             "docs/work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json"]
    return {
        "schema": "mechanism_synthesis.m0_exact_fixtures.v1",
        "status": "EXACT_FIXTURES_PASS_NOT_STRATEGY_RELEASE",
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
        "coordination": {"equilibria": equilibria,
                         "values": [coordination.value(p) for p in equilibria],
                         "pareto": pareto_front({str(p): (coordination.value(p),) for p in equilibria})},
        "unilateral_20_by_27": {"distinct_profiles": len(linear.cache),
                               "deviations": len(cert["deviations"]), "epsilon": cert["epsilon"],
                               "coverage": cert["coverage"]},
        "redundancy": {"difference_rewards": [redundant.difference_reward((1,)*5, i) for i in range(5)],
                       "shapley": shapley},
        "learning_objective_counterexample": {"account_values": [4, 3], "player_shapley": credits},
        "training_runs": 0, "market_rollouts": 0,
    }


def encode(report):
    def default(value):
        if isinstance(value, Fraction):
            return str(value)
        raise TypeError(type(value).__name__)
    return json.dumps(report, default=default, sort_keys=True, indent=2, allow_nan=False)


if __name__ == "__main__":
    print(encode(build_report()))
