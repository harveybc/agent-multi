"""Read pinned source and run bounded, synthetic design counterexamples only."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess


COMMIT = "b4b19ed"
SOURCE = "tools/strategy_behaviour_matrix.py"


def inspect_validity(repo: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args])

    revision = git("rev-parse", COMMIT + "^{commit}").decode().strip()
    raw = git("show", revision + ":" + SOURCE)
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "validity")
    # Execute only the independently inspected dictionary-based verifier.
    space = {"re": re, "TIMING_EXIT_WITNESSES": ()}
    exec(compile(ast.Module(body=[node], type_ignores=[]), SOURCE, "exec"), space)
    validate = space["validity"]
    old = {"pass": False, "events": 1}
    new = {"pass": True, "events": 1}
    witnesses = {"W": old, "W_v2": new}
    empty = validate({}, {}, None)
    vacuous = validate(witnesses, {"W": {"pass": True, "events": 0}}, None)
    space["TIMING_EXIT_WITNESSES"] = ("W",)
    conflicting = validate(witnesses, {"W": old}, {"W": new})
    assert empty["PASS"] and not empty["per_witness"]
    assert vacuous["PASS"] and vacuous["unvalidated"] == ["W"]
    assert conflicting["PASS"] and conflicting["per_witness"]["W"]["status"] == "DEFECT"
    return {
        "commit": revision, "source": SOURCE,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "function_first_line": node.lineno,
        "empty": empty, "vacuous": vacuous, "conflicting_controls": conflicting,
    }


def mathematical_witnesses() -> dict:
    from fractions import Fraction as F
    negative_optimum = F(-10)
    assert not negative_optimum >= F(9, 10) * negative_optimum
    assert F(9) >= F(9, 10) * F(10)
    assert not F(9) - 10 >= F(9, 10) * (F(10) - 10)
    before_shapley = (F(0) + F(4)) / 2
    after_shapley = (F(10) + F(3)) / 2
    assert after_shapley > before_shapley and 3 < 4
    return {
        "negative_optimum": {"best": -10, "threshold": -9, "best_is_rejected": True},
        "translation": {"best": 10, "candidate": 9, "passes": True,
                        "shift": -10, "shifted_candidate_passes": False},
        "shapley_incentive": {
            "potential_table_rows_player_1_columns_player_2": [[0, 0], [0, 4], [10, 3]],
            "deviation": [[1, 1], [2, 1]],
            "account": [4, 3], "player_1_shapley": [str(before_shapley), str(after_shapley)],
        },
        "gap": {"entry": 100, "stop": 99, "first_executable_quote": 95,
                "units": 1, "planned_stop_loss": 1, "realized_loss_before_fees": 5},
        "negative_regression_weight": {"target": 0, "weight": -1,
                                       "predictions": [1, 10], "weighted_squared_losses": [-1, -100]},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    report = {"schema": "guided_mechanism.design_audit_witnesses.v1",
              "source_verifier": inspect_validity(args.source_repo),
              "mathematical_witnesses": mathematical_witnesses(),
              "scope": "Bounded design counterexamples; no training or market-matrix replay"}
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
