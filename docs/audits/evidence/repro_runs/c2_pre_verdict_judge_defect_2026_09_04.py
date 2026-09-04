"""C2 PRE (order @4c1f1532 §3): the CURRENT N2 verdict judge
contradicts the sealed predeclaration rule — "any unlicensed
candidate makes the census INCONCLUSIVE" — in two ways:

A. one unlicensed candidate among eight failures -> judge returns a
   clean NO_TARGET_CANDIDATE_DEMONSTRATED;
B. one unlicensed candidate beside an apparent passer -> judge
   returns TARGET_CANDIDATE_FOUND and even selects.
"""
import importlib.util
import shutil
import sys
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests" / "unit"))

spec = importlib.util.spec_from_file_location(
    "tcn2", REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tcn2)

tspec = importlib.util.spec_from_file_location(
    "tsuite", REPO / "tests/unit/test_target_horizon_census_n2.py")
tsuite = importlib.util.module_from_spec(tspec)
tspec.loader.exec_module(tsuite)

WKS = tsuite.WKS


def starve(result):
    result["class_support"]["score"] = {"0": 5, "1": 60, "2": 151}


def scenario(name, **kwargs):
    home = Path.home() / ".cache" / "c2_pre" / uuid.uuid4().hex
    home.mkdir(parents=True)
    try:
        tsuite._full_run(home, **kwargs)
        trace = tcn2.aggregate_final(home)
        print(f"{name}: verdict={trace['verdict']} "
              f"unlicensed={trace['inconclusive_candidates']} "
              f"selected={trace.get('selection', {}).get('selected')}")
        return trace
    finally:
        shutil.rmtree(home, ignore_errors=True)


# A: bar_h6 unlicensed, the other eight candidates fail
a = scenario("A(unlicensed among failures)",
             overrides={("bar_h6", wk): starve for wk in WKS})
assert a["verdict"] == "NO_TARGET_CANDIDATE_DEMONSTRATED", a["verdict"]

# B: bar_h6 unlicensed while vol_h6 passes
b = scenario("B(unlicensed beside passer)",
             deltas={"vol_h6": 0.15},
             overrides={("bar_h6", wk): starve for wk in WKS})
assert b["verdict"] == "TARGET_CANDIDATE_FOUND", b["verdict"]
assert b["selection"]["selected"] == ["vol_h6"]

print("PRE CONFIRMED: sealed rule 'ANY unlicensed candidate -> "
      "INCONCLUSIVE' is violated in both scenarios")
