"""PRE freeze for order T2 C31-C36 (final screen design): the
residual findings reproduce against agent-multi@fd2be26a.

1. C31: a panel with ONE evaluable extreme series and all others
   NOT_EVALUABLE still ADVANCES — no per-panel extreme-support
   minimum exists in the v4 contract;
2. C35: the three-origin geometry is INFEASIBLE for hospital
   (length 84): the productive rule refuses, and the harness floor
   is 120 — while the ordered TWO-origin geometry yields exactly
   two consecutive 17-observation score windows with all model
   minimums intact (target arithmetic frozen);
3. the v4 draft carries hospital GEOMETRY_LIMITED with 0 admissible
   units and a 202-series population (the state C35 supersedes);
4. current sim digests recorded (both simulations are panel-effect
   level and geometry-independent).

C32/C33/C34 are already enforced by the C25-C30 corrections
(window binding, exact costs, full fresh re-derivation, symlink
root, wiring); this order demands INDEPENDENT per-field mutations,
delivered in the C36 battery. Zero downloads, zero scores, zero
scientific ledger."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tests"))

import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2tests", REPO / "tests/test_t2_harness.py")
tt = ilu.module_from_spec(spec)
spec.loader.exec_module(tt)

STATE = Path.home() / ".local/share/agent-multi"

print("== C31: one evaluable extreme series licenses the panel ==")
d = tt._d3()
recs = []
for uid in d["task_population"]["series_ids"]:
    first_in_panel = uid.endswith("::s0")
    recs.append(tt._rec_for(d, uid, delta=0.10,
                            ex_support=(4 if first_in_panel
                                        else 0)))
out = conf.adjudicate_screen(recs, d)
print("SUPPORT_1_OF_N:", out["verdict"])
for p_, st in out["panels"].items():
    assert st["extreme_state"] == "EVALUATED", (p_, st)
assert out["verdict"] == "ADVANCE_TO_DOMAIN_VALIDATION"
print("panel extreme_state=EVALUATED from a single series; "
      "no per-panel support minimum exists")

print("\n== C35: three-origin geometry infeasible for length 84; "
      "two-origin target arithmetic ==")
for k in (3, 2):
    try:
        bank.origin_windows_for(84, k, 0.6)
        print(f"{k}-origin windows for n=84: ACCEPTED — BUG")
        raise AssertionError
    except SystemExit as exc:
        print(f"{k} origins, n=84:", str(exc)[:70])
# the fixed 120 floor blocks length 84 under ANY origin count —
# the ordered correction derives feasibility from the REAL model
# minimums instead. The C35 target arithmetic, frozen:
base = int(84 * 0.6)
w = (84 - base) // 2
w2 = {"origin0": {"train": [0, base], "score": [base, base + w]},
      "origin1": {"train": [0, base + w], "score": [base + w, 84]}}
print("target 2-origin windows for n=84:", w2)
assert w2 == {"origin0": {"train": [0, 50], "score": [50, 67]},
              "origin1": {"train": [0, 67], "score": [67, 84]}}
for okey, wb in w2.items():
    lo, hi = wb["score"]
    assert hi - lo == 17
    fit_rows = wb["train"][1] - 8 - 1
    scored_rows = hi - lo - 8 - 1
    n_val = max(8, int(fit_rows * 0.2))
    print(f"{okey}: score width 17 | fit rows {fit_rows} "
          f"(mlp val {n_val}, fit {fit_rows - n_val}) | "
          f"scored rows {scored_rows}")
    assert fit_rows - n_val >= 8 and scored_rows >= 1
    assert wb["train"][1] >= 12 + 2   # period 12 mase denominator
src = (REPO / "tools/t2_assay_harness.py").read_text()
print("harness floor today: 'if n < 120' present:",
      "if n < 120" in src, "| ROLLING_ORIGINS = 3:",
      "ROLLING_ORIGINS = 3" in src)
assert "if n < 120" in src and "ROLLING_ORIGINS = 3" in src

print("\n== v4 state to be superseded ==")
v4 = json.loads((STATE / "t2_screen_design_DRAFT_V4_20260906.json"
                 ).read_text())
gl = v4["task_population"]["geometry_limited_panels"]
print("v4 hospital:", gl["hospital"]["n_geometry_admissible"],
      "admissible of", gl["hospital"]["n_census_admissible"],
      "| population:", len(v4["task_population"]["series_ids"]))
assert gl["hospital"]["n_geometry_admissible"] == 0
assert len(v4["task_population"]["series_ids"]) == 202


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("\n== sim digests (geometry-independent, to be re-executed) "
      "==")
print("coverage:", sha(STATE / "t2_coverage_sim_20260906.json")[:16])
print("screen:  ", sha(STATE / "t2_screen_sim_20260906.json")[:16])

print("\nPRE CONFIRMED: C31 positive frozen; C35 target geometry "
      "arithmetic frozen; v4 state recorded")
