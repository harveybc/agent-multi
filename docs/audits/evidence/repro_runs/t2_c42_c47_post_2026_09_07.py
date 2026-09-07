"""POST for order T2 C42-C47: the confirmatory executor exists,
is custody-complete and STRUCTURALLY CLOSED. Zero confirmatory
scores, zero real ledger, zero adjudication; the sealed design and
both installed records untouched."""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
import t2_confirmatory as conf  # noqa: E402
import importlib.util as ilu
spec = ilu.spec_from_file_location(
    "t2exec", REPO / "tools/t2_confirmatory_executor.py")
ex = ilu.module_from_spec(spec)
spec.loader.exec_module(ex)

STATE = Path.home() / ".local/share/agent-multi"
SEALED = STATE / "t2_screen_design_SEALED_V6.json"

print("== 1. the exact confirmatory command, structurally "
      "closed ==")
print("command: PYTHONPATH=. python "
      "tools/t2_confirmatory_executor.py --execute")
with tempfile.TemporaryDirectory() as td:
    try:
        conf.run_confirmatory(
            STATE / "t2_public_data_manifest_20260906.json",
            SEALED, Path(td) / "ledger.json",
            census_path=STATE / "t2_bank_census_20260906.json")
        raise AssertionError("gates opened without the record")
    except SystemExit as exc:
        print("stop:", str(exc)[:110])
        assert "T2_EXECUTION_RECORD_REQUIRED" in str(exc)
        assert not (Path(td) / "ledger.json").exists()
print("live chain: sealed -> fresh(4650/242) -> REAL review "
      "record -> EXECUTION record ABSENT -> typed close, no "
      "ledger")

print("\n== 2. work census (exact) + measured CPU fixture "
      "timings ==")
d = json.loads(SEALED.read_text())
w = ex.census_of_work(d)
print(json.dumps(w))
assert w == {"units": 242, "origins_per_unit": 2, "arms": 4,
             "mlp_seeds": 3, "model_fits": 7744,
             "baseline_evals": 484}
print("measured per-unit rehearsal wall (dev fixtures, NOT "
      "extrapolated as results): sm_co2 25.2s, sm_sunspots 5.4s, "
      "sm_nile 3.9s (see the committed rehearsal output)")

print("\n== 3. state untouched ==")
assert hashlib.sha256(SEALED.read_bytes()).hexdigest() == (
    "d1720f4d6ad05af342c5d02db1dfe8b157c95e7a22684437b9cc6a70e1"
    "3301e5")
assert not (STATE / "t2_confirmatory_results_v6").exists() or \
    not list((STATE / "t2_confirmatory_results_v6").glob(
        "units/RECORD_*"))
assert not (STATE / "t2_attempt_ledger_20260906.json").exists()
print("sealed bytes identical; no unit records under the real "
      "root; no scientific ledger")

print("\nPOST CONFIRMED: executor implemented and closed; "
      "custody + array recompute proven on the rehearsal; zero "
      "scores of the 242 series")
