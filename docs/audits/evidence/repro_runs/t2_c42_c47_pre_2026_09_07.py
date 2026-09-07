"""PRE freeze for order T2 C42-C47 at agent-multi@48037102: the
design is SEALED and verified, but the confirmatory EXECUTOR does
not exist.

Frozen facts:
1. the real sealed v6 exists with EXACTLY the audited identities
   (file d1720f4d..., self e1e3761b..., review record 13310ef8...,
   mode 0600) and no scientific ledger exists;
2. run_confirmatory() with the real sealed design reaches the
   deliberate CONFIRMATORY_EXECUTION_NOT_IMPLEMENTED stop (via a
   THROWAWAY ledger path — the real ledger is never created);
3. no executor tool exists: nothing can produce the 242 units'
   records, no per-unit custody schema exists, and no external
   EXECUTION record gate exists yet.

Zero scores, zero real ledger, zero seal changes."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
import t2_confirmatory as conf  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
SEALED = STATE / "t2_screen_design_SEALED_V6.json"

print("== 1. the real sealed design, audited identities ==")
fsha = hashlib.sha256(SEALED.read_bytes()).hexdigest()
d = json.loads(SEALED.read_text())
print("file:", fsha[:16], "| self:", d["design_sha256"][:16])
assert fsha == ("d1720f4d6ad05af342c5d02db1dfe8b157c95e7a22684437"
                "b9cc6a70e13301e5")
assert d["design_sha256"] == (
    "e1e3761b4c878c390eeac0d220faa48b9669e733507f9b2001a167d31d"
    "6b36d6")
assert d["design_review_record_sha256"] == (
    "13310ef88720f6a50d7f4a106c1490eac5cf65d5fb3659397c03bec125"
    "1d86fb")
assert d["supersedes_draft_sha256"] == conf.T2_V6_DRAFT_FILE_SHA
assert not (STATE / "t2_attempt_ledger_20260906.json").exists()
print("no scientific ledger exists")

print("\n== 2. the single path stops at NOT_IMPLEMENTED ==")
with tempfile.TemporaryDirectory() as td:
    try:
        conf.run_confirmatory(
            STATE / "t2_public_data_manifest_20260906.json",
            SEALED, Path(td) / "throwaway_ledger.json",
            census_path=STATE / "t2_bank_census_20260906.json")
        raise AssertionError("did not stop")
    except SystemExit as exc:
        print("stop:", str(exc)[:90])
        assert "EXECUTION_NOT_IMPLEMENTED" in str(exc)

print("\n== 3. no executor exists ==")
assert not (REPO / "tools/t2_confirmatory_executor.py").exists()
src = (REPO / "tools/t2_confirmatory.py").read_text()
print("execution-record gate in code:",
      "EXECUTION_RECORD" in src)
assert "EXECUTION_RECORD" not in src
print("unit-record custody schema in code:",
      "t2_confirmatory_unit_record" in src)
assert "t2_confirmatory_unit_record" not in src

print("\nPRE CONFIRMED: sealed design verified; executor absent; "
      "execution gate absent; zero scores/ledger")
