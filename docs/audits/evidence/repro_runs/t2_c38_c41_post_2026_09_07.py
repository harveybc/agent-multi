"""POST for order T2 C38-C41: external review custody and the ONE
sealing path stand; the scientific design v6 is untouched. No real
record, no real seal, no ledger, no score."""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
import t2_confirmatory as conf  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"


def dies(label, fn, needle):
    try:
        fn()
    except SystemExit as e:
        ok = needle in str(e)
        print(f"{label}: DIES [{str(e)[:78]}] match={ok}")
        assert ok, (label, str(e))
        return
    raise AssertionError(f"{label}: DID NOT DIE")


print("== 1. C38: review custody is EXTERNAL now ==")
p = conf.T2_REVIEW_RECORD_PATH
try:
    p.relative_to(REPO)
    raise AssertionError("review path still inside the repo")
except ValueError:
    print("productive review path: ~/.config/agent-multi/"
          "reviewer_authority/ (outside the candidate repo)")
look = REPO / ("docs/audits/evidence/"
               "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json")
assert not look.exists()
look.write_text(json.dumps({"schema":
                            "agent_multi.musashi_t2_design_"
                            "review.v2"}))
try:
    dies("REPO_LOOKALIKE",
         lambda: conf.verify_design_review_record(
             {"design_review_record_sha256": "0" * 64},
             "b" * 64, "c" * 64),
         "DESIGN_REVIEW_REQUIRED")
finally:
    look.unlink()
src = (REPO / "tools/t2_confirmatory.py").read_text()
assert "cryptographically" in src
print("honest custody-not-authorship prose present")

print("\n== 2. C39: drafts never score; the CLI names the ONE "
      "sealed identity ==")
mp = STATE / "t2_public_data_manifest_20260906.json"
cp = STATE / "t2_bank_census_20260906.json"
dp = STATE / "t2_screen_design_DRAFT_V6_20260907.json"
import tempfile
with tempfile.TemporaryDirectory() as td:
    dies("DRAFT_TO_SCORING",
         lambda: conf.run_confirmatory(
             mp, dp, Path(td) / "ledger.json", census_path=cp),
         "SEALED_DESIGN_REQUIRED")
hsrc = (REPO / "tools/t2_assay_harness.py").read_text()
assert "t2_confirmatory_design_20260906.json" not in hsrc
assert "t2_screen_design_SEALED_V6.json" in hsrc
print("CLI names t2_screen_design_SEALED_V6.json; the legacy "
      "2026-09-06 filename is gone")
seg = src[src.index("def run_confirmatory"):]
seg = seg[:seg.index("\ndef ", 10)]
assert seg.index("SEALED_DESIGN_REQUIRED") < \
    seg.index("fresh_verify") < \
    seg.index("verify_design_review_record") < \
    seg.index("open_attempt_ledger")
print("single mechanical order: sealed -> fresh -> review -> "
      "ledger (shared by CLI and direct API)")

print("\n== 3. the accepted scientific design v6 is untouched ==")
v6 = json.loads(dp.read_text())
assert hashlib.sha256(dp.read_bytes()).hexdigest() == \
    conf.T2_V6_DRAFT_FILE_SHA
assert v6["design_sha256"] == conf.T2_V6_DRAFT_SELF_SHA
print("draft v6 bytes:", conf.T2_V6_DRAFT_FILE_SHA[:12],
      "| self:", conf.T2_V6_DRAFT_SELF_SHA[:12], "(byte-equal)")
assert not (STATE / "t2_screen_design_SEALED_V6.json").exists()
print("no real sealed design exists; no real review record "
      "exists")

print("\nPOST CONFIRMED: external custody + one sealing path; "
      "scientific v6 frozen; no seal, no score, no ledger")
