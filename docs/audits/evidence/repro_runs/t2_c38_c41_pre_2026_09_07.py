"""PRE freeze for order T2 C38-C41: the two custody/path defects
reproduce at agent-multi@2ecd7915.

1. C38: T2_REVIEW_RECORD_PATH points inside the candidate
   repository (docs/audits/evidence/); a record written there by
   the candidate satisfies the reviewer-string/digest checks —
   honestly classified: DECLARED authorship, not externally
   established custody.
2. C39: the public --confirmatory CLI names the obsolete
   t2_confirmatory_design_20260906.json — not draft v6 nor a
   sealed successor; invoking it refuses on that legacy path
   while the reviewed v6 draft sits unused.

Zero downloads, zero scores, zero seal, zero ledger."""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"

print("== C38: review-record path is inside the repo ==")
p = conf.T2_REVIEW_RECORD_PATH
inside = p.relative_to(REPO)
print("productive review path (repo-relative):", inside)
assert str(inside).startswith("docs/audits/evidence/")
mp = STATE / "t2_public_data_manifest_20260906.json"
cp = STATE / "t2_bank_census_20260906.json"
design = conf.strict_json_load(
    STATE / "t2_screen_design_DRAFT_V6_20260907.json", "d")
manifest_sha = conf._sha_file(mp)
census_sha = conf._sha_file(cp)
draft_sha = design["supersedes_draft_sha256"]
rec = {"schema": "agent_multi.musashi_t2_design_review.v1",
       "reviewed_at_date": "2026-09-07",
       "reviewer": "General Musashi",
       "decision": "SEAL_T2_CONFIRMATORY_DESIGN",
       "design_draft_sha256": draft_sha,
       "manifest_sha256": manifest_sha,
       "census_sha256": census_sha}
assert not p.exists()
p.write_text(json.dumps(rec))
try:
    fake_sealed = json.loads(json.dumps(design))
    fake_sealed["design_review_record_sha256"] = \
        conf._sha_file(p)
    got = conf.verify_design_review_record(
        fake_sealed, manifest_sha, census_sha)
    print("candidate-written repo record ACCEPTED: reviewer =",
          got["reviewer"], "— declared authorship, not external "
          "custody")
    accepted = True
finally:
    p.unlink()
assert accepted

print("\n== C39: the public CLI names the obsolete design ==")
src = (REPO / "tools/t2_assay_harness.py").read_text()
line = next(ln for ln in src.splitlines()
            if "t2_confirmatory_design_20260906.json" in ln)
print("CLI line:", line.strip())
print("draft v6 in the CLI:", "DRAFT_V6" in src,
      "| sealed identity in the CLI:", "SEALED" in src)
assert "DRAFT_V6" not in src and "SEALED" not in src
import subprocess
r = subprocess.run(
    [sys.executable, str(REPO / "tools/t2_assay_harness.py"),
     "--census", str(STATE / "nonexistent_census.json"),
     "--confirmatory"],
    capture_output=True, text=True)
print("CLI invocation refuses (legacy path in play):",
      r.returncode != 0)

print("\nPRE CONFIRMED: declarative review custody + legacy CLI "
      "design path frozen")
