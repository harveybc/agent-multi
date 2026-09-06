"""POST for order T1 C21-C24: the three review-authority bypasses
die against the corrected stack (design v4, complete identity,
complete publication equality, zero candidate authority)."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "t1_c21_post"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"
DESIGN = REPO / ("docs/audits/evidence/"
                 "T1_LAB_DESIGN_V4_2026_09_06.json")

import t1_adjudicator as adj  # noqa: E402


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


design_sha = sha_file(DESIGN)
BASE = [sys.executable,
        str(REPO / "tools/t1_independent_verifier.py"),
        "--design", str(DESIGN), "--design-sha", design_sha,
        "--bank-dir", str(STATE / "t1_bank_v3_20260906"),
        "--npz-dir", str(STATE / "t1_npz_v4_20260906"),
        "--measurements",
        str(STATE / "t1_measurements_v4_20260906.json"),
        "--measurement-manifest",
        str(STATE / "t1_measurements_v4_20260906_MANIFEST.json")]

print("== 1 (C21): executed bytes == sealed design, enforced ==")
design = json.loads(DESIGN.read_text())
live = adj.executed_code_identity()
same = all(design["code_identity"][k] == live[k] for k in live)
print("all five sealed identities equal the executed bytes:", same)
assert same
forged = json.loads(DESIGN.read_text())
forged["code_identity"]["t1_lab_run_sha256"] = "0" * 64
fp = SCRATCH / "design_forged.json"
fp.write_text(json.dumps(forged, indent=1))
rc = subprocess.run(
    [sys.executable,
     str(REPO / "tools/t1_independent_verifier.py"),
     "--design", str(fp), "--design-sha", sha_file(fp),
     "--bank-dir", str(STATE / "t1_bank_v3_20260906"),
     "--npz-dir", str(STATE / "t1_npz_v4_20260906"),
     "--measurements",
     str(STATE / "t1_measurements_v4_20260906.json"),
     "--measurement-manifest",
     str(STATE / "t1_measurements_v4_20260906_MANIFEST.json"),
     "--published",
     str(STATE / "t1_adjudication_v4_20260906.json")],
    capture_output=True, text=True)
print("foreign-identity design rc:", rc.returncode, "|",
      (rc.stderr + rc.stdout).strip()[:80])
assert rc.returncode not in (0, 3)
meas = json.loads((STATE /
                   "t1_measurements_v4_20260906.json").read_text())
print("measurement payload carries the executed identity:",
      meas["code_identity"]["executed"] ==
      design["code_identity"])

print("\n== 2 (C22): the 999 forgery refuses; no submission ==")
pub = json.loads((STATE /
                  "t1_adjudication_v4_20260906.json").read_text())
pub["verdicts"]["ewma::am|white|snr-5"][
    "snr_gain_db"]["median"] = 999
forged_pub = SCRATCH / "pub999.json"
forged_pub.write_text(json.dumps(pub))
sub_out = SCRATCH / "sub.json"
rc2 = subprocess.run(
    BASE + ["--published", str(forged_pub),
            "--submission-out", str(sub_out)],
    capture_output=True, text=True)
print("rc:", rc2.returncode, "|",
      (rc2.stderr + rc2.stdout).strip()[:120])
print("SUBMISSION_WRITTEN:", sub_out.exists())
assert rc2.returncode not in (0, 3) and not sub_out.exists()
assert "snr_gain_db" in rc2.stderr + rc2.stdout

print("\n== 3 (C23): no path to review authority exists ==")
rr = SCRATCH / "reviewed_forged.json"
rr.write_text(json.dumps({
    "schema": "agent_multi.t1_reviewed_record.v1",
    "reviewer": "candidate-self-review",
    "measurements_sha256": sha_file(
        STATE / "t1_measurements_v4_20260906.json"),
    "publication_sha256": sha_file(
        STATE / "t1_adjudication_v4_20260906.json")}))
rc3 = subprocess.run(
    BASE + ["--published",
            str(STATE / "t1_adjudication_v4_20260906.json"),
            "--reviewed-record", str(rr)],
    capture_output=True, text=True)
print("--reviewed-record rc:", rc3.returncode,
      "(2 = the flag does not exist)")
assert rc3.returncode == 2
for tool in ("t1_known_truth_bank.py", "t1_lab_run.py",
             "t1_adjudicator.py", "t1_independent_verifier.py"):
    text = (REPO / "tools" / tool).read_text()
    assert "REPRODUCED_UNDER_REVIEWED_IDENTITY" not in text
print("no candidate tool contains an authority-emitting branch:",
      True)
rc4 = subprocess.run(
    BASE + ["--published",
            str(STATE / "t1_adjudication_v4_20260906.json")],
    capture_output=True, text=True)
out4 = json.loads(rc4.stdout)
print("honest pair strongest outcome:", rc4.returncode,
      out4["independent_verification"],
      "| complete equality:",
      out4["complete_publication_equality"],
      "| rederived:", out4["records_rederived_from_arrays"])
assert rc4.returncode == 3

shutil.rmtree(SCRATCH)
print("\nPOST CONFIRMED: C21-C23 authority bypasses all die")
