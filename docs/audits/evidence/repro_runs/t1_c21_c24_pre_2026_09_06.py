"""PRE freeze for order T1 C21-C24: the three review-authority
bypasses reproduce against agent-multi@e3e817e1 (v3 artifacts).
Read-only on sealed state; mutations on scratch copies."""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
SCRATCH = Path.home() / ".cache" / "t1_c21_pre"
if SCRATCH.exists():
    shutil.rmtree(SCRATCH)
SCRATCH.mkdir(parents=True)
STATE = Path.home() / ".local/share/agent-multi"
DESIGN = REPO / ("docs/audits/evidence/"
                 "T1_LAB_DESIGN_V3_2026_09_06.json")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


print("== C21: executed lab bytes differ from the sealed design ==")
design = json.loads(DESIGN.read_text())
sealed_lab = design["code_identity"]["t1_lab_run_sha256"]
live_lab = sha_file(REPO / "tools/t1_lab_run.py")
meas = json.loads((STATE /
                   "t1_measurements_v3_20260906.json").read_text())
print("sealed  t1_lab_run_sha256:", sealed_lab)
print("live    t1_lab_run_sha256:", live_lab)
print("payload lab_sha256      :",
      meas["code_identity"]["lab_sha256"])
assert sealed_lab != live_lab
assert meas["code_identity"]["lab_sha256"] == live_lab
lab_src = (REPO / "tools/t1_lab_run.py").read_text()
body = lab_src[lab_src.index("def main"):]
print("lab verifies ONLY the causal-operator digest at startup:",
      "causal_operators_sha256" in body
      and "t1_lab_run_sha256" not in body)
for tool in ("t1_adjudicator.py", "t1_independent_verifier.py"):
    tsrc = (REPO / "tools" / tool).read_text()
    print(f"{tool} enforces complete executable identity:",
          "code_identity" in tsrc
          and "t1_adjudicator_sha256" in tsrc)

print("\n== C22: a forged quantitative field still yields a "
      "submission ==")
pub = json.loads((STATE /
                  "t1_adjudication_v3_20260906.json").read_text())
target = pub["verdicts"]["ewma::am|white|snr-5"]
real_median = target["snr_gain_db"]["median"]
target["snr_gain_db"]["median"] = 999
forged_pub = SCRATCH / "publication_forged.json"
forged_pub.write_text(json.dumps(pub))
design_sha = sha_file(DESIGN)
sub_out = SCRATCH / "submission_forged.json"
rc = subprocess.run(
    [sys.executable, str(REPO / "tools/t1_independent_verifier.py"),
     "--design", str(DESIGN), "--design-sha", design_sha,
     "--bank-dir", str(STATE / "t1_bank_v3_20260906"),
     "--npz-dir", str(STATE / "t1_npz_v3_20260906"),
     "--measurements",
     str(STATE / "t1_measurements_v3_20260906.json"),
     "--measurement-manifest",
     str(STATE / "t1_measurements_v3_20260906_MANIFEST.json"),
     "--published", str(forged_pub),
     "--submission-out", str(sub_out)],
    capture_output=True, text=True)
out = json.loads(rc.stdout)
print("verifier label:", out["independent_verification"])
print("records re-derived:", out["records_rederived_from_arrays"])
print("SUBMISSION_WRITTEN_FOR_FORGED_PUBLICATION",
      sub_out.is_file(), "publication_sha256",
      out["publication_sha256"][:16])
print("(real median was", real_median, "-> published 999;",
      "verdict labels unchanged)")
assert rc.returncode == 3 and sub_out.is_file()

print("\n== C23: the candidate can impersonate the reviewer ==")
rr = SCRATCH / "reviewed_forged.json"
rr.write_text(json.dumps({
    "schema": "agent_multi.t1_reviewed_record.v1",
    "reviewer": "candidate-self-review",
    "measurements_sha256": sha_file(
        STATE / "t1_measurements_v3_20260906.json"),
    "publication_sha256": sha_file(forged_pub),
    "authority_looking_extra_field": "TOTALLY_OFFICIAL"}))
rc2 = subprocess.run(
    [sys.executable, str(REPO / "tools/t1_independent_verifier.py"),
     "--design", str(DESIGN), "--design-sha", design_sha,
     "--bank-dir", str(STATE / "t1_bank_v3_20260906"),
     "--npz-dir", str(STATE / "t1_npz_v3_20260906"),
     "--measurements",
     str(STATE / "t1_measurements_v3_20260906.json"),
     "--measurement-manifest",
     str(STATE / "t1_measurements_v3_20260906_MANIFEST.json"),
     "--published", str(forged_pub),
     "--reviewed-record", str(rr)],
    capture_output=True, text=True)
out2 = json.loads(rc2.stdout)
print("verifier label:", out2["independent_verification"])
print("reviewer accepted:", out2.get("reviewer"))
print("VERIFIER_RC", rc2.returncode)
assert rc2.returncode == 0
assert out2["independent_verification"] == \
    "REPRODUCED_UNDER_REVIEWED_IDENTITY"

shutil.rmtree(SCRATCH)
print("\nPRE CONFIRMED: C21-C23 authority bypasses all reproduce")
