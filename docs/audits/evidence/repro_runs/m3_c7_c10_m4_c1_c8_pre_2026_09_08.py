"""PRE freeze for orders M3 C7-C10 + M4 C1-C8 at 16e12a25.

M3: the audit's exact foreign-cell attack — one valid task body
    moved to K=999 with its self-digest, the records-file digest
    and the summary self-digest repaired — is ACCEPTED by the
    verifier after a full 9,800-task regeneration; design and
    summary are parsed with permissive json.loads.
M4: the preflight trains/evaluates only the NEWEST association
    batch (non-cumulative); the 'restart' never continues from
    the reloaded state; heartbeat/stop/wall/RSS/matched-compute
    are declarative only; no verify_preflight exists; a replaced
    checkpoint leaves the report's self-digest valid; retention
    stops on ONE failure while the design says two.

CPU only; the immutable M3 v1-v3 evidence is never modified
(attacks run on copies)."""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import m3_cover_calibration as m3  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="m3c7_pre_"))

print("== M3: the exact foreign-cell attack is ACCEPTED ==")
work = TMP / "runs"
shutil.copytree(m3.RUNS_DIR_V3, work)
lines = (work / "M3_TASK_RECORDS.jsonl").read_text().splitlines()
donor = next(json.loads(ln) for ln in lines
             if json.loads(ln)["kind"] == "task")
foreign = dict(donor)
foreign["K"] = 999
foreign["record_sha256"] = m3._self_sha(foreign, "record_sha256")
lines.append(json.dumps(foreign, sort_keys=True))
(work / "M3_TASK_RECORDS.jsonl").write_text(
    "\n".join(lines) + "\n")
summ = json.loads((work / "M3_SUMMARY.json").read_text())
import hashlib
summ["records_file_sha256"] = hashlib.sha256(
    (work / "M3_TASK_RECORDS.jsonl").read_bytes()).hexdigest()
summ["summary_sha256"] = m3._self_sha(summ, "summary_sha256")
(work / "M3_SUMMARY.json").write_text(json.dumps(summ))
out = m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)
print(json.dumps({"extra_cell_record_accepted": True, **out},
                 indent=1)[:220])
assert out["verified"] is True
src = (REPO / "tools/m3_cover_calibration.py").read_text()
seg = src[src.index("def load_design"):src.index("def _primary_for")]
assert "json.loads(p.read_text())" in seg      # permissive parse
vseg = src[src.index("def verify("):src.index("def main(")]
assert "json.loads((runs / \"M3_SUMMARY.json\")" in vseg
assert "sealed grid keys" not in vseg
print("=> no global cell-population equality; permissive design/"
      "summary parsing (source)")

print("\n== M4: six claimed mechanics are absent or wrong ==")
msrc = (REPO / "tools/m4_residual_capacity.py").read_text()
seg = msrc[msrc.index("def mechanics_preflight"):]
facts = {
    "trains_only_newest_batch":
        "np.vstack([Xtr, Xa])" in seg and
        "cumulative" not in seg.split("def main")[0].lower(),
    "acq_loss_only_newest": "_loss(fork, Xa, ya)" in seg,
    "restart_never_continues":
        "fork2, _ = _load_ckpt(ckf)" in seg and
        "_mlp_sgd(fork2" not in seg,
    "one_retention_failure_not_two":
        "if ret_loss > margin:" in seg and
        "consecutive" not in seg,
    "no_heartbeat_or_stop":
        "M4_HEARTBEAT" not in seg and "M4_STOP" not in seg,
    "no_matched_compute_execution":
        "matched" not in seg.split("def main")[0],
    "no_verify_preflight": "def verify_preflight" not in msrc,
}
print(json.dumps(facts, indent=1))
assert all(facts.values())
# report digest survives checkpoint forgery
out4 = TMP / "m4pf"
r = m4.mechanics_preflight(out4)
ck = next(out4.glob("u0_stop.npz"))
ck.write_bytes(b"forged")
fresh = json.loads((out4 / "M4_PREFLIGHT_REPORT.json").read_text())
ok = m4._self_sha(fresh, "report_sha256") == fresh["report_sha256"]
print(json.dumps(
    {"report_self_valid_after_checkpoint_forgery": ok}, indent=1))
assert ok

shutil.rmtree(TMP)
print("\nPRE CONFIRMED at 16e12a25: foreign-cell acceptance, "
      "permissive parsers, non-cumulative M4 intervention, "
      "no-continuation restart, declarative-only limits, and "
      "forgery-blind reports — all frozen")
