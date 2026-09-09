"""PRE freeze for order M4 C9-C16 at f06008da: the audit's exact
accepted forgery against verify_preflight.

The adversary (byte-faithful to the audit):
1. run the bounded two-unit MECHANICS_ONLY preflight in a fresh
   tmp root;
2. forge the sine unit's first batch record from
   ACQUISITION_ENDPOINT to ACCEPTED and repair its record
   self-digest; lift the unit's accepted-batch fact to match;
3. replace u0_stop.npz with arbitrary NON-NPZ bytes;
4. repair every producer checksum along the way
   (artifacts_sha256 entries + report_sha256).

The current verifier then returns verified:true — it re-derives
digests and recounts the FORGED ledger but never loads a state,
never replays a transition from the deterministic generator, and
trusts producer booleans.

Source facts frozen alongside: batch JSONL lines parse through
permissive json.loads (the strict loader appears only in dead
code); restart_continuation_identical and
matched_compute_executed are consumed as producer booleans;
M4_HEARTBEAT.json is digest-bound in artifacts_sha256 yet
explicitly skipped by the check (bound AND unchecked); the
rehearsal=False diagnostic trains on MINIBATCH-half examples
(8) while the primary arm trains on 16.

CPU only, fresh tmp root, zero scientific claims, committed
evidence untouched."""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import m4_residual_capacity as m4  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="m4_c9_pre_"))
OUT = TMP / "preflight"
try:
    report = m4.mechanics_preflight(OUT)
    v0 = m4.verify_preflight(OUT)
    print("honest_run:", json.dumps(v0))
    assert v0["verified"] is True

    # ---- the audited forgery ----
    led_p = OUT / "u0_batches.jsonl"
    lines = led_p.read_text().splitlines()
    r0 = json.loads(lines[0])
    assert r0["outcome"] == "ACQUISITION_ENDPOINT", r0["outcome"]
    r0["outcome"] = "ACCEPTED"
    r0["record_sha256"] = m4._self_sha(r0, "record_sha256")
    lines[0] = json.dumps(r0, sort_keys=True)
    led_p.write_text("\n".join(lines) + "\n")

    ck = OUT / "u0_stop.npz"
    ck.write_bytes(b"NOT-AN-NPZ-ARBITRARY-BYTES")

    rep_p = OUT / "M4_PREFLIGHT_REPORT.json"
    rep = json.loads(rep_p.read_text())
    rep["units"][0]["accepted_batches_mechanics_only"] += 1
    rep["artifacts_sha256"]["u0_batches.jsonl"] = \
        m4._sha_file(led_p)
    rep["artifacts_sha256"]["u0_stop.npz"] = m4._sha_file(ck)
    rep["report_sha256"] = m4._self_sha(rep, "report_sha256")
    rep_p.write_text(json.dumps(rep, indent=1))

    v1 = m4.verify_preflight(OUT)
    print("forged_run:", json.dumps(v1))
    assert v1["verified"] is True, \
        "PRE expectation: the forgery is ACCEPTED today"

    # ---- source facts ----
    src = (REPO / "tools/m4_residual_capacity.py").read_text()
    vseg = src[src.index("def verify_preflight"):
               src.index("def main")]
    facts = {
        "jsonl_parsed_permissively":
            "json.loads(line)" in vseg,
        "strict_loader_only_dead_code":
            "if False else" in vseg,
        "restart_boolean_trusted":
            'unit["restart_continuation_identical"]' in vseg,
        "matched_compute_boolean_trusted":
            'unit["matched_compute_executed"]' in vseg,
        "heartbeat_digest_bound_but_skipped":
            'if name == "M4_HEARTBEAT.json":' in vseg
            and "continue" in vseg,
        "no_state_load_no_replay":
            "_load_state" not in vseg
            and "apply_batch" not in vseg,
    }
    aseg = src[src.index("def apply_batch"):
               src.index("def _excl_write")]
    facts["no_rehearsal_arm_trains_half_examples"] = (
        'Xmb = st["assoc_X"][ia]' in aseg
        and "size=MINIBATCH - half" in aseg)
    print("source_facts:", json.dumps(facts, indent=1))
    assert all(facts.values())

    print("\nPRE CONFIRMED at f06008da: a forged first-batch "
          "outcome and an arbitrary-bytes checkpoint under "
          "repaired checksums are VERIFIED true; the verifier "
          "reconstructs nothing from the deterministic generator "
          "and trusts producer booleans; the no-rehearsal "
          "diagnostic processes half the primary arm's examples")
finally:
    shutil.rmtree(TMP)
