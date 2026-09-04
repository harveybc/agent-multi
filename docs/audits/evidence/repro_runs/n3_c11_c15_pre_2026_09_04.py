"""PRE freeze for order @13fdf18c (C11-C15): reproduce every
finding exactly as Musashi reported it, before any edit.

C11-A: flipping the v3 registry entry to 'reviewed' in a
       candidate-controlled copy mints N3_PUBLICATION_VERIFIED.
C11-B: a coherent forged POSITIVE bundle, self-added as 'reviewed',
       obtains the gate-bearing label with a fabricated reviewer.
C13:   an arbitrary object replacing v2_correction_map still passes
       as consistent without being named unverified.
C14:   the committed POST output ends in PRE language.
C15:   'science_byte_equal' hashes a field subset, not bytes.
"""
import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "n3f", REPO / "tools" / "n3_fresh_confirmation.py")
n3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3f)
from agent_plugins.experiment_runtime import sha_obj  # noqa: E402

V3 = (REPO / "docs/audits/evidence/"
      "N3_FRESH_CONFIRMATION_BUNDLE_V3_2026_09_04.json")
REG = REPO / n3f.ALLOWLIST
reg_backup = REG.read_bytes()


def redigest(u):
    u["payload_sha256"] = sha_obj(
        {k: v for k, v in u.items() if k != "payload_sha256"})


try:
    # ---- C11-A: candidate flips its own pending entry ----
    reg = json.loads(reg_backup)
    v3_sha = hashlib.sha256(V3.read_bytes()).hexdigest()
    reg["entries"][v3_sha]["status"] = "reviewed"
    reg["entries"][v3_sha]["reviewed_by"] = \
        "candidate self-review"
    REG.write_text(json.dumps(reg, indent=1) + "\n")
    out = n3f.verify(V3, v3_sha)
    print("C11-A:", out["verdict"],
          "| gate_bearing:", out.get("gate_bearing"),
          "| authority:", out.get("authority"))
    assert out["verdict"] == "N3_PUBLICATION_VERIFIED"
    assert out.get("gate_bearing") is True

    # ---- C11-B: coherent forged POSITIVE, self-reviewed ----
    b = json.loads(V3.read_text())
    for u in b["units"]:
        if u["horizon"] == 6:
            y = np.asarray(u["labels"])
            fake = 0.95 * np.asarray(u["arms"]["arm2"]["probs"]) \
                + 0.05 * np.eye(3)[y]
            fake = fake / fake.sum(axis=1, keepdims=True)
            u["arms"]["arm3"]["probs"] = [
                [float(x) for x in row] for row in fake]
            u["arms"]["arm3"]["metrics"] = n3f.unit_metrics(
                y, fake)
            redigest(u)
    contrasts, stats, _ = n3f._rederive(b["units"])
    b["contrasts"] = contrasts
    b["verdict"] = n3f.decide(stats, True, True)
    b["digests"]["code"] = n3f._code_digest()
    forged = Path.home() / ".cache" / "c11_forged_positive.json"
    forged.write_text(json.dumps(b, default=float))
    fsha = hashlib.sha256(forged.read_bytes()).hexdigest()
    reg["entries"][fsha] = {
        "artifact": forged.name, "status": "reviewed",
        "reviewed_by": "candidate self-review",
        "code_digest": b["digests"]["code"],
        "decision": b["verdict"]}
    REG.write_text(json.dumps(reg, indent=1) + "\n")
    out = n3f.verify(forged, fsha)
    print("C11-B: scientific_decision =",
          out["rederived_decision"])
    print("C11-B: verifier_label =", out["verdict"],
          "| gate_bearing:", out.get("gate_bearing"),
          "| authority:", out.get("authority"))
    assert out["rederived_decision"] == \
        "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
    assert out["verdict"] == "N3_PUBLICATION_VERIFIED"
    forged.unlink()
finally:
    REG.write_bytes(reg_backup)

# ---- C13: arbitrary correction map accepted silently ----
b = json.loads(V3.read_text())
b["v2_correction_map"] = {"arbitrary_unverified_claim": True}
b["digests"]["code"] = n3f._code_digest()
p = Path.home() / ".cache" / "c13_arbitrary_map.json"
p.write_text(json.dumps(b, default=float))
sha = hashlib.sha256(p.read_bytes()).hexdigest()
out = n3f.verify(p, sha)
print("C13:", out["verdict"],
      "| names the map unverified?:",
      "informational_unverified_fields" in out)
assert out["verdict"] == "N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST"
assert "informational_unverified_fields" not in out
p.unlink()

# ---- C14: POST output carries PRE language ----
post = (REPO / "docs/audits/evidence/repro_runs/"
        "n3_c6_c10_post_2026_09_04.out").read_text()
last = post.strip().splitlines()[-1]
print("C14 committed POST final line:", last[:100])
assert "is FALSE and a self-chosen checksum" in last

# ---- C15: the name says bytes, the code hashes fields ----
v3 = json.loads(V3.read_text())
print("C15: field is named science_byte_equal:",
      "science_byte_equal" in v3["v2_correction_map"])
assert "science_byte_equal" in v3["v2_correction_map"]

print("\nPRE CONFIRMED: candidate-controlled registry mints "
      "reviewer authority (even for a forged positive); arbitrary "
      "maps pass unnamed; POST speaks PRE; byte-equality is "
      "misnamed")
