#!/usr/bin/env python3
"""C76: generate the resource-only successor of the immutable
sealed v6 design — changing ONLY schema/self-digest, supersession
bindings, the amendment metadata and
resource_contract.max_wall_seconds (216,000 s hard campaign
ceiling; not an ETA or target). Every scientific field stays
byte-equivalent; the executable field-by-field diff
(conf.verify_resource_successor) refuses anything else. This tool
NEVER creates or installs the external Musashi review or
execution record."""
import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))

import t2_confirmatory as conf  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
SEALED = STATE / "t2_screen_design_SEALED_V6.json"
OUT = STATE / "t2_screen_design_RESOURCE_SUCCESSOR_V1.json"


def generate(out_path: Path = OUT) -> dict:
    if Path(out_path).exists():
        raise SystemExit(
            "REFUSED: a resource successor already exists — "
            "immutable, never regenerated in place")
    sealed = conf.strict_json_load(SEALED, "sealed v6 design")
    if conf._self_sha_design(sealed) != sealed["design_sha256"]:
        raise SystemExit("REFUSED: sealed v6 self identity does "
                         "not re-derive")
    succ = copy.deepcopy(sealed)
    succ["schema"] = conf.T2_SUCCESSOR_SCHEMA
    succ["supersedes_sealed_file_sha256"] = conf._sha_file(SEALED)
    succ["supersedes_sealed_self_sha256"] = \
        sealed["design_sha256"]
    succ["resource_amendment"] = {
        "classification": conf.T2_SUCCESSOR_CLASSIFICATION,
        "changed_field": "resource_contract.max_wall_seconds",
        "from_seconds": sealed["resource_contract"]
                              ["max_wall_seconds"],
        "to_seconds": conf.T2_SUCCESSOR_MAX_WALL,
        "chronology_pre_execution": True,
        "chronology_note": (
            "pre-execution: no confirmatory outcome exists; the "
            "60-hour value is a HARD campaign ceiling, never an "
            "ETA or target; runtime still obeys the hard wall "
            "and may stop earlier; the projection basis is "
            "non-authoritative planning evidence and never "
            "grants time"),
        "basis": ("docs/audits/evidence/"
                  "T2_BUDGET_PROJECTION_2026_09_08.json"),
        "amended_at_date": "2026-09-08",
        "authorized_by": ("MUSASHI_AUDIT_T2_C66_C73_M3_C1_C6_"
                          "AND_M4_0_2026_09_08 (resource-only "
                          "successor authorized; execution "
                          "authority NOT granted)")}
    succ["resource_contract"]["max_wall_seconds"] = \
        conf.T2_SUCCESSOR_MAX_WALL
    body = {k: succ[k] for k in sorted(succ)
            if k != "design_sha256"}
    import hashlib
    succ["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    conf.verify_resource_successor(succ, sealed_path=SEALED)
    payload = json.dumps(succ, indent=1).encode()
    fd = os.open(str(out_path), os.O_CREAT | os.O_EXCL
                 | os.O_WRONLY, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return succ


if __name__ == "__main__":
    s = generate()
    print(json.dumps({
        "successor_design_sha256": s["design_sha256"],
        "supersedes_file": s["supersedes_sealed_file_sha256"],
        "max_wall_seconds":
            s["resource_contract"]["max_wall_seconds"]},
        indent=1))
