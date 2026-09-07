#!/usr/bin/env python3
"""C39 (T2): the ONE deterministic sealing path — accepted draft
v6 -> external review record -> sealed v6. Runs ONLY after a valid
EXTERNAL review record exists under the private reviewer-authority
root; changes ONLY the declared seal/chronology/review-binding and
self-digest fields; every scientific field stays byte-equivalent
as a structured value. Draft schemas never score; this tool never
runs the campaign. The real record and the real sealed design are
NOT created by the candidate in this order — tests exercise the
mechanism against fixtures and temporary outputs."""
import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_confirmatory as conf  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"
DRAFT_V6_PATH = STATE / "t2_screen_design_DRAFT_V6_20260907.json"
SEALED_V6_PATH = STATE / "t2_screen_design_SEALED_V6.json"

# The only fields the seal may set or change.
SEAL_ONLY_FIELDS = ("schema", "design_review_record_sha256",
                    "sealed_at_date", "supersedes_draft_sha256",
                    "design_sha256")


def seal_design(draft_path: Path = DRAFT_V6_PATH,
                out_path: Path = SEALED_V6_PATH,
                manifest_path: Path = None,
                census_path: Path = None) -> dict:
    manifest_path = manifest_path or (
        STATE / "t2_public_data_manifest_20260906.json")
    census_path = census_path or (
        STATE / "t2_bank_census_20260906.json")
    draft_raw = Path(draft_path).read_bytes()
    draft_file_sha = hashlib.sha256(draft_raw).hexdigest()
    if draft_file_sha != conf.T2_V6_DRAFT_FILE_SHA:
        raise SystemExit(
            "REFUSED: the draft bytes are not the ACCEPTED v6 "
            "draft — only the reviewed draft can be sealed")
    draft = conf.strict_json_load(draft_path, "draft v6")
    if draft.get("design_sha256") != conf.T2_V6_DRAFT_SELF_SHA:
        raise SystemExit(
            "REFUSED: draft self identity differs from the "
            "accepted v6 identity")
    manifest_sha = conf._sha_file(manifest_path)
    census_sha = conf._sha_file(census_path)
    # the EXTERNAL record must exist and validate BEFORE any seal;
    # probe with the exact bindings the sealed object will carry.
    probe = dict(draft)
    fd = conf._open_private_authority_file(
        conf.T2_REVIEW_RECORD_PATH)
    try:
        raw = b""
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            raw += b
    finally:
        os.close(fd)
    record_sha = hashlib.sha256(raw).hexdigest()
    probe["design_review_record_sha256"] = record_sha
    probe["supersedes_draft_sha256"] = draft_file_sha
    rec = conf.verify_design_review_record(
        probe, manifest_sha, census_sha)
    sealed = dict(draft)
    sealed["schema"] = "agent_multi.t2_screen_design.v6"
    sealed["design_review_record_sha256"] = record_sha
    sealed["supersedes_draft_sha256"] = draft_file_sha
    sealed["sealed_at_date"] = rec["reviewed_at_date"]
    body = {k: sealed[k] for k in sorted(sealed)
            if k != "design_sha256"}
    sealed["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    # every non-seal field must be byte-equivalent as a value
    for k in draft:
        if k not in SEAL_ONLY_FIELDS and sealed[k] != draft[k]:
            raise SystemExit(
                f"REFUSED: sealing altered scientific field {k!r}")
    out_path = Path(out_path)
    fdo = os.open(str(out_path),
                  os.O_CREAT | os.O_EXCL | os.O_WRONLY
                  | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        os.write(fdo, json.dumps(sealed, indent=1).encode())
        os.fsync(fdo)
    finally:
        os.close(fdo)
    return sealed


if __name__ == "__main__":
    s = seal_design()
    print(json.dumps({"sealed_design_sha256": s["design_sha256"],
                      "supersedes_draft_sha256":
                          s["supersedes_draft_sha256"]},
                     indent=1))
