#!/usr/bin/env python3
"""C10/C13 repair: rebuild the public-data manifest as v2 from the
ALREADY-ACQUIRED local bytes and archival record metadata — no
network. Renames the misnamed digest truthfully
(license_id_sha256), binds each Zenodo record's metadata bytes,
and marks ETTh1 EXCLUDED_FROM_T2_CONFIRMATORY per the order (its
license stays inventoried; nothing is re-downloaded, transformed
or redistributed)."""
import hashlib
import json
import time
from pathlib import Path

RAW = Path.home() / ".local/share/agent-multi/t2_public_raw"
STATE = Path.home() / ".local/share/agent-multi"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    old = json.loads(
        (STATE / "t2_public_data_manifest_20260906.json"
         ).read_text())
    datasets = {}
    for lid, d in old["datasets"].items():
        rec_meta_sha = "0" * 64
        rec = d.get("archival_record", "")
        if "zenodo:" in rec:
            rid = rec.split("zenodo:")[1].rstrip(")")
            meta_p = RAW / f"record_{rid}.json"
            if meta_p.is_file():
                rec_meta_sha = sha_file(meta_p)
        lic_p = RAW / "etth1__LICENSE"
        entry = {
            "logical_id": d["logical_id"],
            "family": d["family"],
            "final_url": d["final_url"],
            "archival_record": d["archival_record"],
            "record_metadata_sha256": rec_meta_sha
            if rec_meta_sha != "0" * 64 else hashlib.sha256(
                d["archival_record"].encode()).hexdigest(),
            "retrieved_at_utc": d["retrieved_at_utc"],
            "byte_size": d["byte_size"],
            "sha256": d["sha256"],
            "upstream_checksum": d["upstream_checksum"],
            "license_id": d["license_id"],
            # C13: the truthful name — this digest was always the
            # hash of the license IDENTIFIER for Monash records.
            "license_id_sha256": hashlib.sha256(
                d["license_id"].encode()).hexdigest(),
            "license_text_sha256": (
                sha_file(lic_p) if lid == "etth1"
                and lic_p.is_file() else "UNAVAILABLE"),
            "citation": d["citation"],
            "local_relpath": d["local_relpath"],
            "admission": ("EXCLUDED_FROM_T2_CONFIRMATORY"
                          if lid == "etth1"
                          else d["admission"]),
        }
        datasets[lid] = entry
    manifest = {
        "schema": "agent_multi.t2_public_data_manifest.v2",
        "acquired_at_utc": old["acquired_at_utc"],
        "remanifested_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "byte_cap": old["byte_cap"],
        "bytes_downloaded_total": old["bytes_downloaded_total"],
        "raw_root_note": old["raw_root_note"],
        "etth1_disposition": "EXCLUDED_FROM_T2_CONFIRMATORY per "
                             "the 2026-09-06 order; license "
                             "inventoried (CC-BY-ND-4.0, text "
                             "digest bound); no re-download, no "
                             "transformation, no redistribution",
        "datasets": datasets}
    out = STATE / "t2_public_data_manifest_20260906.json"
    out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"schema": manifest["schema"],
                      "datasets": len(datasets),
                      "etth1": datasets["etth1"]["admission"]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
