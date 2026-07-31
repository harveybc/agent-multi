#!/usr/bin/env python3
"""Validate publication-package structure without external dependencies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAPER_IDS = (
    "p1-doin-protocol",
    "p2-data-first-genome",
    "p3-hierarchical-portfolio",
    "p4-execution-parity",
    "p5-audit-recovery",
)
REQUIRED_FILES = (
    "README.md",
    "paper.tex",
    "references.bib",
    "claims.csv",
    "search_protocol.md",
    "artifact_manifest.json",
)
REQUIRED_DIRS = ("figures", "tables", "supplement")
CLAIMS_HEADERS = (
    "claim_id",
    "manuscript_location",
    "claim_type",
    "claim_text",
    "support_kind",
    "support_ref",
    "artifact_hash",
    "citation_key",
    "technical_verifier",
    "academic_verifier",
    "verification_date",
    "state",
    "notes",
)
MANIFEST_KEYS = {
    "schema",
    "paper_id",
    "state",
    "repositories",
    "datasets",
    "configurations",
    "artifacts",
    "queries",
    "checksums",
}


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    for paper_id in PAPER_IDS:
        paper_dir = root / paper_id
        if not paper_dir.is_dir():
            errors.append(f"{paper_id}: directory missing")
            continue
        for name in REQUIRED_FILES:
            if not (paper_dir / name).is_file():
                errors.append(f"{paper_id}: missing {name}")
        for name in REQUIRED_DIRS:
            if not (paper_dir / name).is_dir():
                errors.append(f"{paper_id}: missing directory {name}")

        claims_path = paper_dir / "claims.csv"
        if claims_path.is_file():
            with claims_path.open(newline="", encoding="utf-8") as handle:
                headers = tuple(next(csv.reader(handle), ()))
            if headers != CLAIMS_HEADERS:
                errors.append(f"{paper_id}: invalid claims.csv headers")

        manifest_path = paper_dir / "artifact_manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"{paper_id}: invalid manifest JSON: {exc}")
            else:
                missing = sorted(MANIFEST_KEYS - set(manifest))
                if missing:
                    errors.append(f"{paper_id}: manifest keys missing: {missing}")
                if manifest.get("paper_id") != paper_id:
                    errors.append(f"{paper_id}: manifest paper_id mismatch")
                if manifest.get("schema") != "agent_multi.publication_artifact_manifest.v1":
                    errors.append(f"{paper_id}: unsupported manifest schema")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "papers",
    )
    args = parser.parse_args()
    errors = validate(args.root)
    if errors:
        for error in errors:
            print(error)
        return 1
    print(f"validated {len(PAPER_IDS)} publication packages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
