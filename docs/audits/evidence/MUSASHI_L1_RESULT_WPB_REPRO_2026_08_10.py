#!/usr/bin/env python3
"""Independent L1 result and repository-presentation reproduction.

The L1 check reads the immutable collection envelope, recomputes the
aggregation from the sealed tree and proves the seal is unchanged.  The WP-B
check validates the committed inventory/snapshots and resolves every relative
README link against the exact delivered Git object.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tools import aggregate_l1_factorial as l1


EXPERIMENT_ID = "2de49ea9225e2baf"
COLLECTION_ROOT = (
    Path.home() / ".local/share/agent-multi/l1_decision_collection_2de49ea9"
)
REPOSITORY_ROOT = Path("/home/harveybc/Documents/GitHub")
README_REVISIONS = {
    "doin-node": "ec5cb130",
    "doin-core": "9c39df4c",
    "doin-plugins": "5c60349d",
    "doin-optimizer": "38720707",
    "doin-evaluator": "c9eb3558",
    "trading-contracts": "3d531f69",
    "lts": "22a1628b",
    "prediction_provider": "7ee76b94",
    "heuristic-strategy": "d8060f69",
    "gym-fx": "b71429a",
    "predictor": "1082c7b",
    "agent-multi": "0d7c937b",
    "feature-eng": "81f6ea12",
    "feature-extractor": "df86252a",
    "preprocessor": "ac14fe70",
    "synthetic-datagen": "176336e6",
    "financial-data": "e85edbee",
    "rl-optimizer": "e306546a",
    "trading-signal": "f3f77141",
    "timeseries-gan": "262a53bd",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    )


def readme_links() -> dict:
    checked = 0
    broken: list[dict[str, str]] = []
    for name, revision in README_REVISIONS.items():
        repo = REPOSITORY_ROOT / name
        text = git(repo, "show", f"{revision}:README.md")
        for raw in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
            target = raw.strip().split()[0].strip("<>")
            if not target or target.startswith(
                ("#", "http://", "https://", "mailto:")
            ):
                continue
            clean = target.split("#", 1)[0].split("?", 1)[0]
            if not clean:
                continue
            checked += 1
            result = subprocess.run(
                ["git", "-C", str(repo), "cat-file", "-e",
                 f"{revision}:{clean}"],
                check=False,
                capture_output=True,
            )
            if result.returncode:
                broken.append({"repository": name, "target": target})
    return {
        "readmes_checked": len(README_REVISIONS),
        "relative_links_checked": checked,
        "broken_relative_links": broken,
    }


def main() -> None:
    envelope = l1.load_collection_envelope(COLLECTION_ROOT, EXPERIMENT_ID)
    recomputed = l1.aggregate(
        COLLECTION_ROOT / "sealed", EXPERIMENT_ID
    )
    published = json.loads(
        (COLLECTION_ROOT / "aggregations" / EXPERIMENT_ID /
         "l1_factorial_aggregation.json").read_text()
    )
    fields = (
        "schema", "experiment_id", "contract_sha256",
        "system_manifest_sha256", "outcome", "outcome_rationale",
        "outcome_domain", "spec_deviation_declared", "refusals", "cells",
        "raw_metrics_per_seed", "subject_execution_revisions",
    )

    audit_root = Path(__file__).resolve().parents[2]
    inventory = json.loads(
        (audit_root / "audits/evidence/"
         "REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json").read_text()
    )
    snapshots = [
        json.loads(line)
        for line in (
            audit_root / "audits/evidence/"
            "REPOSITORY_METADATA_SNAPSHOTS_2026_08_10.jsonl"
        ).read_text().splitlines()
        if line.strip()
    ]
    snapshot_by_repo = {row["repo"]: row for row in snapshots}
    semantic_topic_mismatches = []
    for repository, replacement in {
        "trading-signal": "feature-eng",
        "timeseries-gan": "synthetic-datagen",
    }.items():
        topics = set(snapshot_by_repo[repository]["after"]["topics"])
        if "superseded-by-doin-node" in topics:
            semantic_topic_mismatches.append({
                "repository": repository,
                "incorrect_topic": "superseded-by-doin-node",
                "documented_replacement": replacement,
            })

    result = {
        "l1": {
            "experiment_id": EXPERIMENT_ID,
            "published_outcome": published.get("outcome"),
            "recomputed_outcome": recomputed.get("outcome"),
            "substantive_fields_equal": all(
                recomputed.get(field) == published.get(field)
                for field in fields
            ),
            "cells_total": len(recomputed.get("cells", {})),
            "valid_cells": sum(
                bool(cell.get("valid"))
                for cell in recomputed.get("cells", {}).values()
            ),
            "active_cells": sum(
                bool(cell.get("active"))
                for cell in recomputed.get("cells", {}).values()
            ),
            "refusals_total": len(recomputed.get("refusals", [])),
            "sealed_digest_before": envelope["sealed_input_digest"],
            "sealed_digest_after": l1.tree_digest(
                Path(envelope["sealed_root"])
            ),
            "replica_digest": envelope["replica_tree_digest"],
        },
        "wp_b": {
            "inventory_repositories": len(inventory["repositories"]),
            "metadata_snapshots": len(snapshots),
            "snapshot_invariant_failures": [
                row["repo"] for row in snapshots
                if row.get("invariants_preserved") is not True
            ],
            "snapshot_topic_count_failures": [
                row["repo"] for row in snapshots
                if row.get("topics_exact_20") is not True
            ],
            "semantic_topic_mismatches": semantic_topic_mismatches,
            **readme_links(),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
