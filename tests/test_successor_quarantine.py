"""WP0 quarantine: atomic, idempotent, append-only supersession of the
invalid mechanism_pass successor (AUD-F1-20260808-159)."""
import json
from pathlib import Path

from tools.quarantine_inner_curriculum_successor import (
    SUPERSESSION_SCHEMA,
    quarantine,
)


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "m0root"
    (root / "queue").mkdir(parents=True)
    (root / "queue" / "m0_successor_mechanism_pass.json").write_text(
        json.dumps({"schema": "agent_multi.m0_successor_job.v1",
                    "branch": "mechanism_pass",
                    "launch_eligible": True}))
    (root / "m0_aggregation.json").write_text(json.dumps({"a": 1}))
    (root / "m0_final_table.csv").write_text("seed,arm\n")
    (root / "m0_fleet_manifest.json").write_text(json.dumps({"m": 1}))
    return root


def test_quarantine_supersedes_and_preserves_original(tmp_path):
    root = _root(tmp_path)
    result = quarantine(root)
    assert result["outcome"] == "QUARANTINED"
    superseding = json.loads(
        (root / "queue" / "m0_successor_mechanism_pass.json").read_text())
    assert superseding["schema"] == SUPERSESSION_SCHEMA
    assert superseding["launch_eligible"] is False
    assert superseding["reason_finding"] == "AUD-F1-20260808-159"
    retired = Path(result["retired_path"])
    original = json.loads(retired.read_text())
    assert original["launch_eligible"] is True     # byte-preserved
    envelope = json.loads((root / "m0_correction_envelope_v1.json").read_text())
    assert envelope["historical_evidence_immutable"] is True
    assert envelope["bindings"]["retired_successor_sha256"] == \
        result["original_sha256"]


def test_second_invocation_changes_no_bytes(tmp_path):
    root = _root(tmp_path)
    first = quarantine(root)
    target = root / "queue" / "m0_successor_mechanism_pass.json"
    before = target.read_bytes()
    second = quarantine(root)
    assert second["outcome"] == "ALREADY_QUARANTINED"
    assert second["bytes_changed"] == 0
    assert target.read_bytes() == before
    assert second["retired_original_sha256"] == first["original_sha256"]


def test_missing_successor_refuses(tmp_path):
    root = tmp_path / "empty"
    (root / "queue").mkdir(parents=True)
    assert quarantine(root)["outcome"] == "REFUSED"


def test_aggregation_and_records_untouched(tmp_path):
    root = _root(tmp_path)
    agg_before = (root / "m0_aggregation.json").read_bytes()
    quarantine(root)
    assert (root / "m0_aggregation.json").read_bytes() == agg_before
