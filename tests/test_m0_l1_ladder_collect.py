"""Socket-free tests for the ladder collection/replica/table CLI
(order §3.5-§3.6): seal-no-overwrite, identity-fragmentation refusal,
replica-digest gate, publish-outside-seal and tamper re-proof."""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import collect_l1_factorial as fact  # noqa: E402
from tools import m0_l1_ladder_collect as lad  # noqa: E402

DID = "feedfacefeedface"
ARMS = ("D0_M0_EXACT", "D2_BOUNDARY_ONLY", "D3_COST_PROTECTION",
        "D4_FULL_L1")


def _record(arm: str, *, did: str = DID, active: bool = False,
            handoff: str = "l1_trained_epoch_v4") -> dict:
    val = {"action_raw_mean": 0.2 if active else 0.0,
           "action_raw_std": 0.5 if active else 0.0,
           "action_non_hold_rate": 1.0 if active else 0.0,
           "trades_total": 122 if active else 0,
           "execution_diagnostics": {
               "protected_market_entries": 3 if active else 0}}
    return {
        "arm": arm, "diagnostic_identity": did,
        "arm_identity": f"armid_{arm}", "contract_sha256": "c" * 64,
        "stop_reason": "max_epochs_budget",
        "gradient_updates_total": 19000,
        "m0_activity_facts": {"activity_survived_normal": active},
        "curriculum": {"post_easy": {
            "phase1_handoff_semantics": handoff,
            "best_easy_epoch": 0 if active else 1}},
        "terminal_evaluation_as_run": {"splits_raw": {
            "validation": val}},
    }


def _source_tree(root: Path, *, fragment: str | None = None) -> Path:
    out = root / "out"
    for arm in ARMS:
        arm_dir = out / DID / arm
        arm_dir.mkdir(parents=True)
        did = fragment if (fragment and arm == "D3_COST_PROTECTION") \
            else DID
        record = _record(arm, did=did, active=arm == "D0_M0_EXACT",
                         handoff="m0_epoch0_eligible_v3"
                         if arm == "D0_M0_EXACT"
                         else "l1_trained_epoch_v4")
        (arm_dir / lad.RECORD_NAME).write_text(json.dumps(record))
    (out / DID / "D0_M0_EXACT" / lad.D1_RECORD_NAME).write_text(
        json.dumps({"arm": "D1_EVALUATOR_ONLY",
                    "label_under_m0_definition": "active",
                    "label_under_l1_definition": "active",
                    "labels_agree": True}))
    return out


def _contract(out: Path) -> dict:
    return {"output_root": str(out), "seed": 101,
            "assignments": {arm: {"hostname": "testhost"}
                            for arm in ARMS}}


def _local_fetch(host: str, remote_dir: Path, stage_dir: Path) -> None:
    shutil.copytree(remote_dir, stage_dir)


def _fake_replicate(host: str, sealed: Path, replica: Path) -> None:
    pass


def _verify_equal(host: str, replica: Path, expectations) -> dict:
    # The fake replica "recomputes" the digest over the sealed tree
    # itself — stands in for an identical replica.
    sealed = replica
    return {"tree_digest": fact.tree_digest(sealed),
            "verifier_version": "test", "terminals": []}


def _collect(tmp_path: Path, out: Path, **kw):
    return lad.collect(
        contract=_contract(out), diagnostic_identity=DID,
        collection_root=tmp_path / "col", fetch_fn=_local_fetch,
        replica_host=kw.pop("replica_host", "replicahost"),
        replicate_fn=_fake_replicate,
        replica_verify_fn=kw.pop("verify", _verify_equal), **kw)


def test_seal_replica_and_published_table(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    assert manifest["replica"]["proof"]["tree_digest"] == \
        manifest["collection_tree_digest"]
    assert set(manifest["arms"]) == set(ARMS)
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "TABLE_PUBLISHED"
    assert len(table["rows"]) == 5
    assert table["rows"][1]["arm"] == "D1_EVALUATOR_ONLY"
    assert table["verdict"].startswith(
        "MECHANISM_NAMED: first active-to-inactive transition at "
        "D2_BOUNDARY_ONLY")
    assert table["post_write_digest_reproof"] == \
        manifest["collection_tree_digest"]
    # Second collect on the same root: never overwrite.
    again = _collect(tmp_path, out)
    assert again["outcome"] == "COLLECTION_REFUSED"
    assert "never overwrite" in again["refusals"][0]


def test_identity_fragmentation_refused(tmp_path):
    out = _source_tree(tmp_path, fragment="0123456789abcdef")
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("identity fragmentation" in r
               for r in manifest["refusals"])


def test_replica_digest_mismatch_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(
        tmp_path, out,
        verify=lambda h, r, e: {"tree_digest": "0" * 64})
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("replica is not proof" in r
               for r in manifest["refusals"])
    # A refused collection never publishes.
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"


def test_replicate_crash_is_typed_refusal(tmp_path):
    out = _source_tree(tmp_path)

    def _boom(host, sealed, replica):
        raise RuntimeError("rsync exit 11")

    manifest = lad.collect(
        contract=_contract(out), diagnostic_identity=DID,
        collection_root=tmp_path / "col", fetch_fn=_local_fetch,
        replica_host="replicahost", replicate_fn=_boom,
        replica_verify_fn=_verify_equal)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("replication to replicahost failed" in r
               for r in manifest["refusals"])
    # The sealed tree + manifest still landed (never a crash).
    assert (tmp_path / "col" / "sealed" / DID).is_dir()
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"


def test_publish_without_replica_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out, replica_host=None)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"
    assert any("replicate before interpretation" in r
               for r in table["refusals"])


def test_tampered_sealed_tree_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    sealed = tmp_path / "col" / "sealed" / DID
    (sealed / "D4_FULL_L1" / lad.RECORD_NAME).write_text(
        json.dumps(_record("D4_FULL_L1", active=True)))
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"
    assert any("digest changed" in r for r in table["refusals"])


def test_publish_inside_seal_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    inside = tmp_path / "col" / "sealed" / DID / "table.json"
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=inside)
    assert table["outcome"] == "PUBLISH_REFUSED"
    assert any("inside the sealed root" in r
               for r in table["refusals"])


def test_missing_d1_record_refused(tmp_path):
    out = _source_tree(tmp_path)
    (out / DID / "D0_M0_EXACT" / lad.D1_RECORD_NAME).unlink()
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("D1 evaluator record missing" in r
               for r in manifest["refusals"])
