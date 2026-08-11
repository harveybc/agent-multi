"""Socket-free tests for the ladder collection/replica/table CLI
(order §3.5-§3.6): seal-no-overwrite, identity-fragmentation refusal,
replica-digest gate, publish-outside-seal, tamper re-proof, and the
finding-223 terminal-custody battery — missing/wrong-sha/ambiguous
terminals, per-arm replica-proof cardinality + (arm, seed, relative
path, sha256) binding, the auditor's no-model-fields reproducer, and
the READ-ONLY --verify-terminals-only mode."""
import hashlib
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
            handoff: str = "l1_trained_epoch_v4",
            terminal_path: str | None = None,
            terminal_sha: str | None = None,
            best_path: str | None = None,
            best_sha: str | None = None) -> dict:
    val = {"action_raw_mean": 0.2 if active else 0.0,
           "action_raw_std": 0.5 if active else 0.0,
           "action_non_hold_rate": 1.0 if active else 0.0,
           "trades_total": 122 if active else 0,
           "execution_diagnostics": {
               "protected_market_entries": 3 if active else 0}}
    record = {
        "arm": arm, "diagnostic_identity": did, "seed": 101,
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
    if terminal_path is not None:
        record["terminal_model_path"] = terminal_path
    if terminal_sha is not None:
        record["terminal_model_sha256"] = terminal_sha
    if best_path is not None:
        record["best_model_path"] = best_path
    if best_sha is not None:
        record["best_model_sha256"] = best_sha
    return record


def _source_tree(root: Path, *, fragment: str | None = None,
                 no_model_fields: bool = False) -> Path:
    """Terminal-only inactive D2-D4 + active D0 carrying BOTH best and
    terminal fields — the real post-ladder shape (audit §4/223)."""
    out = root / "out"
    for arm in ARMS:
        arm_dir = out / DID / arm
        attempt = arm_dir / "attempt-01"
        attempt.mkdir(parents=True)
        terminal = attempt / "model.terminal.zip"
        terminal.write_bytes(f"terminal-{arm}".encode())
        terminal_sha = hashlib.sha256(
            terminal.read_bytes()).hexdigest()
        kwargs: dict = {}
        if not no_model_fields:
            kwargs = {"terminal_path": str(terminal),
                      "terminal_sha": terminal_sha}
            if arm == "D0_M0_EXACT":
                best = attempt / "model.zip"
                best.write_bytes(b"best-D0")
                kwargs["best_path"] = str(best)
                kwargs["best_sha"] = hashlib.sha256(
                    best.read_bytes()).hexdigest()
        did = fragment if (fragment and arm == "D3_COST_PROTECTION") \
            else DID
        record = _record(arm, did=did, active=arm == "D0_M0_EXACT",
                         handoff="m0_epoch0_eligible_v3"
                         if arm == "D0_M0_EXACT"
                         else "l1_trained_epoch_v4", **kwargs)
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


def _echo_terminals(expectations) -> list:
    """Stands in for a faithful remote verifier on an identical
    replica: one bound successful load per expectation."""
    return [{"cell": e["cell"], "seed": e["seed"],
             "relative_path": e["relative_path"],
             "sha256": e["expected_sha256"], "loads": True,
             "n_updates": 19000} for e in expectations]


def _verify_with(terminals_fn):
    def _verify(host, replica, expectations):
        # The fake replica "recomputes" the digest over the sealed
        # tree itself — stands in for an identical replica.
        return {"tree_digest": fact.tree_digest(replica),
                "verifier_version": "test",
                "terminals": terminals_fn(expectations)}
    return _verify


_verify_equal = _verify_with(_echo_terminals)


def _collect(tmp_path: Path, out: Path, **kw):
    return lad.collect(
        contract=_contract(out), diagnostic_identity=DID,
        collection_root=tmp_path / "col", fetch_fn=_local_fetch,
        replica_host=kw.pop("replica_host", "replicahost"),
        replicate_fn=_fake_replicate,
        replica_verify_fn=kw.pop("verify", _verify_equal), **kw)


def _manifest_path(tmp_path: Path) -> Path:
    return (tmp_path / "col" /
            f"ladder_collection_manifest_{DID}.json")


# ---------------------------------------------------------------------------
# original battery (fixtures now carry terminal custody)
# ---------------------------------------------------------------------------

def test_seal_replica_and_published_table(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    assert manifest["replica"]["proof"]["tree_digest"] == \
        manifest["collection_tree_digest"]
    assert set(manifest["arms"]) == set(ARMS)
    # Finding 223: terminal custody for EVERY arm, expectations built
    # from terminal fields.
    assert set(manifest["terminals"]) == set(ARMS)
    assert len(manifest["replica"]["expectations"]) == len(ARMS)
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "TABLE_PUBLISHED"
    assert table["terminal_proof_binding_repeated"] is True
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


# ---------------------------------------------------------------------------
# finding-223 terminal custody: staging-side adversaries
# ---------------------------------------------------------------------------

def test_missing_d2_terminal_refused(tmp_path):
    out = _source_tree(tmp_path)
    (out / DID / "D2_BOUNDARY_ONLY" / "attempt-01" /
     "model.terminal.zip").unlink()
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("D2_BOUNDARY_ONLY" in r and
               "terminal artifact not staged" in r
               for r in manifest["refusals"])
    assert not (tmp_path / "col" / "sealed" / DID).exists()


def test_wrong_d3_terminal_sha_refused(tmp_path):
    out = _source_tree(tmp_path)
    rec_path = out / DID / "D3_COST_PROTECTION" / lad.RECORD_NAME
    record = json.loads(rec_path.read_text())
    record["terminal_model_sha256"] = "b" * 64
    rec_path.write_text(json.dumps(record))
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("D3_COST_PROTECTION" in r and
               "differs from the recorded terminal_model_sha256" in r
               for r in manifest["refusals"])


def test_duplicate_relative_path_candidates_refused(tmp_path):
    out = _source_tree(tmp_path)
    arm_dir = out / DID / "D2_BOUNDARY_ONLY"
    # The recorded path contains the arm component TWICE, and BOTH
    # derived relative candidates exist as staged files — a
    # basename-first-match collector would silently pick one.
    nested = arm_dir / "nest" / "D2_BOUNDARY_ONLY"
    nested.mkdir(parents=True)
    (nested / "model.terminal.zip").write_bytes(b"nested-terminal")
    (arm_dir / "model.terminal.zip").write_bytes(b"top-terminal")
    rec_path = arm_dir / lad.RECORD_NAME
    record = json.loads(rec_path.read_text())
    record["terminal_model_path"] = str(
        nested / "model.terminal.zip")
    rec_path.write_text(json.dumps(record))
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("ambiguous staged terminal" in r
               for r in manifest["refusals"])


def test_auditor_reproducer_no_model_fields_refused(tmp_path):
    """The independent counterexample (audit §4/223): four records
    with NO model fields sealed, replicated and published with zero
    model files. It must now REFUSE before sealing."""
    out = _source_tree(tmp_path, no_model_fields=True)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    custody = [r for r in manifest["refusals"]
               if "terminal custody incomplete" in r]
    assert len(custody) == len(ARMS)
    assert not (tmp_path / "col" / "sealed" / DID).exists()
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"


def test_inactive_terminal_only_arms_seal_and_publish(tmp_path):
    """Valid inactive D2-D4 records carrying ONLY terminal fields (no
    best model) must seal and publish with one bound load per arm."""
    out = _source_tree(tmp_path)
    for arm in ARMS[1:]:
        record = json.loads(
            (out / DID / arm / lad.RECORD_NAME).read_text())
        assert "best_model_path" not in record
        assert record["terminal_model_path"]
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    proof_cells = {t["cell"]
                   for t in manifest["replica"]["proof"]["terminals"]}
    assert proof_cells == set(ARMS)
    for exp in manifest["replica"]["expectations"]:
        arm = exp["cell"]
        assert exp["expected_sha256"] == \
            manifest["terminals"][arm]["sha256"]
        assert exp["relative_path"].startswith(arm + "/")
        assert exp["seed"] == 101
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "TABLE_PUBLISHED"


# ---------------------------------------------------------------------------
# finding-223 terminal custody: replica-proof adversaries
# ---------------------------------------------------------------------------

def test_replica_proof_only_d0_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(
        tmp_path, out,
        verify=_verify_with(lambda exps: [
            t for t in _echo_terminals(exps)
            if t["cell"] == "D0_M0_EXACT"]))
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    for arm in ("D2_BOUNDARY_ONLY", "D3_COST_PROTECTION",
                "D4_FULL_L1"):
        assert any(f"NO terminal load for arm {arm}" in r
                   for r in manifest["refusals"])


def test_replica_proof_swapped_d2_d3_refused(tmp_path):
    out = _source_tree(tmp_path)

    def _swap(exps):
        terms = _echo_terminals(exps)
        d2 = next(t for t in terms if t["cell"] == "D2_BOUNDARY_ONLY")
        d3 = next(t for t in terms
                  if t["cell"] == "D3_COST_PROTECTION")
        d2["relative_path"], d3["relative_path"] = \
            d3["relative_path"], d2["relative_path"]
        d2["sha256"], d3["sha256"] = d3["sha256"], d2["sha256"]
        return terms

    manifest = _collect(tmp_path, out, verify=_verify_with(_swap))
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    for arm in ("D2_BOUNDARY_ONLY", "D3_COST_PROTECTION"):
        assert any(arm in r and
                   "not the expected terminal relative path" in r
                   for r in manifest["refusals"])


def test_replica_proof_loads_false_d4_refused(tmp_path):
    out = _source_tree(tmp_path)

    def _d4_broken(exps):
        terms = _echo_terminals(exps)
        for t in terms:
            if t["cell"] == "D4_FULL_L1":
                t["loads"] = False
                t["error"] = "BadZipFile:File is not a zip file"
        return terms

    manifest = _collect(tmp_path, out,
                        verify=_verify_with(_d4_broken))
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("D4_FULL_L1" in r and "did not load" in r
               for r in manifest["refusals"])


def test_replica_proof_duplicate_entry_refused(tmp_path):
    out = _source_tree(tmp_path)

    def _dup(exps):
        terms = _echo_terminals(exps)
        terms.append(dict(next(t for t in terms
                               if t["cell"] == "D0_M0_EXACT")))
        return terms

    manifest = _collect(tmp_path, out, verify=_verify_with(_dup))
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("exactly one bound load is required" in r
               for r in manifest["refusals"])


def test_replica_proof_foreign_entry_refused(tmp_path):
    out = _source_tree(tmp_path)

    def _foreign(exps):
        terms = _echo_terminals(exps)
        terms.append({"cell": "D9_UNKNOWN", "seed": 101,
                      "relative_path": "D9_UNKNOWN/model.zip",
                      "sha256": "f" * 64, "loads": True})
        return terms

    manifest = _collect(tmp_path, out, verify=_verify_with(_foreign))
    assert manifest["outcome"] == "COLLECTION_REFUSED"
    assert any("foreign terminal entry" in r
               for r in manifest["refusals"])


def test_publish_repeats_terminal_binding_from_manifest(tmp_path):
    """Requirement 6: a sealed collection whose STORED manifest proof
    lost its per-arm terminal bindings (legacy best-model-only proof)
    must refuse publication — presence of a generic replica proof is
    not enough."""
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    stored = json.loads(_manifest_path(tmp_path).read_text())
    stored["replica"]["proof"]["terminals"] = [
        {"cell": "D0_M0_EXACT", "seed": 101, "loads": True,
         "sha256": stored["terminals"]["D0_M0_EXACT"]["sha256"]}]
    _manifest_path(tmp_path).write_text(json.dumps(stored))
    table = lad.publish_contrast_table(
        collection_root=tmp_path / "col", diagnostic_identity=DID,
        out_json=tmp_path / "pub" / "table.json")
    assert table["outcome"] == "PUBLISH_REFUSED"
    assert any("NO terminal load for arm D2_BOUNDARY_ONLY" in r
               for r in table["refusals"])
    # The surviving D0 entry lost its path binding too.
    assert any("D0_M0_EXACT" in r and
               "not the expected terminal relative path" in r
               for r in table["refusals"])


# ---------------------------------------------------------------------------
# finding-223: READ-ONLY --verify-terminals-only mode
# ---------------------------------------------------------------------------

def _tree_snapshot(root: Path) -> dict:
    return {str(p.relative_to(root)):
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_verify_terminals_only_verified_and_read_only(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    before = _tree_snapshot(tmp_path / "col")
    report = lad.verify_terminals_only(
        collection_root=tmp_path / "col", diagnostic_identity=DID)
    assert report["outcome"] == "TERMINAL_CUSTODY_VERIFIED"
    assert report["mode"] == "READ_ONLY"
    assert report["manifest_predates_terminal_custody_schema"] is False
    assert report["sealed_tree_digest_fresh"] == \
        report["sealed_tree_digest_recorded"]
    for arm in ARMS:
        assert report["arms"][arm]["fresh_hash_matches_record"] is True
        assert report["arms"][arm]["manifest_files_entry_matches"] \
            is True
    assert report["proof_binding_refusals"] == []
    # Read-only: not one byte changed anywhere under the root.
    assert _tree_snapshot(tmp_path / "col") == before


def test_verify_terminals_only_legacy_manifest_typed_facts(tmp_path):
    """Against a pre-finding-223 manifest (no terminals map, D0-only
    unbound proof — the REAL sealed ladder's shape) the artifacts
    still verify; the impossible binding is a typed legacy fact, not
    an artifact failure."""
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    stored = json.loads(_manifest_path(tmp_path).read_text())
    d0_sha = stored["terminals"]["D0_M0_EXACT"]["sha256"]
    del stored["terminals"]
    stored["schema"] = "agent_multi.m0_l1_ladder_collection.v1"
    stored["replica"]["proof"]["terminals"] = [
        {"cell": "D0_M0_EXACT", "seed": 101, "loads": True,
         "sha256": d0_sha, "n_updates": 19000}]
    _manifest_path(tmp_path).write_text(json.dumps(stored))
    report = lad.verify_terminals_only(
        collection_root=tmp_path / "col", diagnostic_identity=DID)
    assert report["outcome"] == "TERMINAL_CUSTODY_VERIFIED"
    assert report["manifest_predates_terminal_custody_schema"] is True
    assert any("MANIFEST_PREDATES_TERMINAL_CUSTODY_SCHEMA" in f
               for f in report["typed_facts"])
    assert any("LEGACY_REPLICA_PROOF_UNBOUND" in f
               for f in report["typed_facts"])
    assert report["legacy_manifest_binding_gaps"]
    for arm in ARMS:
        assert report["arms"][arm]["fresh_hash_matches_record"] is True


def test_verify_terminals_only_tampered_terminal_refused(tmp_path):
    out = _source_tree(tmp_path)
    manifest = _collect(tmp_path, out)
    assert manifest["outcome"] == "COLLECTION_SEALED"
    sealed = tmp_path / "col" / "sealed" / DID
    (sealed / "D3_COST_PROTECTION" / "attempt-01" /
     "model.terminal.zip").write_bytes(b"swapped-in imposter")
    report = lad.verify_terminals_only(
        collection_root=tmp_path / "col", diagnostic_identity=DID)
    assert report["outcome"] == "TERMINAL_CUSTODY_REFUSED"
    assert any("D3_COST_PROTECTION" in f and
               "differs from the recorded terminal_model_sha256" in f
               for f in report["failures"])
    assert any("sealed tree digest changed" in f
               for f in report["failures"])
