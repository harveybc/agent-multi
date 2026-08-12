"""Socket-free tests for the P1LR sealed collection + typed replica
proof (finding 225).

The collector is the ONLY producer of the proof the screen verdict
demands. These tests prove, with injected transports (no rsync/ssh):

* a complete 16-record source tree stages, seals, replicates and emits
  a typed 16-entry proof file that the runner-side
  ``validate_replica_proof`` accepts and that carries the full
  (experiment_identity, contract_sha256, seed, cell,
  terminal_relative_path, terminal_model_sha256, loads=true) binding;
* the sealed records + emitted proof pass an end-to-end
  ``screen_verdict`` with a boolean-true replica gate;
* 15/16 records, identity fragmentation, duplicate cell identities,
  hash-altered staged terminals, loads=false replica entries, replica
  digest mismatch, a missing replica host and re-collection over an
  existing seal ALL refuse typed;
* staging/sealing never mutate the source tree.
"""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tools import p1lr_collect as p1c  # noqa: E402

EXP_ID = "feedfacefeedface"


def _cell_record(contract, seed: int, cell: str, terminal: Path) -> dict:
    spec = contract["nested_split_contract"]
    return {
        "schema": p1.RECORD_SCHEMA,
        "mode": "screen",
        "evidence_class": "mechanics_screen",
        "experiment_identity": EXP_ID,
        "cell_identity": hashlib.sha256(
            f"{seed}:{cell}".encode()).hexdigest()[:16],
        "seed": seed,
        "cell": cell,
        "factors": dict(contract["cells"][cell]),
        "contract_sha256": contract["_contract_sha256"],
        "terminal_model_path": str(terminal),
        "terminal_model_sha256": hashlib.sha256(
            terminal.read_bytes()).hexdigest(),
        "selection_metric": "paired_generalization_weekly_v1",
        "nested_split_contract_path": spec["path"],
        "nested_split_contract_sha256": spec["sha256"],
        "nested_split_mode": "l1",
        "nested_split_manifest_sha256": "d" * 64,
        "nested_role_facts": {
            role: {key: spec["role_facts"][role].get(key)
                   for key in p1.NESTED_ROLE_FACT_KEYS}
            for role in spec["role_facts"]},
        "handoff_viability": {
            "selected": {"handoff_viability": "VIABLE",
                         "trained_treatment": True},
            "selected_label": "VIABLE",
            "selected_is_collapse": False,
            "per_checkpoint": [
                {"epoch": 1, "handoff_viability": "VIABLE",
                 "trained_treatment": True}],
        },
    }


@pytest.fixture()
def world(tmp_path):
    """Contract copy with tmp output root + a complete 16-cell source
    tree under it."""
    contract = copy.deepcopy(p1.load_contract())
    contract["output_root"] = str(tmp_path / "out")
    source_root = tmp_path / "out" / EXP_ID
    for seed in p1.SEEDS:
        for cell in p1.CELLS:
            attempt = (source_root / f"seed{seed}" / cell /
                       "attempt-01")
            attempt.mkdir(parents=True)
            terminal = attempt / "model.terminal.zip"
            terminal.write_bytes(f"terminal-{seed}-{cell}".encode())
            record = _cell_record(contract, seed, cell, terminal)
            (attempt.parent / p1c.RECORD_NAME).write_text(
                json.dumps(record))
    return contract, source_root


def _local_fetch(host: str, remote_dir: Path, stage_dir: Path) -> None:
    shutil.copytree(remote_dir, stage_dir)


def _noop_replicate(host: str, sealed: Path, replica: Path) -> None:
    pass


def _echo_verify(host: str, replica_root: Path, expectations) -> dict:
    """Faithful replica verifier double: same-digest tree, one bound
    successful load per expectation."""
    return {
        "tree_digest": p1c.tree_digest(replica_root),
        "verifier_version": "test_echo.v1",
        "terminals": [
            {"cell": e["cell"], "seed": e["seed"],
             "relative_path": e["relative_path"],
             "sha256": e["expected_sha256"], "loads": True,
             "n_updates": 40000}
            for e in expectations],
    }


def _collect(contract, tmp_path, *, verify=_echo_verify,
             replica_host="dragon", **kwargs):
    return p1c.collect(
        contract=contract, experiment_identity=EXP_ID,
        collection_root=tmp_path / "collection",
        fetch_fn=_local_fetch, replica_host=replica_host,
        replicate_fn=_noop_replicate, replica_verify_fn=verify,
        **kwargs)


class TestHappyPath:
    def test_seal_proof_and_end_to_end_screen_verdict(
            self, world, tmp_path):
        contract, source_root = world
        manifest = _collect(contract, tmp_path)
        assert manifest["outcome"] == "COLLECTION_SEALED", \
            manifest["refusals"]
        assert manifest["refusals"] == []
        assert len(manifest["terminals"]) == 16
        assert len(manifest["cell_identities"]) == 16

        proof_path = Path(manifest["replica_proof_file"])
        assert proof_path.is_file()
        proof = json.loads(proof_path.read_text())
        assert proof["schema"] == p1.REPLICA_PROOF_SCHEMA
        assert proof["experiment_identity"] == EXP_ID
        assert proof["contract_sha256"] == \
            contract["_contract_sha256"]
        assert len(proof["proofs"]) == 16
        for entry in proof["proofs"]:
            for key in ("experiment_identity", "contract_sha256",
                        "seed", "cell", "terminal_relative_path",
                        "terminal_model_sha256", "loads"):
                assert key in entry
            assert entry["loads"] is True

        # end-to-end: the SEALED records + THIS proof satisfy the
        # runner-side gate and the full screen verdict.
        sealed_root = (tmp_path / "collection" / "sealed" / EXP_ID)
        records = {}
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                records[(seed, cell)] = json.loads(
                    (sealed_root / f"seed{seed}" / cell /
                     p1c.RECORD_NAME).read_text())
        ok, refusals, facts = p1.validate_replica_proof(
            proof, contract=contract, records=records)
        assert (ok, refusals) == (True, [])
        payload, code = p1.screen_verdict(contract, records=records,
                                          replica_proof=proof)
        assert payload["outcome"] == "SCREEN_VIABLE_REGION"
        assert payload["gates"]["replica_terminal_loads"] is True
        assert code == 0

        # source tree untouched by staging/sealing
        for seed in p1.SEEDS:
            for cell in p1.CELLS:
                assert (source_root / f"seed{seed}" / cell /
                        p1c.RECORD_NAME).is_file()

    def test_recollection_over_an_existing_seal_refuses(self, world,
                                                        tmp_path):
        contract, _ = world
        assert _collect(contract, tmp_path)["outcome"] == \
            "COLLECTION_SEALED"
        again = _collect(contract, tmp_path)
        assert again["outcome"] == "COLLECTION_REFUSED"
        assert any("never overwrite" in r for r in again["refusals"])


class TestRefusals:
    def test_fifteen_records_refuse(self, world, tmp_path):
        contract, source_root = world
        (source_root / "seed202" / "P1E_LR1E4" /
         p1c.RECORD_NAME).unlink()
        manifest = _collect(contract, tmp_path)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("staged record missing" in r
                   for r in manifest["refusals"])

    def test_identity_fragmentation_refuses(self, world, tmp_path):
        contract, source_root = world
        rec_path = source_root / "seed303" / "P1N_LR3E5" / \
            p1c.RECORD_NAME
        record = json.loads(rec_path.read_text())
        record["experiment_identity"] = "0" * 16
        rec_path.write_text(json.dumps(record))
        manifest = _collect(contract, tmp_path)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("identity fragmentation" in r
                   for r in manifest["refusals"])

    def test_duplicate_cell_identity_refuses(self, world, tmp_path):
        contract, source_root = world
        a = source_root / "seed101" / "P1N_LR1E4" / p1c.RECORD_NAME
        b = source_root / "seed101" / "P1N_LR3E5" / p1c.RECORD_NAME
        rec_a = json.loads(a.read_text())
        rec_b = json.loads(b.read_text())
        rec_b["cell_identity"] = rec_a["cell_identity"]
        b.write_text(json.dumps(rec_b))
        manifest = _collect(contract, tmp_path)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("duplicate cell identity" in r
                   for r in manifest["refusals"])

    def test_hash_altered_staged_terminal_refuses(self, world,
                                                  tmp_path):
        contract, source_root = world
        terminal = (source_root / "seed404" / "P1E_LR3E5" /
                    "attempt-01" / "model.terminal.zip")
        terminal.write_bytes(b"tampered-after-recording")
        manifest = _collect(contract, tmp_path)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("differs from the recorded" in r
                   for r in manifest["refusals"])

    def test_loads_false_replica_entry_refuses(self, world, tmp_path):
        contract, _ = world

        def bad_verify(host, replica_root, expectations):
            proof = _echo_verify(host, replica_root, expectations)
            proof["terminals"][7]["loads"] = False
            proof["terminals"][7]["error"] = "SAC.load failed"
            return proof
        manifest = _collect(contract, tmp_path, verify=bad_verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("did not load" in r for r in manifest["refusals"])
        assert "replica_proof_file" not in manifest

    def test_replica_digest_mismatch_refuses(self, world, tmp_path):
        contract, _ = world

        def bad_verify(host, replica_root, expectations):
            proof = _echo_verify(host, replica_root, expectations)
            proof["tree_digest"] = "0" * 64
            return proof
        manifest = _collect(contract, tmp_path, verify=bad_verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("replica is not proof" in r
                   for r in manifest["refusals"])

    def test_swapped_replica_entries_refuse(self, world, tmp_path):
        contract, _ = world

        def bad_verify(host, replica_root, expectations):
            proof = _echo_verify(host, replica_root, expectations)
            first, second = proof["terminals"][0], proof["terminals"][1]
            first["sha256"], second["sha256"] = (second["sha256"],
                                                 first["sha256"])
            return proof
        manifest = _collect(contract, tmp_path, verify=bad_verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("does not match" in r or "sha256" in r
                   for r in manifest["refusals"])

    def test_no_replica_host_refuses(self, world, tmp_path):
        contract, _ = world
        manifest = _collect(contract, tmp_path, replica_host=None)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("finding 225" in r for r in manifest["refusals"])
