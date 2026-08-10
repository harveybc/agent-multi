"""WP4 collector tests (order §5): socket-free, transports injected."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import collect_l1_factorial as col  # noqa: E402
from tests.test_aggregate_l1_factorial import (  # noqa: E402
    EXP, SEEDS, make_contract, make_record, make_evidence)


def build_source(tmp_path: Path, contract: dict, *, mutate=None) -> Path:
    """A fleet-shaped source tree with real terminal artifact bytes."""
    root = tmp_path / "fleet_output"
    for seed in SEEDS:
        for cell in contract["cells"]:
            rec = make_record(contract, seed, cell)
            rec_dir = root / EXP / f"seed{seed}" / cell
            attempt = rec_dir / "attempt-cafe-01"
            attempt.mkdir(parents=True)
            terminal = attempt / "model.terminal.zip"
            payload = f"terminal-{seed}-{cell}".encode()
            terminal.write_bytes(payload)
            rec["attempt_dir"] = str(attempt)
            rec["terminal_model_path"] = str(terminal)
            rec["terminal_model_sha256"] = hashlib.sha256(
                payload).hexdigest()
            rec["cell_identity"] = hashlib.sha256(
                f"{seed}/{cell}".encode()).hexdigest()[:16]
            if mutate:
                mutate(rec, seed, cell)
            (rec_dir / "l1_cell_record.json").write_text(
                json.dumps(rec, default=str))
    return root


def local_fetch(host, remote_dir, stage_dir):
    shutil.copytree(remote_dir, stage_dir)


def make_fleet_contract(tmp_path: Path) -> dict:
    contract = make_contract()
    contract["assignments"] = {str(s): {"hostname": f"host{s}",
                                        "gpu_uuid": f"GPU-{s}"}
                               for s in SEEDS}
    contract["output_root"] = str(tmp_path / "fleet_output")
    return contract


def run_collect(tmp_path, contract, **kw):
    return col.collect(
        contract=contract, experiment_id=EXP,
        collection_root=tmp_path / "collection",
        fetch_fn=local_fetch, **kw)


class TestCollection:
    def test_healthy_collection_seals_with_digest(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_SEALED"
        assert manifest["collection_tree_digest"]
        sealed = Path(manifest["sealed_root"])
        assert sealed.is_dir()
        assert (tmp_path / "collection" /
                "collection_manifest.json").is_file()
        assert manifest["source_hosts"] == {str(s): f"host{s}"
                                            for s in SEEDS}
        # Digest is reproducible over the sealed tree.
        assert col.tree_digest(sealed) == \
            manifest["collection_tree_digest"]

    def test_staging_never_overwrites(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        assert run_collect(tmp_path, contract)[
            "outcome"] == "COLLECTION_SEALED"
        # A second collection into the same root must refuse at staging
        # or find the stage occupied — never overwrite.
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_REFUSED"

    def test_hash_mismatch_is_refused(self, tmp_path):
        contract = make_fleet_contract(tmp_path)

        def corrupt(rec, seed, cell):
            if seed == 202 and cell == "L1_E_M03":
                rec["terminal_model_sha256"] = "0" * 64

        build_source(tmp_path, contract, mutate=corrupt)
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("does not match its recorded sha" in r
                   for r in manifest["refusals"])

    def test_duplicate_cell_identity_rejected_across_seeds(self,
                                                           tmp_path):
        contract = make_fleet_contract(tmp_path)

        def dup(rec, seed, cell):
            rec["cell_identity"] = "feedfeedfeedfeed"

        build_source(tmp_path, contract, mutate=dup)
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("duplicate cell identity" in r
                   for r in manifest["refusals"])

    def test_identity_uniformity_enforced(self, tmp_path):
        contract = make_fleet_contract(tmp_path)

        def tamper(rec, seed, cell):
            if seed == 101 and cell == "L1_N_M10":
                rec["subject_code_identity"]["agent-multi"][
                    "commit"] = "9" * 40

        build_source(tmp_path, contract, mutate=tamper)
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("subject_code_identity" in r
                   for r in manifest["refusals"])

    def test_missing_seed_is_refused(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        root = build_source(tmp_path, contract)
        shutil.rmtree(root / EXP / "seed202")
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("missing staged record seed202" in r
                   for r in manifest["refusals"])


class TestReplica:
    def test_replica_verified_by_rehash_and_load(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replica_store = tmp_path / "replica"

        def fake_replicate(host, sealed_root, replica_root):
            shutil.copytree(sealed_root, replica_store)

        def fake_verify(host, replica_root, expectations):
            out = []
            for e in expectations:
                rel = Path(e["replica_path"]).relative_to(
                    Path(str(replica_root)))
                blob = (replica_store / rel).read_bytes()
                out.append({"cell": e["cell"], "seed": e["seed"],
                            "sha256": hashlib.sha256(blob).hexdigest(),
                            "loads": True, "n_updates": 1})
            return out

        manifest = run_collect(
            tmp_path, contract, replica_host="replica-host",
            replica_root=tmp_path / "replica_root",
            replicate_fn=fake_replicate, replica_verify_fn=fake_verify)
        assert manifest["outcome"] == "COLLECTION_SEALED"
        assert manifest["replica"]["host"] == "replica-host"
        assert all(r["loads"] for r in
                   manifest["replica"]["verification"])

    def test_replica_sha_mismatch_refuses(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)

        def fake_replicate(host, sealed_root, replica_root):
            pass

        def bad_verify(host, replica_root, expectations):
            return [{"cell": e["cell"], "seed": e["seed"],
                     "sha256": "0" * 64, "loads": True}
                    for e in expectations]

        manifest = run_collect(
            tmp_path, contract, replica_host="replica-host",
            replica_root=tmp_path / "replica_root",
            replicate_fn=fake_replicate, replica_verify_fn=bad_verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("replica sha mismatch" in r
                   for r in manifest["refusals"])


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
