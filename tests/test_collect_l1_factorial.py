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
            (attempt / "return_traces").mkdir(parents=True)
            (attempt / "nested_splits").mkdir()
            terminal = attempt / "model.terminal.zip"
            payload = f"terminal-{seed}-{cell}".encode()
            terminal.write_bytes(payload)
            (attempt / "model.post_easy.zip").write_bytes(b"phase1")
            (attempt / "results.json").write_text(json.dumps({
                "final_equity": 10_500.0, "mean_weekly_return": 0.001,
                "max_drawdown_pct": 3.2, "sharpe_ratio": 0.4}))
            (attempt / "return_traces" / "evidence.json").write_text(
                json.dumps(make_evidence()))
            (attempt / "nested_splits" /
             "inner_validation.csv").write_text("DATE_TIME\n")
            rec["attempt_dir"] = str(attempt)
            rec["terminal_model_path"] = str(terminal)
            rec["terminal_model_sha256"] = hashlib.sha256(
                payload).hexdigest()
            rec["curriculum"]["post_easy"]["artifact"] = str(
                attempt / "model.post_easy.zip")
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


def replica_store_fns(tmp_path: Path):
    """Faithful local replica: copy + digest computed over the copy."""
    replica_store = tmp_path / "replica_store"

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
        return {"tree_digest": col.tree_digest(replica_store),
                "verifier_version": "test-verifier",
                "terminals": out}

    return fake_replicate, fake_verify, replica_store


class TestCollection:
    def test_healthy_collection_seals_with_digest(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, _ = replica_store_fns(tmp_path)
        manifest = run_collect(tmp_path, contract,
                               replica_host="replica-host",
                               replica_root=tmp_path / "replica_root",
                               replicate_fn=replicate,
                               replica_verify_fn=verify)
        assert manifest["outcome"] == "COLLECTION_SEALED"
        assert manifest["collection_tree_digest"]
        sealed = Path(manifest["sealed_root"])
        assert sealed.is_dir()
        assert (tmp_path / "collection" /
                "collection_manifest.json").is_file()
        assert manifest["source_hosts"] == {str(s): f"host{s}"
                                            for s in SEEDS}
        # Digest is reproducible over the sealed tree and matched on
        # the replica (finding 190).
        assert col.tree_digest(sealed) == \
            manifest["collection_tree_digest"]
        assert manifest["replica"]["digests_match"] is True
        assert manifest["replica"]["replica_tree_digest"] == \
            manifest["collection_tree_digest"]
        assert manifest["replica"]["verifier_version"]
        assert manifest["replica"]["verified_utc"]

    def test_sealing_without_replica_is_typed_partial(self, tmp_path):
        # Finding 190: no replica -> typed partial outcome that
        # aggregation refuses; never plain COLLECTION_SEALED.
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        manifest = run_collect(tmp_path, contract)
        assert manifest["outcome"] == "COLLECTION_SEALED_WITHOUT_REPLICA"
        assert manifest["replica"] is None
        assert any("replica is mandatory" in r
                   for r in manifest["refusals"])

    def test_staging_never_overwrites(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, _ = replica_store_fns(tmp_path)
        assert run_collect(tmp_path, contract,
                           replica_host="replica-host",
                           replica_root=tmp_path / "replica_root",
                           replicate_fn=replicate,
                           replica_verify_fn=verify)[
            "outcome"] == "COLLECTION_SEALED"
        # A second collection into the same root must refuse at staging
        # or find the seal occupied — never overwrite.
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


class TestSealedRootAuthority:
    def test_aggregation_from_sealed_root_after_source_deletion(
            self, tmp_path):
        """Order §4/WP9 acceptance: collect a fleet-shaped source,
        DELETE every source tree, then run the real aggregator from
        the sealed root with real path resolution. INCONCLUSIVE for
        content reasons is legal; reporting files as missing that are
        present in the seal, or reading outside the sealed root, is
        not."""
        from tests.test_aggregate_l1_factorial import (
            healthy_probe, healthy_rollout, make_manifest,
            matching_disk_facts)
        from tools import aggregate_l1_factorial as agg

        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, _ = replica_store_fns(tmp_path)
        manifest = run_collect(tmp_path, contract,
                               replica_host="replica-host",
                               replica_root=tmp_path / "replica_root",
                               replicate_fn=replicate,
                               replica_verify_fn=verify)
        assert manifest["outcome"] == "COLLECTION_SEALED"
        # The collector host does not have the source workers' disks.
        shutil.rmtree(Path(contract["output_root"]))

        sealed_parent = tmp_path / "collection" / "sealed"
        result = agg.aggregate(
            sealed_parent, EXP, contract=contract,
            manifest=make_manifest(), probe_fn=healthy_probe,
            rollout_fn=healthy_rollout,
            disk_facts_fn=matching_disk_facts)
        complaints = list(result["refusals"])
        for facts in result["cells"].values():
            complaints.extend(facts.get("invalid_reasons", []))
        offenders = [c for c in complaints
                     if "missing" in c or "unresolvable" in c]
        assert offenders == [], offenders
        # Every resolved path stayed inside the sealed root.
        sealed_exp = (sealed_parent / EXP).resolve()

        def _contained(path):
            resolved = Path(path).resolve()
            return resolved == sealed_exp or \
                sealed_exp in resolved.parents

        for rec_path in sealed_exp.rglob("l1_cell_record.json"):
            record = json.loads(rec_path.read_text())
            record["_record_path"] = str(rec_path)
            resolver = agg.SealedPathResolver(sealed_parent, EXP)
            resolved, reasons = agg.resolve_record_paths(record,
                                                         resolver)
            assert reasons == []
            assert all(_contained(p) for p in resolved.values())

    def test_real_disk_facts_read_the_sealed_copy(self, tmp_path):
        """terminal_disk_facts rehashes the SEALED artifact (bytes
        match the record) even though the recorded absolute path no
        longer exists."""
        from tools import aggregate_l1_factorial as agg

        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, _ = replica_store_fns(tmp_path)
        run_collect(tmp_path, contract, replica_host="replica-host",
                    replica_root=tmp_path / "replica_root",
                    replicate_fn=replicate, replica_verify_fn=verify)
        shutil.rmtree(Path(contract["output_root"]))
        sealed_parent = tmp_path / "collection" / "sealed"
        rec_path = next((sealed_parent / EXP).rglob(
            "l1_cell_record.json"))
        record = json.loads(rec_path.read_text())
        record["_record_path"] = str(rec_path)
        resolver = agg.SealedPathResolver(sealed_parent, EXP)
        record["_resolved"], _ = agg.resolve_record_paths(record,
                                                          resolver)
        assert not Path(record["terminal_model_path"]).exists()
        facts = agg.terminal_disk_facts(record)
        assert facts is not None
        assert facts["terminal_model_sha256"] == \
            record["terminal_model_sha256"]


class TestReplica:
    def test_replica_verified_by_rehash_and_load(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, _ = replica_store_fns(tmp_path)
        manifest = run_collect(
            tmp_path, contract, replica_host="replica-host",
            replica_root=tmp_path / "replica_root",
            replicate_fn=replicate, replica_verify_fn=verify)
        assert manifest["outcome"] == "COLLECTION_SEALED"
        assert manifest["replica"]["host"] == "replica-host"
        assert all(r["loads"] for r in
                   manifest["replica"]["verification"])

    def test_replica_tree_digest_mismatch_refuses(self, tmp_path):
        # Finding 190: a modified/missing file on the replica (here a
        # tampered results.json) breaks the whole-tree digest and
        # refuses even when every terminal matches.
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)
        replicate, verify, replica_store = replica_store_fns(tmp_path)

        def tampering_replicate(host, sealed_root, replica_root):
            replicate(host, sealed_root, replica_root)
            victim = next(replica_store.rglob("results.json"), None)
            if victim is None:
                victim = next(replica_store.rglob(
                    "l1_cell_record.json"))
            victim.write_text(victim.read_text() + " ")

        manifest = run_collect(
            tmp_path, contract, replica_host="replica-host",
            replica_root=tmp_path / "replica_root",
            replicate_fn=tampering_replicate,
            replica_verify_fn=verify)
        assert manifest["outcome"] == "COLLECTION_REFUSED"
        assert any("tree digest" in r for r in manifest["refusals"])

    def test_replica_sha_mismatch_refuses(self, tmp_path):
        contract = make_fleet_contract(tmp_path)
        build_source(tmp_path, contract)

        def fake_replicate(host, sealed_root, replica_root):
            pass

        def bad_verify(host, replica_root, expectations):
            return {"tree_digest": "0" * 64,
                    "verifier_version": "test-verifier",
                    "terminals": [
                        {"cell": e["cell"], "seed": e["seed"],
                         "sha256": "0" * 64, "loads": True}
                        for e in expectations]}

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
