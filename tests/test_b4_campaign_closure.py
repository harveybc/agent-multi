"""R1 (order 2026-09-11): the B4 v7 closure, and every guard in it.

The closure is the artifact that turns an authorized stop into a
checkable adjudication, so each of its refusals is tested by MUTATING a
synthetic campaign root until exactly that refusal fires. A guard that
is never made to bite is a comment.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import b4_campaign_closure as C  # noqa: E402

#: captured BEFORE any fixture can patch it. A test that means to
#: interrogate the shipped gate must hold the shipped function, not
#: whatever the module attribute points at when the test runs.
REAL_PROVE_RELAUNCH_REFUSES = C.prove_relaunch_refuses


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------- fixture
def build_root(tmp_path: Path, *, cells: int = 3,
               completed: int = 1, partial: int = 1) -> tuple[Path, Path]:
    """A miniature campaign root with the same record shapes as v7."""
    root = tmp_path / "results"
    mat = tmp_path / "materialization"
    root.mkdir()
    mat.mkdir()
    mat_doc = {"materialization": "synthetic"}
    (mat / "B4_MATERIALIZATION.json").write_text(
        json.dumps(mat_doc, sort_keys=True))
    mat_sha = sha_bytes((mat / "B4_MATERIALIZATION.json").read_bytes())

    names = [f"o2022_seed{n}" for n in range(101, 101 + cells)]
    ledger = {
        "schema": "agent_multi.b4_campaign_ledger.v1",
        "population_sha256": "0" * 64,
        "materialization_sha256": mat_sha,
        "generation_provenance": {"campaign_generation": "synthetic_v7"},
        "cells": {n: {"cell_config_sha256": sha_bytes(n.encode()),
                      "status": "PENDING"} for n in names},
    }
    (root / "CAMPAIGN_LEDGER.json").write_text(
        json.dumps(ledger, indent=1))

    for name in names[:completed]:
        seal_completed_cell(root, name, ledger["cells"][name])
    for name in names[completed:completed + partial]:
        make_partial_cell(root, name)
    if partial:
        (root / "CAMPAIGN_STOP").write_text("")
    return root, mat


def seal_completed_cell(root: Path, cell: str, ledger_cell: dict,
                        **overrides) -> Path:
    d = root / cell
    d.mkdir(parents=True, exist_ok=True)
    per_bar = d / f"per_bar_{cell}.csv"
    per_bar.write_text("net_return\n0.1\n0.2\n")
    ckpt = d / "best_model.terminal.zip"
    ckpt.write_bytes(b"checkpoint-bytes")
    attempt = f"attempt_{cell}"
    term = {
        "schema": "agent_multi.b4_cell_terminal.v1",
        "cell": cell, "terminal": "COMPLETED",
        "g1_eligible": False, "checkpoint_promotable": False,
        "attempt_id": attempt,
        "cell_config_sha256": ledger_cell["cell_config_sha256"],
        "artifact_class": "BEST_CHECKPOINT",
        "checkpoint_sha256": sha_bytes(ckpt.read_bytes()),
        "checkpoint_path": str(ckpt),
        "per_bar_csv": str(per_bar),
        "per_bar_sha256": sha_bytes(per_bar.read_bytes()),
        "scored_index_sha256": "1" * 64, "scored_bars": 2,
        "counter_semantics": "synthetic", "sealed_2025_used": False,
        "wall_seconds": 123.4, "effective_limits": {},
        "authorization_record_sha256": "2" * 64,
        "amendment_11_sha256": "3" * 64,
        "campaign_generation": "synthetic_v7",
        "recovery_acta_sha256": "4" * 64,
        "pinned_execution_commit": "deadbeef",
        "latest_amendment_sha256": "5" * 64,
    }
    term.update(overrides)
    term_p = d / "B4_CELL_TERMINAL.json"
    term_p.write_text(json.dumps(term, indent=1))
    term_sha = sha_bytes(term_p.read_bytes())
    (d / f"CLAIM_synthetic.json").write_text(json.dumps({
        "schema": "agent_multi.b4_attempt_claim.v2", "cell": cell,
        "attempt_id": attempt, "claimed_wall": 1788820811.0,
        "terminal_sha256": None}, indent=1))
    intent_p = d / f"SEAL_INTENT_{attempt}.json"
    intent_p.write_text(json.dumps({
        "schema": "agent_multi.b4_seal_intent.v1", "cell": cell,
        "attempt_id": attempt, "terminal_sha256": term_sha}, indent=1))
    (d / f"SEAL_COMPLETE_{attempt}.json").write_text(json.dumps({
        "schema": "agent_multi.b4_seal_completion.v1",
        "intent_sha256": sha_bytes(intent_p.read_bytes()),
        "terminal_sha256": term_sha, "attempt_id": attempt}, indent=1))
    return d


def make_partial_cell(root: Path, cell: str) -> Path:
    d = root / cell
    (d / "cell_runtime").mkdir(parents=True, exist_ok=True)
    attempt = f"attempt_{cell}"
    (d / f"CLAIM_synthetic.json").write_text(json.dumps({
        "schema": "agent_multi.b4_attempt_claim.v2", "cell": cell,
        "attempt_id": attempt, "claimed_wall": 1788856493.8,
        "terminal_sha256": None}, indent=1))
    (d / f"LEASE_{attempt}.json").write_text(json.dumps({
        "schema": "agent_multi.b4_execution_lease.v3", "cell": cell,
        "attempt_id": attempt}, indent=1))
    (d / "best_model.zip").write_bytes(b"partial-model")
    (d / "cell_runtime/status.json").write_text(json.dumps({
        "epoch_completed": 111, "num_timesteps": 2220000,
        "stop_reason": None, "last_durable_artifact": None}))
    (d / "STOP").write_text("")
    return d


@pytest.fixture(autouse=True)
def _open_launch_gate(monkeypatch):
    """The relaunch proof asks the real gate. In unit tests we drive it
    explicitly instead, so the closure never depends on the reviewer
    authority root being present on the machine running the suite."""
    monkeypatch.setattr(C, "prove_relaunch_refuses", lambda: {
        "launch_gate": "stub", "refused": True, "refusal": "REFUSED: stub",
        "accelerator_modules_before": [], "accelerator_modules_after": [],
        "refused_before_any_accelerator_import": True})


# ---------------------------------------------------------- happy path
def test_a_stopped_campaign_closes_as_two_one_nine(tmp_path):
    root, mat = build_root(tmp_path, cells=12, completed=2, partial=1)
    closure = C.build_closure(root, mat)
    assert closure["adjudication"]["counts"] == {
        C.COMPLETED: 2, C.PARTIAL: 1, C.NOT_STARTED: 9}
    assert closure["adjudication"]["verdict"] == \
        "SCIENTIFICALLY_INSUFFICIENT_NO_VERDICT"
    assert all(not c["enters_comparisons"] for c in closure["cells"])


def test_every_completed_cell_is_verified_by_descriptor(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    closure = C.build_closure(root, mat)
    done = [c for c in closure["cells"]
            if c["classification"] == C.COMPLETED][0]
    names = {d["descriptor"] for d in done["descriptors_verified"]}
    for required in ("terminal_schema_exact", "per_bar_digest",
                     "checkpoint_digest", "seal_complete_binds_the_intent",
                     "seal_intent_binds_terminal_digest",
                     "cell_config_digest_binds_ledger",
                     "not_g1_eligible", "checkpoint_not_promotable"):
        assert required in names, required
    assert all(d["verified"] for d in done["descriptors_verified"])


def test_the_partial_is_inventoried_in_full_before_it_is_classified(
        tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    closure = C.build_closure(root, mat)
    part = [c for c in closure["cells"]
            if c["classification"] == C.PARTIAL][0]
    assert part["artifact_count"] == 5
    assert part["artifact_bytes"] > 0
    assert part["scientific_contribution"] == "NONE"
    assert part["last_durable_progress"]["epoch_completed"] == 111


# ------------------------------------------------------------ mutations
def test_a_broken_per_bar_digest_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    (root / "o2022_seed101" / "per_bar_o2022_seed101.csv").write_text(
        "net_return\n9.9\n9.9\n")
    with pytest.raises(SystemExit, match="per_bar_digest"):
        C.build_closure(root, mat)


def test_a_tampered_terminal_breaks_the_seal_chain(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    term_p = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    doc = json.loads(term_p.read_text())
    doc["wall_seconds"] = 999999.0
    term_p.write_text(json.dumps(doc, indent=1))
    with pytest.raises(SystemExit, match="seal_intent_binds_terminal_digest"):
        C.build_closure(root, mat)


def test_a_foreign_cell_config_digest_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    ledger_p = root / "CAMPAIGN_LEDGER.json"
    led = json.loads(ledger_p.read_text())
    led["cells"]["o2022_seed101"]["cell_config_sha256"] = "f" * 64
    ledger_p.write_text(json.dumps(led, indent=1))
    with pytest.raises(SystemExit,
                       match="cell_config_digest_binds_ledger"):
        C.build_closure(root, mat)


def test_an_undeclared_terminal_field_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    term_p = root / "o2022_seed101/B4_CELL_TERMINAL.json"
    doc = json.loads(term_p.read_text())
    doc["promoted_by_operator"] = True
    term_p.write_text(json.dumps(doc, indent=1))
    with pytest.raises(SystemExit, match="terminal_schema_exact"):
        C.build_closure(root, mat)


def test_a_promotable_terminal_refuses(tmp_path):
    """The closure asserts non-promotion; it may never grant it."""
    root, mat = build_root(tmp_path, cells=3, completed=0, partial=1)
    led = json.loads((root / "CAMPAIGN_LEDGER.json").read_text())
    seal_completed_cell(root, "o2022_seed101", led["cells"]["o2022_seed101"],
                        g1_eligible=True)
    with pytest.raises(SystemExit, match="not_g1_eligible"):
        C.build_closure(root, mat)


def test_a_partial_that_carries_a_terminal_refuses(tmp_path):
    """Quarantine is never assumed from a directory's existence."""
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    part = root / "o2022_seed102"
    (part / "results.json").write_text("{}")
    # remove the terminal path so it is classified as partial, then the
    # completion artifact must be what refuses
    with pytest.raises(SystemExit, match="completion artifacts"):
        C.build_closure(root, mat)


def test_a_partial_carrying_a_seal_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    (root / "o2022_seed102/SEAL_COMPLETE_x.json").write_text("{}")
    with pytest.raises(SystemExit, match="carries seals"):
        C.build_closure(root, mat)


def test_a_partial_without_both_stop_signals_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    (root / "CAMPAIGN_STOP").unlink()
    with pytest.raises(SystemExit, match="BOTH durable stop signals"):
        C.build_closure(root, mat)


def test_a_partial_without_a_lease_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    next((root / "o2022_seed102").glob("LEASE_*.json")).unlink()
    with pytest.raises(SystemExit, match="claim and a lease"):
        C.build_closure(root, mat)


def test_a_complete_population_is_not_a_closure(tmp_path):
    root, mat = build_root(tmp_path, cells=2, completed=2, partial=0)
    with pytest.raises(SystemExit, match="not a closure"):
        C.build_closure(root, mat)


def test_a_foreign_materialization_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    (mat / "B4_MATERIALIZATION.json").write_text('{"materialization": "x"}')
    with pytest.raises(SystemExit, match="does not bind this "
                                         "materialization"):
        C.build_closure(root, mat)


def test_a_stop_signal_older_than_the_claim_refuses(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    claim_p = next((root / "o2022_seed102").glob("CLAIM_*.json"))
    doc = json.loads(claim_p.read_text())
    doc["claimed_wall"] = 9_999_999_999.0
    claim_p.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="stop signal predates the claim"):
        C.build_closure(root, mat)


# --------------------------------------------------------------- costs
def test_the_partial_charge_is_declared_a_lower_bound(tmp_path):
    root, mat = build_root(tmp_path, cells=3, completed=1, partial=1)
    closure = C.build_closure(root, mat)
    charge = closure["costs"]["quarantined_partial"]["o2022_seed102"]
    assert charge["bound"] == "LOWER_BOUND"
    assert charge["claim_to_stop_signal_seconds"] > 0
    assert "does not lower any charge" in charge["why"]


def test_completed_costs_come_from_the_terminals(tmp_path):
    root, mat = build_root(tmp_path, cells=12, completed=2, partial=1)
    closure = C.build_closure(root, mat)
    assert closure["costs"]["completed_wall_seconds_total"] == \
        pytest.approx(246.8)


# --------------------------------------------------------- idempotency
def test_re_closing_unchanged_evidence_appends_nothing(tmp_path, capsys):
    root, mat = build_root(tmp_path, cells=12, completed=2, partial=1)
    argv = ["--results-root", str(root), "--materialization-root",
            str(mat), "--emit"]
    C.main(argv)
    first = json.loads(capsys.readouterr().out)
    C.main(argv)
    second = json.loads(capsys.readouterr().out)
    assert first["emitted"] is True
    assert second["emitted"] is False
    assert second["already_closed_at"] == first["closed_at"]
    assert first["adjudication_sha256"] == second["adjudication_sha256"]
    log = (root / C.CLOSURE_LOG).read_text().splitlines()
    assert len(log) == 1


def test_measurement_cost_is_not_part_of_the_adjudication_identity(
        tmp_path):
    root, mat = build_root(tmp_path, cells=12, completed=2, partial=1)
    a = C.build_closure(root, mat)
    b = C.build_closure(root, mat)
    assert a["adjudication_sha256"] == b["adjudication_sha256"]
    assert "measurement" in a
    assert "inventory_seconds" in a["measurement"]


def test_emitting_changes_no_training_artifact(tmp_path, capsys):
    root, mat = build_root(tmp_path, cells=12, completed=2, partial=1)
    C.main(["--results-root", str(root), "--materialization-root",
            str(mat), "--emit"])
    out = json.loads(capsys.readouterr().out)
    proof = out["artifact_immutability_proof"]
    assert proof["unchanged"] is True
    assert proof["tree_sha256_before"] == proof["tree_sha256_after"]
    assert proof["files_digested"] > 0


# ------------------------------------------------------ relaunch proof
def test_the_real_launch_gate_refuses_before_any_accelerator_import():
    """Not stubbed: this is the one test that asks the shipped gate."""
    assert REAL_PROVE_RELAUNCH_REFUSES is not C.prove_relaunch_refuses, (
        "the autouse stub must be in place, or this test proves nothing "
        "about it being bypassed")
    proof = REAL_PROVE_RELAUNCH_REFUSES()
    assert proof["refused"] is True
    assert proof["refused_before_any_accelerator_import"] is True
    assert proof["accelerator_modules_after"] == []
    assert "REFUSED" in proof["refusal"]
