"""Tests for the durable terminal-to-next-job queue (order 2026-08-15 §3).

Socket-free and clock-free: every record lives under ``tmp_path`` and
every timestamp is injected, so a reboot is simulated exactly the way it
really behaves — the process, the heartbeats and the in-memory state are
gone, and ONLY the files on disk remain.

The five scenarios the order names are covered explicitly:
completion before reboot, reboot during dispatch, duplicate dispatch
attempts, one node unavailable, and a conflicting chain already present.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import experiment_transition_queue as etq  # noqa: E402

NOW = datetime(2026, 8, 15, 6, 0, 0, tzinfo=timezone.utc)
EXPERIMENT = "p1_difficulty_lr_factorial_20260811_v1"
MODE = "decision"
IDENTITY = "c0e53cf18b7d60dd"          # the real terminal decision run
CHAIN = "chain-l2-en-0001"


class FakeEmitter:
    def __init__(self, ok=True):
        self.ok = ok
        self.observed = []
        self.recovered = []

    def observe(self, event_code, severity, summary, payload,
                affected_object="-"):
        self.observed.append({"event_code": event_code,
                              "severity": severity, "summary": summary,
                              "payload": payload,
                              "affected_object": affected_object})
        return self.ok

    def recover(self, event_code, evidence, affected_object="-"):
        self.recovered.append({"event_code": event_code,
                               "evidence": evidence,
                               "affected_object": affected_object})
        return self.ok


def _queue(tmp_path):
    return tmp_path / "transition-queue"


def _terminal(tmp_path, *, now=NOW, budget=3600.0, records=16, cells=16):
    """The fleet-level observer enrols a TERMINAL experiment."""
    return etq.ensure_terminal_record(
        _queue(tmp_path), experiment=EXPERIMENT, mode=MODE,
        identity=IDENTITY, records_landed=records, cells_total=cells,
        output_root="~/.local/share/agent-multi/p1lr_decision",
        terminal_utc=now.isoformat(), now=now,
        transition_budget_seconds=budget, observed_by="omega")


def _approved(tmp_path, record, *, now=NOW, materialized=True,
              chain_id=CHAIN):
    record = etq.approve_successor(
        record, job_id="l2-frozen-l1-en-v1", experiment="l2_en_20260815_v1",
        contract_sha256="a" * 64, approved_by="owner",
        approval_reference="document-38 §L2", chain_id=chain_id, now=now)
    if materialized:
        record = etq.set_materialization(record, "materialized", now=now)
    etq.save_record(_queue(tmp_path), record, now=now)
    return record


# ── the durable record itself ─────────────────────────────────────────

def test_record_carries_every_field_the_order_names(tmp_path):
    record = _terminal(tmp_path)
    assert record["schema"] == etq.RECORD_SCHEMA
    assert record["current_job"]["experiment"] == EXPERIMENT
    assert record["current_job"]["mode"] == MODE
    assert record["current_job"]["identity"] == IDENTITY
    assert record["terminal_result"]["identity"] == IDENTITY
    assert record["terminal_result"]["records_landed"] == 16
    assert record["terminal_result"]["complete"] is True
    assert record["next_job"] is None
    assert record["materialization_state"] == "not_started"
    assert record["dispatch_state"] == "undispatched"
    assert record["blockers"] == []
    # …and it is on disk, byte-identical, under its own identity
    path = etq.record_path(_queue(tmp_path), record["transition_id"])
    assert json.loads(path.read_text())["transition_id"] == \
        record["transition_id"]


def test_enrolment_is_idempotent_and_never_restarts_the_budget(tmp_path):
    """Four hosts observing the same completion converge on ONE record;
    a later observation must not reset the transition clock."""
    first = _terminal(tmp_path, now=NOW)
    later = _terminal(tmp_path, now=NOW + timedelta(hours=5))
    assert first["transition_id"] == later["transition_id"]
    assert later["terminal_result"]["terminal_utc"] == \
        first["terminal_result"]["terminal_utc"]
    assert len(list((_queue(tmp_path)).glob("*.json"))) == 1


def test_terminal_utc_basis_is_never_dressed_up_as_a_completion_time(
        tmp_path):
    record = etq.ensure_terminal_record(
        _queue(tmp_path), experiment=EXPERIMENT, mode=MODE,
        identity=IDENTITY, records_landed=16, cells_total=16,
        terminal_utc=None, now=NOW)
    assert record["terminal_result"]["terminal_utc_basis"] == \
        "first_terminal_observation_by_queue"


# ── §3 bullet 1: completed_untransitioned ─────────────────────────────

def test_terminal_without_successor_is_completed_untransitioned(tmp_path):
    status = etq.transition_status(_terminal(tmp_path), now=NOW)
    assert status["value"] == "completed_untransitioned"
    assert status["successor_dispatched"] is False
    assert [b["code"] for b in status["blockers"]] == \
        ["NO_APPROVED_SUCCESSOR"]


def test_missing_record_is_untransitioned_not_healthy(tmp_path):
    """Absent bookkeeping is never rendered as a healthy fleet."""
    status = etq.transition_status(None, now=NOW)
    assert status["value"] == "completed_untransitioned"
    assert status["basis"] == "no_durable_record"
    assert [b["code"] for b in status["blockers"]] == \
        ["NO_DURABLE_TRANSITION_RECORD"]


def test_dispatched_successor_is_transitioned(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    assert etq.transition_status(record, now=NOW)["value"] == \
        "transition_dispatch_in_progress"
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    status = etq.transition_status(record, now=NOW)
    assert status["value"] == "transitioned"
    assert status["successor_dispatched"] is True
    assert status["chain_id"] == CHAIN


# ── scenario: completion BEFORE reboot ────────────────────────────────

def test_completion_before_reboot_survives_as_completed_untransitioned(
        tmp_path):
    """The experiment finished, then Omega rebooted. Nothing in memory
    survives; the durable record still names the un-transitioned fleet."""
    _terminal(tmp_path, now=NOW)
    after_reboot = NOW + timedelta(hours=9)     # the observed 7.5h+ gap
    view = etq.reconstruct_transitions(_queue(tmp_path), now=after_reboot)
    assert view["records_total"] == 1
    assert view["fleet_idle_after_terminal_completion"] is True
    only = view["completed_untransitioned"][0]
    assert only["current_job"]["identity"] == IDENTITY
    assert only["elapsed_since_terminal_seconds"] == 9 * 3600.0
    assert only["over_budget"] is True
    assert "durable transition records on disk ONLY" in view["basis"]


def test_reconstruction_reads_nothing_but_the_queue_directory(tmp_path,
                                                              monkeypatch):
    """§3 bullet 4 literally: no heartbeat, no shell process, no chat
    message, no operator memory. Any subprocess/socket use is a bug."""
    _approved(tmp_path, _terminal(tmp_path))

    def _forbidden(*args, **kwargs):     # pragma: no cover - must not run
        raise AssertionError("reconstruction consulted a live source")

    import socket as _socket
    import subprocess as _subprocess
    monkeypatch.setattr(_subprocess, "run", _forbidden)
    monkeypatch.setattr(_subprocess, "Popen", _forbidden)
    monkeypatch.setattr(_socket, "socket", _forbidden)
    view = etq.reconstruct_transitions(_queue(tmp_path),
                                       now=NOW + timedelta(hours=2))
    assert view["transitions"][0]["next_job_id"] == "l2-frozen-l1-en-v1"


# ── scenario: reboot DURING dispatch ──────────────────────────────────

def test_reboot_during_dispatch_is_reconstructed_not_assumed_in_flight(
        tmp_path):
    """A claim is a LEASE. The claimer rebooted and stopped renewing it,
    so after the lease expires the transition is honestly un-transitioned
    again — never a dispatch that is somehow still in flight."""
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    etq.save_record(_queue(tmp_path), record, now=NOW)   # …then the reboot

    during = NOW + timedelta(seconds=300)
    live = etq.reconstruct_transitions(_queue(tmp_path), now=during)
    assert live["transitions"][0]["value"] == \
        "transition_dispatch_in_progress"

    after = NOW + timedelta(hours=9)
    view = etq.reconstruct_transitions(_queue(tmp_path), now=after)
    stuck = view["transitions"][0]
    assert stuck["value"] == "completed_untransitioned"
    assert stuck["dispatch_claim_expired"] is True
    assert len(view["expired_dispatch_claims"]) == 1

    # …and the recovering host may now take the lease, journalled as a
    # reclaim rather than as a fresh dispatch.
    reloaded = etq.load_record(
        etq.record_path(_queue(tmp_path), record["transition_id"]))
    retaken = etq.claim_dispatch(reloaded, claim_id="c2", host="omega",
                                 chain_id=CHAIN, now=after)
    events = [e["event"] for e in retaken["journal"]]
    assert "dispatch_claim_reclaimed_after_lease_expiry" in events
    assert etq.transition_status(retaken, now=after)["value"] == \
        "transition_dispatch_in_progress"


# ── scenario: DUPLICATE dispatch attempts (must fail closed) ──────────

def test_duplicate_dispatch_attempt_fails_closed(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    etq.save_record(_queue(tmp_path), record, now=NOW)

    with pytest.raises(etq.TransitionRefusal) as excinfo:
        etq.claim_dispatch(record, claim_id="c2", host="dragon",
                           chain_id=CHAIN, now=NOW + timedelta(seconds=30))
    assert excinfo.value.code == "TRANSITION_DUPLICATE_DISPATCH"
    # the durable record is UNCHANGED: a refusal dispatches nothing
    on_disk = etq.load_record(
        etq.record_path(_queue(tmp_path), record["transition_id"]))
    assert on_disk["dispatch"]["claim_id"] == "c1"
    assert on_disk["dispatch"]["claimed_by"] == "omega"


def test_duplicate_dispatch_after_confirmation_also_fails_closed(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    with pytest.raises(etq.TransitionRefusal) as excinfo:
        etq.claim_dispatch(record, claim_id="c2", host="gamma",
                           chain_id=CHAIN, now=NOW + timedelta(hours=4))
    assert excinfo.value.code == "TRANSITION_DUPLICATE_DISPATCH"
    assert "join_chain" in excinfo.value.reason
    # …and someone else's dispatch can never be confirmed as one's own
    with pytest.raises(etq.TransitionRefusal) as confirm_refusal:
        etq.confirm_dispatch(record, claim_id="c2", now=NOW)
    assert confirm_refusal.value.code == "TRANSITION_DUPLICATE_DISPATCH"


def test_same_claim_id_is_idempotent_not_a_duplicate(tmp_path):
    """A retry after a lost response must renew, never refuse."""
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    renewed = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                 chain_id=CHAIN,
                                 now=NOW + timedelta(seconds=120))
    assert renewed["dispatch"]["claim_id"] == "c1"
    assert renewed["dispatch"]["renewed_utc"] != \
        renewed["dispatch"]["claimed_utc"]


def test_unprovable_claim_is_neither_in_flight_nor_free(tmp_path):
    """A corrupted claim timestamp must not be guessed in either
    direction: the status stops claiming a dispatch is in flight, and a
    second claimer is still refused."""
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    record["dispatch"]["claimed_utc"] = "not-a-timestamp"
    record["dispatch"]["renewed_utc"] = "not-a-timestamp"

    status = etq.transition_status(record, now=NOW)
    assert status["value"] == "completed_untransitioned"
    assert status["dispatch_claim_liveness_provable"] is False
    with pytest.raises(etq.TransitionRefusal) as excinfo:
        etq.claim_dispatch(record, claim_id="c2", host="dragon",
                           chain_id=CHAIN, now=NOW + timedelta(days=1))
    assert excinfo.value.code == "TRANSITION_DUPLICATE_DISPATCH"
    assert excinfo.value.facts["claim_liveness_provable"] is False


def test_unmaterialized_or_blocked_successor_is_never_dispatched(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path), materialized=False)
    with pytest.raises(etq.TransitionRefusal) as not_material:
        etq.claim_dispatch(record, claim_id="c1", host="omega", now=NOW)
    assert not_material.value.code == "TRANSITION_NOT_MATERIALIZED"

    record = etq.set_materialization(record, "materialized", now=NOW)
    record = etq.add_blocker(record, code="DATASET_HASH_UNVERIFIED",
                             detail="ETH 4h manifest not verified on gamma",
                             now=NOW)
    with pytest.raises(etq.TransitionRefusal) as blocked:
        etq.claim_dispatch(record, claim_id="c1", host="omega", now=NOW)
    assert blocked.value.code == "TRANSITION_BLOCKED"

    record = etq.clear_blocker(record, "DATASET_HASH_UNVERIFIED", now=NOW)
    assert etq.claim_dispatch(record, claim_id="c1", host="omega",
                              now=NOW)["dispatch_state"] == \
        "dispatch_claimed"


def test_no_approved_successor_cannot_be_dispatched(tmp_path):
    with pytest.raises(etq.TransitionRefusal) as excinfo:
        etq.claim_dispatch(_terminal(tmp_path), claim_id="c1",
                           host="omega", now=NOW)
    assert excinfo.value.code == "TRANSITION_NO_APPROVED_SUCCESSOR"


# ── scenario: ONE NODE UNAVAILABLE ────────────────────────────────────

def test_one_unavailable_node_degrades_breadth_without_blocking_healthy(
        tmp_path):
    """A dead node is a typed fact, not a veto: the healthy nodes keep
    collaborating on the ONE proven chain and the record shows exactly
    which breadth was achieved."""
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    record = etq.mark_node(record, host="gamma", state="unavailable",
                           reason="ssh transport failure (exit 255)",
                           now=NOW)
    record = etq.join_chain(record, host="dragon", chain_id=CHAIN, now=NOW)
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    etq.save_record(_queue(tmp_path), record, now=NOW)

    status = etq.transition_status(record, now=NOW)
    assert status["value"] == "transitioned"
    assert status["nodes"]["gamma"]["state"] == "unavailable"
    assert status["nodes"]["gamma"]["reason"].startswith("ssh transport")
    assert status["nodes"]["dragon"]["state"] == "dispatched"
    assert status["nodes"]["dragon"]["chain_id"] == CHAIN

    # gamma comes back and joins the SAME proven chain — no second chain
    record = etq.join_chain(record, host="gamma", chain_id=CHAIN,
                            now=NOW + timedelta(hours=1))
    assert record["nodes"]["gamma"]["state"] == "dispatched"
    assert {n["chain_id"] for n in record["nodes"].values()} == {CHAIN}


# ── scenario: a CONFLICTING CHAIN already present ─────────────────────

def test_conflicting_chain_is_refused_on_dispatch_and_on_join(tmp_path):
    """"Never create parallel independent chains for one arm": a second
    chain identity is refused whether it arrives as a dispatch or as a
    node trying to join."""
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    record = etq.confirm_dispatch(record, claim_id="c1", now=NOW)
    etq.save_record(_queue(tmp_path), record, now=NOW)

    with pytest.raises(etq.TransitionRefusal) as dispatch_conflict:
        etq.claim_dispatch(record, claim_id="c9", host="dragon",
                           chain_id="chain-rogue-0002",
                           now=NOW + timedelta(hours=9))
    assert dispatch_conflict.value.code == "TRANSITION_CHAIN_CONFLICT"

    with pytest.raises(etq.TransitionRefusal) as join_conflict:
        etq.join_chain(record, host="dragon", chain_id="chain-rogue-0002",
                       now=NOW)
    assert join_conflict.value.code == "TRANSITION_CHAIN_CONFLICT"
    assert join_conflict.value.facts["proven_chain_id"] == CHAIN

    # re-approving a different chain over a live dispatch is refused too
    with pytest.raises(etq.TransitionRefusal) as approve_conflict:
        etq.approve_successor(record, job_id="l2-other",
                              approved_by="owner",
                              chain_id="chain-rogue-0002", now=NOW)
    assert approve_conflict.value.code == "TRANSITION_CHAIN_CONFLICT"

    # the durable record still holds exactly one chain
    on_disk = etq.load_record(
        etq.record_path(_queue(tmp_path), record["transition_id"]))
    assert on_disk["next_job"]["chain_id"] == CHAIN
    assert on_disk["dispatch"]["claim_id"] == "c1"


def test_a_node_cannot_join_a_chain_that_no_dispatch_established(tmp_path):
    record = _terminal(tmp_path)
    with pytest.raises(etq.TransitionRefusal) as excinfo:
        etq.join_chain(record, host="dragon", chain_id=CHAIN, now=NOW)
    assert excinfo.value.code == "TRANSITION_NO_PROVEN_CHAIN"


# ── §3 bullet 6: ONE deduplicated incident, closed by recovery ────────

def test_one_incident_per_stuck_transition_and_recovery_closes_it(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path, budget=3600.0))
    emitter = FakeEmitter()
    over = NOW + timedelta(seconds=7200)

    for offset in (0, 900, 1800):
        updated = etq.sweep_transition_incidents(
            _queue(tmp_path), emitter=emitter,
            now=over + timedelta(seconds=offset))
    assert len(emitter.observed) == 1
    incident = emitter.observed[0]
    assert incident["event_code"] == \
        f"experiment_transition_undispatched.{record['transition_id']}"
    assert incident["severity"] == "P1"
    assert incident["payload"]["over_budget_seconds"] == 3600.0
    assert "fleet is idle after a terminal completion" in \
        incident["summary"]
    assert updated["actions"][0]["action"] == \
        "incident_already_open_deduplicated"

    # the marker is DURABLE, so a rebooted host cannot re-emit it
    on_disk = etq.load_record(
        etq.record_path(_queue(tmp_path), record["transition_id"]))
    assert on_disk["incident"]["open"] is True
    fresh_emitter = FakeEmitter()
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=fresh_emitter,
                                   now=over + timedelta(days=1))
    assert fresh_emitter.observed == []

    # recovery closes the SAME event code, exactly once
    dispatched = etq.claim_dispatch(on_disk, claim_id="c1", host="omega",
                                    chain_id=CHAIN, now=over)
    dispatched = etq.confirm_dispatch(dispatched, claim_id="c1", now=over)
    etq.save_record(_queue(tmp_path), dispatched, now=over)
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=emitter,
                                   now=over + timedelta(seconds=3600))
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=emitter,
                                   now=over + timedelta(seconds=7200))
    assert [r["event_code"] for r in emitter.recovered] == [
        f"experiment_transition_undispatched.{record['transition_id']}"]


def test_incident_waits_for_the_declared_budget(tmp_path):
    _approved(tmp_path, _terminal(tmp_path, budget=7200.0))
    emitter = FakeEmitter()
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=emitter,
                                   now=NOW + timedelta(seconds=7000))
    assert emitter.observed == []
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=emitter,
                                   now=NOW + timedelta(seconds=7300))
    assert len(emitter.observed) == 1


def test_failed_emission_leaves_the_marker_closed_so_the_next_poll_retries(
        tmp_path):
    _approved(tmp_path, _terminal(tmp_path, budget=60.0))
    failing = FakeEmitter(ok=False)
    result = etq.sweep_transition_incidents(
        _queue(tmp_path), emitter=failing, now=NOW + timedelta(seconds=600))
    assert result["actions"][0]["action"] == \
        "incident_emission_failed_will_retry"
    working = FakeEmitter()
    etq.sweep_transition_incidents(_queue(tmp_path), emitter=working,
                                   now=NOW + timedelta(seconds=900))
    assert len(working.observed) == 1


def test_failed_dispatch_returns_to_untransitioned_with_a_blocker(tmp_path):
    record = _approved(tmp_path, _terminal(tmp_path))
    record = etq.claim_dispatch(record, claim_id="c1", host="omega",
                                chain_id=CHAIN, now=NOW)
    record = etq.fail_dispatch(record, claim_id="c1",
                               reason="gamma refused the plan hash",
                               now=NOW)
    status = etq.transition_status(record, now=NOW)
    assert status["value"] == "completed_untransitioned"
    assert [b["code"] for b in status["blockers"]] == ["DISPATCH_FAILED"]


# ── hygiene: foreign files, typed refusals, CLI ───────────────────────

def test_foreign_files_are_never_coerced_into_transitions(tmp_path):
    _terminal(tmp_path)
    (_queue(tmp_path) / "notes.json").write_text('{"hello": "world"}')
    (_queue(tmp_path) / "broken.json").write_text("{not json")
    view = etq.reconstruct_transitions(_queue(tmp_path), now=NOW)
    assert view["records_total"] == 1
    assert sorted(Path(u["path"]).name for u in view["unreadable"]) == \
        ["broken.json", "notes.json"]


def test_unknown_states_are_typed_refusals(tmp_path):
    record = _terminal(tmp_path)
    with pytest.raises(etq.TransitionRefusal) as materialization:
        etq.set_materialization(record, "almost", now=NOW)
    assert materialization.value.code == \
        "TRANSITION_MATERIALIZATION_STATE_INVALID"
    with pytest.raises(etq.TransitionRefusal) as node:
        etq.mark_node(record, host="omega", state="maybe", now=NOW)
    assert node.value.code == "TRANSITION_NODE_STATE_INVALID"


def test_cli_reconstruct_and_claim_round_trip(tmp_path, capsys):
    record = _approved(tmp_path, _terminal(tmp_path))
    tid = record["transition_id"]
    assert etq.main(["--queue-dir", str(_queue(tmp_path)), "claim", tid,
                     "--claim-id", "c1", "--chain-id", CHAIN]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == \
        "transition_dispatch_in_progress"
    # a second, different claim exits non-zero and prints the refusal
    assert etq.main(["--queue-dir", str(_queue(tmp_path)), "claim", tid,
                     "--claim-id", "c2", "--chain-id", CHAIN]) == 2
    refusal = json.loads(capsys.readouterr().out)
    assert refusal["error_code"] == "TRANSITION_DUPLICATE_DISPATCH"
    assert etq.main(["--queue-dir", str(_queue(tmp_path)),
                     "reconstruct"]) == 0
    view = json.loads(capsys.readouterr().out)
    assert view["schema"] == etq.RECONSTRUCTION_SCHEMA
    assert view["records_total"] == 1
