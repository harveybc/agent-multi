"""WO4 proofs — preserve decision v2, repair identity-blind supervision
(finding AUD-GEN-20260815-250).

Musashi's independent observation, reproduced here BEFORE the fix
(req 6): the active idle guard reported v1 identity ``c0e53cf18b7d60dd``
as ALIVE because process matching keyed on SEED ALONE — a v2 decision
PID with ``--seed 101`` made terminal v1 seed 101 look busy. All
deployed ``p1lr-decision@*.service`` units were inactive and pinned
legacy gate/config paths, so a reboot would not have reconstructed v2
and bounded recovery could have restarted the wrong identity.

Every test is socket-free: process facts, unit facts, telemetry and
ledger emissions are injected; durable queue records live under
``tmp_path``; the only real-file interactions are tmp trees and the
repo's own SHIPPED files (env/drop-in byte-pinned to their generator).
Nothing here touches systemd, nvidia-smi, the network or any live
process — the live v2 run is untouchable by construction.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import experiment_transition_queue as etq  # noqa: E402
from tools import p1lr_identity_supervision as sup  # noqa: E402
from tools import p1lr_idle_guard as guard  # noqa: E402
from tools.l1_fleet_launcher import ExclusiveClaim  # noqa: E402

NOW = datetime(2026, 8, 16, 6, 0, 0, tzinfo=timezone.utc)

# The REAL identities of the finding (Musashi §2/§4).
V1_SCREEN_IDENTITY = "886b776e022d0d7c"
V1_DECISION_IDENTITY = "c0e53cf18b7d60dd"
V2_SCREEN_IDENTITY = "14e7ce8208ac9776"
V2_CHAIN = "cdf30aebf585385b"
V1_EXPERIMENT = "p1_difficulty_lr_factorial_20260811_v1"
V2_EXPERIMENT = "p1_difficulty_lr_factorial_20260815_v2"
V2_DECISION_EXPERIMENT = "p1_difficulty_lr_factorial_20260815_v2_decision"

CELLS = ["P1N_LR1E4", "P1N_LR3E5", "P1E_LR1E4", "P1E_LR3E5"]
ASSIGNMENTS = {
    "101": {"hostname": "omega",
            "gpu_uuid": "GPU-612d1e0c-33de-d5cc-56eb-06c0ae424326"},
    "202": {"hostname": "dragon",
            "gpu_uuid": "GPU-a8bd1b2c-26c4-f3a9-0fc0-fc3dfc6780f9"},
    "303": {"hostname": "gamma",
            "gpu_uuid": "GPU-b77fc3ad-db77-b648-dc15-ec79b65e2519"},
    "404": {"hostname": "gamma",
            "gpu_uuid": "GPU-a9f35631-d36a-6cc6-c23b-eb0b36d50fb8"},
}

REAL_V2_CONTRACT = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                           "p1_difficulty_lr_factorial_v2.json")
SHIPPED_ENV_DIR = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                          "p1lr_env_v2")
SHIPPED_DROPIN = (REPO / "examples/systemd/p1lr-decision@.service.d/"
                         "20-v2-identity.conf")
SHIPPED_CONTROL_MANIFEST = (REPO / "examples/systemd/p1lr-control/"
                                   "CONTROL_MANIFEST.sha256")
INSTALL_SCRIPT = (REPO / "examples/systemd/"
                         "install_p1lr_v2_identity_supervision.sh")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# fixtures: contracts, roots, process facts, durable records
# ---------------------------------------------------------------------------

def _contract(tmp_path, *, experiment, tag, seeds=(101,)):
    return {
        "schema": "agent_multi.p1_difficulty_lr_factorial.v1",
        "experiment": experiment,
        "cells": {c: {} for c in CELLS},
        "seeds": list(seeds),
        "assignments": {str(s): ASSIGNMENTS[str(s)] for s in seeds},
        "cell_order": {str(s): list(CELLS) for s in seeds},
        "output_root": str(tmp_path / f"{tag}_screen"),
        "decision_run": {"output_root": str(tmp_path / f"{tag}_decision")},
    }


def _write_contract(tmp_path, contract, name):
    path = tmp_path / name
    path.write_text(json.dumps(contract, indent=1, sort_keys=True))
    return path


def _terminal_seed(tmp_path, contract, identity, seed, *, mode="decision",
                   age_seconds=7200, now=NOW):
    """All four cell records landed for one seed (a TERMINAL seed)."""
    root = Path(contract["decision_run"]["output_root"] if mode ==
                "decision" else contract["output_root"])
    for cell in CELLS:
        path = root / identity / f"seed{seed}" / cell / "cell_record.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"schema": "agent_multi.p1_difficulty_lr_cell_record.v1",
             "seed": seed, "cell": cell, "mode": mode}))
        stamp = (now - timedelta(seconds=age_seconds)).timestamp()
        os.utime(path, (stamp, stamp))


def _v2_runtime_worktree(tmp_path):
    """A tmp stand-in for the immutable runtime worktree: the REAL v2
    contract bytes at the REAL relative path (so the observed live
    cmdline resolves and hashes exactly as on the fleet)."""
    runtime = tmp_path / "runtime-agent-multi-p1lr-v2-924910fe"
    rel = Path(sup.V2_CONTRACT_RELPATH)
    target = runtime / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(REAL_V2_CONTRACT.read_bytes())
    (runtime / "tools").mkdir(exist_ok=True)
    (runtime / "tools/p1_difficulty_lr_factorial.py").write_text("# stub\n")
    return runtime


def _live_v2_process(tmp_path, seed=101):
    """Musashi's exact observation, as injected process facts: the live
    v2 decision PID (nohup, immutable runtime worktree, relative
    contract path, --seed 101)."""
    runtime = _v2_runtime_worktree(tmp_path)
    return {
        "pid": 731019,
        "cwd": str(runtime),
        "cmdline": [
            "/home/harveybc/anaconda3/envs/trading-stack/bin/python",
            "tools/p1_difficulty_lr_factorial.py",
            "--seed", str(seed),
            "--mode", "decision",
            "--contract", sup.V2_CONTRACT_RELPATH,
            "--screen-gate",
            "/home/harveybc/.local/share/agent-multi/"
            "p1lr_v2_collections_20260815/screen_14e7ce82/"
            "screen_verdict.json",
        ],
    }


def _enrol_v1_dispatched(queue_dir, *, now):
    """The historical v1 record: screen 886b776e terminal, decision
    chain c0e53cf18b7d60dd retro-recorded as dispatched (older)."""
    record = etq.ensure_terminal_record(
        queue_dir, experiment=V1_EXPERIMENT, mode="screen",
        identity=V1_SCREEN_IDENTITY, records_landed=16, cells_total=16,
        now=now, observed_by="test")
    record = etq.approve_successor(
        record, job_id=f"p1lr-v1-decision-{V1_DECISION_IDENTITY}",
        experiment=V1_EXPERIMENT, approved_by="historical",
        chain_id=V1_DECISION_IDENTITY, now=now)
    record = etq.set_materialization(record, "materialized", now=now)
    record = etq.claim_dispatch(record, claim_id="historical-v1-decision",
                                host="omega", chain_id=V1_DECISION_IDENTITY,
                                now=now)
    record = etq.confirm_dispatch(record, claim_id="historical-v1-decision",
                                  now=now)
    etq.save_record(queue_dir, record, now=now)
    return record


def _enrol_v2_dispatched(queue_dir, *, now, contract_path,
                         evidence_root=None):
    """The v2 record replicating durable record 15cbfec7ac8bbf66:
    screen 14e7ce82 terminal 16/16 -> decision chain cdf30aebf585385b
    DISPATCHED (newest)."""
    evidence_root = evidence_root or str(
        Path.home() / ".local/share/agent-multi/"
                      "p1lr_v2_collections_20260815/screen_14e7ce82")
    record = etq.ensure_terminal_record(
        queue_dir, experiment=V2_EXPERIMENT, mode="screen",
        identity=V2_SCREEN_IDENTITY, records_landed=16, cells_total=16,
        evidence_root=evidence_root,
        contract_path=str(contract_path),
        contract_sha256=_sha(Path(contract_path).read_bytes()),
        now=now, observed_by="test")
    record = etq.approve_successor(
        record, job_id=f"p1lr-v2-decision-{V2_CHAIN}",
        experiment=V2_DECISION_EXPERIMENT,
        contract_path=str(contract_path),
        contract_sha256=_sha(Path(contract_path).read_bytes()),
        approved_by="MUSASHI order 2026-08-15 §8.5",
        chain_id=V2_CHAIN, now=now)
    record = etq.set_materialization(record, "materialized", now=now)
    record = etq.claim_dispatch(record,
                                claim_id="v2-decision-dispatch-20260815",
                                host="omega", chain_id=V2_CHAIN, now=now)
    record = etq.confirm_dispatch(
        record, claim_id="v2-decision-dispatch-20260815", now=now)
    etq.save_record(queue_dir, record, now=now)
    return record


class FakeEmitter:
    def __init__(self):
        self.observed = []
        self.recovered = []

    def observe(self, event_code, severity, summary, payload,
                affected_object="-"):
        self.observed.append(event_code)
        return True

    def recover(self, event_code, evidence, affected_object="-"):
        self.recovered.append(event_code)
        return True


def _poll_v1(tmp_path, *, process_facts=None, process_alive_fn=None,
             queue_dir=None, enforce=True, restart_calls=None,
             utilization=0):
    """A guard cycle over the TERMINAL v1 decision identity on omega —
    the exact configuration of Musashi's observation."""
    contract = _contract(tmp_path, experiment=V1_EXPERIMENT, tag="v1")
    contract_path = _write_contract(tmp_path, contract, "v1_contract.json")
    _terminal_seed(tmp_path, contract, V1_DECISION_IDENTITY, 101)
    emitter = FakeEmitter()
    restart_calls = restart_calls if restart_calls is not None else []

    def restart_fn(unit):
        restart_calls.append(unit)
        return {"ok": True, "returncode": 0, "stderr": ""}

    report = guard.poll(
        contract=contract, identity=V1_DECISION_IDENTITY,
        state=guard.default_state(), now=NOW, local_hostname="omega",
        emitter=emitter,
        process_alive_fn=process_alive_fn or (lambda seed: False),
        process_facts_fn=(None if process_facts is None
                          else lambda: process_facts),
        expected_contract_sha256=_sha(contract_path.read_bytes()),
        gpu_telemetry_fn=lambda uuid: {"gpu_utilization_pct": utilization,
                                       "gpu_temperature_c": 40},
        unit_exists_fn=lambda unit: True,
        restart_fn=restart_fn, mode="decision",
        transition_queue_dir=queue_dir,
        transition_emitter=emitter,
        enforce_transition_authority=enforce)
    return report, emitter, restart_calls


# ═══════════════════════════════════════════════════════════════════════
# REQ 6 — BEFORE: reproduce Musashi's identity-blind observation; AFTER:
# a v2 PID can never make terminal v1 look alive.
# ═══════════════════════════════════════════════════════════════════════

def test_req6_before_legacy_seed_only_matching_reports_terminal_v1_alive(
        tmp_path):
    """The retired matcher keyed on SEED ALONE: the live v2 decision
    cmdline satisfies the v1 guard's pattern, and a poll wired that way
    renders terminal v1 seed 101 ``busy_process_alive`` — the defect
    exactly as the auditor observed it."""
    v2_proc = _live_v2_process(tmp_path)
    pattern = guard.legacy_seed_only_pattern(101)
    joined = " ".join(v2_proc["cmdline"])
    # The seed-only pattern MATCHES the v2 process: this is the bug.
    assert re.search(pattern, joined), (
        "the legacy pattern no longer reproduces the defect basis")

    def legacy_alive(seed):
        return bool(re.search(guard.legacy_seed_only_pattern(seed), joined))

    report, _, _ = _poll_v1(tmp_path, process_alive_fn=legacy_alive)
    entry = report["seeds"]["101"]
    assert entry["process_alive"] is True            # the false ALIVE
    assert entry["idle_class"] == "busy_process_alive"
    assert entry["idle"] is False                    # fleet idle time hidden


def test_req6_after_v2_pid_cannot_make_terminal_v1_look_alive(tmp_path):
    """Identity-aware matching over the SAME injected process facts: the
    v2 PID is typed foreign (contract identity mismatch), terminal v1
    renders completed_untransitioned, and no restart targets v1."""
    v2_proc = _live_v2_process(tmp_path)
    report, emitter, restarts = _poll_v1(
        tmp_path, process_facts=[v2_proc], queue_dir=None)
    entry = report["seeds"]["101"]
    assert entry["process_alive"] is False
    assert entry["idle_class"] == "completed_untransitioned"
    assert entry["idle"] is True                     # idle time SURFACES
    match = entry["process_match"]
    assert match["matched"] == []
    assert len(match["foreign_same_seed"]) == 1
    foreign = match["foreign_same_seed"][0]
    assert foreign["pid"] == 731019
    assert "contract_identity_mismatch" in foreign["mismatch"]
    assert foreign["contract_sha256"].startswith("f5544a5f")
    assert restarts == []                            # terminal, never restarted


def test_req6_positive_control_v2_guard_still_sees_its_own_worker(tmp_path):
    """The fix must not blind the guard to its OWN workers: the v2 guard
    (v2 contract identity) proves the same PID alive."""
    v2_proc = _live_v2_process(tmp_path)
    contract = json.loads(REAL_V2_CONTRACT.read_text())
    binding = {"mode": "decision",
               "output_root": str(contract["decision_run"]["output_root"])}
    result = guard.match_seed_processes(
        [v2_proc], seed=101, binding=binding,
        expected_contract_sha256=_sha(REAL_V2_CONTRACT.read_bytes()))
    assert result["alive"] is True
    assert result["matched"][0]["pid"] == 731019
    assert result["foreign_same_seed"] == []


# ═══════════════════════════════════════════════════════════════════════
# REQ 2 — matching requires contract identity + mode + seed + output
# root; NEVER seed alone.
# ═══════════════════════════════════════════════════════════════════════

def test_req2_match_requires_contract_mode_seed_never_seed_alone(tmp_path):
    v2_proc = _live_v2_process(tmp_path)
    v2_sha = _sha(REAL_V2_CONTRACT.read_bytes())
    contract = json.loads(REAL_V2_CONTRACT.read_text())
    binding = {"mode": "decision",
               "output_root": str(contract["decision_run"]["output_root"])}

    # seed mismatch: not even a candidate
    result = guard.match_seed_processes(
        [v2_proc], seed=202, binding=binding,
        expected_contract_sha256=v2_sha)
    assert result["alive"] is False
    assert (result["matched"] == result["foreign_same_seed"]
            == result["unprovable_same_seed"] == [])

    # mode mismatch: typed foreign, never alive
    screen_binding = dict(binding, mode="screen")
    result = guard.match_seed_processes(
        [v2_proc], seed=101, binding=screen_binding,
        expected_contract_sha256=v2_sha)
    assert result["alive"] is False
    assert "mode_mismatch" in result["foreign_same_seed"][0]["mismatch"]

    # contract identity mismatch: typed foreign, never alive
    result = guard.match_seed_processes(
        [v2_proc], seed=101, binding=binding,
        expected_contract_sha256="0" * 64)
    assert result["alive"] is False
    assert "contract_identity_mismatch" in \
        result["foreign_same_seed"][0]["mismatch"]

    # unreadable contract: UNPROVEN is not mine — typed, never alive
    ghost = dict(v2_proc, cmdline=list(v2_proc["cmdline"]))
    idx = ghost["cmdline"].index("--contract") + 1
    ghost["cmdline"][idx] = "examples/config/does_not_exist.json"
    result = guard.match_seed_processes(
        [ghost], seed=101, binding=binding,
        expected_contract_sha256=v2_sha)
    assert result["alive"] is False
    assert "contract_unprovable" in \
        result["unprovable_same_seed"][0]["mismatch"]

    # full identity: alive, with the output root bound into the match
    result = guard.match_seed_processes(
        [v2_proc], seed=101, binding=binding,
        expected_contract_sha256=v2_sha)
    assert result["alive"] is True
    assert result["matched"][0]["output_root"] == str(
        Path(binding["output_root"]).expanduser())


def test_req2_poll_requires_the_contract_identity_with_process_facts(
        tmp_path):
    """process_facts without the guarded contract sha is a programming
    error, refused loudly — matching can never degrade to seed-only."""
    contract = _contract(tmp_path, experiment=V1_EXPERIMENT, tag="v1")
    with pytest.raises(ValueError, match="never seed alone"):
        guard.poll(
            contract=contract, identity=V1_DECISION_IDENTITY,
            state=guard.default_state(), now=NOW, local_hostname="omega",
            emitter=FakeEmitter(), process_alive_fn=lambda seed: False,
            process_facts_fn=lambda: [],
            expected_contract_sha256=None,
            gpu_telemetry_fn=lambda uuid: {},
            unit_exists_fn=lambda unit: True,
            restart_fn=lambda unit: {"ok": True}, mode="decision")


def test_req2_bash_wrappers_and_ssh_lines_never_parse_as_runners():
    """The nohup bash/ssh wrapper lines that carry the runner string
    inside ONE argv element are not runners: the script name must be its
    own argv element."""
    wrapper = ["/bin/bash", "-c",
               "cd $W && nohup setsid env CUDA_VISIBLE_DEVICES=GPU-x "
               "python tools/p1_difficulty_lr_factorial.py --seed 101 "
               "--mode decision > log 2>&1 &"]
    assert guard.parse_runner_cmdline(wrapper) is None
    assert guard.parse_runner_cmdline(None) is None
    assert guard.parse_runner_cmdline([]) is None


# ═══════════════════════════════════════════════════════════════════════
# REQ 3 — the guard discovers the ACTIVE durable transition and refuses
# a conflicting identity (typed refusal, no silent adoption).
# ═══════════════════════════════════════════════════════════════════════

def _queue_with_both_records(tmp_path):
    queue_dir = tmp_path / "queue"
    contract_path = tmp_path / "v2_contract.json"
    contract_path.write_bytes(REAL_V2_CONTRACT.read_bytes())
    _enrol_v1_dispatched(queue_dir, now=NOW - timedelta(days=3))
    _enrol_v2_dispatched(queue_dir, now=NOW - timedelta(hours=5),
                         contract_path=contract_path)
    return queue_dir, contract_path


def test_req3_guard_refuses_v1_identity_against_active_v2_transition(
        tmp_path):
    queue_dir, _ = _queue_with_both_records(tmp_path)
    report, emitter, restarts = _poll_v1(tmp_path, process_facts=[],
                                         queue_dir=queue_dir)
    refusal = report["refusal"]
    assert refusal["error_code"] == \
        "P1LR_GUARD_IDENTITY_CONFLICTS_ACTIVE_TRANSITION"
    # v1's own historical dispatched record does NOT save it: the ruling
    # authority is the NEWEST dispatch (the v2 chain).
    assert report["transition_authority"]["value"] == \
        "identity_is_superseded_older_chain"
    assert refusal["guarded_identity"] == V1_DECISION_IDENTITY
    ruling = report["transition_authority"]["authorities"][0]
    assert ruling["chain_id"] == V2_CHAIN and ruling["ruling"] is True
    # no silent adoption, no incident, no restart, no seed actions
    assert report["seeds"] == {}
    assert report["identity"] == V1_DECISION_IDENTITY
    assert emitter.observed == [] and restarts == []
    assert V2_CHAIN in refusal["corrective_command"]


def test_req3_guard_passes_the_active_chain_and_its_predecessor(tmp_path):
    queue_dir, contract_path = _queue_with_both_records(tmp_path)
    records, _ = etq.load_records(queue_dir)
    # the active v2 chain passes
    verdict = guard.guard_transition_authority(
        records, identity=V2_CHAIN, experiment=V2_DECISION_EXPERIMENT,
        now=NOW)
    assert verdict["conflict"] is False
    assert verdict["value"] == "identity_is_active_chain"
    # the terminal v2 screen (the predecessor) passes
    verdict = guard.guard_transition_authority(
        records, identity=V2_SCREEN_IDENTITY, experiment=V2_EXPERIMENT,
        now=NOW)
    assert verdict["conflict"] is False
    assert verdict["value"] == \
        "identity_is_terminal_predecessor_of_active_chain"
    # an unrelated family sees no authority at all
    verdict = guard.guard_transition_authority(
        records, identity="feedcafe00000000",
        experiment="l1_matched_factorial_20260807", now=NOW)
    assert verdict["conflict"] is False
    assert verdict["value"] == "no_active_authority_for_family"


# ═══════════════════════════════════════════════════════════════════════
# REQ 4 — ONE writer per seed under concurrent timer + manual
# activation (lock/lease).
# ═══════════════════════════════════════════════════════════════════════

def test_req4_one_writer_per_seed_under_concurrent_activation(tmp_path):
    """Timer-activated and manually-activated starts converge on the
    runner's per-seed/cell flock: the second acquirer loses and reads
    the holder's identity (exit-class ALREADY_RUNNING, clean no-op)."""
    lock = tmp_path / "locks" / "exclusive_claim.seed101.P1N_LR3E5.lock"
    timer_activation = ExclusiveClaim(lock)
    manual_activation = ExclusiveClaim(lock)
    assert timer_activation.acquire() is True
    assert manual_activation.acquire() is False       # ONE writer
    holder = manual_activation.holder()
    assert holder["pid"] == os.getpid()               # loser sees the winner
    timer_activation.release()
    assert manual_activation.acquire() is True        # boundary reached
    manual_activation.release()


def test_req4_one_writer_across_processes(tmp_path):
    """The flock holds across PROCESS boundaries: while a child process
    holds the seed lock, this process cannot acquire it."""
    lock = tmp_path / "locks" / "exclusive_claim.seed101.P1E_LR3E5.lock"
    lock.parent.mkdir(parents=True)
    child = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import fcntl, sys, time
            fd = open({str(lock)!r}, "w")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            print("held", flush=True)
            time.sleep(30)
        """)], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "held"
        assert ExclusiveClaim(lock).acquire() is False
    finally:
        child.kill()
        child.wait()
    assert ExclusiveClaim(lock).acquire() is True


def test_req4_dispatch_lease_is_single_writer_too(tmp_path):
    """The fleet-level analogue: a second, different dispatch claim
    against the durable record fails closed while the first is live."""
    queue_dir = tmp_path / "queue"
    contract_path = tmp_path / "v2_contract.json"
    contract_path.write_bytes(REAL_V2_CONTRACT.read_bytes())
    record = etq.ensure_terminal_record(
        queue_dir, experiment=V2_EXPERIMENT, mode="screen",
        identity=V2_SCREEN_IDENTITY, records_landed=16, cells_total=16,
        now=NOW)
    record = etq.approve_successor(
        record, job_id=f"p1lr-v2-decision-{V2_CHAIN}",
        approved_by="test", chain_id=V2_CHAIN, now=NOW)
    record = etq.set_materialization(record, "materialized", now=NOW)
    record = etq.claim_dispatch(record, claim_id="timer-activation",
                                host="omega", chain_id=V2_CHAIN, now=NOW)
    with pytest.raises(etq.TransitionRefusal) as exc:
        etq.claim_dispatch(record, claim_id="manual-activation",
                           host="omega", chain_id=V2_CHAIN,
                           now=NOW + timedelta(seconds=30))
    assert exc.value.code == "TRANSITION_DUPLICATE_DISPATCH"


# ═══════════════════════════════════════════════════════════════════════
# REQ 5 — reboot reconstruction from the durable record + shipped
# files ALONE (unit-generation logic, not systemd).
# ═══════════════════════════════════════════════════════════════════════

def test_req5_reboot_plan_reconstructs_the_v2_unit_graph(tmp_path):
    queue_dir, contract_path = _queue_with_both_records(tmp_path)
    plan = sup.plan_reboot_reconstruction(queue_dir, contract_path,
                                          now=NOW)
    assert plan["chain_id"] == V2_CHAIN
    assert plan["contract_sha256"].startswith("f5544a5f")
    assert plan["experiment"] == V2_DECISION_EXPERIMENT
    assert {h: [u["unit"] for u in us]
            for h, us in plan["hosts"].items()} == {
        "omega": ["p1lr-decision@101.service"],
        "dragon": ["p1lr-decision@202.service"],
        "gamma": ["p1lr-decision@303.service",
                  "p1lr-decision@404.service"],
    }
    assert plan["unit_total"] == 4
    seed101 = plan["hosts"]["omega"][0]
    assert seed101["gpu_uuid"] == ASSIGNMENTS["101"]["gpu_uuid"]
    assert seed101["expected"]["chain_id"] == V2_CHAIN
    assert seed101["expected"]["screen_gate"].endswith(
        "screen_14e7ce82/screen_verdict.json")
    assert seed101["expected"]["working_directory"] == \
        sup.DEFAULT_RUNTIME_DIR_SPEC
    assert "durable transition records" in plan["basis"]
    # v1 is visibly SUPERSEDED, never part of the reconstruction
    assert [a["chain_id"] for a in plan["superseded_authorities"]] == \
        [V1_DECISION_IDENTITY]
    # per-host narrowing (what one booting host would derive)
    gamma = sup.plan_reboot_reconstruction(queue_dir, contract_path,
                                           host="gamma", now=NOW)
    assert list(gamma["hosts"]) == ["gamma"]
    assert gamma["unit_total"] == 2
    # deploy commands are PRINTED text bound to the safe boundary
    commands = "\n".join(sup.deploy_commands(plan))
    assert "NEXT SAFE PROCESS BOUNDARY ONLY" in commands
    assert "systemctl --user start p1lr-decision@101.service" in commands


def test_req5_reboot_plan_refuses_unproven_reconstruction(tmp_path):
    contract_path = tmp_path / "v2_contract.json"
    contract_path.write_bytes(REAL_V2_CONTRACT.read_bytes())
    empty_queue = tmp_path / "empty-queue"
    empty_queue.mkdir()
    with pytest.raises(sup.SupervisionRefusal) as exc:
        sup.plan_reboot_reconstruction(empty_queue, contract_path, now=NOW)
    assert exc.value.code == "SUPERVISION_NO_ACTIVE_AUTHORITY"

    queue_dir, _ = _queue_with_both_records(tmp_path)
    tampered = tmp_path / "tampered_contract.json"
    tampered.write_text(REAL_V2_CONTRACT.read_text() + "\n")
    with pytest.raises(sup.SupervisionRefusal) as exc:
        sup.plan_reboot_reconstruction(queue_dir, tampered, now=NOW)
    assert exc.value.code == "SUPERVISION_CONTRACT_SHA_MISMATCH"


# ═══════════════════════════════════════════════════════════════════════
# REQ 7 — an old v1 unit cannot restart while v2 owns the lease.
# ═══════════════════════════════════════════════════════════════════════

def test_req7_guard_never_restarts_v1_while_v2_owns_the_lease(tmp_path):
    """Even a v1 seed that LOOKS stalled (pending cells, idle GPU, aged
    heartbeat) triggers no restart while the durable queue says the v2
    chain owns the family: the authority refusal precedes all recovery
    machinery."""
    queue_dir, _ = _queue_with_both_records(tmp_path)
    contract = _contract(tmp_path, experiment=V1_EXPERIMENT, tag="v1")
    # pending, aged: only 1 of 4 records — a classic §7.8 stall picture
    root = Path(contract["decision_run"]["output_root"])
    path = (root / V1_DECISION_IDENTITY / "seed101" / CELLS[0]
            / "cell_record.json")
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    stamp = (NOW - timedelta(seconds=7200)).timestamp()
    os.utime(path, (stamp, stamp))
    emitter = FakeEmitter()
    restart_calls = []
    report = guard.poll(
        contract=contract, identity=V1_DECISION_IDENTITY,
        state=guard.default_state(), now=NOW, local_hostname="omega",
        emitter=emitter, process_alive_fn=lambda seed: False,
        process_facts_fn=lambda: [],
        expected_contract_sha256="1" * 64,
        gpu_telemetry_fn=lambda uuid: {"gpu_utilization_pct": 0,
                                       "gpu_temperature_c": 35},
        unit_exists_fn=lambda unit: True,
        restart_fn=lambda unit: restart_calls.append(unit) or {"ok": True},
        mode="decision", transition_queue_dir=queue_dir,
        transition_emitter=emitter)
    assert report["refusal"]["error_code"] == \
        "P1LR_GUARD_IDENTITY_CONFLICTS_ACTIVE_TRANSITION"
    assert restart_calls == [] and emitter.observed == []


def test_req7_lease_gate_refuses_old_v1_unit_and_passes_v2(tmp_path):
    queue_dir, _ = _queue_with_both_records(tmp_path)
    # the old v1 unit (its env declares the v1 chain): REFUSED
    verdict = sup.lease_gate_verdict(
        queue_dir, expected_chain_id=V1_DECISION_IDENTITY,
        experiment=V1_EXPERIMENT, mode="decision", now=NOW)
    assert verdict["verdict"] == "REFUSED_LEASE_HELD_BY_OTHER_CHAIN"
    assert verdict["passed"] is False
    assert verdict["ruling_authority"]["chain_id"] == V2_CHAIN
    # a legacy unit with NO declared chain (empty env expansion): REFUSED
    verdict = sup.lease_gate_verdict(
        queue_dir, expected_chain_id="",
        experiment=V1_EXPERIMENT, mode="decision", now=NOW)
    assert verdict["verdict"] == "REFUSED_UNIT_IDENTITY_UNDECLARED"
    # the v2 unit: PASS
    verdict = sup.lease_gate_verdict(
        queue_dir, expected_chain_id=V2_CHAIN,
        experiment=V2_DECISION_EXPERIMENT, mode="decision", now=NOW)
    assert verdict["verdict"] == "PASS" and verdict["passed"] is True
    # no durable authority at all: nothing authorizes a start
    empty = tmp_path / "empty-queue"
    empty.mkdir()
    verdict = sup.lease_gate_verdict(
        empty, expected_chain_id=V2_CHAIN,
        experiment=V2_DECISION_EXPERIMENT, mode="decision", now=NOW)
    assert verdict["verdict"] == "REFUSED_NO_DURABLE_AUTHORITY"


def test_req7_lease_gate_cli_exit_codes(tmp_path):
    """ExecStartPre semantics: PASS exit 0; refusal exit 4 — the class
    RestartPreventExitStatus=4 never retries."""
    queue_dir, _ = _queue_with_both_records(tmp_path)
    assert sup.main(["--queue-dir", str(queue_dir), "lease-gate",
                     "--expected-chain-id", V2_CHAIN,
                     "--experiment", V2_DECISION_EXPERIMENT,
                     "--mode", "decision"]) == 0
    assert sup.main(["--queue-dir", str(queue_dir), "lease-gate",
                     "--expected-chain-id", V1_DECISION_IDENTITY,
                     "--experiment", V1_EXPERIMENT,
                     "--mode", "decision"]) == 4


# ═══════════════════════════════════════════════════════════════════════
# REQ 1 — shipped identity-specific env files + drop-in + install
# script that ENABLES NOTHING.
# ═══════════════════════════════════════════════════════════════════════

def test_req1_shipped_env_files_are_byte_pinned_to_the_generator(tmp_path):
    """The committed seed identity files, the control manifest and the
    drop-in are EXACTLY what the generator produces from the real v2
    contract + a durable record bound to it: no hand-edit can drift them
    silently, and — finding AUD-GEN-20260816-256 — they EXIST in git."""
    queue_dir = tmp_path / "queue"
    contract_path = tmp_path / "v2_contract.json"
    contract_path.write_bytes(REAL_V2_CONTRACT.read_bytes())
    record = _enrol_v2_dispatched(queue_dir, now=NOW,
                                  contract_path=contract_path)
    contract = json.loads(REAL_V2_CONTRACT.read_text())
    outdir = tmp_path / "generated"
    manifest = sup.materialize_supervision(
        contract, record, contract_path=contract_path, outdir=outdir)
    assert manifest["chain_id"] == V2_CHAIN
    assert manifest["contract_sha256"].startswith("f5544a5f")
    for seed in (101, 202, 303, 404):
        name = f"seed{seed}{sup.SEED_ENV_SUFFIX}"
        generated = (outdir / sup.SEED_ENV_DIR_RELPATH / name).read_bytes()
        shipped = (SHIPPED_ENV_DIR / name).read_bytes()
        assert generated == shipped, f"{name} drifted"
    generated = (outdir / sup.CONTROL_BUNDLE_RELPATH /
                 sup.CONTROL_MANIFEST_NAME).read_bytes()
    assert generated == SHIPPED_CONTROL_MANIFEST.read_bytes(), \
        "control manifest drifted"
    generated = (outdir / "examples/systemd/p1lr-decision@.service.d/"
                          "20-v2-identity.conf").read_bytes()
    assert generated == SHIPPED_DROPIN.read_bytes(), "drop-in drifted"


def test_req1_shipped_identity_files_escape_the_credential_ignore_rules():
    """Finding AUD-GEN-20260816-256: the payload used to be ``seed*.env``,
    which the repository's ``*.env`` credential rule swallowed, so the
    branch shipped nothing. The suffix must stay OUTSIDE those rules and
    the files must be TRACKED, not merely present in a working tree."""
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"],
                            cwd=REPO, capture_output=True, text=True)
    if inside.returncode != 0:
        pytest.skip("not a git worktree: repository tracking is unprovable "
                    "here, and an unprovable claim is not a passing one")
    assert sup.SEED_ENV_SUFFIX == ".env.conf"
    names = sorted(p.name for p in SHIPPED_ENV_DIR.iterdir())
    assert names == [f"seed{s}{sup.SEED_ENV_SUFFIX}"
                     for s in (101, 202, 303, 404)]
    for name in names:
        ignored = subprocess.run(
            ["git", "check-ignore", "-q",
             str((SHIPPED_ENV_DIR / name).relative_to(REPO))],
            cwd=REPO, capture_output=True)
        assert ignored.returncode != 0, f"{name} is git-ignored again"
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--",
         str(SHIPPED_ENV_DIR.relative_to(REPO)),
         str(SHIPPED_CONTROL_MANIFEST.relative_to(REPO))],
        cwd=REPO, capture_output=True, text=True)
    assert tracked.returncode == 0, tracked.stderr
    listed = sorted(Path(p).name for p in tracked.stdout.split())
    assert listed == sorted(names + [sup.CONTROL_MANIFEST_NAME])
    # and they carry no credential-shaped value
    for name in names:
        text = (SHIPPED_ENV_DIR / name).read_text().lower()
        for forbidden in ("token", "secret", "password", "api_key",
                          "private_key", "passwd"):
            assert forbidden not in text, f"{name} mentions {forbidden}"


def test_req1_env_files_carry_the_full_v2_identity():
    for seed, assignment in ASSIGNMENTS.items():
        text = (SHIPPED_ENV_DIR /
                f"seed{seed}{sup.SEED_ENV_SUFFIX}").read_text()
        values = dict(line.split("=", 1) for line in text.splitlines()
                      if line and not line.startswith("#"))
        assert values["P1LR_SEED"] == seed
        assert values["P1LR_HOST"] == assignment["hostname"]
        assert values["CUDA_VISIBLE_DEVICES"] == assignment["gpu_uuid"]
        assert values["P1LR_MODE"] == "decision"
        assert values["P1LR_CONTRACT"] == sup.V2_CONTRACT_RELPATH
        assert values["P1LR_CONTRACT_SHA256"] == _sha(
            REAL_V2_CONTRACT.read_bytes())
        assert values["P1LR_EXPECTED_CHAIN_ID"] == V2_CHAIN
        assert values["P1LR_EXPERIMENT"] == V2_DECISION_EXPERIMENT
        assert values["P1LR_OUTPUT_ROOT"].endswith(
            "p1_difficulty_lr_factorial_20260815_v2_decision")


def test_req1_dropin_pins_worktree_gate_and_lease_gate():
    text = SHIPPED_DROPIN.read_text()
    effective = [line for line in text.splitlines()
                 if line and not line.startswith("#")]
    assert ("WorkingDirectory=%h/Documents/GitHub/.runtime/"
            "agent-multi-p1lr-v2-924910fe") in effective
    assert ("Environment=P1LR_SCREEN_GATE=%h/.local/share/agent-multi/"
            "p1lr_v2_collections_20260815/screen_14e7ce82/"
            "screen_verdict.json") in effective
    # resets, then identity-specific env
    assert "EnvironmentFile=" in effective and "ExecStartPre=" in effective
    assert "ExecStart=" in effective
    assert any(f"p1lr-v2/seed%i{sup.SEED_ENV_SUFFIX}" in line
               for line in effective)
    # both gates precede the runner
    joined = "\n".join(effective)
    assert "p1lr_decision_gate_check.sh ${P1LR_SCREEN_GATE} " \
           "${P1LR_CONTRACT}" in joined
    assert "p1lr_identity_supervision.py lease-gate" in joined
    assert joined.index("lease-gate") < joined.index("ExecStart=%h")
    # the runner line is the v2 invocation, mode pinned literally
    assert ("--seed %i --mode decision --contract ${P1LR_CONTRACT} "
            "--screen-gate ${P1LR_SCREEN_GATE}") in joined


def test_req1_install_script_enables_and_starts_nothing():
    """Every systemctl invocation the script EXECUTES is daemon-reload;
    enable/start appear only inside echo'd operator guidance."""
    lines = INSTALL_SCRIPT.read_text().splitlines()
    executed = [ln.strip() for ln in lines
                if ln.strip() and not ln.strip().startswith("#")
                and not ln.strip().startswith("echo")]
    for line in executed:
        if "systemctl" in line:
            assert "daemon-reload" in line, (
                f"install script executes a systemctl mutation: {line!r}")
        assert "enable --now" not in line
        assert not re.search(r"systemctl --user (start|restart|enable)\b",
                             line)
    # and the guidance it prints names the boundary rule
    text = INSTALL_SCRIPT.read_text()
    assert "NEXT SAFE BOUNDARY" in text
    assert "NEVER while the seed's matching v2 PID is alive" in text


# ═══════════════════════════════════════════════════════════════════════
# FINDING AUD-GEN-20260816-261 — the restart admission gate must not be
# mutable canonical-checkout code.
# ═══════════════════════════════════════════════════════════════════════

def _dropin_pinned_manifest_sha() -> str:
    match = re.search(r"p1lr-control/([0-9a-f]{64})/",
                      SHIPPED_DROPIN.read_text())
    assert match, "the drop-in pins no control-manifest digest"
    return match.group(1)


def test_261_dropin_never_executes_the_mutable_canonical_checkout():
    """The BEFORE state: ExecStartPre ran
    ``%h/Documents/GitHub/agent-multi/tools/p1lr_identity_supervision.py``
    — a ``git pull`` could change restart admission. AFTER: no effective
    line references the canonical checkout at all."""
    effective = [ln for ln in SHIPPED_DROPIN.read_text().splitlines()
                 if ln and not ln.startswith("#")]
    for line in effective:
        assert sup.DEFAULT_CANONICAL_REPO_SPEC not in line, (
            f"the unit still executes mutable checkout code: {line!r}")
    gate_lines = [ln for ln in effective if "lease-gate" in ln]
    assert len(gate_lines) == 1
    pinned = _dropin_pinned_manifest_sha()
    assert f"{sup.CONTROL_INSTALL_ROOT_SPEC}/{pinned}/tools/" \
           "p1lr_identity_supervision.py" in gate_lines[0]


def test_261_pinned_digest_binds_the_shipped_manifest_and_its_members():
    """Two-level binding: the unit's literal == sha256(manifest), and
    every manifest line == sha256(the reviewed repo module)."""
    pinned = _dropin_pinned_manifest_sha()
    assert pinned == _sha(SHIPPED_CONTROL_MANIFEST.read_bytes())
    assert pinned == sup.control_manifest_sha256(REPO)
    lines = SHIPPED_CONTROL_MANIFEST.read_text().splitlines()
    members = dict(reversed(ln.split("  ", 1)) for ln in lines if ln)
    assert sorted(members) == sorted(sup.CONTROL_BUNDLE_MEMBERS)
    for member, digest in members.items():
        assert digest == _sha((REPO / member).read_bytes()), member
    # cheap early signal: no NEW top-level non-stdlib dependency
    for member in members:
        for line in (REPO / member).read_text().splitlines():
            match = re.match(r"^from (\S+) import|^import (\S+)", line)
            if not match:
                continue
            module = (match.group(1) or match.group(2)).split(".")[0]
            if module in {"tools", "__future__"}:
                continue
            assert module in sys.stdlib_module_names, (
                f"{member} imports non-stdlib {module!r}: the control "
                "bundle would be incomplete")


def test_261_the_installed_bundle_runs_the_gate_with_no_repo_on_sys_path(
        tmp_path):
    """The regex scan above cannot see a lazy import inside a function.
    This does: the gate is EXECUTED from the isolated bundle, with the
    repository unreachable (cwd ``/``, empty PYTHONPATH). A member the
    manifest forgot would surface as ImportError, not as a silent
    dependency on the mutable checkout."""
    control = _install_control_bundle(tmp_path / "p1lr-control")
    queue_dir, _ = _queue_with_both_records(tmp_path)
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = ""
    def run(chain, experiment):
        return subprocess.run(
            [sys.executable,
             str(control / "tools/p1lr_identity_supervision.py"),
             "--queue-dir", str(queue_dir), "lease-gate",
             "--expected-chain-id", chain, "--experiment", experiment,
             "--mode", "decision"],
            cwd="/", env=env, capture_output=True, text=True, timeout=120)
    ok = run(V2_CHAIN, V2_DECISION_EXPERIMENT)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["verdict"] == "PASS"
    refused = run(V1_DECISION_IDENTITY, V1_EXPERIMENT)
    assert refused.returncode == 4, refused.stderr
    assert json.loads(refused.stdout)["verdict"] == \
        "REFUSED_LEASE_HELD_BY_OTHER_CHAIN"
    # and the isolation is real: nothing resolved through the checkout
    assert str(REPO) not in ok.stdout + ok.stderr


def _install_control_bundle(dest: Path) -> Path:
    """Reproduce what the installer lays down, in a tmp tree."""
    control = dest / _dropin_pinned_manifest_sha()
    (control / "tools").mkdir(parents=True)
    (control / sup.CONTROL_MANIFEST_NAME).write_bytes(
        SHIPPED_CONTROL_MANIFEST.read_bytes())
    for member in sup.CONTROL_BUNDLE_MEMBERS:
        (control / member).write_bytes((REPO / member).read_bytes())
    return control


def test_261_verification_fails_closed_on_every_tamper(tmp_path):
    pinned = _dropin_pinned_manifest_sha()
    control = _install_control_bundle(tmp_path / "p1lr-control")
    assert sup.verify_control_bundle(control, pinned)["verdict"] == "PASS"

    # absent bundle
    verdict = sup.verify_control_bundle(tmp_path / "nowhere", pinned)
    assert verdict["verdict"] == "REFUSED_CONTROL_MANIFEST_ABSENT"
    assert verdict["passed"] is False
    # a drifted MEMBER (the classic "someone edited the gate in place")
    member = control / "tools/p1lr_identity_supervision.py"
    original = member.read_bytes()
    member.write_bytes(original + b"\n# silently widened admission\n")
    verdict = sup.verify_control_bundle(control, pinned)
    assert verdict["verdict"] == "REFUSED_CONTROL_MEMBER_DRIFT"
    assert verdict["passed"] is False
    member.write_bytes(original)
    # a drifted MANIFEST that "covers" the tampered member: the unit's
    # own literal digest still refuses it
    member.write_bytes(original + b"\n# widened\n")
    (control / sup.CONTROL_MANIFEST_NAME).write_text(
        sup.control_manifest_content(control))
    verdict = sup.verify_control_bundle(control, pinned)
    assert verdict["verdict"] == "REFUSED_CONTROL_MANIFEST_DRIFT"
    assert verdict["passed"] is False
    # a missing member
    member.unlink()
    (control / sup.CONTROL_MANIFEST_NAME).write_bytes(
        SHIPPED_CONTROL_MANIFEST.read_bytes())
    assert sup.verify_control_bundle(control, pinned)["verdict"] == \
        "REFUSED_CONTROL_MEMBER_ABSENT"


def test_261_the_units_own_shell_verification_command_really_works(tmp_path):
    """Not a paraphrase: the EXACT ExecStartPre string from the shipped
    drop-in is run, with ``%h`` expanded as systemd would, against a real
    installed bundle. It must exit 0 clean and 4 on drift."""
    home = tmp_path / "home"
    control_root = home / ".local/lib/agent-multi/p1lr-control"
    control_root.mkdir(parents=True)
    control = _install_control_bundle(control_root)
    line = next(ln for ln in SHIPPED_DROPIN.read_text().splitlines()
                if ln.startswith("ExecStartPre=/bin/sh -c "))
    command = line[len("ExecStartPre=/bin/sh -c "):].strip()
    assert command.startswith("'") and command.endswith("'")
    command = command[1:-1].replace("%h", str(home))
    assert "$" not in command, ("systemd would expand a shell variable "
                                "before /bin/sh sees it")
    assert subprocess.run(["/bin/sh", "-c", command]).returncode == 0
    tampered = control / "tools/experiment_transition_queue.py"
    tampered.write_bytes(tampered.read_bytes() + b"\n# drift\n")
    assert subprocess.run(["/bin/sh", "-c", command]).returncode == 4
    # and an absent bundle is refused with the same never-retried class
    for path in sorted(control.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    control.rmdir()
    assert subprocess.run(["/bin/sh", "-c", command]).returncode == 4


def test_261_verify_control_cli_exit_codes(tmp_path):
    pinned = _dropin_pinned_manifest_sha()
    control = _install_control_bundle(tmp_path / "p1lr-control")
    assert sup.main(["verify-control", "--control-dir", str(control),
                     "--expected-manifest-sha256", pinned]) == 0
    assert sup.main(["verify-control",
                     "--control-dir", str(tmp_path / "nowhere"),
                     "--expected-manifest-sha256", pinned]) == 4


# ═══════════════════════════════════════════════════════════════════════
# REQ 5 (order §4.5) — the installer, executed in a TEMPORARY HOME:
# all four identities land, and NOTHING is enabled or started.
# ═══════════════════════════════════════════════════════════════════════

def test_installer_in_a_temp_home_lands_four_identities_and_starts_nothing(
        tmp_path):
    """The installer really runs, with a recording ``systemctl`` stub on
    PATH. Proof obligations (order §4.5): the four seed identities land
    byte-identically, the read-only control bundle lands and verifies,
    and the ONLY systemctl invocation is ``daemon-reload``."""
    home = tmp_path / "home"
    home.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "systemctl.log"
    stub = bindir / "systemctl"
    stub.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "$@" >> {log}
        exit 0
        """))
    stub.chmod(0o755)
    env = dict(os.environ)
    env.update(HOME=str(home), REPO_DIR=str(REPO),
               P1LR_V2_RUNTIME_DIR=str(REPO),
               P1LR_PYTHON=sys.executable,
               PATH=f"{bindir}:{env['PATH']}")
    result = subprocess.run(["bash", str(INSTALL_SCRIPT)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr

    env_dir = home / ".config/agent-multi/p1lr-v2"
    landed = sorted(p.name for p in env_dir.iterdir())
    assert landed == [f"seed{s}{sup.SEED_ENV_SUFFIX}"
                      for s in (101, 202, 303, 404)], landed
    for name in landed:
        assert (env_dir / name).read_bytes() == \
            (SHIPPED_ENV_DIR / name).read_bytes()
    assert (home / ".config/systemd/user/p1lr-decision@.service.d/"
                   "20-v2-identity.conf").read_bytes() == \
        SHIPPED_DROPIN.read_bytes()

    pinned = _dropin_pinned_manifest_sha()
    control = home / f".local/lib/agent-multi/p1lr-control/{pinned}"
    assert sup.verify_control_bundle(control, pinned)["verdict"] == "PASS"
    assert not (os.stat(control).st_mode & 0o222), "control dir is writable"
    for member in sup.CONTROL_BUNDLE_MEMBERS:
        assert not (os.stat(control / member).st_mode & 0o222), member

    # NOTHING enabled, NOTHING started — the runtime proof, not a lint
    invocations = [ln for ln in log.read_text().splitlines() if ln.strip()]
    assert invocations == ["--user daemon-reload"], invocations
    assert "installed (NOTHING enabled, NOTHING started)" in result.stdout
    # the printed boundary guidance is guidance, never execution
    assert "systemctl --user start  p1lr-decision@101.service" in \
        result.stdout


def test_installer_refuses_when_the_pinned_digest_and_manifest_disagree(
        tmp_path):
    """Fail-closed install: a drifted control manifest that the reviewed
    drop-in does not pin must abort BEFORE anything is verified as
    deployable."""
    home = tmp_path / "home"
    home.mkdir()
    fake_repo = tmp_path / "repo"
    subprocess.run(["cp", "-a", str(REPO / "examples"), str(fake_repo)],
                   check=True)
    fake_repo_root = tmp_path / "root"
    fake_repo_root.mkdir()
    (fake_repo_root / "examples").symlink_to(fake_repo)
    (fake_repo_root / "tools").symlink_to(REPO / "tools")
    manifest = (fake_repo_root / "examples/systemd/p1lr-control"
                / sup.CONTROL_MANIFEST_NAME)
    manifest.write_text(manifest.read_text().replace("0", "1", 1))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "systemctl"
    stub.write_text("#!/bin/sh\nexit 0\n")
    stub.chmod(0o755)
    env = dict(os.environ)
    env.update(HOME=str(home), REPO_DIR=str(fake_repo_root),
               P1LR_V2_RUNTIME_DIR=str(REPO),
               P1LR_PYTHON=sys.executable,
               PATH=f"{bindir}:{env['PATH']}")
    result = subprocess.run(["bash", str(INSTALL_SCRIPT)], env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 2
    assert "drop-in pins control manifest" in result.stderr


# ═══════════════════════════════════════════════════════════════════════
# REQ 8 — terminal artifacts are preserved across the WHOLE transition
# path: nothing is deleted, moved or rewritten.
# ═══════════════════════════════════════════════════════════════════════

def _tree_snapshot(*roots):
    snapshot = {}
    for root in roots:
        for path in sorted(Path(root).rglob("*")):
            if path.is_file():
                snapshot[str(path)] = _sha(path.read_bytes())
    return snapshot


def test_req8_transition_path_preserves_terminal_artifacts(tmp_path):
    # terminal screen artifacts + sealed collection + decision artifacts
    contract = _contract(tmp_path, experiment=V2_EXPERIMENT, tag="v2",
                         seeds=(101,))
    contract_path = _write_contract(tmp_path, contract,
                                    "v2_local_contract.json")
    _terminal_seed(tmp_path, contract, V2_SCREEN_IDENTITY, 101,
                   mode="screen")
    collection = tmp_path / "collection_screen_14e7ce82"
    (collection / "seed101").mkdir(parents=True)
    (collection / "screen_verdict.json").write_text(json.dumps(
        {"schema": "agent_multi.p1_difficulty_lr_screen_verdict.v1",
         "outcome": "SCREEN_VIABLE_REGION"}))
    (collection / "seed101" / "terminal.json").write_text("{}")
    _terminal_seed(tmp_path, contract, V2_CHAIN, 101, mode="decision")
    screen_root = Path(contract["output_root"])
    decision_root = Path(contract["decision_run"]["output_root"])
    before = _tree_snapshot(screen_root, decision_root, collection)
    assert len(before) >= 10

    # the ENTIRE transition path, end to end
    queue_dir = tmp_path / "queue"
    record = etq.ensure_terminal_record(
        queue_dir, experiment=V2_EXPERIMENT, mode="screen",
        identity=V2_SCREEN_IDENTITY, records_landed=16, cells_total=16,
        evidence_root=str(collection),
        contract_path=str(contract_path),
        contract_sha256=_sha(contract_path.read_bytes()), now=NOW)
    record = etq.approve_successor(
        record, job_id=f"p1lr-v2-decision-{V2_CHAIN}",
        experiment=V2_DECISION_EXPERIMENT,
        contract_path=str(contract_path),
        contract_sha256=_sha(contract_path.read_bytes()),
        approved_by="test", chain_id=V2_CHAIN, now=NOW)
    record = etq.set_materialization(record, "materialized", now=NOW)
    record = etq.claim_dispatch(record, claim_id="claim-1", host="omega",
                                chain_id=V2_CHAIN, now=NOW)
    record = etq.confirm_dispatch(record, claim_id="claim-1", now=NOW)
    etq.save_record(queue_dir, record, now=NOW)
    etq.evaluate_transition_incident(record, emitter=FakeEmitter(),
                                     now=NOW)
    etq.reconstruct_transitions(queue_dir, now=NOW)
    guard.poll(
        contract=contract, identity=V2_CHAIN,
        state=guard.default_state(), now=NOW, local_hostname="omega",
        emitter=FakeEmitter(), process_alive_fn=lambda seed: False,
        process_facts_fn=lambda: [],
        expected_contract_sha256=_sha(contract_path.read_bytes()),
        gpu_telemetry_fn=lambda uuid: {"gpu_utilization_pct": 0},
        unit_exists_fn=lambda unit: True,
        restart_fn=lambda unit: {"ok": True}, mode="decision",
        transition_queue_dir=queue_dir, transition_emitter=FakeEmitter())
    sup.plan_reboot_reconstruction(queue_dir, contract_path, now=NOW)
    sup.materialize_supervision(
        json.loads(contract_path.read_text()), record,
        contract_path=contract_path, outdir=tmp_path / "generated")
    sup.lease_gate_verdict(queue_dir, expected_chain_id=V2_CHAIN,
                           experiment=V2_DECISION_EXPERIMENT,
                           mode="decision", now=NOW)

    after = _tree_snapshot(screen_root, decision_root, collection)
    assert after == before, (
        "the transition path deleted, moved or rewrote terminal "
        "artifacts — forbidden by WO4 requirement 8")
