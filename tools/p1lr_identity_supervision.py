#!/usr/bin/env python3
"""Identity-specific P1LR v2 supervision: env/unit materializer, reboot
planner and transition lease gate (WO4, finding AUD-GEN-20260815-250).

THE DEFECT: the corrected v2 decision chain ``cdf30aebf585385b`` runs
healthy under nohup from the immutable runtime worktree, but every
deployed ``p1lr-decision@*.service`` unit is inactive and still pins
LEGACY paths — the v1 contract, the old cd823e2b screen gate, the
legacy per-seed env. A reboot would therefore NOT reconstruct v2
through systemd, and bounded recovery could start the WRONG identity.

This module makes the supervision identity-specific and durable:

1. **Materializer** — generates, from the v2 contract plus the DURABLE
   transition record (the only authority, order 2026-08-15 §3), the
   per-seed environment files (v2 contract path + sha256, v2 screen
   gate, chain id, CUDA UUID per the contract assignment) and the
   ``20-v2-identity.conf`` drop-in that repoints the whole
   ``p1lr-decision@`` template to the pinned immutable runtime worktree
   and the v2 gate. Shipped as files; the install script ENABLES
   NOTHING.
2. **Reboot planner** — :func:`plan_reboot_reconstruction` derives, from
   the durable queue record and the shipped contract ALONE, the exact
   per-host v2 unit graph a reboot must reconstruct. It is the
   unit-generation logic under test — systemd itself is never touched.
3. **Lease gate** — the ``lease-gate`` CLI (ExecStartPre of every
   ``p1lr-decision@`` instance via the drop-in) refuses, exit 4 and
   never retried, any unit start whose declared chain does not hold the
   RULING active transition of its experiment family. An old v1 unit
   cannot restart while v2 owns the lease; an undeclared unit cannot
   start at all while any chain owns the family.
4. **Control artifact** (finding AUD-GEN-20260816-261) — the lease gate
   is a RESTART ADMISSION GATE, so it never executes from the mutable
   canonical checkout. ``CONTROL_MANIFEST.sha256`` pins the sha256 of
   every module the gate needs; the manifest's own sha256 names a
   read-only, CONTENT-ADDRESSED install path and is written LITERALLY
   into the drop-in, which re-verifies both digests (``sha256sum -c``,
   exit 4 on any drift or absence) immediately before running the gate.
   Changing what may start therefore requires changing a reviewed,
   versioned unit — not a ``git pull``.

Socket-free by construction: every function is pure over a contract
dict, a record dict and paths; the CLI reads only local files. Nothing
here starts, stops, restarts, enables or disables any unit or process —
the ABSOLUTE rule of the order is that the live v2 PIDs continue
untouched, and deployment happens only at the next safe process
boundary, by the operator, with the printed commands.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools import experiment_transition_queue as etq  # noqa: E402

PLAN_SCHEMA = "agent_multi.p1lr_reboot_reconstruction_plan.v1"
MANIFEST_SCHEMA = "agent_multi.p1lr_identity_supervision_manifest.v1"
GATE_SCHEMA = "agent_multi.p1lr_transition_lease_gate_verdict.v1"

# ── The v2 deployment identity (all values re-proven against the
# durable record and the contract at generation time, never trusted
# from these constants alone) ───────────────────────────────────────
V2_CONTRACT_RELPATH = ("examples/config/phase_3_eth_sac_dynamics/"
                       "p1_difficulty_lr_factorial_v2.json")
#: The immutable runtime worktree the live v2 PIDs execute from
#: (WP0 pinned-worktree convention). %h-relative for systemd files.
DEFAULT_RUNTIME_DIR_SPEC = (
    "%h/Documents/GitHub/.runtime/agent-multi-p1lr-v2-924910fe")
DEFAULT_PYTHON_SPEC = "%h/anaconda3/envs/trading-stack/bin/python"
#: Kept for reference ONLY. Nothing generated here may execute from the
#: mutable canonical checkout at unit-start time — see
#: :data:`CONTROL_INSTALL_ROOT_SPEC` and finding AUD-GEN-20260816-261.
DEFAULT_CANONICAL_REPO_SPEC = "%h/Documents/GitHub/agent-multi"
#: Installed (host-local) per-seed env dir — OUTSIDE the immutable
#: runtime worktree, which predates these files and must never change.
ENV_INSTALL_DIR_SPEC = "%h/.config/agent-multi/p1lr-v2"
DROPIN_NAME = "20-v2-identity.conf"

#: Per-seed identity files are SHIPPED (versioned) deliverables, so they
#: must not be swallowed by the repository's credential ignore rules
#: (``*.env``, ``.env``, ``.env.*`` under "Credentials and machine-local
#: authority"). Finding AUD-GEN-20260816-256: the generated ``seed*.env``
#: files were silently absent from the repository while the installer
#: required their glob. The suffix below is deliberately NOT matched by
#: those rules, so a newly generated seed is visible to ``git status``
#: instead of vanishing; systemd's ``EnvironmentFile=`` parses by content,
#: never by extension, so the deployed semantics are unchanged.
SEED_ENV_SUFFIX = ".env.conf"
SEED_ENV_DIR_RELPATH = ("examples/config/phase_3_eth_sac_dynamics/"
                        "p1lr_env_v2")

# ── Lease-gate control artifact (finding AUD-GEN-20260816-261) ───────
#: The lease gate is a RESTART ADMISSION GATE. It must never execute from
#: the mutable canonical checkout, where a routine ``git pull`` could
#: change who may start without changing the unit's declared identity.
#: Instead the two supervision modules are installed READ-ONLY under a
#: CONTENT-ADDRESSED control path whose directory name IS the sha256 of
#: the manifest that pins every member's sha256. The drop-in carries that
#: manifest digest LITERALLY, so:
#:   * changing the supervision code changes the manifest digest, hence
#:     the control path, hence the unit — a reviewed, versioned edit;
#:   * tampering with an installed copy breaks ``sha256sum -c`` and the
#:     unit refuses (exit 4, never retried) before any training process
#:     exists;
#:   * a missing bundle refuses too — the gate fails CLOSED.
CONTROL_BUNDLE_RELPATH = "examples/systemd/p1lr-control"
CONTROL_MANIFEST_NAME = "CONTROL_MANIFEST.sha256"
#: Sorted, exhaustive: the lease gate imports nothing outside these two
#: modules and the standard library.
CONTROL_BUNDLE_MEMBERS = (
    "tools/experiment_transition_queue.py",
    "tools/p1lr_identity_supervision.py",
)
CONTROL_INSTALL_ROOT_SPEC = "%h/.local/lib/agent-multi/p1lr-control"
#: Absolute on purpose: an ExecStartPre inherits systemd's PATH, and a
#: verification tool resolved through PATH is not a verification tool.
SHA256SUM_BIN = "/usr/bin/sha256sum"

GENERATED_HEADER = (
    "# GENERATED by tools/p1lr_identity_supervision.py materialize "
    "(WO4, finding\n"
    "# AUD-GEN-20260815-250) — regenerate, never hand-edit: tests pin "
    "these bytes\n"
    "# to the generator.\n")


class SupervisionRefusal(ValueError):
    """Typed refusal: generation/planning fails closed, mutating
    nothing, when the durable record and the shipped contract disagree."""

    def __init__(self, code: str, reason: str, **facts: Any):
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.facts = facts

    def as_block(self, **extra: Any) -> dict[str, Any]:
        block: dict[str, Any] = {
            "source": "p1lr_identity_supervision",
            "basis": "refusal",
            "error_code": self.code,
            "reason": self.reason,
        }
        block.update(self.facts)
        block.update(extra)
        return block


def _now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# lease-gate control artifact (finding AUD-GEN-20260816-261)
# ---------------------------------------------------------------------------

def control_manifest_content(source_root: Path = REPO_ROOT) -> str:
    """The ``sha256sum -c`` manifest of the lease-gate control bundle.

    Deliberately free of comments and of any timestamp: the bytes are a
    pure function of the member contents, so the manifest's own digest —
    the value the unit pins and the control directory is named after — is
    reproducible by anyone with the reviewed source.
    """
    source_root = Path(source_root)
    lines = []
    for member in CONTROL_BUNDLE_MEMBERS:
        path = source_root / member
        if not path.is_file():
            raise SupervisionRefusal(
                "SUPERVISION_CONTROL_MEMBER_MISSING",
                f"lease-gate control bundle member {member!r} is absent "
                f"under {source_root}; the restart admission gate cannot "
                "be pinned to an artifact that does not exist",
                member=member, source_root=str(source_root))
        lines.append(f"{sha256_file(path)}  {member}")
    return "".join(f"{line}\n" for line in lines)


def control_manifest_sha256(source_root: Path = REPO_ROOT) -> str:
    """The digest the drop-in pins and the control directory is named
    after. One value binds the whole admission-gate implementation."""
    return hashlib.sha256(
        control_manifest_content(source_root).encode()).hexdigest()


def control_dir_spec(manifest_sha256: str,
                     root_spec: str = CONTROL_INSTALL_ROOT_SPEC) -> str:
    """The content-addressed, read-only install path of the bundle."""
    return f"{root_spec}/{manifest_sha256}"


def verify_control_bundle(control_dir: Path, expected_manifest_sha256: str,
                          ) -> dict[str, Any]:
    """The Python mirror of the unit's ``sha256sum -c`` chain, for tests
    and for read-only operator verification. FAILS CLOSED: an absent
    manifest, a drifted manifest, an absent member or a drifted member
    all return ``passed=False`` with a typed verdict."""
    control_dir = Path(control_dir)
    manifest_path = control_dir / CONTROL_MANIFEST_NAME
    base = {
        "control_dir": str(control_dir),
        "expected_manifest_sha256": expected_manifest_sha256,
    }
    if not manifest_path.is_file():
        return dict(base, passed=False,
                    verdict="REFUSED_CONTROL_MANIFEST_ABSENT",
                    reason=(f"{manifest_path} does not exist: no reviewed "
                            "control artifact backs this restart"))
    observed = sha256_file(manifest_path)
    if observed != expected_manifest_sha256:
        return dict(base, passed=False, observed_manifest_sha256=observed,
                    verdict="REFUSED_CONTROL_MANIFEST_DRIFT",
                    reason=(f"{manifest_path} hashes to {observed}, not the "
                            f"pinned {expected_manifest_sha256}"))
    members: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, _, member = line.partition("  ")
        members[member] = digest
    for member, digest in sorted(members.items()):
        member_path = control_dir / member
        if not member_path.is_file():
            return dict(base, passed=False, member=member,
                        verdict="REFUSED_CONTROL_MEMBER_ABSENT",
                        reason=f"{member_path} does not exist")
        member_observed = sha256_file(member_path)
        if member_observed != digest:
            return dict(base, passed=False, member=member,
                        observed_member_sha256=member_observed,
                        verdict="REFUSED_CONTROL_MEMBER_DRIFT",
                        reason=(f"{member_path} hashes to {member_observed}, "
                                f"not the manifest's {digest}"))
    return dict(base, passed=True, verdict="PASS", members=members,
                reason=("the manifest matches its pinned digest and every "
                        "member matches the manifest"))


# ---------------------------------------------------------------------------
# durable-record derivations (the record is the ONLY authority)
# ---------------------------------------------------------------------------

def screen_gate_path_from_record(record: dict) -> str:
    """The v2 screen-gate path, derived from the durable record's sealed
    terminal evidence root — never typed in by an operator."""
    evidence_root = str((record.get("terminal_result") or {})
                        .get("evidence_root") or "").rstrip("/")
    if not evidence_root:
        raise SupervisionRefusal(
            "SUPERVISION_NO_EVIDENCE_ROOT",
            "the durable record carries no terminal_result.evidence_root; "
            "the screen gate path cannot be derived",
            transition_id=record.get("transition_id"))
    return f"{evidence_root}/screen_verdict.json"


def verify_record_binds_contract(record: dict, contract_path: Path,
                                 ) -> dict[str, Any]:
    """Prove the durable record's approved successor runs EXACTLY the
    shipped contract file (sha256 equality), or refuse."""
    next_job = dict(record.get("next_job") or {})
    declared = next_job.get("contract_sha256")
    observed = sha256_file(contract_path)
    if not declared:
        raise SupervisionRefusal(
            "SUPERVISION_RECORD_WITHOUT_CONTRACT_SHA",
            f"durable record {record.get('transition_id')} approves "
            f"successor {next_job.get('id')!r} without a contract sha256; "
            "an unpinned successor cannot be materialized",
            transition_id=record.get("transition_id"))
    if declared != observed:
        raise SupervisionRefusal(
            "SUPERVISION_CONTRACT_SHA_MISMATCH",
            f"durable record {record.get('transition_id')} binds successor "
            f"contract sha {declared} but {contract_path} hashes to "
            f"{observed}: the shipped contract is NOT the approved one",
            transition_id=record.get("transition_id"),
            declared_sha256=declared, observed_sha256=observed,
            contract_path=str(contract_path))
    chain_id = next_job.get("chain_id")
    if not chain_id:
        raise SupervisionRefusal(
            "SUPERVISION_RECORD_WITHOUT_CHAIN",
            f"durable record {record.get('transition_id')} approves a "
            "successor without a chain id; identity-specific supervision "
            "needs the chain identity",
            transition_id=record.get("transition_id"))
    return {"contract_sha256": observed, "chain_id": str(chain_id),
            "experiment": str(next_job.get("experiment") or ""),
            "next_job_id": str(next_job.get("id") or "")}


# ---------------------------------------------------------------------------
# generation (pure text; ENABLES NOTHING)
# ---------------------------------------------------------------------------

def seed_env_content(contract: dict, seed: int, *, contract_sha256: str,
                     chain_id: str, experiment: str) -> str:
    """One seed's identity-specific environment file.

    CUDA_VISIBLE_DEVICES equals the CONTRACT assignment (the runner
    refuses REFUSED_GPU_UNBOUND on mismatch — the env file is
    convenience, the contract check is authority). The contract path is
    RELATIVE because WorkingDirectory is pinned to the immutable runtime
    worktree by the same drop-in that loads this file, which also makes
    the resulting cmdline byte-identical to the live nohup workers'.
    """
    assignment = dict((contract.get("assignments") or {})
                      .get(str(seed)) or {})
    host = assignment.get("hostname")
    gpu = assignment.get("gpu_uuid")
    if not (host and gpu):
        raise SupervisionRefusal(
            "SUPERVISION_SEED_UNASSIGNED",
            f"seed {seed} carries no hostname/gpu_uuid assignment in the "
            "contract; no identity-specific binding can be generated",
            seed=seed)
    decision = dict(contract.get("decision_run") or {})
    output_root = str(decision.get("output_root") or "")
    if not output_root:
        raise SupervisionRefusal(
            "SUPERVISION_NO_DECISION_ROOT",
            "contract declares no decision_run.output_root",
            seed=seed)
    return (
        f"# P1LR v2 DECISION seed {seed} — identity-specific worker "
        "binding.\n"
        + GENERATED_HEADER +
        "# CUDA_VISIBLE_DEVICES must EQUAL the contract assignment; the "
        "runner\n"
        "# refuses (REFUSED_GPU_UNBOUND) on any mismatch.\n"
        f"P1LR_SEED={seed}\n"
        f"P1LR_HOST={host}\n"
        f"CUDA_VISIBLE_DEVICES={gpu}\n"
        "P1LR_MODE=decision\n"
        f"P1LR_CONTRACT={V2_CONTRACT_RELPATH}\n"
        f"P1LR_CONTRACT_SHA256={contract_sha256}\n"
        f"P1LR_EXPECTED_CHAIN_ID={chain_id}\n"
        f"P1LR_EXPERIMENT={experiment}\n"
        f"P1LR_OUTPUT_ROOT={output_root}\n")


def dropin_content(*, runtime_dir_spec: str = DEFAULT_RUNTIME_DIR_SPEC,
                   python_spec: str = DEFAULT_PYTHON_SPEC,
                   env_dir_spec: str = ENV_INSTALL_DIR_SPEC,
                   control_root_spec: str = CONTROL_INSTALL_ROOT_SPEC,
                   control_manifest_sha256: str,
                   gate_path_spec: str) -> str:
    """The ``20-v2-identity.conf`` drop-in for ``p1lr-decision@.service``.

    It applies to EVERY ``p1lr-decision@<seed>`` instance on the host:
    the template is repointed WHOLESALE to the v2 identity, so there is
    no legacy-path instance left for an operator or a recovery loop to
    start by accident. Base-template semantics that stay correct
    (Restart classes, RestartPreventExitStatus=4, memory bounds) are
    inherited, not repeated.

    The lease gate executes from the CONTENT-ADDRESSED read-only control
    bundle (finding AUD-GEN-20260816-261), never from the mutable
    canonical checkout, and its manifest digest is verified by the unit
    itself immediately before the gate runs.
    """
    control_dir = control_dir_spec(control_manifest_sha256,
                                   root_spec=control_root_spec)
    manifest_spec = f"{control_dir}/{CONTROL_MANIFEST_NAME}"
    # No `$VAR` and no `%` beyond systemd's own `%h`: systemd expands
    # both before /bin/sh ever sees the string, so the verification must
    # not depend on shell variables.
    verify_cmd = (
        f"/bin/sh -c 'echo \"{control_manifest_sha256}  {manifest_spec}\" "
        f"| {SHA256SUM_BIN} --status -c - "
        f"&& cd {control_dir} "
        f"&& {SHA256SUM_BIN} --status -c {CONTROL_MANIFEST_NAME} "
        f"|| exit 4'")
    return (
        "# P1LR v2 identity pin for p1lr-decision@.service — applies to "
        "every seed\n"
        "# instance on this host: the template is repointed WHOLESALE to "
        "the v2\n"
        "# identity, so no legacy-path instance is left to restart by "
        "accident.\n"
        + GENERATED_HEADER +
        "[Service]\n"
        "# 1. Pinned IMMUTABLE runtime worktree (WP0): a restart derives "
        "the same\n"
        "#    experiment identity as the live nohup workers.\n"
        f"WorkingDirectory={runtime_dir_spec}\n"
        "# 2. The v2 screen gate (screen identity 14e7ce82, "
        "SCREEN_VIABLE_REGION,\n"
        "#    sealed collection), replacing the legacy cd823e2b gate of "
        "the base\n"
        "#    template. ExecStartPre re-verifies it against the V2 "
        "contract.\n"
        f"Environment=P1LR_SCREEN_GATE={gate_path_spec}\n"
        "# 3. Identity-specific per-seed env (v2 contract path+sha, "
        "chain id,\n"
        "#    CUDA UUID). The legacy p1lr_env binding is RESET, not "
        "shadowed.\n"
        "EnvironmentFile=\n"
        f"EnvironmentFile={env_dir_spec}/seed%i{SEED_ENV_SUFFIX}\n"
        "EnvironmentFile=-%h/.config/agent-multi/p1lr-decision@%i.env\n"
        "# 4. Gates, in order:\n"
        "#    (a) screen-gate verification bound to the V2 contract, run "
        "from the\n"
        "#        pinned immutable runtime worktree;\n"
        "#    (b) CONTROL-ARTIFACT VERIFICATION (finding "
        "AUD-GEN-20260816-261).\n"
        "#        A restart admission gate must NOT be mutable checkout "
        "code, so\n"
        "#        the lease gate is installed READ-ONLY under a "
        "CONTENT-ADDRESSED\n"
        "#        control path whose directory name IS the sha256 of the "
        "manifest\n"
        "#        that pins every member. This line re-derives both "
        "digests and\n"
        "#        exits 4 — the never-retried refusal class — on drift or "
        "absence,\n"
        "#        so the gate fails CLOSED. Changing the supervision "
        "implementation\n"
        "#        changes this literal digest, i.e. changes this reviewed "
        "unit.\n"
        "#    (c) the durable-transition LEASE gate itself — a unit whose "
        "chain\n"
        "#        does not hold the ruling active transition refuses "
        "(exit 4,\n"
        "#        never retried) BEFORE any training process exists.\n"
        "ExecStartPre=\n"
        f"ExecStartPre={runtime_dir_spec}/examples/systemd/"
        "p1lr_decision_gate_check.sh ${P1LR_SCREEN_GATE} "
        "${P1LR_CONTRACT}\n"
        f"ExecStartPre={verify_cmd}\n"
        f"ExecStartPre={python_spec} {control_dir}/tools/"
        "p1lr_identity_supervision.py lease-gate "
        "--expected-chain-id ${P1LR_EXPECTED_CHAIN_ID} "
        "--experiment ${P1LR_EXPERIMENT} --mode decision\n"
        "# 5. The v2 invocation itself — byte-compatible with the live "
        "nohup\n"
        "#    workers (same relative contract path, same mode, same "
        "gate).\n"
        "ExecStart=\n"
        f"ExecStart={python_spec} tools/p1_difficulty_lr_factorial.py "
        "--seed %i --mode decision --contract ${P1LR_CONTRACT} "
        "--screen-gate ${P1LR_SCREEN_GATE}\n")


def materialize_supervision(contract: dict, record: dict, *,
                            contract_path: Path,
                            outdir: Path,
                            runtime_dir_spec: str =
                            DEFAULT_RUNTIME_DIR_SPEC,
                            gate_path_spec: Optional[str] = None,
                            control_source_root: Path = REPO_ROOT,
                            ) -> dict[str, Any]:
    """Write every identity-specific supervision file under ``outdir``
    (repo-shippable layout) and return a typed manifest. PURE FILE
    OUTPUT: nothing is installed, enabled, started or reloaded here.

    ``control_source_root`` is the reviewed source of the lease-gate
    control bundle (default: this repository). Its members' digests
    become ``CONTROL_MANIFEST.sha256``, whose own digest is written
    literally into the drop-in and names the read-only install path.
    """
    binding = verify_record_binds_contract(record, contract_path)
    if gate_path_spec is None:
        gate = screen_gate_path_from_record(record)
        home = str(Path.home())
        if gate.startswith("~/"):
            gate_spec = "%h/" + gate[2:]
        elif gate.startswith(home + "/"):
            gate_spec = "%h" + gate[len(home):]
        else:
            gate_spec = gate
    else:
        gate_spec = gate_path_spec
    outdir = Path(outdir)
    env_dir = outdir / SEED_ENV_DIR_RELPATH
    dropin_dir = outdir / "examples/systemd/p1lr-decision@.service.d"
    control_dir = outdir / CONTROL_BUNDLE_RELPATH
    env_dir.mkdir(parents=True, exist_ok=True)
    dropin_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for seed in [int(s) for s in (contract.get("seeds") or [])]:
        content = seed_env_content(
            contract, seed, contract_sha256=binding["contract_sha256"],
            chain_id=binding["chain_id"],
            experiment=binding["experiment"])
        path = env_dir / f"seed{seed}{SEED_ENV_SUFFIX}"
        path.write_text(content, encoding="utf-8")
        written[str(path.relative_to(outdir))] = hashlib.sha256(
            content.encode()).hexdigest()
    manifest_text = control_manifest_content(control_source_root)
    manifest_sha = hashlib.sha256(manifest_text.encode()).hexdigest()
    manifest_path = control_dir / CONTROL_MANIFEST_NAME
    manifest_path.write_text(manifest_text, encoding="utf-8")
    written[str(manifest_path.relative_to(outdir))] = manifest_sha
    dropin = dropin_content(runtime_dir_spec=runtime_dir_spec,
                            control_manifest_sha256=manifest_sha,
                            gate_path_spec=gate_spec)
    dropin_path = dropin_dir / DROPIN_NAME
    dropin_path.write_text(dropin, encoding="utf-8")
    written[str(dropin_path.relative_to(outdir))] = hashlib.sha256(
        dropin.encode()).hexdigest()
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": _now().isoformat(),
        "transition_id": record.get("transition_id"),
        "chain_id": binding["chain_id"],
        "experiment": binding["experiment"],
        "contract_sha256": binding["contract_sha256"],
        "screen_gate_spec": gate_spec,
        "runtime_dir_spec": runtime_dir_spec,
        "control_manifest_sha256": manifest_sha,
        "control_dir_spec": control_dir_spec(manifest_sha),
        "files": written,
        "mutation_contract": ("files only — nothing installed, enabled, "
                              "started or reloaded; the live v2 PIDs are "
                              "untouched by construction"),
    }


# ---------------------------------------------------------------------------
# reboot reconstruction planning (requirement 5 — logic, not systemd)
# ---------------------------------------------------------------------------

def plan_reboot_reconstruction(queue_dir: Path, contract_path: Path, *,
                               runtime_dir_spec: str =
                               DEFAULT_RUNTIME_DIR_SPEC,
                               host: Optional[str] = None,
                               now: Optional[datetime] = None,
                               lease_seconds: float =
                               etq.DEFAULT_CLAIM_LEASE_SECONDS,
                               ) -> dict[str, Any]:
    """The unit graph a reboot must reconstruct, derived from the
    DURABLE queue records plus the shipped contract ALONE (order
    2026-08-15 §3 durability contract): no heartbeat, no process table,
    no systemd state, no operator memory.

    Refuses (typed) when no ruling active transition exists for the
    contract's family, or when the ruling successor's contract sha does
    not match the shipped contract file — a reboot must never
    reconstruct an unproven identity.
    """
    moment = now or _now()
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    family = etq.experiment_family(str(contract.get("experiment") or ""))
    records, unreadable = etq.load_records(queue_dir)
    authorities = etq.active_chain_authorities(
        records, now=moment, lease_seconds=lease_seconds, family=family)
    if not authorities:
        raise SupervisionRefusal(
            "SUPERVISION_NO_ACTIVE_AUTHORITY",
            f"no durable record proves an active dispatched chain in "
            f"experiment family {family!r}: a reboot has nothing proven "
            "to reconstruct (enrol/dispatch the transition first — the "
            "durable record is the ONLY authority)",
            family=family, queue_dir=str(queue_dir),
            records_total=len(records), unreadable=unreadable)
    ruling = authorities[0]
    record = next((r for r in records
                   if r.get("transition_id") == ruling["transition_id"]),
                  None)
    binding = verify_record_binds_contract(record or {}, contract_path)
    gate_path = screen_gate_path_from_record(record or {})
    assignments = dict(contract.get("assignments") or {})
    hosts: dict[str, list[dict]] = {}
    for seed_str, assignment in assignments.items():
        if not (isinstance(assignment, dict)
                and assignment.get("hostname")):
            continue
        seed = int(seed_str)
        hostname = str(assignment["hostname"])
        if host is not None and hostname != host:
            continue
        hosts.setdefault(hostname, []).append({
            "seed": seed,
            "unit": f"p1lr-decision@{seed}.service",
            "gpu_uuid": assignment.get("gpu_uuid"),
            "env_file": (f"{ENV_INSTALL_DIR_SPEC}/"
                         f"seed{seed}{SEED_ENV_SUFFIX}"),
            "expected": {
                "chain_id": binding["chain_id"],
                "contract_sha256": binding["contract_sha256"],
                "mode": "decision",
                "screen_gate": gate_path,
                "working_directory": runtime_dir_spec,
                "output_root": str(dict(contract.get("decision_run")
                                        or {}).get("output_root")),
            },
        })
    for units in hosts.values():
        units.sort(key=lambda u: u["seed"])
    return {
        "schema": PLAN_SCHEMA,
        "generated_at": moment.isoformat(),
        "basis": ("durable transition records + shipped contract ONLY; "
                  "no heartbeat, process table, systemd state or "
                  "operator memory (order 2026-08-15 §3)"),
        "family": family,
        "ruling_authority": ruling,
        "transition_id": ruling["transition_id"],
        "chain_id": binding["chain_id"],
        "experiment": binding["experiment"],
        "contract_path": str(contract_path),
        "contract_sha256": binding["contract_sha256"],
        "screen_gate": gate_path,
        "runtime_dir_spec": runtime_dir_spec,
        "hosts": hosts,
        "unit_total": sum(len(u) for u in hosts.values()),
        "superseded_authorities": authorities[1:],
    }


def deploy_commands(plan: dict) -> list[str]:
    """The EXACT next-safe-boundary commands, per host, NEVER executed
    here. Enable (without --now) arms reboot reconstruction; start is
    legal only once the seed's live v2 PID has exited on its own."""
    lines = [
        "# NEXT SAFE PROCESS BOUNDARY ONLY (finding 250 / WO4).",
        "# ABSOLUTE RULE: while a matching v2 PID is alive for a seed, "
        "NO systemctl",
        "# start/restart of that seed's unit. Check per seed:",
        "#   pgrep -af 'p1_difficulty_lr_factorial.py --seed <seed> "
        "--mode decision'",
    ]
    for hostname in sorted(plan.get("hosts") or {}):
        units = plan["hosts"][hostname]
        lines.append(f"# ── {hostname} "
                     f"({len(units)} unit(s)) ─────────────────────────")
        lines.append(f"#   on {hostname}: bash examples/systemd/"
                     "install_p1lr_v2_identity_supervision.sh")
        for unit in units:
            lines.append(
                f"#   arm reboot reconstruction (does NOT start "
                f"anything): systemctl --user enable {unit['unit']}")
            lines.append(
                f"#   ONLY at seed {unit['seed']}'s boundary (its v2 PID "
                f"exited): systemctl --user start {unit['unit']}")
    lines.append("# guard timer (safe anytime; touches no worker): "
                 "systemctl --user enable --now p1lr-idle-guard.timer")
    return lines


# ---------------------------------------------------------------------------
# transition lease gate (requirement 7 — the unit-side check)
# ---------------------------------------------------------------------------

def lease_gate_verdict(queue_dir: Path, *, expected_chain_id: str,
                       experiment: str, mode: str,
                       now: Optional[datetime] = None,
                       lease_seconds: float =
                       etq.DEFAULT_CLAIM_LEASE_SECONDS,
                       ) -> dict[str, Any]:
    """PASS only when the unit's declared chain HOLDS the ruling active
    transition of its experiment family. Fail-closed refusals:

    ``REFUSED_UNIT_IDENTITY_UNDECLARED``
        the unit start declared no chain (a legacy env without
        ``P1LR_EXPECTED_CHAIN_ID`` expands to an empty argument): an
        unidentified unit never starts while identity supervision is
        deployed.
    ``REFUSED_LEASE_HELD_BY_OTHER_CHAIN``
        a DIFFERENT chain rules the family's active transition — the
        old-v1-unit-during-v2 case, refused before any training process
        exists.
    ``REFUSED_NO_DURABLE_AUTHORITY``
        no durable record proves ANY active chain: nothing authorizes
        this start (the durable record is the only authority; enrol and
        dispatch it first).
    """
    moment = now or _now()
    family = etq.experiment_family(experiment)
    base = {
        "schema": GATE_SCHEMA,
        "generated_at": moment.isoformat(),
        "queue_dir": str(Path(queue_dir).expanduser()),
        "family": family,
        "expected_chain_id": expected_chain_id or None,
        "experiment": experiment,
        "mode": mode,
    }
    if not str(expected_chain_id or "").strip():
        return dict(base, verdict="REFUSED_UNIT_IDENTITY_UNDECLARED",
                    passed=False,
                    reason=("this unit start declares NO chain identity "
                            "(empty P1LR_EXPECTED_CHAIN_ID — a legacy "
                            "environment); an unidentified unit never "
                            "starts under identity supervision"))
    records, unreadable = etq.load_records(queue_dir)
    authorities = etq.active_chain_authorities(
        records, now=moment, lease_seconds=lease_seconds, family=family)
    if not authorities:
        return dict(base, verdict="REFUSED_NO_DURABLE_AUTHORITY",
                    passed=False, unreadable=unreadable,
                    reason=(f"no durable record proves an active "
                            f"dispatched chain for family {family!r}: "
                            "nothing authorizes this start — enrol and "
                            "dispatch the transition first (the durable "
                            "record is the ONLY authority)"))
    ruling = authorities[0]
    if ruling.get("chain_id") == expected_chain_id:
        return dict(base, verdict="PASS", passed=True,
                    ruling_authority=ruling,
                    reason=(f"chain {expected_chain_id} holds the ruling "
                            f"{ruling['transition_state']} transition "
                            f"{ruling['transition_id']} of family "
                            f"{family!r}"))
    return dict(base, verdict="REFUSED_LEASE_HELD_BY_OTHER_CHAIN",
                passed=False, ruling_authority=ruling,
                superseded_authorities=authorities[1:],
                reason=(f"chain {ruling.get('chain_id')} holds the ruling "
                        f"{ruling['transition_state']} transition "
                        f"{ruling['transition_id']} of family {family!r}; "
                        f"a start declaring chain {expected_chain_id} is "
                        "refused before any training process exists — an "
                        "old unit cannot restart while another chain owns "
                        "the lease (finding 250)"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print(value: Any) -> None:
    print(json.dumps(value, indent=1, sort_keys=True))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path,
                        default=etq.DEFAULT_QUEUE_DIR,
                        help="durable transition records (the ONLY "
                             "authority)")
    sub = parser.add_subparsers(dest="command", required=True)

    mat = sub.add_parser("materialize",
                         help="write identity-specific env files + "
                              "drop-in (files only; enables NOTHING)")
    mat.add_argument("--contract", type=Path,
                     default=REPO_ROOT / V2_CONTRACT_RELPATH)
    mat.add_argument("--transition-id", default=None,
                     help="durable record to bind; defaults to the "
                          "ruling active authority of the contract's "
                          "family")
    mat.add_argument("--outdir", type=Path, default=REPO_ROOT,
                     help="write under this repo-shaped root "
                          "(default: this repo)")
    mat.add_argument("--runtime-dir-spec",
                     default=DEFAULT_RUNTIME_DIR_SPEC)

    plan = sub.add_parser("plan",
                          help="reboot-reconstruction unit graph from "
                               "the durable records alone")
    plan.add_argument("--contract", type=Path,
                      default=REPO_ROOT / V2_CONTRACT_RELPATH)
    plan.add_argument("--host", default=None)
    plan.add_argument("--runtime-dir-spec",
                      default=DEFAULT_RUNTIME_DIR_SPEC)
    plan.add_argument("--with-deploy-commands", action="store_true")

    gate = sub.add_parser("lease-gate",
                          help="unit-start transition lease gate "
                               "(exit 0 PASS / 4 typed refusal)")
    gate.add_argument("--expected-chain-id", default="")
    gate.add_argument("--experiment", required=True)
    gate.add_argument("--mode", required=True)

    ctl = sub.add_parser("verify-control",
                         help="read-only verification of the installed "
                              "lease-gate control bundle against its "
                              "pinned manifest digest (exit 0/4)")
    ctl.add_argument("--control-dir", type=Path, default=None,
                     help="installed bundle; default: the "
                          "content-addressed path of THIS checkout's "
                          "manifest under ~/.local/lib/agent-multi")
    ctl.add_argument("--expected-manifest-sha256", default=None,
                     help="default: recomputed from this checkout")
    ctl.add_argument("--source-root", type=Path, default=REPO_ROOT)

    args = parser.parse_args(argv)
    try:
        if args.command == "verify-control":
            expected = (args.expected_manifest_sha256
                        or control_manifest_sha256(args.source_root))
            control_dir = args.control_dir or Path(
                control_dir_spec(expected).replace(
                    "%h", str(Path.home()), 1))
            verdict = verify_control_bundle(control_dir, expected)
            _print(verdict)
            return 0 if verdict["passed"] else 4
        if args.command == "lease-gate":
            verdict = lease_gate_verdict(
                args.queue_dir, expected_chain_id=args.expected_chain_id,
                experiment=args.experiment, mode=args.mode)
            _print(verdict)
            return 0 if verdict["passed"] else 4
        if args.command == "plan":
            result = plan_reboot_reconstruction(
                args.queue_dir, args.contract, host=args.host,
                runtime_dir_spec=args.runtime_dir_spec)
            if args.with_deploy_commands:
                result["deploy_commands_next_safe_boundary"] = \
                    deploy_commands(result)
            _print(result)
            return 0
        if args.command == "materialize":
            records, _unreadable = etq.load_records(args.queue_dir)
            if args.transition_id:
                record = next((r for r in records
                               if r.get("transition_id")
                               == args.transition_id), None)
                if record is None:
                    raise SupervisionRefusal(
                        "SUPERVISION_RECORD_NOT_FOUND",
                        f"no durable record {args.transition_id} under "
                        f"{args.queue_dir}",
                        transition_id=args.transition_id)
            else:
                contract = json.loads(args.contract.read_text())
                family = etq.experiment_family(
                    str(contract.get("experiment") or ""))
                authorities = etq.active_chain_authorities(
                    records, family=family)
                if not authorities:
                    raise SupervisionRefusal(
                        "SUPERVISION_NO_ACTIVE_AUTHORITY",
                        f"no active dispatched chain for family "
                        f"{family!r}; name --transition-id explicitly "
                        "or dispatch the transition first",
                        family=family)
                record = next(
                    r for r in records
                    if r.get("transition_id")
                    == authorities[0]["transition_id"])
            contract = json.loads(args.contract.read_text())
            manifest = materialize_supervision(
                contract, record, contract_path=args.contract,
                outdir=args.outdir,
                runtime_dir_spec=args.runtime_dir_spec)
            _print(manifest)
            return 0
        raise SupervisionRefusal("SUPERVISION_UNKNOWN_COMMAND",
                                 f"unknown command {args.command!r}")
    except SupervisionRefusal as refusal:
        _print(refusal.as_block())
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
