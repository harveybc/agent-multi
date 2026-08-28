"""Typed GPU dispatch authorization + executable identity (H1/H3,
final SAC dispatch hardening order 2026-08-28; findings 377/379).

DATA-SOTA-377: authorization is a TYPED, CONTENT-BOUND artifact —
`agent_multi.paired_sac_dispatch_authorization.v1` — parsed and
verified field by field BEFORE any CUDA probe or model construction.
Path existence proves nothing; a generic file (the auditor's
counterexample: /etc/hosts) refuses at the parse step.

DATA-SOTA-379: the execution identity binds the WHOLE executable tree:
exact 40-hex HEAD, tracked+untracked cleanliness, and a canonical
sha256 allowlist over every executing file (driver, nested trainer,
SAC agent, grouped materializer + extractor, pretrained loader,
custody, authorization module itself, environment and execution
envelope modules resolved from the installed environment, split
contract, strong config, cost manifest, design, candidate manifest,
envelope calibration). Hashes are computed from file bytes at call
time — no mutable ``.sha256`` sidecar is ever an authority.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

AUTHORIZATION_SCHEMA = "agent_multi.paired_sac_dispatch_authorization.v1"
AUTHORIZATION_SCOPE = "EXECUTE_EIGHT_PAIRED_SAC_CELLS"
EXPECTED_AUDITOR = "General Musashi"
REQUIRED_FIELDS = (
    "schema", "campaign_id", "trial_ids",
    "reviewed_correction_commit", "paired_design_sha256",
    "candidate_seal_manifest_sha256", "launch_manifest_sha256",
    "executable_allowlist_sha256", "authorization_scope",
    "issued_utc", "auditor", "audit_order_commit_digest")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AuthorizationRefused(RuntimeError):
    """Typed refusal: the presented document does not authorize."""


def _sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_authorization(path: Path) -> dict[str, Any]:
    """Parse the document as the typed artifact — nothing else counts.
    Non-JSON content (any generic file) refuses; unknown keys refuse;
    missing fields refuse; malformed digests refuse."""
    path = Path(path)
    if not path.is_file():
        raise AuthorizationRefused(
            "authorization document does not exist")
    try:
        payload = json.loads(path.read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthorizationRefused(
            "authorization is not the typed artifact — the document "
            f"does not parse as JSON ({type(exc).__name__}); a generic "
            "file can never authorize (DATA-SOTA-377)") from exc
    if not isinstance(payload, dict):
        raise AuthorizationRefused(
            "authorization must be a JSON object (DATA-SOTA-377)")
    unknown = sorted(set(payload) - set(REQUIRED_FIELDS))
    if unknown:
        raise AuthorizationRefused(
            f"authorization carries unknown keys {unknown} — refused "
            "(DATA-SOTA-377)")
    missing = sorted(set(REQUIRED_FIELDS) - set(payload))
    if missing:
        raise AuthorizationRefused(
            f"authorization is missing required fields {missing} — "
            "refused (DATA-SOTA-377)")
    if payload["schema"] != AUTHORIZATION_SCHEMA:
        raise AuthorizationRefused(
            f"authorization schema {payload['schema']!r} is not "
            f"{AUTHORIZATION_SCHEMA!r} — refused")
    return payload


def verify_authorization(path: Path, *, campaign_id: str,
                         trial_ids: list[str],
                         paired_design_sha256: str,
                         candidate_seal_manifest_sha256: str,
                         launch_manifest_sha256: dict[str, str],
                         executable_allowlist_sha256: str
                         ) -> dict[str, Any]:
    """Every field must match the CURRENT campaign facts exactly.
    A stale design digest, wrong campaign, wrong seal, wrong trial set,
    unfilled template blanks or a foreign auditor refuse."""
    auth = load_authorization(path)
    checks = (
        ("campaign_id", campaign_id, auth["campaign_id"]),
        ("paired_design_sha256", paired_design_sha256,
         auth["paired_design_sha256"]),
        ("candidate_seal_manifest_sha256",
         candidate_seal_manifest_sha256,
         auth["candidate_seal_manifest_sha256"]),
        ("authorization_scope", AUTHORIZATION_SCOPE,
         auth["authorization_scope"]),
        ("auditor", EXPECTED_AUDITOR, auth["auditor"]),
        ("executable_allowlist_sha256", executable_allowlist_sha256,
         auth["executable_allowlist_sha256"]),
    )
    for name, expected, actual in checks:
        if actual != expected:
            raise AuthorizationRefused(
                f"authorization field {name!r} does not match the "
                f"campaign fact — got {str(actual)[:64]!r}, expected "
                f"{str(expected)[:64]!r} (stale or foreign "
                "authorization refuses, DATA-SOTA-377)")
    if sorted(auth["trial_ids"]) != sorted(trial_ids):
        raise AuthorizationRefused(
            "authorization trial set differs from the design's eight "
            "trials — refused")
    if not (isinstance(auth["reviewed_correction_commit"], str)
            and _HEX40.match(auth["reviewed_correction_commit"])):
        raise AuthorizationRefused(
            "reviewed_correction_commit is not an exact 40-hex commit "
            "— an unfilled template or malformed binding refuses")
    if not (isinstance(auth["audit_order_commit_digest"], str)
            and (_HEX40.match(auth["audit_order_commit_digest"])
                 or _HEX64.match(auth["audit_order_commit_digest"]))):
        raise AuthorizationRefused(
            "audit_order_commit_digest is not a commit/content digest")
    if auth["launch_manifest_sha256"] != launch_manifest_sha256:
        raise AuthorizationRefused(
            "authorization launch-manifest digests differ from the "
            "committed manifests — refused")
    if not str(auth["issued_utc"]).strip() or "TO_BE" in str(
            auth["issued_utc"]).upper():
        raise AuthorizationRefused(
            "issued_utc is blank or an unfilled template placeholder")
    return auth


# --- DATA-SOTA-379: executable identity -----------------------------

def _module_file(module_name: str) -> Path:
    import importlib
    module = importlib.import_module(module_name)
    return Path(module.__file__).resolve()


def executable_manifest(repo: Path) -> dict[str, str]:
    """Canonical {logical name: sha256} over every executing file,
    computed from bytes NOW. The environment and envelope modules are
    resolved from the installed environment exactly as the trainer
    will import them."""
    repo = Path(repo)
    files = {
        "driver": repo / "tools/dispatch_paired_pretrain_comparison.py",
        "nested_pipeline":
            repo / "pipeline_plugins/rl_pipeline_with_validation.py",
        "nested_splits": repo / "pipeline_plugins/_nested_splits.py",
        "observation_contract":
            repo / "pipeline_plugins/_observation_contract.py",
        "sac_agent": repo / "agent_plugins/sac_agent.py",
        "grouped_architecture":
            repo / "agent_plugins/grouped_architecture.py",
        "grouped_features_extractor":
            repo / "agent_plugins/grouped_features_extractor.py",
        "pretrained_branch_loader":
            repo / "agent_plugins/pretrained_branch_loader.py",
        "dispatch_custody": repo / "agent_plugins/dispatch_custody.py",
        "dispatch_authorization": Path(__file__).resolve(),
        "env_gym_fx": _module_file("env_plugins.gym_fx_env"),
        "strategy_shared_execution_envelope":
            _module_file("strategy_plugins.shared_execution_envelope"),
        "split_contract": repo / ("examples/config/"
                                  "phase_3_eth_sac_dynamics/splits/"
                                  "eth_nested_split_contract_o2022_"
                                  "paired_v1.json"),
        "strong_config": repo / ("examples/config/"
                                 "project3_ethusdt_4h_sac_grouped_"
                                 "strong_v1.json"),
        "cost_manifest": repo / ("examples/config/"
                                 "phase_3_eth_sac_dynamics/"
                                 "cost_manifest_eth_h4_v2.json"),
        "design": repo / ("docs/audits/evidence/"
                          "PAIRED_PRETRAIN_COMPARISON_DESIGN_"
                          "2026_08_27.json"),
        "candidate_manifest": repo / ("docs/audits/evidence/"
                                      "CANDIDATE_GENERATION_MANIFEST_"
                                      "2026_08_28.json"),
        "envelope_calibration": repo / ("docs/audits/evidence/"
                                        "screen_b_rule_arms_v3_"
                                        "20260826/ENVELOPE_"
                                        "CALIBRATION_o2022.json"),
    }
    return {name: _sha_file(path)
            for name, path in sorted(files.items())}


def executable_manifest_digest(manifest: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def verify_worktree_identity(repo: Path, *,
                             expected_commit: str) -> dict[str, Any]:
    """Exact 40-hex HEAD equality plus tracked AND untracked
    cleanliness — a modified trainer can never run under an accepted
    cell genesis (DATA-SOTA-379)."""
    repo = Path(repo)
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    if not _HEX40.match(head):
        raise AuthorizationRefused(f"unparseable HEAD {head!r}")
    if head != expected_commit:
        raise AuthorizationRefused(
            f"executing HEAD {head[:12]} != authorized reviewed "
            f"commit {str(expected_commit)[:12]} — refused "
            "(DATA-SOTA-379)")
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain",
         "--untracked-files=normal"],
        capture_output=True, text=True, check=True).stdout.strip()
    if status:
        raise AuthorizationRefused(
            "worktree is not clean (tracked or untracked changes "
            "present) — an unpinned tree never executes "
            "(DATA-SOTA-379):\n" + status[:800])
    return {"head": head, "clean": True}
