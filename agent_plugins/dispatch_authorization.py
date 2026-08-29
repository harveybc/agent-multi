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
        # the EXECUTING gym-fx core (fill-truth order 2026-08-28):
        # trade accounting lives in these files — they are part of the
        # executable identity like every other executing module
        "gym_fx_core_env":
            _module_file("gym_fx").parent.parent / "app/env.py",
        "gym_fx_core_bridge":
            _module_file("gym_fx").parent.parent / "app/bt_bridge.py",
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
    manifest = {name: _sha_file(path)
                for name, path in sorted(files.items())}
    # DATA-SOTA-382: the executing environment's entry-point metadata
    # is part of the executable identity — PYTHONPATH alone is not
    manifest["entry_point_metadata"] = resolve_required_entry_points(
        repo)["entry_point_metadata_digest"]
    return manifest


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


# --- DATA-SOTA-382: executing environmental preflight ----------------

REQUIRED_ENTRY_POINTS = (
    ("pipeline.plugins", "rl_pipeline_with_validation"),
    ("agent.plugins", "sac_agent"),
    ("env.plugins", "gym_fx_env"),
    ("preprocessor.plugins", "feature_window_preprocessor"),
    ("strategy.plugins", "shared_execution_envelope"),
    ("feature_branch.plugins", "patchtst_branch"),
    ("feature_branch.plugins", "tft_branch"),
    ("feature_branch.plugins", "timesnet_branch"),
    ("feature_branch.plugins", "tcn_branch"),
    ("feature_branch.plugins", "gru_branch"),
    ("feature_branch.plugins", "mlp_branch"),
    ("feature_fusion.plugins", "cross_family_attention"),
)


def resolve_required_entry_points(repo: Path) -> dict[str, Any]:
    """DATA-SOTA-382: the preflight must prove that the EXECUTING
    environment's installed entry-point metadata resolves every
    required plugin — `PYTHONPATH` visibility alone proved nothing on
    the fleet. Missing metadata, duplicated registrations, or a
    resolution outside the pinned worktree/installation roots refuse.
    Returns per-plugin {distribution, version, file, sha256} plus a
    canonical digest of the whole resolution."""
    import importlib
    import importlib.metadata as md
    import site
    import sysconfig

    repo = Path(repo).resolve()
    allowed_roots = [repo]
    for root in {sysconfig.get_paths().get("purelib"),
                 sysconfig.get_paths().get("platlib"),
                 *site.getsitepackages()}:
        if root:
            allowed_roots.append(Path(root).resolve())
    # editable installs execute from their checkout: pin the sibling
    # checkout roots (…/GitHub and …/GitHub/.worktrees layouts)
    allowed_roots.append(repo.parent.resolve())
    allowed_roots.append(repo.parent.parent.resolve())
    resolution: dict[str, Any] = {}
    for group, name in REQUIRED_ENTRY_POINTS:
        matches = [ep for ep in md.entry_points(group=group)
                   if ep.name == name]
        if not matches:
            raise AuthorizationRefused(
                f"entry point {group}:{name} is NOT registered in the "
                "executing environment's installed metadata — "
                "PYTHONPATH visibility alone does not execute "
                "(DATA-SOTA-382)")
        if len(matches) > 1:
            dists = sorted({(ep.dist.name if ep.dist else "?")
                            for ep in matches})
            raise AuthorizationRefused(
                f"entry point {group}:{name} is registered "
                f"{len(matches)} times (distributions {dists}) — a "
                "duplicated registration makes resolution "
                "order-dependent and refuses (DATA-SOTA-382)")
        ep = matches[0]
        module = importlib.import_module(
            ep.value.split(":")[0])
        resolved = Path(module.__file__).resolve()
        if not any(str(resolved).startswith(str(root) + "/")
                   or resolved == root
                   for root in allowed_roots):
            raise AuthorizationRefused(
                f"{group}:{name} resolves to {resolved.name} OUTSIDE "
                "the pinned worktree/installation roots — refused "
                "(DATA-SOTA-382)")
        resolution[f"{group}:{name}"] = {
            "distribution": ep.dist.name if ep.dist else None,
            "version": ep.dist.version if ep.dist else None,
            "file": str(resolved),
            "sha256": _sha_file(resolved),
        }
    digest = hashlib.sha256(json.dumps(
        {k: {"distribution": v["distribution"],
             "version": v["version"], "sha256": v["sha256"]}
         for k, v in resolution.items()},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"entry_points": resolution,
            "entry_point_metadata_digest": digest}


def bounded_extractor_forward(repo: Path, device: str) -> dict[str, Any]:
    """DATA-SOTA-382: instantiate the SAME architecture materializer
    the trainer uses and run one bounded forward on the SELECTED
    device — construction-only proof that the resolved environment
    actually executes there."""
    import torch

    from agent_plugins.grouped_architecture import (
        materialize_from_config, snapshot_effective_config)
    from agent_plugins.grouped_features_extractor import (
        build_grouped_extractor_class)

    snapshot = snapshot_effective_config(
        Path(repo) / "examples/config/"
        "project3_ethusdt_4h_sac_grouped_strong_v1.json")
    materialized = snapshot["materialized"]
    architecture = materialized["architecture"]
    window = int(snapshot["env_config"]["window_size"])
    feature_count = len(snapshot["env_config"]["feature_columns"])
    state_keys = architecture.get("state_keys") or []
    import gymnasium as gym
    import numpy as np
    spaces = {"features": gym.spaces.Box(-np.inf, np.inf,
                                         (window, feature_count),
                                         dtype=np.float32)}
    for key in state_keys:
        # live_stationary_v2: each agent-state key is one scalar box
        spaces[key] = gym.spaces.Box(-np.inf, np.inf, (1,),
                                     dtype=np.float32)
    observation_space = gym.spaces.Dict(spaces)
    extractor_cls = build_grouped_extractor_class()
    torch.manual_seed(0)
    extractor = extractor_cls(observation_space,
                              architecture=architecture).to(device)
    extractor.eval()
    batch = {key: torch.zeros((2,) + space.shape,
                              device=device)
             for key, space in spaces.items()}
    with torch.no_grad():
        out = extractor(batch)
    if not torch.isfinite(out).all():
        raise AuthorizationRefused(
            "bounded preflight forward produced non-finite output "
            "(DATA-SOTA-382)")
    return {"device": device, "output_shape": list(out.shape),
            "architecture_digest": materialized[
                "architecture_digest"]}


# --- DATA-SOTA-383: proven device binding + cuDNN micro-preflight ----

PRIVATE_BINDING_PATH = (Path.home() / ".local/share/agent-multi/"
                        "restricted_evidence/"
                        "paired_sac_fleet_private_binding_20260828"
                        ".json")


def verify_device_binding(logical_slot: str,
                          binding_path: Path | None = None
                          ) -> dict[str, Any]:
    """DATA-SOTA-383: `CUDA_VISIBLE_DEVICES` proves visibility, not
    identity — on one fleet host ordinal 0 is a different physical
    class for PyTorch than for nvidia-smi. The OPERATOR's private
    plan (restricted store, never committed) binds each logical slot
    to an expected physical class and a local device identity; after
    the environment is applied, PyTorch must see exactly ONE device
    whose class and local identity match the slot. Public evidence
    receives ONLY the sanitized class and the slot."""
    import torch

    if torch.cuda.device_count() != 1:
        raise AuthorizationRefused(
            f"{torch.cuda.device_count()} CUDA devices visible — "
            "exactly ONE per cell process (DATA-SOTA-380/383)")
    path = Path(binding_path or PRIVATE_BINDING_PATH)
    if not path.is_file():
        raise AuthorizationRefused(
            "operator private device-binding plan is absent — GPU "
            "execution requires the slot→physical binding "
            "(DATA-SOTA-383)")
    plan = json.loads(path.read_text())
    entry = (plan.get("slots") or {}).get(logical_slot)
    if not entry:
        raise AuthorizationRefused(
            f"slot {logical_slot!r} is not in the operator binding "
            "plan (DATA-SOTA-383)")
    name = torch.cuda.get_device_name(0)
    expected_class = str(entry.get("expected_device_class") or "")
    if not expected_class or "TO_BE" in expected_class.upper():
        raise AuthorizationRefused(
            "operator binding plan is unfilled for "
            f"{logical_slot} (DATA-SOTA-383)")
    if expected_class not in name:
        raise AuthorizationRefused(
            f"visible device class {name!r} does not match the "
            f"slot's expected class {expected_class!r} — wrong "
            "physical device (DATA-SOTA-383)")
    expected_local = entry.get("local_identity")
    if expected_local:
        props = torch.cuda.get_device_properties(0)
        local = str(getattr(props, "uuid", "")) or None
        if local != str(expected_local):
            raise AuthorizationRefused(
                "visible device local identity does not match the "
                f"slot binding for {logical_slot} — wrong physical "
                "device (DATA-SOTA-383); identities stay in the "
                "restricted plan and are never published")
    # sanitized: class + slot ONLY — no UUID, no bus id, no host
    return {"logical_slot": logical_slot,
            "device_class_sanitized": name,
            "local_identity_verified": bool(expected_local)}


def cudnn_micro_preflight(device: str = "cuda") -> dict[str, Any]:
    """DATA-SOTA-383: a REAL Conv2d forward/backward plus device
    synchronization on the bound device BEFORE any custody
    reservation — a transient cuDNN failure refuses without spending
    an attempt identity."""
    import time

    import torch

    start = time.perf_counter()
    conv = torch.nn.Conv2d(3, 8, 3).to(device)
    x = torch.randn(8, 3, 32, 32, device=device, requires_grad=True)
    y = conv(x)
    loss = y.square().mean()
    loss.backward()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    if not (torch.isfinite(y).all()
            and torch.isfinite(x.grad).all()
            and all(torch.isfinite(p.grad).all()
                    for p in conv.parameters())):
        raise AuthorizationRefused(
            "cuDNN micro-preflight produced non-finite tensors — "
            "the device is not fit for the cell (DATA-SOTA-383)")
    return {"device": device, "conv2d_forward_backward_ok": True,
            "wall_ms": round((time.perf_counter() - start) * 1e3, 1)}
