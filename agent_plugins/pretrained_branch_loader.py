"""Pretrained branch-encoder transfer loader (Musashi dispatch
2026-08-27, automatic consequence of accepted DATA-SOTA-353..356).

Smallest reusable loader consistent with the plugin architecture: no
loading logic lives in an experiment driver, and NOTHING about the
topology is inferred from checkpoint shapes — the declared v4 contract
is the only authority.

Verification chain before any tensor moves:

* generation seal (torn/substituted artifacts refuse);
* the referenced contract re-validates as v4 and matches its sealed
  digest; source-data digest, preprocessing identity (module sha +
  scaling-config digest), ordered 83-feature partition, per-family
  ordered digests, branch-assignment/topology digest and origin-plan
  digest all match the sealed identity — any drift refuses;
* training-code identity: the sealed library/runner file digests must
  equal the CURRENT files (refusal on drift); the git commit is bound
  and reported (a later commit that touches neither file is
  legitimate and is reported, not refused).

Loading contract:

* encoder state ONLY, by NAMED family, each file digest-bound to its
  family in the sealed manifest — a valid tensor under the wrong
  family digest refuses;
* strict key/shape/dtype equality against the freshly constructed
  declared module: missing, extra, renamed, reordered, duplicated,
  cross-family and head-injected keys refuse; optimizer/calibration
  payloads refuse as a typed category;
* post-load bit-for-bit tensor parity against the sealed source state
  is proven by re-serialization comparison;
* the forward output must be finite; NaN/Inf is a typed failure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_plugins.branch_pretraining import (
    canonical_feature_digest, load_generation, sha256_file, sha256_obj,
    validate_branch_partition, validate_contract)

TRANSFER_STATUS = "MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE"


class TransferLoadError(ValueError):
    """Typed refusal: the pretrained artifact, its identity or the
    offered state is invalid for transfer. Never load, never forward."""


def verify_source(pretrain_dir: Path, repo_root: Path,
                  data_path: Path) -> dict[str, Any]:
    """Run the full identity chain; returns {manifest, contract,
    parsed, partition, code_identity_report} or refuses."""
    _ckpt, manifest, _generation = load_generation(pretrain_dir)
    # M0 quarantine (order 2026-08-27): a generation whose seal is in
    # the quarantine register NEVER loads, whatever its manifest says
    import hashlib as _hashlib

    seal_path = pretrain_dir / "generation.json"
    seal = json.loads(seal_path.read_text())
    register_path = (Path(__file__).resolve().parents[1]
                     / "docs/audits/evidence/"
                       "GENERATION_QUARANTINE_REGISTER.json")
    if register_path.exists():
        register = json.loads(register_path.read_text())
        entry = (register.get("entries") or {}).get(
            seal.get("manifest_sha256"))
        if entry:
            raise TransferLoadError(
                f"generation QUARANTINED as {entry['class']}: "
                f"{entry.get('reason', '')[:80]} — loading refused "
                f"(M0)")
    identity = manifest.get("identity") or {}
    contract_path = repo_root / str(identity.get("contract_path") or "")
    if not contract_path.is_file():
        raise TransferLoadError(
            f"sealed contract absent: {identity.get('contract_path')}")
    actual_contract_sha = sha256_file(contract_path)
    if actual_contract_sha != identity.get("contract_sha256"):
        raise TransferLoadError(
            "contract identity drift: the file at the sealed "
            "contract_path no longer matches the sealed digest")
    contract = json.loads(contract_path.read_text())
    from agent_plugins.branch_pretraining import PretrainContractError
    try:
        parsed = validate_contract(contract)  # v4-only: v3 refuses here
    except PretrainContractError as exc:
        raise TransferLoadError(
            f"sealed contract does not validate for the v4 loader: "
            f"{exc}") from exc

    if not data_path.is_file():
        raise TransferLoadError(f"source dataset absent: {data_path}")
    actual_data_sha = sha256_file(data_path)
    if actual_data_sha != identity.get("data_sha256"):
        raise TransferLoadError(
            "source-data digest drift: the dataset no longer matches "
            "the sealed data_sha256")

    partition = validate_branch_partition(
        list(contract["feature_columns"]), contract["branches"])
    checks = {
        "feature_columns_sha256": partition["global_ordered_digest"],
        "family_ordered_digests_sha256": sha256_obj(
            partition["family_ordered_digests"]),
        "origin_plan_sha256": sha256_obj(contract["origin_plan"]),
        "normalization_policies_digest": sha256_obj(
            parsed["normalization_policies"]),
    }
    for key, actual in checks.items():
        if identity.get(key) != actual:
            raise TransferLoadError(
                f"identity drift on {key}: sealed "
                f"{identity.get(key)!r} vs current {actual!r}")
    assignment = [{"name": b["name"], "plugin": b["plugin"],
                   "params": b.get("params") or {},
                   "features": list(b["features"])}
                  for b in contract["branches"]]
    if identity.get("branch_assignment_sha256") != sha256_obj(assignment):
        raise TransferLoadError(
            "topology digest drift: sealed branch assignment differs "
            "from the contract's declared branches")

    # preprocessing identity against the CURRENT executing plugin
    import inspect

    from app.plugin_loader import load_plugin
    plugin_name = contract["observation_pipeline"]["preprocessor_plugin"]
    plugin_class, _ = load_plugin("preprocessor.plugins", plugin_name)
    if identity.get("preprocessor_module_sha256") != sha256_file(
            Path(inspect.getfile(plugin_class))):
        raise TransferLoadError(
            "preprocessing identity drift: the executing preprocessor "
            "module differs from the sealed one")
    source_config_path = Path(
        contract["observation_pipeline"]["source_config"])
    if not source_config_path.is_absolute():
        source_config_path = repo_root / source_config_path
    env_config = json.loads(source_config_path.read_text())
    keys = ("window_size", "feature_scaling", "feature_scaling_window",
            "feature_binary_columns", "feature_clip")
    preprocessing_identity = {
        key: env_config.get(key, plugin_class.plugin_params.get(key))
        for key in keys}
    if identity.get("preprocessing_config_digest") != sha256_obj(
            preprocessing_identity):
        raise TransferLoadError(
            "preprocessing identity drift: scaling configuration "
            "differs from the sealed digest")

    # training-code identity: file digests refuse on drift; the commit
    # is bound and reported (see module docstring for the rule). The
    # code files live in THIS code tree — repo_root only resolves the
    # artifact's sealed contract path.
    code_tree = Path(__file__).resolve().parents[1]
    library_sha = sha256_file(
        code_tree / "agent_plugins/branch_pretraining.py")
    runner_sha = sha256_file(code_tree / "tools/pretrain_branches.py")
    if identity.get("library_sha256") != library_sha:
        raise TransferLoadError(
            "training-code identity drift: branch_pretraining.py "
            "differs from the sealed library_sha256")
    if identity.get("runner_sha256") != runner_sha:
        raise TransferLoadError(
            "training-code identity drift: pretrain_branches.py "
            "differs from the sealed runner_sha256")
    import subprocess
    current_commit = subprocess.run(
        ["git", "-C", str(code_tree), "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip()
    code_identity_report = {
        "sealed_code_commit": identity.get("code_commit"),
        "current_code_commit": current_commit,
        "commits_equal": identity.get("code_commit") == current_commit,
        "library_sha_equal": True, "runner_sha_equal": True}
    return {"manifest": manifest, "contract": contract,
            "parsed": parsed, "partition": partition,
            "env_config": env_config,
            "code_identity_report": code_identity_report}


def refuse_non_encoder_payload(state: Any, family: str) -> None:
    """Typed category refusals BEFORE key comparison: optimizer,
    replay and calibration payloads are structurally not module
    state."""
    if not isinstance(state, dict) or not state:
        raise TransferLoadError(
            f"{family}: offered transfer state is not a non-empty "
            f"state dict")
    if "param_groups" in state:
        raise TransferLoadError(
            f"{family}: OPTIMIZER state offered as transfer state "
            f"(param_groups present) — refused as a category")
    for marker in ("effective_weights", "generator_state",
                   "replay_buffer", "calibration"):
        if marker in state:
            raise TransferLoadError(
                f"{family}: non-encoder payload key {marker!r} offered "
                f"as transfer state — refused as a category")


def state_accounting(state: dict[str, Any]) -> dict[str, int]:
    import torch

    tensors = [v for v in state.values() if torch.is_tensor(v)]
    return {"tensors": len(state),
            "bytes": int(sum(t.numel() * t.element_size()
                             for t in tensors))}


def strict_load_encoder(module, state: dict[str, Any],
                        family: str) -> int:
    """Strict key/shape/dtype load. Returns the number of tensors
    loaded; every mismatch is a typed refusal."""
    import torch

    refuse_non_encoder_payload(state, family)
    expected = module.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    if missing or extra:
        raise TransferLoadError(
            f"{family}: encoder key set mismatch — missing "
            f"{missing[:4]}, extra/injected {extra[:4]} (renamed, "
            f"head-injected or cross-family keys refuse)")
    for key in expected:
        tensor = state[key]
        if not torch.is_tensor(tensor):
            raise TransferLoadError(
                f"{family}: {key} is not a tensor")
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise TransferLoadError(
                f"{family}: {key} shape {tuple(tensor.shape)} differs "
                f"from declared {tuple(expected[key].shape)}")
        if tensor.dtype != expected[key].dtype:
            raise TransferLoadError(
                f"{family}: {key} dtype {tensor.dtype} differs from "
                f"declared {expected[key].dtype}")
    module.load_state_dict(state, strict=True)
    return len(expected)


def verify_architecture_matches_contract(materialized: dict[str, Any],
                                         contract: dict[str, Any]
                                         ) -> None:
    """DATA-SOTA-357: the transfer target is the EFFECTIVE configured
    architecture. Every temporal branch of the sealed contract must
    equal the materialized architecture's branch EXACTLY (name, plugin,
    params, ordered features) — a weak-route or otherwise different
    architecture refuses instead of being silently replaced."""
    arch_branches = materialized["architecture"]["branches"]
    contract_branches = contract["branches"]
    if len(arch_branches) != len(contract_branches):
        raise TransferLoadError(
            f"architecture/contract branch count mismatch: "
            f"{len(arch_branches)} vs {len(contract_branches)}")
    for index, (arch, sealed) in enumerate(zip(arch_branches,
                                               contract_branches)):
        for field in ("name", "plugin"):
            if str(arch.get(field)) != str(sealed.get(field)):
                raise TransferLoadError(
                    f"branch[{index}] {field} mismatch: effective "
                    f"architecture declares {arch.get(field)!r}, "
                    f"sealed contract {sealed.get(field)!r} — the "
                    f"pretrained encoders do NOT fit the configured "
                    f"architecture (DATA-SOTA-357)")
        if (arch.get("params") or {}) != (sealed.get("params") or {}):
            raise TransferLoadError(
                f"branch[{index}] ({arch.get('name')}) params differ "
                f"between effective architecture and sealed contract")
        if list(arch.get("features") or []) != list(
                sealed.get("features") or []):
            raise TransferLoadError(
                f"branch[{index}] ({arch.get('name')}) ordered "
                f"features differ between effective architecture and "
                f"sealed contract")


def load_family_encoders(pretrain_dir: Path, manifest: dict[str, Any],
                         contract: dict[str, Any],
                         extractor) -> dict[str, Any]:
    """Load every temporal branch encoder from its digest-bound file;
    prove bit parity by re-serialization comparison. Returns per-family
    reports plus a DERIVED accounting object with a conservation
    assertion (DATA-SOTA-357: never a printed literal)."""
    import io

    import torch

    artifacts = manifest.get("artifacts") or {}
    report: dict[str, Any] = {}
    accounting = {"offered_tensors": 0, "offered_bytes": 0,
                  "loaded_tensors": 0, "loaded_bytes": 0,
                  "loaded_per_family": {},
                  "rejected_keys_by_reason": {},
                  "excluded_categories": [
                      "objective_heads/adapters (separate artifact, "
                      "never offered)",
                      "optimizer/replay/calibration payloads (typed "
                      "category refusal before key comparison)"]}
    for index, branch in enumerate(contract["branches"]):
        family = str(branch["name"])
        entry = artifacts.get(family)
        if not entry:
            raise TransferLoadError(
                f"{family}: no sealed artifact entry")
        encoder_path = pretrain_dir / str(entry["encoder_file"])
        if not encoder_path.is_file():
            raise TransferLoadError(
                f"{family}: encoder file absent")
        actual_sha = sha256_file(encoder_path)
        if actual_sha != entry.get("encoder_sha256"):
            raise TransferLoadError(
                f"{family}: encoder file digest {actual_sha[:12]} does "
                f"not match the sealed family digest — a valid tensor "
                f"under the WRONG family refuses")
        state = torch.load(encoder_path, weights_only=True)
        offered = state_accounting(state)
        accounting["offered_tensors"] += offered["tensors"]
        accounting["offered_bytes"] += offered["bytes"]
        module = extractor.temporal_branches[index]
        loaded = strict_load_encoder(module, state, family)
        accounting["loaded_tensors"] += loaded
        accounting["loaded_bytes"] += offered["bytes"]
        accounting["loaded_per_family"][family] = {
            "tensors": loaded, "bytes": offered["bytes"]}
        # bit parity: re-serialize the LOADED module and compare every
        # tensor against the sealed source state. DATA-SOTA-381: the
        # target module may live on CUDA while the sealed state loads
        # on CPU — torch.equal across devices raises, which blocked
        # every treatment cell on the GPU fleet. The comparison happens
        # in ONE common domain by moving EXCLUSIVELY the verification
        # copy of the sealed tensor to the target tensor's device;
        # dtype and content are never altered (a device transfer of an
        # identical-dtype tensor is bit-exact), and the module's own
        # tensors never move.
        buffer = io.BytesIO()
        torch.save(module.state_dict(), buffer)
        buffer.seek(0)
        reloaded = torch.load(buffer, weights_only=True)
        for key in state:
            sealed = state[key]
            if reloaded[key].dtype != sealed.dtype:
                raise TransferLoadError(
                    f"{family}: post-load dtype drift at {key} "
                    f"({reloaded[key].dtype} != {sealed.dtype})")
            verification_copy = sealed.to(reloaded[key].device)
            if not torch.equal(reloaded[key], verification_copy):
                raise TransferLoadError(
                    f"{family}: post-load bit parity FAILED at {key}")
        report[family] = {
            "encoder_file": entry["encoder_file"],
            "encoder_sha256": actual_sha,
            "tensors_loaded": loaded,
            "bit_parity": True,
            "family_feature_digest": canonical_feature_digest(
                list(branch["features"]))}
    rejected_total = sum(accounting["rejected_keys_by_reason"].values())
    if accounting["offered_tensors"] != (accounting["loaded_tensors"]
                                         + rejected_total):
        raise TransferLoadError(
            f"accounting conservation violated: offered "
            f"{accounting['offered_tensors']} != loaded "
            f"{accounting['loaded_tensors']} + rejected "
            f"{rejected_total}")
    accounting["conservation"] = (
        f"offered({accounting['offered_tensors']}) == "
        f"loaded({accounting['loaded_tensors']}) + "
        f"rejected({rejected_total}) — DERIVED, asserted")
    accounting["rejected_total_derived"] = rejected_total
    return {"families": report, "accounting": accounting}


def check_finite_forward(output, label: str = "forward output"):
    import torch

    if not torch.isfinite(output).all():
        raise TransferLoadError(
            f"{label} contains NaN/Inf — typed run failure")
    return output


def load_into_sac_policy(model, pretrain_dir: Path, repo_root: Path,
                         data_path: Path, *,
                         expected_seal_manifest_sha256: str | None = None,
                         materialized: dict[str, Any] | None = None
                         ) -> dict[str, Any]:
    """C3 (SAC driver correction order 2026-08-28): initialize an
    already-constructed SB3 SAC model's grouped feature extractors from
    a sealed pretraining generation — the ONLY treatment difference in
    the paired comparison.

    * runs the full identity chain (verify_source: seal, contract
      revalidation, data digest, quarantine register) before touching
      the model;
    * refuses a shared actor/critic extractor — the accepted strong
      route builds separate extractors (share_features_extractor
      False) and the design binds that;
    * loads the five family encoders into actor, critic AND
      critic_target extractors, each with per-tensor bit parity against
      the sealed artifacts (tensor parity BEFORE the first update);
    * proves trainability: every temporal-branch parameter of actor and
      critic requires_grad AND belongs to that network's optimizer
      param groups (critic_target is polyak-tracked by SAC semantics,
      identical in both arms, and is recorded as such);
    * heads/optimizers from pretraining are never offered (encoder-only
      files; typed category refusal inside load_family_encoders).
    """
    source = verify_source(Path(pretrain_dir), Path(repo_root),
                           Path(data_path))
    contract = source["contract"]
    manifest = source["manifest"]
    seal = json.loads(
        (Path(pretrain_dir) / "generation.json").read_text())
    if expected_seal_manifest_sha256 and \
            seal["manifest_sha256"] != expected_seal_manifest_sha256:
        raise TransferLoadError(
            "sealed generation manifest digest "
            f"{seal['manifest_sha256'][:12]} does not match the design "
            "binding — the cell refuses before any weight moves")
    if materialized is not None:
        verify_architecture_matches_contract(materialized, contract)
    policy = model.policy
    actor_ex = policy.actor.features_extractor
    critic_ex = policy.critic.features_extractor
    target_ex = policy.critic_target.features_extractor
    if actor_ex is critic_ex:
        raise TransferLoadError(
            "actor and critic share one features extractor — the "
            "design requires share_features_extractor False (separate "
            "extractors); refusing")
    reports: dict[str, Any] = {}
    for name, extractor in (("actor", actor_ex),
                            ("critic", critic_ex),
                            ("critic_target", target_ex)):
        if not hasattr(extractor, "temporal_branches"):
            raise TransferLoadError(
                f"{name}: features extractor is not the grouped route")
        reports[name] = load_family_encoders(
            Path(pretrain_dir), manifest, contract, extractor)
    trainability: dict[str, Any] = {}
    for name, network, extractor in (
            ("actor", policy.actor, actor_ex),
            ("critic", policy.critic, critic_ex)):
        optimizer_param_ids = {
            id(p) for group in network.optimizer.param_groups
            for p in group["params"]}
        branch_params = [p for branch in extractor.temporal_branches
                         for p in branch.parameters()]
        if not branch_params:
            raise TransferLoadError(f"{name}: no encoder parameters")
        frozen = sum(1 for p in branch_params if not p.requires_grad)
        outside = sum(1 for p in branch_params
                      if id(p) not in optimizer_param_ids)
        if frozen or outside:
            raise TransferLoadError(
                f"{name}: {frozen} encoder params frozen, {outside} "
                f"outside the optimizer — all encoder parameters must "
                f"remain trainable (order C3)")
        trainability[name] = {
            "encoder_params": len(branch_params),
            "all_requires_grad": True,
            "all_in_optimizer": True}
    trainability["critic_target"] = {
        "polyak_tracked": True,
        "note": "SB3 SAC target network — never optimizer-trained, "
                "identical semantics in both arms"}
    return {
        "seal_manifest_sha256": seal["manifest_sha256"],
        "extractors": reports,
        "trainability": trainability,
        "code_identity": source["code_identity_report"],
    }
