#!/usr/bin/env python3
"""Zero-update 2,660-input SAC genesis artifacts for P1LR v2.

Order 2026-08-15 §3 (AUD-P1LR-20260815-235): the clean start for the
corrected-observation L1 factorial is a **zero-update genesis artifact
per seed** — NOT an adapted 2,724-input anchor, and NOT ad hoc
independent model construction ("independent model construction
without a persisted tensor identity is insufficient").

Per seed {101, 202, 303, 404} this tool:

1. derives the corrected observation dimension from the v2 contract's
   observation declaration and the held-fixed base config's ORDERED
   feature columns (32 bars x 83 features + 4 agent state = 2,660),
   refusing any drift of the pinned ``feature_columns_sha256`` or of
   the declared expected dimension;
2. constructs a SAC model DETERMINISTICALLY on CPU through the very
   agent plugin the factorial trains with
   (``agent_plugins.sac_agent.Plugin.build``), seeded with the seed,
   against a spaces-only environment whose ``step`` RAISES — zero
   gradient updates and zero replay transitions are structural, not
   asserted;
3. proves zero updates (``num_timesteps == 0``, ``_n_updates == 0``,
   replay buffer position 0), proves construction DETERMINISM (a
   second in-process build hashes to the same policy tensors), and
   persists the artifact;
4. records the policy-tensor sha256 (the digest family the P1LR
   runner uses for terminals: ``agent_plugins.sac_agent.
   _policy_tensor_hash``) AND the container sha256, and proves the
   tensor identity survives the save/load round trip;
5. writes a typed per-seed metadata file and one manifest proving
   (i) all four cells of a seed begin from this ONE persisted tensor
   (the v2 contract binds exactly this artifact into every cell, and
   the runner re-proves the pin before each cell materializes) and
   (ii) different seeds carry pairwise DISTINCT genesis tensors.

The artifact type is ``zero_update_genesis``. It is NEVER a trained
champion, NEVER a handoff, NEVER promotable evidence — it is the
paired initial condition of a cold start with an exact, persisted,
hash-bound tensor identity.

Typed outcomes: GENESIS_MATERIALIZED | GENESIS_CHECK_PASS |
GENESIS_REFUSED (exit 4).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from tools import m0_l1_mechanism_ladder as ladder  # noqa: E402
from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tools.l1_factorial_screen import atomic_write_json  # noqa: E402

ARTIFACT_SCHEMA = "agent_multi.p1lr_zero_update_genesis.v1"
MANIFEST_SCHEMA = "agent_multi.p1lr_zero_update_genesis_manifest.v1"
ARTIFACT_TYPE = p1.GENESIS_ARTIFACT_TYPE  # "zero_update_genesis"

#: Construction-only replay capacity. The buffer stays EMPTY (step()
#: raises); a tiny capacity avoids allocating gigabytes for a buffer
#: that must never hold a transition. The artifact is consumed as a
#: WEIGHT SOURCE only: the runner's warm start builds the training
#: model from the cell's own config and copies the policy tensors
#: exactly (sac_agent.load_for_training), so no genesis-side
#: hyperparameter reaches training.
CONSTRUCTION_BUFFER_SIZE = 100


def _spaces_only_env(observation_dim: int):
    """Minimal Gymnasium env carrying ONLY the contract spaces.

    ``step`` raises: a genesis construction that tried to collect even
    one transition would crash instead of silently training.
    """
    import gymnasium
    import numpy as np
    from gymnasium import spaces

    class _SpacesOnlyEnv(gymnasium.Env):
        metadata: dict = {}

        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(int(observation_dim),), dtype=np.float32)
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.render_mode = None

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return (np.zeros(self.observation_space.shape,
                             dtype=np.float32), {})

        def step(self, action):
            raise RuntimeError(
                "zero-update genesis: the spaces-only environment can "
                "never be stepped — no replay transition may exist "
                "(order 2026-08-15 §3)")

    return _SpacesOnlyEnv()


def load_v2_contract(path: Path) -> dict:
    """The v2 contract via the runner's fail-closed loader (the
    observation and genesis declarations are validated there)."""
    contract = p1.load_contract(path)
    if p1.contract_version(contract) != 2:
        raise ValueError(
            "genesis artifacts exist only for the corrected v2 "
            f"contract; {path} is not schema {p1.CONTRACT_SCHEMA_V2}")
    return contract


def resolve_observation_dimension(contract: dict,
                                  bindings: dict) -> dict:
    """Derive the corrected dimension from the contract + the
    held-fixed base config's ordered feature columns, fail-closed."""
    from pipeline_plugins._observation_contract import (
        feature_columns_sha256)

    common = bindings["common"]
    base_path = REPO / common["base_config"]["path"]
    actual_base_sha = p1._sha_file(base_path)
    if actual_base_sha != common["base_config"]["sha256"]:
        raise RuntimeError(
            "held-fixed base config drifted from the ladder binding — "
            "the genesis would be constructed against an unpinned "
            "feature set")
    base_config = json.loads(base_path.read_text())
    columns = base_config.get("feature_columns") or []
    columns_sha = feature_columns_sha256(columns)
    obs = contract["observation_contract"]
    if columns_sha != obs["feature_columns_sha256"]:
        raise RuntimeError(
            f"ordered feature columns hash to {columns_sha[:16]}… but "
            f"the v2 contract pins "
            f"{obs['feature_columns_sha256'][:16]}… — the ordered "
            "feature contract moved (finding 235)")
    expected = contract["expected_observation"]
    if len(columns) != int(expected["feature_count"]):
        raise RuntimeError(
            f"base config carries {len(columns)} feature columns; the "
            f"contract expects {expected['feature_count']}")
    derived = (int(obs["window_size"]) * len(columns)
               + int(expected["agent_state_dims"]))
    if derived != int(expected["expected_dimension"]):
        raise RuntimeError(
            f"derived observation dimension {derived} != declared "
            f"{expected['expected_dimension']} (finding 235)")
    return {
        "observation_dim": derived,
        "window_size": int(obs["window_size"]),
        "feature_count": len(columns),
        "agent_state_dims": int(expected["agent_state_dims"]),
        "feature_columns_sha256": columns_sha,
        "base_config_sha256": actual_base_sha,
        "net_arch": list(base_config.get("net_arch") or (256, 256)),
        "ent_coef": base_config.get("ent_coef"),
    }


def _build_model(seed: int, observation_dim: int,
                 obs_facts: dict):
    """One deterministic CPU construction through the real agent
    plugin. Returns (model, plugin, env)."""
    from agent_plugins.sac_agent import Plugin as SacPlugin

    env = _spaces_only_env(observation_dim)
    plugin = SacPlugin()
    config = {
        "device": "cpu",
        "train_seed": int(seed),
        "net_arch": tuple(obs_facts["net_arch"]),
        "ent_coef": obs_facts["ent_coef"],
        "buffer_size": CONSTRUCTION_BUFFER_SIZE,
        "use_sde": False,
    }
    model = plugin.build(env, config)
    return model, plugin, env


def _zero_update_proof(model) -> dict:
    """Typed structural proof: zero gradient updates, zero replay
    transitions. Refuses (never records) anything else."""
    n_updates = int(getattr(model, "_n_updates", -1))
    num_timesteps = int(getattr(model, "num_timesteps", -1))
    buffer = getattr(model, "replay_buffer", None)
    replay_positions = int(buffer.size()) if buffer is not None else 0
    if n_updates != 0 or num_timesteps != 0 or replay_positions != 0:
        raise RuntimeError(
            f"GENESIS_NOT_ZERO_UPDATE: n_updates={n_updates}, "
            f"num_timesteps={num_timesteps}, "
            f"replay_positions={replay_positions} — a genesis with "
            "any update or transition is not a genesis (order §3)")
    return {
        "gradient_updates": 0,
        "num_timesteps": 0,
        "replay_transitions_written": 0,
        "environment_steps_possible": False,
        "proof": ("constructed against a spaces-only environment "
                  "whose step() raises; n_updates, num_timesteps and "
                  "replay position verified 0 before save"),
    }


def build_seed_genesis(contract: dict, bindings: dict, seed: int,
                       output_root: Path) -> dict:
    """Construct, prove, persist and hash ONE seed's genesis."""
    from agent_plugins.sac_agent import _policy_tensor_hash

    obs_facts = resolve_observation_dimension(contract, bindings)
    dim = obs_facts["observation_dim"]

    model, plugin, env = _build_model(seed, dim, obs_facts)
    zero_proof = _zero_update_proof(model)
    tensor_sha = _policy_tensor_hash(model.policy)

    # Determinism proof: a second independent construction with the
    # same seed reproduces the identical policy tensors — the four
    # cells of this seed therefore begin from ONE tensor identity no
    # matter which of them loads the artifact first.
    rebuilt, _plugin2, env2 = _build_model(seed, dim, obs_facts)
    rebuilt_sha = _policy_tensor_hash(rebuilt.policy)
    if rebuilt_sha != tensor_sha:
        raise RuntimeError(
            f"GENESIS_NONDETERMINISTIC: seed {seed} constructions "
            f"hash {tensor_sha[:16]}… vs {rebuilt_sha[:16]}… — a "
            "non-reproducible genesis cannot pair the four cells")
    del rebuilt
    env2.close()

    seed_dir = output_root / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    artifact = seed_dir / f"zero_update_genesis_seed{seed}.zip"
    if artifact.exists():
        raise RuntimeError(
            f"GENESIS_EXISTS: {artifact} already exists — a persisted "
            "genesis identity is immutable; use a fresh output root "
            "instead of overwriting a pinned artifact")
    plugin.save(model, str(artifact))
    del model
    env.close()

    container_sha = p1._sha_file(artifact)
    # Round-trip identity proof with the runner's OWN digest path.
    reloaded_sha = p1._genesis_tensor_sha(artifact)
    if reloaded_sha != tensor_sha:
        raise RuntimeError(
            f"GENESIS_IDENTITY_LOST: seed {seed} tensors hash "
            f"{tensor_sha[:16]}… in memory but {reloaded_sha[:16]}… "
            "after save/load — the persisted artifact is not the "
            "constructed identity")
    observed_dim = p1.policy_observation_dim(artifact)
    if observed_dim != dim:
        raise RuntimeError(
            f"GENESIS_WRONG_DIMENSION: persisted artifact carries a "
            f"{observed_dim}-input first layer, expected {dim}")

    entry = {
        "schema": ARTIFACT_SCHEMA,
        "artifact_type": ARTIFACT_TYPE,
        "never_a_trained_champion_or_handoff": True,
        "seed": int(seed),
        "path": str(artifact),
        "container_sha256": container_sha,
        "policy_tensor_sha256": tensor_sha,
        "observation_dim": observed_dim,
        "action_dim": 1,
        "net_arch": list(obs_facts["net_arch"]),
        "ent_coef": obs_facts["ent_coef"],
        "construction_buffer_size": CONSTRUCTION_BUFFER_SIZE,
        "construction_hyperparameters_inert": (
            "the runner's warm start builds the training model from "
            "the CELL's config and copies policy tensors exactly "
            "(sac_agent.load_for_training); nothing but the tensor "
            "identity leaves this artifact"),
        "zero_update_proof": zero_proof,
        "construction_deterministic": True,
        "identity_preserved_after_save_load": True,
        "observation_binding": {
            key: obs_facts[key]
            for key in ("window_size", "feature_count",
                        "agent_state_dims", "feature_columns_sha256",
                        "base_config_sha256")},
        "observation_contract_sha256":
            p1.observation_contract_sha256(contract),
        "contract_schema": contract.get("schema"),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(seed_dir / f"zero_update_genesis_seed{seed}.json",
                      entry)
    return entry


def materialize(contract: dict, bindings: dict,
                output_root: Path, seeds=None) -> dict:
    """Build every requested seed's genesis + the typed manifest."""
    seeds = [int(s) for s in (seeds or p1.SEEDS)]
    entries: dict = {}
    for seed in seeds:
        entries[str(seed)] = build_seed_genesis(
            contract, bindings, seed, output_root)

    tensor_shas = {seed: entry["policy_tensor_sha256"]
                   for seed, entry in entries.items()}
    distinct = len(set(tensor_shas.values())) == len(tensor_shas)
    if len(seeds) > 1 and not distinct:
        raise RuntimeError(
            f"GENESIS_SEEDS_NOT_DISTINCT: {tensor_shas} — different "
            "seeds must have DISTINCT genesis tensors (order §3)")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "outcome": "GENESIS_MATERIALIZED",
        "artifact_type": ARTIFACT_TYPE,
        "never_a_trained_champion_or_handoff": True,
        "finding": "AUD-P1LR-20260815-235",
        "contract_schema": contract.get("schema"),
        "contract_sha256_at_generation": contract.get(
            "_contract_sha256"),
        "observation_contract_sha256":
            p1.observation_contract_sha256(contract),
        "seeds": entries,
        "same_tensor_within_seed": {
            "cells": list(p1.CELLS),
            "claim": ("all four cells of a seed begin from the SAME "
                      "policy tensor"),
            "proof": ("one persisted artifact per seed; the v2 "
                      "contract pins its container AND policy-tensor "
                      "sha into every cell of that seed; the runner "
                      "re-proves both pins before each cell "
                      "materializes and the warm start copies the "
                      "policy state dict exactly "
                      "(policy_hash_matches_source_after_transfer); "
                      "construction is additionally proven "
                      "deterministic per seed"),
            "per_seed_policy_tensor_sha256": tensor_shas,
        },
        "distinct_across_seeds": {
            "claim": "different seeds have DISTINCT genesis tensors",
            "pairwise_distinct": distinct,
            "distinct_tensor_count": len(set(tensor_shas.values())),
            "seed_count": len(tensor_shas),
        },
        "subject_code_identity": ladder.source_identities(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(output_root / "genesis_manifest.json", manifest)
    return manifest


def check(contract: dict, *, seeds=None) -> dict:
    """Verify the contract's pinned genesis artifacts on THIS host
    (existence, container sha, tensor sha, dimension, distinctness)
    without building anything — the distribution-side custody check."""
    seeds = [int(s) for s in (seeds or p1.SEEDS)]
    report: dict = {}
    refusals: list = []
    for seed in seeds:
        try:
            facts = p1._verify_initialization(contract, seed)
            report[str(seed)] = {
                "verified": True,
                "path": facts["path"],
                "container_sha256": facts["container_sha256"],
                "policy_tensor_sha256": facts["policy_tensor_sha256"],
                "observation_dim": facts["observation_dim"],
            }
        except RuntimeError as exc:
            refusals.append(str(exc))
            report[str(seed)] = {"verified": False, "error": str(exc)}
    return {
        "schema": MANIFEST_SCHEMA,
        "outcome": ("GENESIS_CHECK_PASS" if not refusals
                    else "GENESIS_REFUSED"),
        "mode": "check",
        "seeds": report,
        "refusals": refusals,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path,
                        default=p1.CONTRACT_PATH_V2)
    parser.add_argument("--output-root", type=Path, default=None,
                        help="defaults to the contract's "
                             "genesis.output_root")
    parser.add_argument("--seed", type=int, action="append",
                        default=None, choices=list(p1.SEEDS),
                        help="restrict to specific seeds (repeatable); "
                             "default: all four")
    parser.add_argument("--check", action="store_true",
                        help="verify the PINNED artifacts instead of "
                             "building (distribution custody check)")
    args = parser.parse_args()
    contract = load_v2_contract(args.contract)
    if args.check:
        payload = check(contract, seeds=args.seed)
        print(json.dumps(payload, default=str), flush=True)
        return 0 if payload["outcome"] == "GENESIS_CHECK_PASS" else 4
    bindings = p1.load_bindings()
    output_root = Path(
        args.output_root
        or contract["genesis"]["output_root"]).expanduser()
    try:
        manifest = materialize(contract, bindings, output_root,
                               seeds=args.seed)
    except RuntimeError as exc:
        print(json.dumps({"outcome": "GENESIS_REFUSED",
                          "error": str(exc)}, default=str), flush=True)
        return 4
    print(json.dumps(manifest, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
