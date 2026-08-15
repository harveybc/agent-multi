#!/usr/bin/env python3
"""L2 curriculum arms: L2_N versus L2_EN (doc 38 §6/§8, order 2026-08-15 §2).

The executable contract is
``examples/config/phase_3_eth_sac_dynamics/l2_curriculum_arms_v1.json``.
This runner is the ONLY path from that contract to a running candidate,
and it deliberately reuses the machinery that is already proven instead
of building a second ad-hoc runner:

  contracts/identity   the P1LR runner's fail-closed loader style,
                       content-addressed identities and per-mode roots;
  splits/evidence      ``pipeline_plugins._nested_splits`` (ONE
                       implementation) plus the P1LR nested binding and
                       role-fact verifiers, and the ONE context-prefix
                       replay ``p1._outer_validation_final_eval`` for
                       BOTH members of the L2 pair;
  comparator           ``pipeline_plugins._paired_generalization``
                       (``paired_generalization_weekly_v1`` and
                       ``PairedStoppingState``);
  optimizer            ``optimizer_plugins.l2_curriculum_optimizer``,
                       itself a subclass of the shared-population DEAP
                       optimizer doin-node already drives;
  custody/claims       ``tools.l1_fleet_launcher.ExclusiveClaim`` and the
                       atomic-json/heartbeat/GPU-gate primitives used by
                       the P1LR fleet.

The ten executable requirements of the dispatch order and where they are
enforced:

 1  L1 FROZEN            ``_l2.assert_l1_recipe_frozen`` +
                         ``_l2.assert_no_l1_gene_in_l2_space`` +
                         ``_l2.assert_stage_difficulty_binding``
 2  easy invalidation    ``_l2.invalidate_easy_fitness``,
                         ``_l2.assert_all_elites_reevaluated``,
                         ``_l2.assert_normal_leaderboard``,
                         ``_l2.assert_promotion_eligible``
 3  identical budgets    ``_l2.assert_arms_identical_budget`` +
                         ``assert_arms_comparable``
 4  typed genes only     ``_l2.assert_typed_genes_only`` +
                         ``_l2.assert_common_anchor_not_inherited``
 5  paired L2 patience   ``_l2.l2_stopping_state`` +
                         ``_l2.l2_generation_stop_decision``
 6  diversity logging    ``_l2.population_diversity`` +
                         ``_l2.assert_diversity_logged``
 7  data roles           ``materialize_l2_nested_roles`` (l2 mode; the
                         sealed 2025 year is never materialized)
 8  node identity        ``node_identity_facts`` + ``verify_node_identity``
 9  one chain per arm    ``assert_single_chain_per_arm`` +
                         ``assert_sequential_arm_dispatch`` +
                         ``open_or_join_chain``
10  four-GPU smoke       ``--mode smoke`` + ``smoke_dispatch_commands``

Three ADDITIONAL gates close the failure mode that voided three prior
experiments (finding D8-20260815). That failure was never a measurement
bug: when no checkpoint passed the trade/activity gate, selection fell
back to the untouched warm-start anchor, so every arm scored the SAME
object and the campaign silently measured the anchor instead of the
treatment. In ``eth_curriculum_decision_20260807_v2`` the arms E4,
EN4_10 and N14 all selected checkpoints whose POLICY TENSOR is
bit-identical to ``anchor_seed101.zip`` (``8137b224…``), so the paired
EN-minus-N difference was subtracting the anchor from itself.

11  anchor-fallback      ``selected_checkpoint_anchor_facts`` (per
                         candidate, inside ``evaluate_candidate``) +
                         ``assert_arm_not_anchor_fallback`` (per arm,
                         inside ``champion_of_chain``)
12  arm differentiation  ``compare_arms`` →
                         ``arm_differentiation.assert_arms_differentiated``
13  campaign verdict     ``compare_arms(require_informative=True)``, so an
                         all-degenerate campaign is typed
                         CAMPAIGN_UNANSWERABLE instead of published as a
                         scientific null; surfaced by ``smoke_acceptance``

Identity of a checkpoint is ALWAYS its policy TENSOR hash, never the
``.zip`` container digest: Stable-Baselines3 containers embed member
timestamps, so the same weights saved twice hash differently and a
container rule would have missed every one of those three sites.

Nothing here authorizes a launch. A smoke or decision dispatch still
requires the GPU readiness gate, the contract-assigned host and the
contract-assigned GPU UUID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from optimizer_plugins import l2_curriculum_optimizer as _l2  # noqa: E402
from pipeline_plugins import _nested_splits  # noqa: E402
from pipeline_plugins import _paired_generalization as _paired  # noqa: E402
from pipeline_plugins import _system_config as sysid  # noqa: E402
from tools import arm_differentiation as _armdiff  # noqa: E402
from tools import gpu_readiness_probe as gpu_probe  # noqa: E402
from tools import m0_l1_mechanism_ladder as ladder  # noqa: E402
from tools import p1_difficulty_lr_factorial as p1  # noqa: E402
from tools.l1_factorial_screen import atomic_write_json  # noqa: E402
from tools.l1_fleet_launcher import (  # noqa: E402
    ExclusiveClaim,
    _atomic_json,
    _pid_start_identity,
    visible_gpu_uuids,
)

CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "l2_curriculum_arms_v1.json")
CONTRACT_SCHEMA = "agent_multi.l2_curriculum_arms.v1"
CHAIN_SCHEMA = "agent_multi.l2_chain_registry.v1"
CANDIDATE_RECORD_SCHEMA = "agent_multi.l2_candidate_record.v1"
GENERATION_RECORD_SCHEMA = _l2.GENERATION_RECORD_SCHEMA
HEARTBEAT_SCHEMA = "agent_multi.l2_curriculum_heartbeat.v1"
PREFLIGHT_SCHEMA = "agent_multi.l2_curriculum_preflight.v1"
DISPATCH_SCHEMA = "agent_multi.l2_smoke_dispatch.v1"
ANCHOR_GATE_SCHEMA = "agent_multi.l2_anchor_fallback_gate.v1"
ARM_ANCHOR_SCHEMA = "agent_multi.l2_arm_anchor_fallback.v1"
BEHAVIOR_FACTS_SCHEMA = "agent_multi.l2_scored_behavior_facts.v1"
CAMPAIGN_VERDICT_SCHEMA = "agent_multi.l2_campaign_verdict.v1"
ACCEPTANCE_SCHEMA = "agent_multi.l2_smoke_acceptance.v1"
HEARTBEAT_INTERVAL_S = 60

ARMS = ("L2_N", "L2_EN")
MODES = ("smoke", "decision")
MODE_PROFILES = {"smoke": "l2_curriculum_mechanics_smoke",
                 "decision": "l2_curriculum_decision_run"}
WORKERS = ("w1", "w2", "w3", "w4")

SELECTION_METRIC = _paired.METRIC_NAME
EVIDENCE_PAIR = ("inner_validation", "outer_validation")

#: The metric tuple the arm-differentiation ladder compares. It is the
#: DECISION-side truth (outer validation) plus the paired objective and
#: the inner member, so two arms that separate on any of them separate
#: here. ``arm_differentiation.DEFAULT_METRIC_KEYS`` names the P1LR
#: fields; the L2 pair reports its own, hence an explicit tuple.
L2_METRIC_KEYS = ("paired_score", "outer_mean_weekly_rap",
                  "outer_mean_weekly_return", "outer_total_return",
                  "outer_max_drawdown_fraction", "outer_trades_total",
                  "inner_mean_weekly_rap")

EXIT_CLASS = {
    "PREFLIGHT_PASS": 0,
    "CHAIN_JOINED": 0,
    "ARM_COMPLETE": 0,
    "WORKER_COMPLETE": 0,
    "GENERATION_COMPLETE": 0,
    "STOPPED_ON_PATIENCE": 0,
    "DISPATCH_PLAN": 0,
    "CAMPAIGN_VERDICT": 0,
    "ACCEPTANCE_PASS": 0,
    "ALREADY_RUNNING": 3,
    "PREFLIGHT_REFUSED": 4,
    "REFUSED_BAD_CONTRACT": 4,
    "REFUSED_WRONG_HOST": 4,
    "REFUSED_GPU_UNBOUND": 4,
    "REFUSED_PARALLEL_CHAIN": 4,
    "REFUSED_NODE_IDENTITY_MISMATCH": 4,
    "REFUSED_OUT_OF_ARM_ORDER": 4,
    "REFUSED_ANCHOR_UNVERIFIED": 4,
    "REFUSED_ANCHOR_FALLBACK": 4,
    "REFUSED_ARMS_NOT_DIFFERENTIATED": 4,
    "CAMPAIGN_UNANSWERABLE": 4,
    "CAMPAIGN_INCOMPLETE": 4,
    "ACCEPTANCE_REFUSED": 4,
    "WORKER_FAILED": 1,
}


class L2AnchorFallbackRefusal(RuntimeError):
    """Typed refusal: every selected checkpoint IS the frozen anchor.

    Carries the arm's facts so the refusal is auditable without
    re-deriving them (finding D8-20260815).
    """

    def __init__(self, facts: dict) -> None:
        self.facts = dict(facts)
        super().__init__(
            "REFUSED_ANCHOR_FALLBACK: arm "
            f"{facts.get('arm')!r} selected {facts.get('checked_candidates')}"
            " checkpoint(s) and EVERY one of them is bit-identical to the"
            f" frozen warm-start anchor policy tensor"
            f" {str(facts.get('anchor_policy_tensor_sha256'))[:16]}… — no"
            " candidate learned anything, so this arm measured the anchor,"
            " not the treatment. A null result here would be an artefact"
            " of selection fallback and is refused rather than reported")


def _sha_file(path: Path) -> str:
    return sysid.sha_file(path)


def _canonical_sha(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# contract loading — every typed refusal happens BEFORE model construction
# ---------------------------------------------------------------------------

def load_contract(path: Path = CONTRACT_PATH) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(
            f"unknown L2 curriculum contract schema "
            f"{contract.get('schema')!r}")
    if tuple(contract.get("arms") or ()) != ARMS:
        raise ValueError(f"contract arms must be exactly {list(ARMS)}")
    if contract.get("selection_metric") != SELECTION_METRIC:
        raise ValueError(
            "the L2 objective must be the paired comparator "
            f"{SELECTION_METRIC!r}; the legacy lexicographic branch is "
            "forbidden (doc 38 §4.2)")
    if tuple(contract.get("l2_evidence_pair") or ()) != EVIDENCE_PAIR:
        raise ValueError(
            f"the L2 evidence pair must be exactly {list(EVIDENCE_PAIR)}")
    if contract.get("decision_evidence_difficulty") != _l2.NORMAL:
        raise ValueError(
            "decision evidence must be normal-realistic for BOTH arms")

    # requirement 1 — L1 frozen, and no L1 curriculum field is a gene
    recipe = contract.get("frozen_l1_recipe") or {}
    _l2.assert_l1_recipe_frozen(recipe)
    frozen_names = list(recipe.get("l1_frozen_gene_names") or ())
    if not frozen_names:
        raise ValueError(
            "frozen_l1_recipe.l1_frozen_gene_names must list the fields "
            "that may never become L2 genes")
    shared = contract.get("shared") or {}
    gene_space = [tuple(item) for item in (shared.get("gene_space") or ())]
    if not gene_space:
        raise ValueError("shared.gene_space is empty")
    gene_names = [str(item[0]) for item in gene_space]
    if len(set(gene_names)) != len(gene_names):
        raise ValueError("shared.gene_space has duplicate gene names")
    for name, low, high, kind in gene_space:
        if kind not in ("int", "float"):
            raise ValueError(f"gene {name!r} has unknown kind {kind!r}")
        if float(low) >= float(high):
            raise ValueError(f"gene {name!r} has low >= high")
    _l2.assert_no_l1_gene_in_l2_space(gene_names, frozen_names)

    binding = contract.get("l2_stage_difficulty_binding") or {}
    if binding.get("mechanism") not in tuple(
            binding.get("allowed_mechanisms") or ()):
        raise ValueError("unknown l2_stage_difficulty_binding.mechanism")
    if binding.get("training_difficulty_frozen") != \
            recipe.get("phase1_dynamics"):
        raise ValueError(
            "l2_stage_difficulty_binding.training_difficulty_frozen must "
            "equal the frozen L1 phase-1 dynamics")

    # requirement 3 — identical budgets/population/seeds across arms
    for mode in MODES:
        assert_arms_comparable(contract, mode=mode)

    # requirement 7 — the nested role contract shape
    spec = contract.get("nested_split_contract") or {}
    if spec.get("mode") != "l2":
        raise ValueError(
            "the L2 program must consume the nested contract in "
            "nested_split_mode 'l2'")
    if int(spec.get("context_bars", -1)) != 256:
        raise ValueError("the causal context declaration must be 256 bars")
    sealed = (spec.get("role_facts") or {}).get("sealed_test") or {}
    if sealed.get("status") != "SEALED":
        raise ValueError("sealed_test must be pinned SEALED")

    contract["_contract_sha256"] = _sha_file(Path(path))
    contract["_contract_path"] = str(path)
    return contract


def load_bindings(path: Path | None = None) -> dict:
    """The held-fixed data/plugin/cost recipe: the SAME ladder contract
    the L1 program proved, loaded through its own fail-closed loader."""
    return ladder.load_contract(path or ladder.CONTRACT_PATH)


def mode_block(contract: dict, mode: str) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    return dict((contract.get("modes") or {}).get(mode) or {})


def mode_setting(contract: dict, mode: str, key: str):
    """A mode may override any shared setting; the decision mode simply
    declares no overrides, so both modes read through ONE resolver."""
    block = mode_block(contract, mode)
    if key in block:
        return block[key]
    if key == "stage_schedules":
        return contract["stage_schedules"]
    if key == "l2_stopping":
        return contract["l2_stopping"]
    return (contract.get("shared") or {})[key]


def output_root_for_mode(contract: dict, mode: str) -> Path:
    root = mode_block(contract, mode).get("output_root")
    if not root:
        raise ValueError(f"mode {mode!r} declares no output_root")
    return Path(root).expanduser()


def arm_schedule(contract: dict, arm: str, mode: str) -> list:
    if arm not in ARMS:
        raise ValueError(f"unknown L2 arm {arm!r}")
    schedules = mode_setting(contract, mode, "stage_schedules")
    gene_names = [str(item[0])
                  for item in (contract["shared"]["gene_space"])]
    return _l2.expand_stage_schedule(schedules[arm], gene_names=gene_names)


def assert_arms_comparable(contract: dict, *, mode: str) -> dict:
    """Requirement 3, executable: identical total candidate-evaluation
    budget, population size, population seed, initial genome seeds, L1
    recipe, data roles and normal-realistic decision evidence."""
    population_size = int(mode_setting(contract, mode, "population_size"))
    declared_total = int(
        mode_setting(contract, mode, "total_candidate_evaluations"))
    ledgers = {
        arm: _l2.arm_budget_ledger(arm_schedule(contract, arm, mode),
                                   population_size=population_size)
        for arm in ARMS}
    summary = _l2.assert_arms_identical_budget(
        ledgers, declared_total=declared_total,
        population_size=population_size)

    seeds = list((contract["shared"] or {}).get("initial_genome_seeds") or ())
    if len(seeds) != int(contract["shared"]["population_size"]):
        raise ValueError(
            "shared.initial_genome_seeds must hold exactly "
            "shared.population_size entries — both arms start from the "
            "SAME initial genome seeds (requirement 3)")
    if len(set(seeds)) != len(seeds):
        raise ValueError("shared.initial_genome_seeds has duplicates")

    # Both arms must end in a normal-realistic decision stage, and only
    # L2_EN may contain an easy stage at all.
    for arm in ARMS:
        schedule = arm_schedule(contract, arm, mode)
        if schedule[-1]["evaluation_difficulty"] != _l2.NORMAL:
            raise ValueError(
                f"arm {arm}: the final stage must be normal-realistic")
        easy = [stage["name"] for stage in schedule
                if stage["evaluation_difficulty"] == _l2.EASY]
        if arm == "L2_N" and easy:
            raise ValueError(
                f"L2_N declares easy stage(s) {easy} — L2_N evaluates "
                "every generation under normal-realistic conditions")
        if arm == "L2_EN":
            if not easy:
                raise ValueError(
                    "L2_EN declares no easy triage stage; it would be a "
                    "duplicate of L2_N")
            kinds = [stage["stage_kind"] for stage in schedule]
            if "reevaluation" not in kinds:
                raise ValueError(
                    "L2_EN has no reevaluation stage — surviving elites "
                    "MUST be re-scored under normal-realistic conditions "
                    "at the transition (requirement 2)")
            first_normal = next(
                index for index, stage in enumerate(schedule)
                if stage["evaluation_difficulty"] == _l2.NORMAL)
            if schedule[first_normal]["stage_kind"] != "reevaluation":
                raise ValueError(
                    "L2_EN's first normal-realistic stage must be the "
                    "mandatory re-evaluation, before any normal search "
                    "generation (requirement 2)")
    summary["mode"] = mode
    summary["ledgers"] = ledgers
    return summary


# ---------------------------------------------------------------------------
# requirement 7 — nested data roles, sealed 2025 never materialized
# ---------------------------------------------------------------------------

def materialize_l2_nested_roles(contract: dict, bindings: dict,
                                out_dir: Path) -> dict:
    """Materialize the per-role CSVs through the ONE typed
    implementation in nested_split_mode ``l2`` and verify every derived
    fact against the pinned role facts BEFORE model construction. Wrong
    role path, wrong sha, wrong scored/context count, a swapped role or
    ANY sealed-test materialization refuses here."""
    binding = p1.verify_nested_split_binding(contract, bindings)
    if binding["mode"] != "l2":
        raise RuntimeError(
            f"nested binding mode {binding['mode']!r} != 'l2' — the L2 "
            "program never runs under the L1 split mode")
    nested = _nested_splits.load_contract(p1.nested_contract_path(contract))
    split_dir = Path(out_dir) / "nested_splits"
    manifest = _nested_splits.materialize_nested_splits(
        nested, split_dir, mode="l2")
    refusals = p1.verify_role_facts(manifest["roles"],
                                    binding["role_facts_pinned"])
    if (split_dir / "sealed_test.csv").exists():
        refusals.append(
            "sealed_test.csv exists on disk — sealed 2025 was touched")
    if refusals:
        raise RuntimeError("nested role facts refused before training: "
                           + "; ".join(refusals))
    manifest_path = Path(manifest["manifest_path"])
    return {
        "binding": binding,
        "split_dir": str(split_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha_file(manifest_path),
        "roles": {role: dict(entry)
                  for role, entry in manifest["roles"].items()},
    }


def assert_sealed_test_inaccessible(config: dict) -> None:
    if config.get("evaluate_test_split"):
        raise RuntimeError(
            "evaluate_test_split leaked true — sealed 2025 is release-only")
    if config.get("nested_split_mode") != "l2":
        raise RuntimeError("executable config lost nested_split_mode 'l2'")
    for key in ("test_start", "test_end", "test_years", "test_days"):
        if config.get(key) is not None:
            raise RuntimeError(
                f"legacy sealed-window field {key!r} survived "
                "materialization — refused")


def split_identity(nested_roles: dict) -> str:
    """The identity a stopping state is bound to: an artifact scored
    under one split contract can never continue under another."""
    binding = nested_roles["binding"]
    return _canonical_sha({
        "contract_sha256": binding["sha256"],
        "source_sha256": binding["source_sha256"],
        "context_bars": binding["context_bars"],
    })[:16]


# ---------------------------------------------------------------------------
# identities, requirement 8 node identity, requirement 9 chain identity
# ---------------------------------------------------------------------------

def domain_payload(contract: dict, arm: str, mode: str) -> dict:
    shared = contract["shared"]
    return {
        "arm": arm,
        "mode": mode,
        "gene_space": [list(item) for item in shared["gene_space"]],
        "stage_schedule": arm_schedule(contract, arm, mode),
        "frozen_l1_recipe": contract["frozen_l1_recipe"],
        "l2_stage_difficulty_binding":
            contract["l2_stage_difficulty_binding"],
        "selection_metric": SELECTION_METRIC,
        "l2_evidence_pair": list(EVIDENCE_PAIR),
        "l2_gap_penalty_beta": contract["l2_gap_penalty_beta"],
        "population_size": int(mode_setting(contract, mode,
                                            "population_size")),
        "population_seed": int(shared["population_seed"]),
        "initial_genome_seeds": list(shared["initial_genome_seeds"]),
        "total_candidate_evaluations": int(
            mode_setting(contract, mode, "total_candidate_evaluations")),
        "candidate_budget_knobs": mode_setting(contract, mode,
                                               "candidate_budget_knobs"),
        "candidate_stopping_knobs": mode_setting(
            contract, mode, "candidate_stopping_knobs"),
        "nested_split_contract": contract["nested_split_contract"],
    }


def experiment_identity(contract: dict, bindings: dict,
                        sources: dict | None = None,
                        mode: str = "smoke") -> str:
    """One content-addressed identity per mode: contract sha + the
    held-fixed ladder binding + the nested split contract + the paired
    metric + the frozen L1 recipe + code identities + the mode profile.
    The smoke identity can never collide with the decision identity."""
    if mode not in MODES:
        raise ValueError(f"unknown execution mode {mode!r}")
    sources = sources or ladder.source_identities()
    payload = {
        "contract": contract["_contract_sha256"],
        "held_fixed_bindings_contract": bindings["_contract_sha256"],
        "nested_split_contract_sha256":
            contract["nested_split_contract"]["sha256"],
        "selection_metric": SELECTION_METRIC,
        "frozen_l1_recipe": contract["frozen_l1_recipe"],
        "code": {name: {"commit": s["commit"],
                        "dirty_untracked_digest":
                            s["dirty_untracked_digest"]}
                 for name, s in sorted(sources.items())},
        "profile": MODE_PROFILES[mode],
    }
    return _canonical_sha(payload)[:16]


def node_identity_facts(contract: dict, bindings: dict, *, arm: str,
                        mode: str, sources: dict | None = None) -> dict:
    """Requirement 8: the exact facts every node must prove EQUAL before
    it evaluates anything — source revisions, plan hash, domain hash,
    dataset hash, population seed and the chain genesis."""
    sources = sources or ladder.source_identities()
    plan_hash = contract["_contract_sha256"]
    domain_hash = _canonical_sha(domain_payload(contract, arm, mode))
    dataset_sha256 = str(
        (contract["nested_split_contract"].get("dataset_sha256")
         or bindings["common"]["data"]["sha256"]))
    population_seed = int(contract["shared"]["population_seed"])
    genesis = _canonical_sha({
        "plan_hash": plan_hash,
        "domain_hash": domain_hash,
        "dataset_sha256": dataset_sha256,
        "population_seed": population_seed,
        "arm": arm,
        "mode": mode,
        "experiment_identity": experiment_identity(
            contract, bindings, sources=sources, mode=mode),
    })[:32]
    return {
        "source_revisions": {
            name: {"commit": s["commit"],
                   "dirty_untracked_digest": s["dirty_untracked_digest"]}
            for name, s in sorted(sources.items())},
        "plan_hash": plan_hash,
        "domain_hash": domain_hash,
        "dataset_sha256": dataset_sha256,
        "population_seed": population_seed,
        "arm": arm,
        "mode": mode,
        "genesis": genesis,
        "experiment_identity": experiment_identity(
            contract, bindings, sources=sources, mode=mode),
    }


NODE_IDENTITY_KEYS = ("source_revisions", "plan_hash", "domain_hash",
                      "dataset_sha256", "population_seed", "genesis",
                      "experiment_identity")


def verify_node_identity(observed: dict, registered: dict, *,
                         finalized_ancestry: str | None = None,
                         registered_ancestry: str | None = None) -> list:
    """Fail-closed equality of every identity fact, returning exact
    refusal strings. A node that cannot prove all of them does not
    evaluate (requirement 8)."""
    refusals: list = []
    for key in NODE_IDENTITY_KEYS:
        if observed.get(key) != registered.get(key):
            refusals.append(
                f"node identity mismatch on {key}: this node computes "
                f"{observed.get(key)!r}, the chain registers "
                f"{registered.get(key)!r}")
    if finalized_ancestry is not None or registered_ancestry is not None:
        if finalized_ancestry != registered_ancestry:
            refusals.append(
                "finalized ancestry mismatch: this node recomputes "
                f"{finalized_ancestry!r}, the chain registers "
                f"{registered_ancestry!r} — a node may not evaluate on a "
                "different finalized history")
    return refusals


def chain_identity(contract: dict, bindings: dict, *, arm: str,
                   mode: str, sources: dict | None = None) -> str:
    facts = node_identity_facts(contract, bindings, arm=arm, mode=mode,
                               sources=sources)
    return facts["genesis"][:16]


def arm_root(contract: dict, exp_id: str, arm: str, mode: str) -> Path:
    return output_root_for_mode(contract, mode) / exp_id / arm


def chain_dir(contract: dict, exp_id: str, arm: str, mode: str,
              chain_id: str) -> Path:
    return arm_root(contract, exp_id, arm, mode) / f"chain_{chain_id}"


def compute_finalized_ancestry(chain_path: Path) -> str:
    """A rolling digest over the finalized generation records, so a node
    proves it descends from the same finalized history rather than
    merely sharing a genesis."""
    registry = json.loads((chain_path / "chain.json").read_text())
    ancestry = str(registry["genesis"])
    generation = 0
    while True:
        record = chain_path / f"gen{generation:03d}" / \
            "generation_record.json"
        if not record.is_file():
            break
        ancestry = hashlib.sha256(
            (ancestry + _sha_file(record)).encode()).hexdigest()
        generation += 1
    return ancestry[:32]


def assert_single_chain_per_arm(arm_path: Path, chain_id: str) -> None:
    """Requirement 9: NEVER create parallel independent chains for one
    arm. A second, different chain directory is a typed refusal."""
    if not arm_path.exists():
        return
    others = sorted(
        entry.name for entry in arm_path.iterdir()
        if entry.is_dir() and entry.name.startswith("chain_")
        and entry.name != f"chain_{chain_id}")
    if others:
        raise RuntimeError(
            f"REFUSED_PARALLEL_CHAIN: arm {arm_path.name} already owns "
            f"chain(s) {others}; a second independent chain "
            f"'chain_{chain_id}' would split the population. All four "
            "workers collaborate on ONE chain per arm (requirement 9)")


def assert_sequential_arm_dispatch(contract: dict, exp_id: str, arm: str,
                                   mode: str) -> None:
    """Arms run SEQUENTIALLY in the contract's declared order; an arm may
    not start while an earlier arm's chain is unfinished."""
    order = list((contract["workers"] or {}).get("arm_order") or ARMS)
    if arm not in order:
        raise RuntimeError(f"arm {arm!r} is not in the dispatch order")
    for earlier in order[:order.index(arm)]:
        path = arm_root(contract, exp_id, earlier, mode)
        chains = sorted(path.glob("chain_*")) if path.exists() else []
        complete = any((chain / "arm_complete.json").is_file()
                       for chain in chains)
        if not complete:
            raise RuntimeError(
                f"REFUSED_OUT_OF_ARM_ORDER: arm {arm} may not start while "
                f"{earlier} has no completed chain — the fleet works on "
                "ONE arm at a time (requirement 9)")


def open_or_join_chain(contract: dict, bindings: dict, *, arm: str,
                       mode: str, worker_id: str,
                       sources: dict | None = None) -> dict:
    """Create the arm's ONE chain or join it, proving node identity and
    finalized ancestry first (requirements 8 and 9)."""
    sources = sources or ladder.source_identities()
    exp_id = experiment_identity(contract, bindings, sources=sources,
                                 mode=mode)
    facts = node_identity_facts(contract, bindings, arm=arm, mode=mode,
                               sources=sources)
    chain_id = facts["genesis"][:16]
    arm_path = arm_root(contract, exp_id, arm, mode)
    assert_sequential_arm_dispatch(contract, exp_id, arm, mode)
    assert_single_chain_per_arm(arm_path, chain_id)
    path = chain_dir(contract, exp_id, arm, mode, chain_id)
    path.mkdir(parents=True, exist_ok=True)
    registry_path = path / "chain.json"

    claim = ExclusiveClaim(path / "locks" / "chain.lock")
    if not claim.acquire():
        # Another worker on this host is registering; joining is still
        # correct as soon as the registry exists.
        if not registry_path.is_file():
            return {"outcome": "ALREADY_RUNNING",
                    "reason": "chain registration in progress",
                    "holder": claim.holder()}
    try:
        if registry_path.is_file():
            registry = json.loads(registry_path.read_text())
            refusals = verify_node_identity(
                facts, registry.get("node_identity") or {},
                finalized_ancestry=compute_finalized_ancestry(path),
                registered_ancestry=registry.get("finalized_ancestry"))
            if registry.get("chain_identity") != chain_id:
                refusals.append(
                    "chain identity mismatch: registry "
                    f"{registry.get('chain_identity')!r} != {chain_id!r}")
            if refusals:
                raise RuntimeError(
                    "REFUSED_NODE_IDENTITY_MISMATCH: " + "; ".join(refusals))
            joined = dict(registry)
            joined["_joined_by"] = worker_id
            joined["_chain_dir"] = str(path)
            joined["outcome"] = "CHAIN_JOINED"
            return joined
        registry = {
            "schema": CHAIN_SCHEMA,
            "chain_identity": chain_id,
            "arm": arm,
            "mode": mode,
            "experiment_identity": exp_id,
            "genesis": facts["genesis"],
            "node_identity": facts,
            "created_utc": _utc(),
            "created_by": {"worker_id": worker_id,
                           "hostname": socket.gethostname()},
            "population_size": int(mode_setting(contract, mode,
                                                "population_size")),
            "total_candidate_evaluations": int(mode_setting(
                contract, mode, "total_candidate_evaluations")),
            "stage_schedule": arm_schedule(contract, arm, mode),
            "budget_ledger": _l2.arm_budget_ledger(
                arm_schedule(contract, arm, mode),
                population_size=int(mode_setting(contract, mode,
                                                 "population_size"))),
            "finalized_ancestry": facts["genesis"][:32],
            "finalized_generations": 0,
        }
        _atomic_json(registry_path, registry)
        registry["_chain_dir"] = str(path)
        registry["_joined_by"] = worker_id
        registry["outcome"] = "CHAIN_JOINED"
        return registry
    finally:
        try:
            claim.release()
        except Exception:                                   # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# candidate materialization — the ONLY path from contract to a run config
# ---------------------------------------------------------------------------

def materialize_candidate_config(contract: dict, bindings: dict, *,
                                 arm: str, stage: dict, genome: dict,
                                 out_dir: Path, mode: str) -> dict:
    """Turn ONE typed genome into a runnable candidate config under the
    FROZEN L1 recipe, the contract's nested roles and the stage's
    declared difficulty. Every refusal fires before model construction.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown L2 arm {arm!r}")
    recipe = contract["frozen_l1_recipe"]
    shared = contract["shared"]
    gene_names = [str(item[0]) for item in shared["gene_space"]]

    # requirement 4 — typed genes only, no inherited weights/topology
    parameters = _l2.assert_typed_genes_only(genome, gene_names=gene_names)
    _l2.assert_common_anchor_not_inherited(recipe, genome)

    common = bindings["common"]
    base_path = REPO / common["base_config"]["path"]
    if _sha_file(base_path) != common["base_config"]["sha256"]:
        raise RuntimeError(
            "base config drifted from the held-fixed ladder binding")
    manifest_path = REPO / bindings["l1_reference"]["system_manifest_path"]
    if _sha_file(manifest_path) != bindings["l1_reference"][
            "system_manifest_sha256"]:
        raise RuntimeError(
            "L1 system manifest drifted from the ladder contract")
    manifest = json.loads(manifest_path.read_text())
    v3_costs = manifest["costs"]["config_bindings"]
    if v3_costs != bindings["arms"]["D3_COST_PROTECTION"]["delta"]:
        raise RuntimeError(
            "the held-fixed cost/protection contract is unbound")
    sysid.validate_normal_contract(v3_costs)

    nested_binding = p1.verify_nested_split_binding(contract, bindings)
    budget = mode_setting(contract, mode, "candidate_budget_knobs")
    stopping = mode_setting(contract, mode, "candidate_stopping_knobs")

    config = json.loads(base_path.read_text())
    observation_contract = contract.get("observation_contract")
    if not isinstance(observation_contract, dict) or not observation_contract:
        raise RuntimeError(
            "the L2 experiment declares no observation_contract; refusing "
            "before candidate identity or model construction")
    # Bind the declaration explicitly into the materialized candidate. The
    # pipeline still validates and applies it, but no worker has to rediscover
    # its own contract by scanning repository files from an identity digest.
    config["observation_contract"] = dict(observation_contract)
    for field in ("train_years", "val_years", "test_years", "train_days",
                  "val_days", "test_days", "train_start", "train_end",
                  "validation_start", "validation_end", "val_start",
                  "val_end", "test_start", "test_end", "split_anchor"):
        config.pop(field, None)
    config["split_contract_note"] = (
        "typed nested split contract governs (mode l2); legacy split "
        "fields removed")
    config["nested_split_contract"] = nested_binding["path"]
    config["nested_split_mode"] = "l2"
    config["nested_split_dir"] = str(Path(out_dir) / "nested_splits")
    config["input_data_file"] = str(Path(common["data"]["path"]))
    config["env_mode"] = "training"
    config["evaluate_test_split"] = False
    config["selection_metric"] = SELECTION_METRIC
    config["l1_gap_penalty_beta"] = float(contract["l2_gap_penalty_beta"])
    config["selection_min_trades"] = int(common["selection_min_trades"])
    config["quiet_mode"] = True
    config["write_results_sidecar"] = False
    config["save_model"] = str(Path(out_dir) / "model.zip")
    config["return_trace_dir"] = str(Path(out_dir) / "return_traces")
    config["inactive_terminal_is_typed_result"] = True

    seed = int(recipe["candidate_seed"])
    config["eval_seed"] = seed
    config["train_seed"] = seed
    config["ga_seed"] = seed

    # --- the FROZEN L1 recipe (requirement 1) -------------------------
    config["phase1_mode"] = str(recipe["phase1_dynamics"])
    config["phase1_learning_rate"] = float(recipe["phase1_learning_rate"])
    config["easy_learning_rate"] = float(recipe["phase1_learning_rate"])
    config["learning_rate"] = float(recipe["phase2_learning_rate"])
    config["phase1_handoff_semantics"] = str(
        recipe["phase1_handoff_semantics"])
    config.update(v3_costs)
    if config.get("ent_coef") != float(recipe["entropy"]["value"]):
        raise RuntimeError(
            "base config entropy is not the frozen recipe entropy")
    config["epoch_timesteps"] = int(budget["epoch_timesteps"])
    config["easy_max_epochs"] = int(budget["phase1_epochs"])
    config["max_epochs"] = int(budget["phase2_max_epochs"])
    config["total_max_passes"] = int(budget["total_max_passes"])
    config["phase1_max_fraction"] = float(budget["phase1_max_fraction"])
    config["normal_phase_min_passes"] = int(budget["normal_phase_min_passes"])
    config["l1_min_checkpoint_timesteps"] = 1
    config["easy_patience"] = 10_000
    for key in ("l1_patience", "l1_patience_start_epoch",
                "l1_activity_patience",
                "l1_activity_patience_start_epoch"):
        config[key] = int(stopping[key])
    config["execution_cost_curriculum_epochs"] = max(
        2, int(budget["phase2_max_epochs"]))

    if recipe["candidate_initialization"] == "frozen_common_anchor":
        anchor = recipe["common_anchor"]
        anchor_path = Path(anchor["path"]).expanduser()
        for other in MODES:
            try:
                anchor_path.resolve().relative_to(
                    output_root_for_mode(contract, other).resolve())
            except ValueError:
                continue
            raise RuntimeError(
                "the common anchor lives under an L2 output root — an L2 "
                "run terminal can never anchor a candidate")
        config["warm_start_model"] = str(anchor_path)
        config["warm_start_model_sha256"] = anchor["sha256"]
    else:
        config["warm_start_model"] = None
        config["warm_start_model_sha256"] = None
    config["load_model"] = None

    config["agent_plugin"] = str(common["plugins"]["agent_plugin"])
    for key in ("env_plugin", "strategy_plugin"):
        bound = common["plugins"].get(key)
        if bound and config.get(key) != bound:
            raise RuntimeError(
                f"plugin drift: config[{key!r}]={config.get(key)!r} != "
                f"held-fixed binding {bound!r}")

    # --- the genome's typed genes, and NOTHING else --------------------
    for name, low, high, kind in shared["gene_space"]:
        value = parameters[name]
        if not (float(low) <= float(value) <= float(high)):
            raise RuntimeError(
                f"gene {name!r}={value!r} is outside its declared bounds "
                f"[{low}, {high}]")
        config[name] = int(round(float(value))) if kind == "int" \
            else float(value)

    # --- the stage's declared difficulty (requirement 1 + 2) -----------
    config["l2_arm"] = arm
    config["l2_stage"] = str(stage["name"])
    config["l2_stage_kind"] = str(stage["stage_kind"])
    config["l2_evaluation_difficulty"] = str(stage["evaluation_difficulty"])
    binding_spec = contract["l2_stage_difficulty_binding"]
    if binding_spec["mechanism"] == "training_and_evidence_v1":
        config["phase1_mode"] = str(stage["evaluation_difficulty"])
    _l2.assert_stage_difficulty_binding(config, stage, binding_spec)
    assert_sealed_test_inaccessible(config)

    config["_identity"] = {
        "experiment_contract_sha256": contract["_contract_sha256"],
        "held_fixed_bindings_contract_sha256":
            bindings["_contract_sha256"],
        "base_config_sha256": _sha_file(base_path),
        "data_sha256": common["data"]["sha256"],
        "mode": mode,
        "arm": arm,
        "stage": dict(stage),
        "selection_metric": SELECTION_METRIC,
        "l2_evidence_pair": list(EVIDENCE_PAIR),
        "nested_split_contract_path":
            nested_binding["contract_relative_path"],
        "nested_split_contract_sha256": nested_binding["sha256"],
        "nested_split_mode": "l2",
        "nested_context_bars": nested_binding["context_bars"],
        "nested_role_facts_pinned": nested_binding["role_facts_pinned"],
        "l1_system_manifest_sha256":
            bindings["l1_reference"]["system_manifest_sha256"],
        "frozen_l1_recipe": {
            "phase1_dynamics": config["phase1_mode"],
            "phase1_learning_rate": config["phase1_learning_rate"],
            "phase2_learning_rate": config["learning_rate"],
            "phase1_handoff_semantics": config["phase1_handoff_semantics"],
            "entropy": dict(recipe["entropy"]),
            "candidate_initialization":
                recipe["candidate_initialization"],
            "candidate_seed": seed,
        },
        "genome": {"parameters": dict(parameters)},
        "genome_fingerprint": _l2.genome_fingerprint(
            parameters, gene_names=gene_names),
        "plugins": dict(common["plugins"]),
    }
    return config


# ---------------------------------------------------------------------------
# gate 11 — the anchor-fallback gate (finding D8-20260815)
#
# The highest-value check in this file. Selection that finds no
# activity-eligible checkpoint used to fall back to the untouched
# warm-start anchor, so the arm scored the anchor and the treatment was
# never measured. Identity is the POLICY TENSOR hash: SB3 ``.zip``
# containers embed member mtimes, so the same weights saved twice yield
# different container digests and a container rule proves nothing.
# ---------------------------------------------------------------------------

def default_policy_tensor_sha256(path: Path | str) -> str:
    """The ONE checkpoint identity used by this program."""
    return _armdiff.policy_tensor_sha256(path)


def anchor_binding(contract: dict) -> dict | None:
    """The frozen common anchor, or ``None`` for a cold-start contract."""
    recipe = contract.get("frozen_l1_recipe") or {}
    if recipe.get("candidate_initialization") != "frozen_common_anchor":
        return None
    anchor = dict(recipe.get("common_anchor") or {})
    if not anchor.get("path"):
        raise RuntimeError(
            "REFUSED_ANCHOR_UNVERIFIED: the contract warm-starts from a "
            "frozen common anchor but declares no anchor path")
    anchor["path"] = str(Path(str(anchor["path"])).expanduser())
    return anchor


def anchor_policy_tensor_sha256(contract: dict, *,
                                policy_tensor_sha_fn=None) -> dict:
    """Hash the frozen anchor's POLICY TENSORS, fail-closed.

    An anchor whose tensors cannot be read is a typed
    ``REFUSED_ANCHOR_UNVERIFIED``: without the anchor's tensor hash the
    fallback gate cannot fire, and a gate that cannot fire is worse than
    no gate at all because it looks like a pass.
    """
    anchor = anchor_binding(contract)
    if anchor is None:
        return {"declared": False, "anchor_path": None,
                "anchor_container_sha256": None,
                "anchor_policy_tensor_sha256": None,
                "reason": "the contract declares a cold start; there is no "
                          "warm-start anchor a candidate could fall back to"}
    path = Path(anchor["path"])
    if not path.is_file():
        raise RuntimeError(
            f"REFUSED_ANCHOR_UNVERIFIED: the frozen common anchor {path} "
            "does not exist, so no candidate's selected checkpoint can be "
            "proven different from it")
    container = _sha_file(path)
    declared = anchor.get("sha256")
    if declared and container != declared:
        raise RuntimeError(
            f"REFUSED_ANCHOR_UNVERIFIED: the frozen common anchor {path} "
            f"hashes to {container} but the contract pins {declared}")
    fn = policy_tensor_sha_fn or default_policy_tensor_sha256
    try:
        tensor_sha = fn(path)
    except Exception as exc:                                # noqa: BLE001
        raise RuntimeError(
            f"REFUSED_ANCHOR_UNVERIFIED: cannot read the policy tensors of "
            f"the frozen common anchor {path}: {type(exc).__name__}: {exc} "
            "— the anchor-fallback gate would silently pass") from exc
    return {"declared": True, "anchor_path": str(path),
            "anchor_container_sha256": container,
            "anchor_policy_tensor_sha256": str(tensor_sha),
            "reason": "frozen common anchor policy tensors hashed"}


def selected_checkpoint_anchor_facts(contract: dict,
                                     selected_path: str | Path | None, *,
                                     policy_tensor_sha_fn=None,
                                     anchor_facts: dict | None = None
                                     ) -> dict:
    """Gate 11, per candidate: is the SELECTED checkpoint the anchor?

    Returns a typed fact, never a bare boolean, so the candidate record
    carries BOTH tensor shas and the reason. ``selected_equals_anchor``
    is ``None`` only when there was no selected checkpoint at all (an
    inactive candidate) or when the contract declares a cold start —
    both are separately typed outcomes, never a silent pass.
    """
    facts = anchor_facts or anchor_policy_tensor_sha256(
        contract, policy_tensor_sha_fn=policy_tensor_sha_fn)
    payload = {
        "schema": ANCHOR_GATE_SCHEMA,
        "anchor_declared": bool(facts.get("declared")),
        "anchor_path": facts.get("anchor_path"),
        "anchor_container_sha256": facts.get("anchor_container_sha256"),
        "anchor_policy_tensor_sha256":
            facts.get("anchor_policy_tensor_sha256"),
        "selected_model_path": (str(selected_path) if selected_path
                                else None),
        "selected_container_sha256": None,
        "selected_policy_tensor_sha256": None,
        "selected_equals_anchor": None,
        "checked": False,
        "identity": "policy_tensor_sha256",
        "reason": "",
    }
    if selected_path is None:
        payload["reason"] = (
            "no checkpoint was selected, so nothing could fall back to the "
            "anchor; the candidate is typed inactive instead")
        return payload
    if not facts.get("declared"):
        payload["reason"] = str(facts.get("reason"))
        return payload
    path = Path(str(selected_path))
    if not path.is_file():
        raise RuntimeError(
            f"REFUSED_ANCHOR_UNVERIFIED: the selected checkpoint {path} "
            "vanished before the anchor-fallback gate could hash it")
    fn = policy_tensor_sha_fn or default_policy_tensor_sha256
    try:
        tensor_sha = str(fn(path))
    except Exception as exc:                                # noqa: BLE001
        raise RuntimeError(
            f"REFUSED_ANCHOR_UNVERIFIED: cannot read the policy tensors of "
            f"the selected checkpoint {path}: {type(exc).__name__}: {exc}"
        ) from exc
    payload["selected_container_sha256"] = _sha_file(path)
    payload["selected_policy_tensor_sha256"] = tensor_sha
    payload["checked"] = True
    equal = tensor_sha == payload["anchor_policy_tensor_sha256"]
    payload["selected_equals_anchor"] = equal
    payload["reason"] = (
        "the selected checkpoint's policy tensor is BIT-IDENTICAL to the "
        "frozen warm-start anchor: this candidate did not learn, it IS the "
        "anchor, and any score computed from it measures the anchor rather "
        "than the treatment (finding D8-20260815)" if equal else
        "the selected checkpoint's policy tensor differs from the frozen "
        "warm-start anchor: training moved the weights that were scored")
    return payload


def evidence_behavior_facts(evidence: dict) -> dict:
    """Behaviour evidence for the SCORED policy — the L2 analogue of
    ``arm_differentiation.trace_behavior_facts``.

    The L2 role replay (``p1._outer_validation_final_eval``) keeps its
    trace in memory and returns the per-week return vector instead of
    writing a CSV, so the behavioural evidence available here is that
    vector plus the closed-trade count. The degeneracy predicate is the
    trace helper's, transposed: a scored window with no trades, or whose
    weekly return vector carries a single distinct value, could not have
    depended on its observations, and two such arms are ENTITLED to
    identical metrics. Degeneracy is proven from the evidence, never
    assumed.

    Identity labels (seed, run_id, episode_id) are excluded by
    construction: only the weekly vector, the trade count and the role
    name enter the digest.
    """
    roles = []
    trades = []
    distinct = 0
    weeks = 0
    for role in EVIDENCE_PAIR:
        payload = (evidence or {}).get(role) or {}
        vector = [float(value) for value in
                  (payload.get("weekly_return_vector") or [])]
        total = payload.get("trades_total")
        roles.append({"role": role,
                      "weekly_return_vector": vector,
                      "trades_total": total})
        weeks = max(weeks, len(vector))
        distinct = max(distinct, len({repr(value) for value in vector}))
        if total is not None:
            trades.append(int(total))
    if not any(role["weekly_return_vector"] for role in roles) and not trades:
        return {"schema": BEHAVIOR_FACTS_SCHEMA,
                "behavior_fingerprint": None,
                "behavior_degenerate": True,
                "trades_total": 0,
                "distinct_weekly_returns": 0,
                "weeks": 0,
                "reason": "no scored evidence exists for this candidate"}
    fingerprint = hashlib.sha256(json.dumps(
        roles, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()
    trades_total = min(trades) if trades else 0
    degenerate = trades_total <= 0 or distinct <= 1
    return {
        "schema": BEHAVIOR_FACTS_SCHEMA,
        "behavior_fingerprint": fingerprint,
        "behavior_degenerate": bool(degenerate),
        "trades_total": trades_total,
        "distinct_weekly_returns": distinct,
        "weeks": weeks,
        "reason": ("the scored window produced no trades or a constant "
                   "weekly return vector, so this policy's output could "
                   "not have depended on its observations"
                   if degenerate else
                   "the scored window varies week to week under a live "
                   "trade count"),
    }


def candidate_identity(chain_id: str, generation: int, index: int,
                       genome_fingerprint: str, stage_name: str) -> str:
    return _canonical_sha({
        "chain_identity": chain_id,
        "generation": int(generation),
        "index": int(index),
        "genome_fingerprint": genome_fingerprint,
        "stage": stage_name,
    })[:16]


# ---------------------------------------------------------------------------
# GPU gates and heartbeats (the proven fleet primitives)
# ---------------------------------------------------------------------------

def check_gpu_binding(contract: dict, worker_id: str, *,
                      hostname: str | None = None,
                      cuda_env: str | None = "<from-environment>",
                      observed_uuids: list | None = None,
                      enforce: bool = True) -> dict | None:
    assignment = ((contract.get("workers") or {}).get("assignments")
                  or {}).get(worker_id) or {}
    hostname = hostname or socket.gethostname()
    if cuda_env == "<from-environment>":
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if assignment.get("hostname") != hostname:
        return {"outcome": "REFUSED_WRONG_HOST",
                "reason": (f"worker {worker_id} is assigned to "
                           f"{assignment.get('hostname')!r}, this is "
                           f"{hostname!r}")}
    if not enforce:
        return None
    assigned = assignment.get("gpu_uuid")
    observed = (visible_gpu_uuids() if observed_uuids is None
                else observed_uuids)
    if assigned not in observed:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": f"assigned GPU {assigned} not visible on "
                          f"{hostname}"}
    if cuda_env is None:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": "CUDA_VISIBLE_DEVICES is unset; visibility is "
                          "not an execution binding (WP13)"}
    if cuda_env != assigned:
        return {"outcome": "REFUSED_GPU_UNBOUND",
                "reason": (f"CUDA_VISIBLE_DEVICES={cuda_env!r} != the "
                           f"assignment {assigned!r}")}
    return None


def gpu_launch_gate(assigned_gpu_uuid: str, *,
                    heartbeat: dict | None = None) -> tuple:
    if heartbeat is None:
        heartbeat = gpu_probe.collect_heartbeat()
    payload, exit_code = gpu_probe.launch_gate(heartbeat, assigned_gpu_uuid)
    if exit_code != gpu_probe.EXIT_OK:
        return payload, {
            "outcome": "REFUSED_GPU_UNBOUND",
            "reason": ("gpu_readiness_probe launch gate refused: "
                       + "; ".join(payload.get("reasons")
                                   or payload.get("blocking")
                                   or ["unknown"])),
            "gate": payload}
    return payload, None


class CandidateHeartbeat:
    """Atomic heartbeat with assigned/bound/observed CUDA facts plus the
    chain coordinates, so fleet status reads the chain, not a guess."""

    def __init__(self, path: Path, *, contract: dict, worker_id: str,
                 arm: str, mode: str, chain_id: str, exp_id: str):
        self.path = path
        self.contract = contract
        self.worker_id = worker_id
        self.arm = arm
        self.mode = mode
        self.chain_id = chain_id
        self.exp_id = exp_id
        self._state: dict = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def beat(self, **update) -> None:
        self._state.update(update)
        pid = os.getpid()
        assignment = ((self.contract.get("workers") or {}).get(
            "assignments") or {}).get(self.worker_id) or {}
        self._state.update({
            "schema": HEARTBEAT_SCHEMA,
            "mode": self.mode,
            "arm": self.arm,
            "worker_id": self.worker_id,
            "chain_identity": self.chain_id,
            "experiment_identity": self.exp_id,
            "pid": pid,
            "pid_start_identity": _pid_start_identity(pid),
            "hostname": socket.gethostname(),
            "assigned_gpu_uuid": assignment.get("gpu_uuid"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "observed_gpu_uuids": visible_gpu_uuids(),
            **ladder.gpu_telemetry(assignment.get("gpu_uuid")),
            "updated_utc": _utc(),
        })
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.path, self._state)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(HEARTBEAT_INTERVAL_S):
                self.beat()
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# candidate evaluation
# ---------------------------------------------------------------------------

def _default_pipeline_factory(config: dict):
    from pipeline_plugins.rl_pipeline_with_solvency_curriculum import (
        PipelinePlugin as CurriculumPipeline)
    return CurriculumPipeline(config)


def _role_eval(*, config: dict, agent, model_path: str, role: str,
               nested_roles: dict, seed: int, solvency_mode: str) -> dict:
    """ONE context-prefix replay implementation for BOTH members of the
    L2 pair — the P1LR outer-truth replay, parameterized by role."""
    return p1._outer_validation_final_eval(
        config=config, agent=agent, model_path=model_path,
        artifact_role="best_checkpoint", nested_roles=nested_roles,
        seed=int(seed), role=role, solvency_mode=solvency_mode,
        run_id="l2_curriculum")


def evaluate_candidate(contract: dict, bindings: dict, *, arm: str,
                       stage: dict, genome: dict, index: int,
                       generation: int, chain_id: str, out_dir: Path,
                       mode: str, pipeline_factory=None,
                       agent_loader=None, nested_roles_fn=None,
                       role_eval_fn=None, policy_tensor_sha_fn=None,
                       anchor_facts: dict | None = None) -> dict:
    """Train ONE candidate under the frozen L1 recipe and score it with
    the paired inner/outer comparator under the stage's difficulty.

    Easy stages are scored under easy solvency dynamics and the record is
    stamped ``evaluation_difficulty=easy_chronological_continuation`` so
    it can never be mistaken for decision evidence.

    Gate 11 fires BEFORE the paired evidence is computed: if the selected
    checkpoint's policy tensor is bit-identical to the frozen warm-start
    anchor, this candidate did not learn — it IS the anchor — and scoring
    it would produce the anchor's numbers under the treatment's name.
    Such a candidate is typed as a REJECTED, MEASURED outcome (never an
    imputed zero), exactly like the inactive case beside it.
    """
    started = _utc()
    config = materialize_candidate_config(
        contract, bindings, arm=arm, stage=stage, genome=genome,
        out_dir=out_dir, mode=mode)
    identity = config.pop("_identity")
    nested_roles = (nested_roles_fn or materialize_l2_nested_roles)(
        contract, bindings, out_dir)
    agent = (agent_loader or ladder._agent_plugin)(
        identity["plugins"]["agent_plugin"])
    pipeline = (pipeline_factory or _default_pipeline_factory)(config)
    result = pipeline.run_pipeline(config=config, env_plugin=None,
                                   agent_plugin=agent, mode="train")

    terminal_path = result.get("terminal_model_path")
    if not terminal_path or not Path(terminal_path).is_file():
        raise RuntimeError(
            "candidate finished without a terminal artifact — refused")
    terminal_sha = _sha_file(Path(terminal_path))
    best_path = result.get("best_model_path")
    active = bool(best_path) and Path(str(best_path)).is_file()

    # --- gate 11: the selected checkpoint may not BE the anchor --------
    anchor_gate = selected_checkpoint_anchor_facts(
        contract, str(best_path) if active else None,
        policy_tensor_sha_fn=policy_tensor_sha_fn,
        anchor_facts=anchor_facts)
    anchor_fallback = anchor_gate["selected_equals_anchor"] is True

    difficulty = str(stage["evaluation_difficulty"])
    solvency_mode = (_l2.EASY if difficulty == _l2.EASY else _l2.NORMAL)
    evidence: dict = {}
    paired: dict
    rejected_reason = None
    if anchor_fallback:
        paired = {"schema": _paired.SCHEMA, "metric": SELECTION_METRIC,
                  "eligible": False, "paired_score": None,
                  "ineligibility_reasons": [
                      "selected checkpoint is bit-identical to the frozen"
                      " warm-start anchor — the candidate did not learn,"
                      " so its score would measure the anchor and not the"
                      " treatment (finding D8-20260815)"]}
        rejected_reason = "selected_checkpoint_is_the_frozen_warm_start_anchor"
    elif active:
        for role in EVIDENCE_PAIR:
            evidence[role] = (role_eval_fn or _role_eval)(
                config=config, agent=agent, model_path=str(best_path),
                role=role, nested_roles=nested_roles,
                seed=int(config["eval_seed"]),
                solvency_mode=solvency_mode)
        paired = _l2.paired_l2_fitness(
            _paired_summary(evidence["inner_validation"]),
            _paired_summary(evidence["outer_validation"]),
            beta=float(contract["l2_gap_penalty_beta"]),
            min_trades=int(contract.get("l2_min_trades", 1)),
            candidate_id=f"{chain_id}:{generation}:{index}")
    else:
        paired = {"schema": _paired.SCHEMA, "metric": SELECTION_METRIC,
                  "eligible": False, "paired_score": None,
                  "ineligibility_reasons": [
                      "candidate produced no activity-eligible checkpoint"
                      " — a measured outcome, never an imputed zero"]}
        rejected_reason = "no_activity_eligible_checkpoint"

    fitness = paired.get("paired_score") if paired.get("eligible") else None
    behavior = evidence_behavior_facts(evidence)
    record = {
        "schema": CANDIDATE_RECORD_SCHEMA,
        "candidate_identity": candidate_identity(
            chain_id, generation, index, identity["genome_fingerprint"],
            str(stage["name"])),
        "chain_identity": chain_id,
        "arm": arm,
        "mode": mode,
        "generation": int(generation),
        "index": int(index),
        "stage": dict(stage),
        # The single most load-bearing field in the whole program: an
        # easy score is permanently labelled and can never silently be
        # compared with a normal-realistic score.
        "evaluation_difficulty": difficulty,
        "decision_evidence": difficulty == _l2.NORMAL,
        "promotion_eligible": bool(active and difficulty == _l2.NORMAL
                                   and paired.get("eligible")),
        "requires_normal_reevaluation": False,
        "parameters": dict(identity["genome"]["parameters"]),
        "genome_fingerprint": identity["genome_fingerprint"],
        "fitness": fitness,
        "paired": paired,
        "candidate_rejected": rejected_reason is not None,
        "candidate_rejected_reason": rejected_reason,
        "activity_status": "active" if active else "inactive",
        "evidence": evidence,
        "terminal_model_path": str(terminal_path),
        "terminal_model_sha256": terminal_sha,
        "best_model_path": str(best_path) if active else None,
        "best_model_sha256": (_sha_file(Path(str(best_path)))
                              if active else None),
        # --- gate 11 facts (finding D8-20260815) -----------------------
        # The identity of the weights that produced `paired`, hashed as
        # TENSORS. `best_model_sha256` above is the CONTAINER digest and
        # is unusable as an identity: SB3 zips embed member mtimes, so
        # the same weights saved twice hash differently there.
        "scored_policy_tensor_sha256":
            anchor_gate["selected_policy_tensor_sha256"],
        "anchor_policy_tensor_sha256":
            anchor_gate["anchor_policy_tensor_sha256"],
        "selected_equals_anchor": anchor_gate["selected_equals_anchor"],
        "anchor_fallback_gate": anchor_gate,
        # --- behaviour evidence for the differentiation ladder ---------
        "behavior": behavior,
        "behavior_fingerprint": behavior["behavior_fingerprint"],
        "behavior_degenerate": behavior["behavior_degenerate"],
        "config_identity": identity,
        "resolved_config_sha256": sysid.resolved_config_sha256(config),
        "nested_split_manifest_sha256": nested_roles["manifest_sha256"],
        "nested_role_facts": {
            role: {key: (nested_roles["roles"].get(role) or {}).get(key)
                   for key in p1.NESTED_ROLE_FACT_KEYS}
            for role in _nested_splits.ROLES},
        "split_identity": split_identity(nested_roles),
        "started_utc": started,
        "finished_utc": _utc(),
        "hostname": socket.gethostname(),
    }
    return record


def _paired_summary(role_payload: dict) -> dict:
    """Flatten one role replay into the summary shape the comparator
    reads, keeping the raw facts visible."""
    metrics = dict(role_payload.get("metrics") or {})
    metrics["trades_total"] = role_payload.get("trades_total")
    metrics.setdefault("nested_role", role_payload.get("role"))
    return metrics


# ---------------------------------------------------------------------------
# generation lifecycle
# ---------------------------------------------------------------------------

def generation_dir(chain_path: Path, generation: int) -> Path:
    return chain_path / f"gen{int(generation):03d}"


def materialize_generation_population(contract: dict, chain_path: Path,
                                      generation: int, *,
                                      optimizer=None) -> dict:
    """Write ``population.json`` for one generation, either the genesis
    population (deterministic from the contract's population seed) or the
    reproduction of the previous finalized generation."""
    registry = json.loads((chain_path / "chain.json").read_text())
    arm = registry["arm"]
    mode = registry["mode"]
    schedule = registry["stage_schedule"]
    population_size = int(registry["population_size"])
    shared = contract["shared"]
    gene_space = [tuple(item) for item in shared["gene_space"]]
    gene_names = [str(item[0]) for item in gene_space]
    plugin = optimizer or _build_optimizer(contract, arm, mode)

    gen_path = generation_dir(chain_path, generation)
    gen_path.mkdir(parents=True, exist_ok=True)
    population_path = gen_path / "population.json"
    if population_path.is_file():
        return json.loads(population_path.read_text())

    stage = _l2.stage_for_generation(schedule, generation)
    if generation == 0:
        state = plugin.create_shared_population(
            population_size, seed=int(shared["population_seed"]))
        payload = {
            "generation": 0,
            "stage": stage,
            "stage_idx": 0,
            "population": state["population"],
            "innovation_tracker": state["innovation_tracker"],
            "param_defaults": state["param_defaults"],
            "config_snapshot": state["config_snapshot"],
            "difficulty_transition": False,
            "easy_fitness_invalidated": False,
        }
    else:
        previous = json.loads(
            (generation_dir(chain_path, generation - 1)
             / "generation_record.json").read_text())
        payload = {
            "generation": int(generation),
            "stage": stage,
            "stage_idx": int(previous["next_stage_idx"]),
            "population": previous["next_population"],
            "innovation_tracker": previous["innovation_tracker"],
            "param_defaults": previous["param_defaults"],
            "config_snapshot": previous.get("config_snapshot"),
            "difficulty_transition": bool(
                previous.get("difficulty_transition")),
            "easy_fitness_invalidated": bool(
                previous.get("easy_fitness_invalidated")),
        }
    for genome in payload["population"]:
        _l2.assert_typed_genes_only(genome, gene_names=gene_names)
    payload["schema"] = "agent_multi.l2_generation_population.v1"
    payload["chain_identity"] = registry["chain_identity"]
    payload["population_fingerprint"] = _canonical_sha(
        {"generation": payload["generation"],
         "stage_idx": payload["stage_idx"],
         "population": [genome["parameters"]
                        for genome in payload["population"]]})[:32]
    _atomic_json(population_path, payload)
    return payload


def _build_optimizer(contract: dict, arm: str, mode: str):
    plugin = _l2.Plugin({})
    plugin.bind_l2_contract(
        gene_space=contract["shared"]["gene_space"],
        stage_schedule=mode_setting(contract, mode, "stage_schedules")[arm],
        frozen_gene_names=contract["frozen_l1_recipe"][
            "l1_frozen_gene_names"],
        population_seed=int(contract["shared"]["population_seed"]),
        elitism=int(contract["shared"].get("elitism", 1)),
        ga_cxpb=float(contract["shared"].get("ga_cxpb", 0.5)),
        ga_mutpb=float(contract["shared"].get("ga_mutpb", 0.2)),
        patience=int(mode_setting(contract, mode,
                                  "l2_stopping")["patience"]))
    return plugin


def finalize_generation(contract: dict, chain_path: Path, generation: int,
                        *, optimizer=None) -> dict:
    """Close ONE generation: collect its candidate records, log diversity,
    apply the paired L2 patience decision, reproduce the next population
    (invalidating easy fitness at a difficulty transition) and extend the
    finalized ancestry. Fail-closed while any record is missing."""
    registry = json.loads((chain_path / "chain.json").read_text())
    arm, mode = registry["arm"], registry["mode"]
    schedule = registry["stage_schedule"]
    population_size = int(registry["population_size"])
    gene_space = [tuple(item) for item in contract["shared"]["gene_space"]]
    gene_names = [str(item[0]) for item in gene_space]
    gene_bounds = {str(n): (lo, hi) for n, lo, hi, _k in gene_space}
    gen_path = generation_dir(chain_path, generation)
    record_path = gen_path / "generation_record.json"
    if record_path.is_file():
        return json.loads(record_path.read_text())

    population = json.loads((gen_path / "population.json").read_text())
    stage = population["stage"]
    records = []
    for index in range(population_size):
        path = gen_path / f"candidate{index:03d}" / "candidate_record.json"
        if not path.is_file():
            raise RuntimeError(
                f"generation {generation} is not complete: candidate "
                f"{index} has no record — refusing to close it")
        records.append(json.loads(path.read_text()))

    # requirement 2 — the difficulty label of every record must equal the
    # stage's declared difficulty; a mixed generation is refused.
    difficulties = {record["evaluation_difficulty"] for record in records}
    if difficulties != {stage["evaluation_difficulty"]}:
        raise RuntimeError(
            f"generation {generation} mixes evaluation difficulties "
            f"{sorted(difficulties)} against the declared stage "
            f"{stage['evaluation_difficulty']!r} — easy and normal scores "
            "never share one comparable leaderboard")

    diversity = _l2.population_diversity(
        records, gene_names=gene_names, gene_bounds=gene_bounds)

    # requirement 5 — paired inner/outer patience after the floor. Only a
    # normal-realistic generation may move the decision-side patience.
    stopping_spec = mode_setting(contract, mode, "l2_stopping")
    identity = str(records[0].get("split_identity"))
    state = _paired.PairedStoppingState(
        patience=int(stopping_spec["patience"]),
        floor=int(stopping_spec["minimum_generations"]),
        min_delta=float(stopping_spec["min_delta"]),
        split_identity=identity)
    history_path = chain_path / "stopping_state.json"
    if history_path.is_file():
        state = _paired.PairedStoppingState.from_state(
            json.loads(history_path.read_text()),
            split_identity=identity)
    if stage["evaluation_difficulty"] == _l2.NORMAL:
        eligible = [record for record in records
                    if (record.get("paired") or {}).get("eligible")]
        best = max(eligible,
                   key=lambda r: float(r["paired"]["paired_score"]),
                   default=None)
        best_paired = (best["paired"] if best is not None
                       else {"eligible": False})
        stop = _l2.l2_generation_stop_decision(state, best_paired,
                                               generation)
        _atomic_json(history_path, state.to_state())
    else:
        best = None
        stop = {"generation": int(generation), "improved": False,
                "stop": False,
                "reason": "easy triage generation: easy fitness may never "
                          "touch the normal decision patience",
                "minimum_generations": state.floor,
                "patience": state.patience, "waited": state.waited,
                "best_paired_score": state.best_score,
                "floor_reached": int(generation) >= state.floor,
                "evidence_pair": list(EVIDENCE_PAIR)}

    plugin = optimizer or _build_optimizer(contract, arm, mode)
    evaluated = [{
        "parameters": dict(record["parameters"]),
        "fitness": (record["fitness"] if record["fitness"] is not None
                    else float("-inf")),
        "evaluation_difficulty": record["evaluation_difficulty"],
        "candidate_id": record["candidate_identity"],
        "candidate_rejected": bool(record.get("candidate_rejected")),
        "paired": record.get("paired"),
    } for record in records]

    total_budget = int(registry["total_candidate_evaluations"])
    spent = (int(generation) + 1) * population_size
    budget_exhausted = spent >= total_budget
    reproduction = plugin.reproduce_shared(
        evaluated, int(generation), _reproduction_seed(registry, generation,
                                                       evaluated),
        population["innovation_tracker"], schedule,
        population["param_defaults"],
        current_stage_idx=int(population["stage_idx"]),
        no_improve_count=int(state.waited))

    record = {
        "schema": GENERATION_RECORD_SCHEMA,
        "chain_identity": registry["chain_identity"],
        "arm": arm,
        "mode": mode,
        "generation": int(generation),
        "stage": stage,
        "evaluation_difficulty": stage["evaluation_difficulty"],
        "population_size": population_size,
        "population_fingerprint": population["population_fingerprint"],
        "candidate_identities": [record_["candidate_identity"]
                                 for record_ in records],
        "diversity": diversity,
        "stopping": stop,
        "budget": {"spent_candidate_evaluations": spent,
                   "total_candidate_evaluations": total_budget,
                   "exhausted": budget_exhausted},
        "best_candidate": (
            {"candidate_identity": best["candidate_identity"],
             "fitness": best["fitness"],
             "evaluation_difficulty": best["evaluation_difficulty"],
             "parameters": best["parameters"]}
            if best is not None else None),
        "difficulty_transition": bool(
            reproduction.get("difficulty_transition")),
        "easy_fitness_invalidated": bool(
            reproduction.get("easy_fitness_invalidated")),
        "invalidation": reproduction.get("invalidation"),
        "requires_normal_reevaluation": int(
            reproduction.get("requires_normal_reevaluation") or 0),
        "next_population": reproduction["population"],
        "next_stage_idx": int(reproduction["stage_idx"]),
        "innovation_tracker": population["innovation_tracker"],
        "param_defaults": population["param_defaults"],
        "config_snapshot": population.get("config_snapshot"),
        "terminal": bool(stop["stop"]) or budget_exhausted,
        "finalized_utc": _utc(),
    }
    _l2.assert_diversity_logged(record,
                                contract["diversity"]["required_fields"])
    _atomic_json(record_path, record)

    registry["finalized_generations"] = int(generation) + 1
    registry["finalized_ancestry"] = compute_finalized_ancestry(chain_path)
    _atomic_json(chain_path / "chain.json", registry)
    if record["terminal"]:
        _atomic_json(chain_path / "arm_complete.json", {
            "schema": "agent_multi.l2_arm_complete.v1",
            "chain_identity": registry["chain_identity"],
            "arm": arm, "mode": mode,
            "generations_finalized": int(generation) + 1,
            "reason": ("budget_exhausted" if budget_exhausted
                       else "paired_generation_patience"),
            "finalized_ancestry": registry["finalized_ancestry"],
            "completed_utc": _utc(),
        })
    return record


def _reproduction_seed(registry: dict, generation: int,
                       evaluated: list) -> int:
    """The doin-node convention: the reproduction seed is derived from
    the FULLY OBSERVED generation, so every node reproduces the same next
    population without a privileged coordinator."""
    payload = json.dumps(
        {str(index): entry["fitness"]
         for index, entry in enumerate(evaluated)}, sort_keys=True)
    digest = hashlib.sha256(
        f"{registry['chain_identity']}:{generation}:{payload}".encode())
    return int.from_bytes(digest.digest()[:4], "big")


def chain_candidate_records(chain_path: Path, *,
                            difficulty: str | None = None) -> list:
    """Every finalized candidate record of ONE chain, in chain order."""
    registry = json.loads((chain_path / "chain.json").read_text())
    entries = []
    for generation in range(int(registry["finalized_generations"])):
        gen_path = generation_dir(chain_path, generation)
        record = json.loads((gen_path / "generation_record.json").read_text())
        if difficulty is not None and \
                record["evaluation_difficulty"] != difficulty:
            continue
        for index in range(int(registry["population_size"])):
            path = gen_path / f"candidate{index:03d}" / \
                "candidate_record.json"
            if path.is_file():
                entries.append(json.loads(path.read_text()))
    return entries


def arm_anchor_fallback_facts(chain_path: Path) -> dict:
    """Gate 11, per ARM: how many selected checkpoints ARE the anchor.

    ``checked_candidates`` counts only the candidates that actually
    selected a checkpoint and whose tensor identity was established, so
    an inactive arm is never mistaken for an anchor-fallback arm — those
    are different diagnoses with different remedies.
    """
    registry = json.loads((chain_path / "chain.json").read_text())
    records = chain_candidate_records(chain_path)
    checked = [record for record in records
               if record.get("selected_equals_anchor") is not None]
    equal = [record for record in checked
             if record["selected_equals_anchor"] is True]
    anchor_shas = {str(record.get("anchor_policy_tensor_sha256"))
                   for record in checked
                   if record.get("anchor_policy_tensor_sha256")}
    facts = {
        "schema": ARM_ANCHOR_SCHEMA,
        "arm": registry.get("arm"),
        "mode": registry.get("mode"),
        "chain_identity": registry.get("chain_identity"),
        "chain_dir": str(chain_path),
        "candidates": len(records),
        "checked_candidates": len(checked),
        "selected_equals_anchor_count": len(equal),
        "selected_equals_anchor_ids": [record["candidate_identity"]
                                       for record in equal],
        "anchor_policy_tensor_sha256": (sorted(anchor_shas)[0]
                                        if len(anchor_shas) == 1 else
                                        sorted(anchor_shas) or None),
        "distinct_scored_policy_tensors": len({
            str(record.get("scored_policy_tensor_sha256"))
            for record in checked
            if record.get("scored_policy_tensor_sha256")}),
        "inactive_candidates": sum(
            1 for record in records
            if record.get("activity_status") != "active"),
        "all_selected_equal_anchor": bool(checked) and len(equal) == len(
            checked),
    }
    facts["verdict"] = ("ANCHOR_FALLBACK" if facts["all_selected_equal_anchor"]
                        else "NOT_EVALUATED" if not checked
                        else "PARTIAL_ANCHOR_FALLBACK" if equal
                        else "OK")
    return facts


def assert_arm_not_anchor_fallback(chain_path: Path) -> dict:
    """Refuse an arm that measured the anchor instead of the treatment.

    The diagnosis of finding D8-20260815 states this single check "would
    have caught sites 1 and 2 on day one". An arm whose every selected
    checkpoint is the frozen anchor has produced NO evidence about its
    treatment, so it refuses rather than publishing a null.
    """
    facts = arm_anchor_fallback_facts(chain_path)
    if facts["all_selected_equal_anchor"]:
        raise L2AnchorFallbackRefusal(facts)
    return facts


def champion_of_chain(chain_path: Path) -> dict:
    """The ONLY champion resolver. It refuses first: an arm that measured
    only the anchor (gate 11), then an easy-evaluated or
    pending-re-evaluation entry, can never become champion (requirements
    2 and 11).
    """
    anchor = assert_arm_not_anchor_fallback(chain_path)
    entries = chain_candidate_records(chain_path, difficulty=_l2.NORMAL)
    ranked = _l2.normal_leaderboard(entries)
    if not ranked:
        return {"champion": None,
                "reason": "no eligible normal-realistic candidate",
                "anchor_fallback": anchor}
    _l2.assert_promotion_eligible(ranked[0], action="champion")
    return {"champion": ranked[0], "leaderboard_size": len(ranked),
            "anchor_fallback": anchor}


# ---------------------------------------------------------------------------
# gates 12 and 13 — the ARM-COMPARISON boundary
#
# The only place where L2_N and L2_EN are put beside each other. Nothing
# may cross this boundary without proving that the two arms could have
# differed at all: `arm_differentiation.assert_arms_differentiated` reads
# the same per-candidate facts the records already carry and answers,
# with a typed verdict, whether an identical result is a real outcome or
# a lost distinction. Its semantics are respected exactly — a legitimately
# dead pair of arms (DEGENERATE_IDENTICAL) is NOT a defect, and is typed
# CAMPAIGN_UNANSWERABLE by require_informative rather than published as a
# scientific null.
# ---------------------------------------------------------------------------

def arm_treatment(contract: dict, arm: str, mode: str) -> dict:
    """The DECLARED treatment of one arm: its stage difficulty policy.

    This is what the campaign claims to vary. If two arms render the
    same treatment key the ladder returns SAME_TREATMENT and refuses to
    call the pair a comparison at all.
    """
    schedule = arm_schedule(contract, arm, mode)
    return {
        "arm": arm,
        "stage_names": [str(stage["name"]) for stage in schedule],
        "stage_kinds": [str(stage["stage_kind"]) for stage in schedule],
        "stage_difficulties": [str(stage["evaluation_difficulty"])
                               for stage in schedule],
        "declares_easy_triage": any(
            stage["evaluation_difficulty"] == _l2.EASY
            for stage in schedule),
        "decision_evidence_difficulty": _l2.NORMAL,
    }


def candidate_metric_tuple(record: dict | None) -> dict:
    """The comparable metric tuple of ONE candidate record.

    Absent evidence renders as ``None`` rather than an imputed zero, so a
    dead arm never fabricates a number it did not measure.
    """
    evidence = (record or {}).get("evidence") or {}
    outer = dict((evidence.get("outer_validation") or {}).get("metrics")
                 or {})
    inner = dict((evidence.get("inner_validation") or {}).get("metrics")
                 or {})
    paired = (record or {}).get("paired") or {}
    return {
        "paired_score": paired.get("paired_score"),
        "outer_mean_weekly_rap": outer.get("mean_weekly_rap"),
        "outer_mean_weekly_return": outer.get("mean_weekly_return"),
        "outer_total_return": outer.get("total_return"),
        "outer_max_drawdown_fraction": outer.get("max_drawdown_fraction"),
        "outer_trades_total": (evidence.get("outer_validation")
                               or {}).get("trades_total"),
        "inner_mean_weekly_rap": inner.get("mean_weekly_rap"),
    }


def arm_observation(contract: dict, chain_path: Path, *, arm: str,
                    mode: str) -> tuple:
    """Build ONE arm's :class:`ArmObservation` from REAL candidate facts.

    Every field comes from the arm's finalized records — no field is
    synthesized, and the scored identity is the policy TENSOR sha the
    per-candidate gate already recorded, never the container digest.
    Returns ``(observation, facts)``; ``facts`` is the audit trail.
    """
    resolved = champion_of_chain(chain_path)
    champion = resolved.get("champion")
    anchor = resolved.get("anchor_fallback") or {}
    records = chain_candidate_records(chain_path)
    registry = json.loads((chain_path / "chain.json").read_text())
    active = champion is not None
    behavior = (champion or {}).get("behavior") or {}
    metrics = candidate_metric_tuple(champion)
    trades = metrics.get("outer_trades_total")
    observation = _armdiff.ArmObservation(
        arm=arm,
        treatment=arm_treatment(contract, arm, mode),
        metrics=metrics,
        scored_policy_tensor_sha256=(
            (champion or {}).get("scored_policy_tensor_sha256")),
        behavior_fingerprint=behavior.get("behavior_fingerprint"),
        behavior_degenerate=(behavior.get("behavior_degenerate")
                             if champion is not None else True),
        active=active,
        trades_total=(int(trades) if trades is not None else
                      (None if active else 0)),
        provenance={
            "mode": mode,
            "chain_identity": registry.get("chain_identity"),
            "chain_dir": str(chain_path),
            "experiment_identity": registry.get("experiment_identity"),
            "finalized_generations": registry.get("finalized_generations"),
            "finalized_ancestry": registry.get("finalized_ancestry"),
            "candidate_identity": (champion or {}).get("candidate_identity"),
            "genome_fingerprint": (champion or {}).get("genome_fingerprint"),
            "evaluation_difficulty": (
                (champion or {}).get("evaluation_difficulty")),
            "candidates": len(records),
            "anchor_policy_tensor_sha256":
                anchor.get("anchor_policy_tensor_sha256"),
            "selected_equals_anchor_count":
                anchor.get("selected_equals_anchor_count"),
            "distinct_scored_policy_tensors":
                anchor.get("distinct_scored_policy_tensors"),
        })
    facts = {
        "arm": arm,
        "champion_resolved": active,
        "champion_reason": resolved.get("reason"),
        "anchor_fallback": anchor,
        "metrics": metrics,
        "scored_policy_tensor_sha256":
            observation.scored_policy_tensor_sha256,
        "behavior_fingerprint": observation.behavior_fingerprint,
        "behavior_degenerate": observation.is_degenerate(),
        "trades_total": observation.trades_total,
        "provenance": dict(observation.provenance),
    }
    return observation, facts


def compare_arms(contract: dict, bindings: dict | None = None, *,
                 mode: str = "smoke", sources: dict | None = None,
                 require_informative: bool = True) -> tuple:
    """Gates 12 + 13: the campaign verdict, or a typed refusal.

    ``require_informative=True`` is the default and the point: a campaign
    in which no cross-treatment pair separates is UNANSWERABLE, not a
    null result. A null published from two arms that both collapsed —
    or worse, that both scored the same object — is exactly the artefact
    that voided three prior experiments.
    """
    bindings = bindings or load_bindings()
    sources = sources or ladder.source_identities()
    exp_id = experiment_identity(contract, bindings, sources=sources,
                                 mode=mode)
    payload: dict = {
        "schema": CAMPAIGN_VERDICT_SCHEMA,
        "generated_utc": _utc(),
        "mode": mode,
        "experiment_identity": exp_id,
        "contract_sha256": contract["_contract_sha256"],
        "metric_keys": list(L2_METRIC_KEYS),
        "require_informative": bool(require_informative),
        "checkpoint_identity": "policy_tensor_sha256",
        "arms": {},
        "refusals": [],
    }
    observations = []
    for arm in ARMS:
        chain = chain_dir(contract, exp_id, arm, mode,
                          chain_identity(contract, bindings, arm=arm,
                                         mode=mode, sources=sources))
        if not (chain / "chain.json").is_file():
            payload["refusals"].append(
                f"arm {arm} has no chain under {chain} — the comparison "
                "cannot be made from an unfinished campaign")
            payload["arms"][arm] = {"arm": arm, "chain_dir": str(chain),
                                    "state": "MISSING"}
            continue
        if not (chain / "arm_complete.json").is_file():
            payload["refusals"].append(
                f"arm {arm} has not published arm_complete.json — the "
                "comparison cannot be made from an unfinished campaign")
            payload["arms"][arm] = {"arm": arm, "chain_dir": str(chain),
                                    "state": "INCOMPLETE"}
            continue
        try:
            observation, facts = arm_observation(contract, chain, arm=arm,
                                                 mode=mode)
        except L2AnchorFallbackRefusal as exc:
            payload["arms"][arm] = {"arm": arm, "chain_dir": str(chain),
                                    "state": "ANCHOR_FALLBACK",
                                    "anchor_fallback": exc.facts,
                                    "refusal": str(exc)}
            payload["refusals"].append(str(exc))
            payload["outcome"] = "REFUSED_ANCHOR_FALLBACK"
            continue
        facts["state"] = "COMPLETE"
        facts["chain_dir"] = str(chain)
        payload["arms"][arm] = facts
        observations.append(observation)

    if payload.get("outcome") == "REFUSED_ANCHOR_FALLBACK":
        return payload, EXIT_CLASS["REFUSED_ANCHOR_FALLBACK"]
    if payload["refusals"]:
        payload["outcome"] = "CAMPAIGN_INCOMPLETE"
        return payload, EXIT_CLASS["CAMPAIGN_INCOMPLETE"]

    try:
        report = _armdiff.assert_arms_differentiated(
            observations, metric_keys=L2_METRIC_KEYS,
            require_informative=require_informative)
        payload["differentiation"] = report
        payload["outcome"] = "CAMPAIGN_VERDICT"
        payload["statement"] = (
            "the two arms declare different treatments and their measured "
            "results separate; the comparison measures the treatment")
        return payload, EXIT_CLASS["CAMPAIGN_VERDICT"]
    except _armdiff.ArmDifferentiationRefusal as exc:
        payload["differentiation"] = exc.report
        verdicts = {pair["verdict"] for pair in exc.refusals}
        payload["refusals"].extend(
            f"{pair['arms'][0]} vs {pair['arms'][1]}: {pair['verdict']} — "
            f"{pair['detail']}" for pair in exc.refusals)
        if verdicts == {"NO_INFORMATIVE_CONTRAST"}:
            payload["outcome"] = "CAMPAIGN_UNANSWERABLE"
            payload["statement"] = (
                "the measurement is sound but every cross-treatment pair "
                "is degenerate: this campaign cannot answer its question, "
                "and it is typed UNANSWERABLE rather than published as a "
                "scientific null")
            return payload, EXIT_CLASS["CAMPAIGN_UNANSWERABLE"]
        payload["outcome"] = "REFUSED_ARMS_NOT_DIFFERENTIATED"
        payload["statement"] = (
            "the arms cannot be told apart by the measurement pipeline "
            f"({sorted(verdicts)}); no verdict may be published")
        return payload, EXIT_CLASS["REFUSED_ARMS_NOT_DIFFERENTIATED"]


# ---------------------------------------------------------------------------
# the worker loop
# ---------------------------------------------------------------------------

def run_worker(worker_id: str, *, contract: dict, arm: str, mode: str,
               bindings: dict | None = None, enforce_gpu: bool = True,
               max_candidates: int | None = None,
               gate_heartbeat: dict | None = None,
               pipeline_factory=None, agent_loader=None,
               nested_roles_fn=None, role_eval_fn=None,
               policy_tensor_sha_fn=None) -> dict:
    """One worker collaborating on the arm's ONE chain.

    It joins (never forks) the chain, proves node identity, claims one
    unevaluated candidate at a time with a flock-backed exclusive claim,
    writes an immutable candidate record, and closes a generation only
    when every candidate record exists.

    The frozen anchor's policy tensors are hashed ONCE per worker, before
    any candidate runs: an unverifiable anchor is a typed
    ``REFUSED_ANCHOR_UNVERIFIED`` here rather than a gate that silently
    never fires (gate 11).
    """
    if worker_id not in WORKERS:
        raise ValueError(f"unknown worker {worker_id!r}")
    if arm not in ARMS:
        raise ValueError(f"unknown L2 arm {arm!r}")
    bindings = bindings or load_bindings()
    sources_before = ladder.source_identities()
    exp_id = experiment_identity(contract, bindings, sources=sources_before,
                                 mode=mode)
    try:
        anchor_facts = anchor_policy_tensor_sha256(
            contract, policy_tensor_sha_fn=policy_tensor_sha_fn)
    except RuntimeError as exc:
        return {"outcome": "REFUSED_ANCHOR_UNVERIFIED", "reason": str(exc),
                "worker_id": worker_id, "arm": arm,
                "experiment_identity": exp_id}

    refusal = check_gpu_binding(contract, worker_id, enforce=enforce_gpu)
    if refusal:
        return {**refusal, "worker_id": worker_id, "arm": arm,
                "experiment_identity": exp_id}
    gate_payload = None
    if enforce_gpu:
        assignment = contract["workers"]["assignments"][worker_id]
        gate_payload, gate_refusal = gpu_launch_gate(
            assignment["gpu_uuid"], heartbeat=gate_heartbeat)
        if gate_refusal:
            return {**gate_refusal, "worker_id": worker_id, "arm": arm,
                    "experiment_identity": exp_id}

    try:
        chain = open_or_join_chain(contract, bindings, arm=arm, mode=mode,
                                   worker_id=worker_id,
                                   sources=sources_before)
    except RuntimeError as exc:
        text = str(exc)
        outcome = ("REFUSED_PARALLEL_CHAIN"
                   if "REFUSED_PARALLEL_CHAIN" in text else
                   "REFUSED_OUT_OF_ARM_ORDER"
                   if "REFUSED_OUT_OF_ARM_ORDER" in text
                   else "REFUSED_NODE_IDENTITY_MISMATCH")
        return {"outcome": outcome, "reason": text, "worker_id": worker_id,
                "arm": arm, "experiment_identity": exp_id}
    if chain.get("outcome") == "ALREADY_RUNNING":
        return {**chain, "worker_id": worker_id, "arm": arm}

    chain_path = Path(chain["_chain_dir"])
    chain_id = chain["chain_identity"]
    population_size = int(chain["population_size"])
    total_generations = sum(int(stage["generations"])
                            for stage in chain["stage_schedule"])
    heartbeat = CandidateHeartbeat(
        chain_path / "workers" / f"{worker_id}.heartbeat.json",
        contract=contract, worker_id=worker_id, arm=arm, mode=mode,
        chain_id=chain_id, exp_id=exp_id)
    heartbeat.beat(terminal_state="RUNNING", progress="joined")
    heartbeat.start()

    evaluated = 0
    generations_closed = 0
    try:
        for generation in range(total_generations):
            if (chain_path / "arm_complete.json").is_file():
                break
            # Every node re-proves identity and finalized ancestry BEFORE
            # it evaluates in a new generation (requirement 8).
            registry = json.loads((chain_path / "chain.json").read_text())
            facts = node_identity_facts(contract, bindings, arm=arm,
                                        mode=mode, sources=sources_before)
            refusals = verify_node_identity(
                facts, registry.get("node_identity") or {},
                finalized_ancestry=compute_finalized_ancestry(chain_path),
                registered_ancestry=registry.get("finalized_ancestry"))
            if refusals:
                heartbeat.beat(
                    terminal_state="REFUSED_NODE_IDENTITY_MISMATCH",
                    error="; ".join(refusals))
                return {"outcome": "REFUSED_NODE_IDENTITY_MISMATCH",
                        "reason": "; ".join(refusals),
                        "worker_id": worker_id, "arm": arm,
                        "experiment_identity": exp_id}

            population = materialize_generation_population(
                contract, chain_path, generation)
            stage = population["stage"]
            gen_path = generation_dir(chain_path, generation)

            for index in range(population_size):
                if max_candidates is not None and evaluated >= max_candidates:
                    break
                cell = gen_path / f"candidate{index:03d}"
                record_path = cell / "candidate_record.json"
                if record_path.is_file():
                    continue
                claim = ExclusiveClaim(
                    chain_path / "locks" /
                    f"candidate.g{generation:03d}.c{index:03d}.lock")
                if not claim.acquire():
                    continue
                try:
                    if record_path.is_file():
                        continue
                    cell.mkdir(parents=True, exist_ok=True)
                    heartbeat.beat(progress="evaluating",
                                   generation=generation, candidate=index,
                                   stage=stage["name"],
                                   evaluation_difficulty=stage[
                                       "evaluation_difficulty"])
                    record = evaluate_candidate(
                        contract, bindings, arm=arm, stage=stage,
                        genome=population["population"][index],
                        index=index, generation=generation,
                        chain_id=chain_id, out_dir=cell, mode=mode,
                        pipeline_factory=pipeline_factory,
                        agent_loader=agent_loader,
                        nested_roles_fn=nested_roles_fn,
                        role_eval_fn=role_eval_fn,
                        policy_tensor_sha_fn=policy_tensor_sha_fn,
                        anchor_facts=anchor_facts)
                    record["worker_id"] = worker_id
                    sources_after = ladder.source_identities()
                    for name in sources_before:
                        sysid.assert_source_identity_unmoved(
                            sources_before[name], sources_after[name])
                    _atomic_json(record_path, record)
                    evaluated += 1
                finally:
                    try:
                        claim.release()
                    except Exception:                       # noqa: BLE001
                        pass

            complete = all(
                (gen_path / f"candidate{index:03d}" /
                 "candidate_record.json").is_file()
                for index in range(population_size))
            if not complete:
                heartbeat.beat(progress="awaiting-peers",
                               generation=generation)
                break
            close = ExclusiveClaim(
                chain_path / "locks" / f"generation.{generation:03d}.lock")
            if close.acquire():
                try:
                    heartbeat.beat(progress="finalizing-generation",
                                   generation=generation)
                    finalize_generation(contract, chain_path, generation)
                    generations_closed += 1
                finally:
                    try:
                        close.release()
                    except Exception:                       # noqa: BLE001
                        pass
            if (chain_path / "arm_complete.json").is_file():
                break
    finally:
        heartbeat.beat(terminal_state="WORKER_COMPLETE",
                       candidates_evaluated=evaluated,
                       generations_closed=generations_closed)
        heartbeat.stop()

    complete = (chain_path / "arm_complete.json").is_file()
    return {
        "outcome": "ARM_COMPLETE" if complete else "WORKER_COMPLETE",
        "worker_id": worker_id,
        "arm": arm,
        "mode": mode,
        "experiment_identity": exp_id,
        "chain_identity": chain_id,
        "chain_dir": str(chain_path),
        "candidates_evaluated": evaluated,
        "generations_closed": generations_closed,
        "gpu_gate": gate_payload,
        "anchor_policy_tensor_sha256":
            anchor_facts.get("anchor_policy_tensor_sha256"),
        "anchor_fallback": (arm_anchor_fallback_facts(chain_path)
                            if complete else None),
    }


# ---------------------------------------------------------------------------
# preflight and the smoke dispatch plan
# ---------------------------------------------------------------------------

def preflight(contract: dict, bindings: dict | None = None) -> tuple:
    """No-training acceptance boundary: really materialize the nested
    roles, prove sealed 2025 stays SEALED, prove both arms carry an
    identical budget/population/seed set, prove the frozen L1 recipe
    reaches an executable candidate config and prove the smoke and
    decision identities do not collide."""
    import tempfile

    bindings = bindings or load_bindings()
    payload: dict = {
        "schema": PREFLIGHT_SCHEMA,
        "generated_utc": _utc(),
        "contract_sha256": contract["_contract_sha256"],
        "held_fixed_bindings_contract_sha256":
            bindings["_contract_sha256"],
        "training_used": False,
        "refusals": [],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="l2_preflight_") as scratch:
            nested_roles = materialize_l2_nested_roles(
                contract, bindings, Path(scratch))
            payload["nested_split_mode"] = nested_roles["binding"]["mode"]
            payload["nested_context_bars"] = \
                nested_roles["binding"]["context_bars"]
            payload["nested_role_facts"] = {
                role: {key: (nested_roles["roles"].get(role) or {}).get(key)
                       for key in p1.NESTED_ROLE_FACT_KEYS}
                for role in _nested_splits.ROLES}
            payload["sealed_test_state"] = (
                nested_roles["roles"].get("sealed_test") or {}).get("status")
            payload["sealed_test_csv_absent"] = not (
                Path(scratch) / "nested_splits" / "sealed_test.csv").exists()
            if not payload["sealed_test_csv_absent"]:
                payload["refusals"].append(
                    "sealed_test.csv exists — sealed 2025 was touched")
            payload["split_identity"] = split_identity(nested_roles)

            sources = ladder.source_identities()
            identities = {}
            for mode in MODES:
                exp_id = experiment_identity(contract, bindings,
                                             sources=sources, mode=mode)
                comparability = assert_arms_comparable(contract, mode=mode)
                arms_block = {}
                for arm in ARMS:
                    schedule = arm_schedule(contract, arm, mode)
                    plugin = _build_optimizer(contract, arm, mode)
                    state = plugin.create_shared_population(
                        int(mode_setting(contract, mode, "population_size")),
                        seed=int(contract["shared"]["population_seed"]))
                    configs = []
                    for stage in schedule:
                        if stage["stage_kind"] != "search":
                            continue
                        config = materialize_candidate_config(
                            contract, bindings, arm=arm, stage=stage,
                            genome=state["population"][0],
                            out_dir=Path(scratch) / mode / arm /
                            stage["name"], mode=mode)
                        config.pop("_identity")
                        configs.append(config)
                        if config["selection_metric"] != SELECTION_METRIC:
                            payload["refusals"].append(
                                f"{mode}/{arm}/{stage['name']}: selection "
                                "metric drifted from the paired comparator")
                        if config["evaluate_test_split"]:
                            payload["refusals"].append(
                                f"{mode}/{arm}: evaluate_test_split leaked")
                        if config["phase1_learning_rate"] != 3e-05 or \
                                config["learning_rate"] != 3e-05:
                            payload["refusals"].append(
                                f"{mode}/{arm}/{stage['name']}: the frozen "
                                "phase-1/phase-2 LR moved")
                    arms_block[arm] = {
                        "genesis": node_identity_facts(
                            contract, bindings, arm=arm, mode=mode,
                            sources=sources)["genesis"],
                        "budget_ledger": comparability["ledgers"][arm],
                        "stage_schedule": [
                            {"name": stage["name"],
                             "stage_kind": stage["stage_kind"],
                             "evaluation_difficulty":
                                 stage["evaluation_difficulty"],
                             "generations": stage["generations"]}
                            for stage in schedule],
                        "population_fingerprint": _canonical_sha(
                            [genome["parameters"]
                             for genome in state["population"]])[:32],
                        "materialized_search_configs": len(configs),
                    }
                if arms_block["L2_N"]["population_fingerprint"] != \
                        arms_block["L2_EN"]["population_fingerprint"]:
                    payload["refusals"].append(
                        f"{mode}: the two arms do not start from the same "
                        "initial population (requirement 3)")
                if arms_block["L2_N"]["genesis"] == \
                        arms_block["L2_EN"]["genesis"]:
                    payload["refusals"].append(
                        f"{mode}: both arms derive one genesis — their "
                        "chains would be indistinguishable")
                identities[mode] = {
                    "experiment_identity": exp_id,
                    "output_root": str(output_root_for_mode(contract, mode)),
                    "comparability": comparability,
                    "arms": arms_block,
                }
            payload["modes"] = identities
            if identities["smoke"]["experiment_identity"] == \
                    identities["decision"]["experiment_identity"]:
                payload["refusals"].append(
                    "smoke and decision identities collide")

            # --- gates 11-13, declared before any GPU is touched -------
            treatments = {arm: arm_treatment(contract, arm, "smoke")
                          for arm in ARMS}
            distinguishable = (
                json.dumps(treatments["L2_N"], sort_keys=True) !=
                json.dumps(treatments["L2_EN"], sort_keys=True))
            if not distinguishable:
                payload["refusals"].append(
                    "the two arms render the same treatment key, so the "
                    "differentiation ladder would type every pair "
                    "SAME_TREATMENT and the campaign could never be a "
                    "comparison")
            payload["differentiation_precondition"] = {
                "arm_treatments": treatments,
                "treatments_distinguishable": distinguishable,
                "metric_keys": list(L2_METRIC_KEYS),
                "checkpoint_identity": "policy_tensor_sha256",
                "container_identity_rejected": (
                    "SB3 .zip containers embed member mtimes; the same "
                    "weights saved twice hash differently, so a container "
                    "digest can never prove two arms scored different "
                    "policies"),
                "gates_enforced": {
                    "anchor_fallback_per_candidate":
                        "selected_checkpoint_anchor_facts",
                    "anchor_fallback_per_arm":
                        "assert_arm_not_anchor_fallback",
                    "arm_differentiation": "compare_arms",
                    "require_informative": True,
                },
            }
            anchor = anchor_binding(contract)
            payload["anchor_declared"] = anchor is not None
            payload["anchor_path"] = (anchor or {}).get("path")
            payload["anchor_present"] = bool(
                anchor and Path(anchor["path"]).is_file())
            if anchor is not None and not payload["anchor_present"]:
                payload["refusals"].append(
                    f"the frozen common anchor {anchor['path']} is absent, "
                    "so the anchor-fallback gate could never fire")
    except Exception as exc:                                # noqa: BLE001
        payload["refusals"].append(f"{type(exc).__name__}: {exc}")

    payload["outcome"] = ("PREFLIGHT_PASS" if not payload["refusals"]
                          else "PREFLIGHT_REFUSED")
    return payload, EXIT_CLASS[payload["outcome"]]


def smoke_acceptance(contract: dict, bindings: dict | None = None, *,
                     mode: str = "smoke", sources: dict | None = None,
                     require_informative: bool = True) -> tuple:
    """The smoke's acceptance verdict, with the three gates VISIBLE.

    The smoke exists to prove the machinery, so its acceptance output
    must state plainly which of two things happened: the machinery
    distinguished its two arms, or it could not — and in the second case,
    exactly why. A smoke that runs to completion while every arm scores
    the same object is the failure this file exists to prevent, and it
    must not be able to look like a pass.
    """
    bindings = bindings or load_bindings()
    sources = sources or ladder.source_identities()
    pre, _ = preflight(contract, bindings)
    exp_id = experiment_identity(contract, bindings, sources=sources,
                                 mode=mode)
    payload: dict = {
        "schema": ACCEPTANCE_SCHEMA,
        "generated_utc": _utc(),
        "mode": mode,
        "experiment_identity": exp_id,
        "contract_sha256": contract["_contract_sha256"],
        "preflight": {"outcome": pre["outcome"],
                      "refusals": list(pre["refusals"]),
                      "differentiation_precondition": pre.get(
                          "differentiation_precondition")},
        "refusals": [],
        "gates": {},
    }
    if pre["outcome"] != "PREFLIGHT_PASS":
        payload["refusals"].extend(
            f"preflight: {reason}" for reason in pre["refusals"])

    verdict, _ = compare_arms(contract, bindings, mode=mode,
                              sources=sources,
                              require_informative=require_informative)
    payload["campaign_verdict"] = verdict
    per_arm = {}
    for arm, facts in (verdict.get("arms") or {}).items():
        anchor = facts.get("anchor_fallback") or {}
        per_arm[arm] = {
            "state": facts.get("state"),
            "checked_candidates": anchor.get("checked_candidates"),
            "selected_equals_anchor_count":
                anchor.get("selected_equals_anchor_count"),
            "distinct_scored_policy_tensors":
                anchor.get("distinct_scored_policy_tensors"),
            "anchor_policy_tensor_sha256":
                anchor.get("anchor_policy_tensor_sha256"),
            "scored_policy_tensor_sha256":
                facts.get("scored_policy_tensor_sha256"),
            "verdict": anchor.get("verdict"),
        }
    anchor_states = [entry.get("verdict") for entry in per_arm.values()]
    if verdict["outcome"] == "REFUSED_ANCHOR_FALLBACK":
        anchor_status = "REFUSED"
        anchor_statement = (
            "at least one arm selected ONLY checkpoints that are "
            "bit-identical to the frozen warm-start anchor: that arm "
            "measured the anchor, not its treatment")
    elif not anchor_states or all(state in (None, "NOT_EVALUATED")
                                  for state in anchor_states):
        anchor_status = "NOT_EVALUATED"
        anchor_statement = (
            "no arm has finalized a selected checkpoint yet, so the "
            "anchor-fallback gate has nothing to compare")
    elif "PARTIAL_ANCHOR_FALLBACK" in anchor_states:
        anchor_status = "PASS_WITH_FINDINGS"
        anchor_statement = (
            "some candidates selected the frozen anchor and were typed as "
            "rejected, measured outcomes; no arm is wholly anchor-bound")
    else:
        anchor_status = "PASS"
        anchor_statement = (
            "every scored checkpoint's policy tensor differs from the "
            "frozen warm-start anchor: the arms scored what they trained")
    payload["gates"]["anchor_fallback"] = {
        "gate": "anchor_fallback",
        "enforced_by": ["selected_checkpoint_anchor_facts",
                        "assert_arm_not_anchor_fallback"],
        "identity": "policy_tensor_sha256",
        "status": anchor_status,
        "statement": anchor_statement,
        "arms": per_arm,
    }

    report = verdict.get("differentiation") or {}
    diff_status = {
        "CAMPAIGN_VERDICT": "PASS",
        "CAMPAIGN_UNANSWERABLE": "UNANSWERABLE",
        "REFUSED_ARMS_NOT_DIFFERENTIATED": "REFUSED",
    }.get(verdict["outcome"], "NOT_EVALUATED")
    payload["gates"]["arm_differentiation"] = {
        "gate": "arm_differentiation",
        "enforced_by": ["compare_arms",
                        "arm_differentiation.assert_arms_differentiated"],
        "status": diff_status,
        "require_informative": bool(require_informative),
        "metric_keys": list(L2_METRIC_KEYS),
        "statement": verdict.get("statement") or (
            "the arm-differentiation ladder was not reached: "
            + "; ".join(verdict.get("refusals") or ["unknown"])),
        "pairs": [{"arms": pair["arms"], "verdict": pair["verdict"],
                   "detail": pair["detail"],
                   "metric_tuple_identical": pair["metric_tuple_identical"]}
                  for pair in (report.get("pairs") or ())],
        "informative": report.get("informative"),
        "contrast_pairs": report.get("contrast_pairs"),
        "degenerate_identical": report.get("degenerate_identical"),
    }

    if verdict["outcome"] != "CAMPAIGN_VERDICT":
        payload["refusals"].extend(verdict.get("refusals") or [])
    payload["outcome"] = ("ACCEPTANCE_PASS" if not payload["refusals"]
                          else "ACCEPTANCE_REFUSED")
    payload["proved_arms_are_distinguishable"] = (
        payload["outcome"] == "ACCEPTANCE_PASS"
        and diff_status == "PASS")
    payload["statement"] = (
        "the L2 smoke PROVED its machinery can distinguish L2_N from "
        "L2_EN: the anchor-fallback gate passed on every arm and the "
        "arm-differentiation ladder separated the pair"
        if payload["proved_arms_are_distinguishable"] else
        "the L2 smoke CANNOT claim its machinery distinguishes its arms — "
        + "; ".join(payload["refusals"] or ["no reason recorded"]))
    return payload, EXIT_CLASS[payload["outcome"]]


def smoke_dispatch_commands(contract: dict, bindings: dict | None = None,
                            *, mode: str = "smoke") -> dict:
    """The exact per-host commands for the four-GPU dispatch. Emitting
    them authorizes nothing; each command still enforces its own host,
    GPU UUID and readiness gate."""
    bindings = bindings or load_bindings()
    exp_id = experiment_identity(contract, bindings, mode=mode)
    assignments = contract["workers"]["assignments"]
    order = list(contract["workers"]["arm_order"])
    python = sys.executable
    plan = {
        "schema": DISPATCH_SCHEMA,
        "outcome": "DISPATCH_PLAN",
        "mode": mode,
        "experiment_identity": exp_id,
        "contract_sha256": contract["_contract_sha256"],
        "output_root": str(output_root_for_mode(contract, mode)),
        "arm_order": order,
        "sequential": ("all four workers collaborate on ONE arm at a "
                       "time; start the next arm only after the previous "
                       "arm publishes arm_complete.json"),
        "preflight": (f"{python} {REPO}/tools/l2_curriculum_arms.py "
                      f"--preflight --contract {contract['_contract_path']}"),
        "arms": {},
    }
    for arm in order:
        chain = chain_identity(contract, bindings, arm=arm, mode=mode)
        commands = {}
        for worker_id in WORKERS:
            assignment = assignments[worker_id]
            commands[worker_id] = {
                "hostname": assignment["hostname"],
                "gpu_uuid": assignment["gpu_uuid"],
                "command": (
                    f"CUDA_VISIBLE_DEVICES={assignment['gpu_uuid']} "
                    f"{python} {REPO}/tools/l2_curriculum_arms.py "
                    f"--worker {worker_id} --arm {arm} --mode {mode} "
                    f"--contract {contract['_contract_path']}"),
            }
        plan["arms"][arm] = {
            "chain_identity": chain,
            "chain_dir": str(chain_dir(contract, exp_id, arm, mode, chain)),
            "workers": commands,
        }
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--worker", choices=list(WORKERS),
                       help="join this arm's ONE chain as this worker and "
                            "evaluate candidates until the arm completes")
    group.add_argument("--preflight", action="store_true",
                       help="no-training materialization proof: nested "
                            "roles, sealed-test absence, identical arm "
                            "budgets, frozen L1 recipe, distinct "
                            "per-mode identities")
    group.add_argument("--dispatch-plan", action="store_true",
                       help="print the exact per-host commands for the "
                            "four-GPU dispatch (authorizes nothing)")
    group.add_argument("--compare-arms", action="store_true",
                       help="the arm-comparison boundary: refuse unless "
                            "the two arms scored different policies and "
                            "their results actually separate (gates "
                            "11-13); an all-degenerate campaign is typed "
                            "CAMPAIGN_UNANSWERABLE, never a null")
    group.add_argument("--acceptance", action="store_true",
                       help="the smoke's acceptance verdict with the "
                            "anchor-fallback and arm-differentiation "
                            "gates made visible")
    group.add_argument("--champion", action="store_true",
                       help="resolve the arm's champion; refuses if any "
                            "easy-evaluated or pending-re-evaluation "
                            "entry would enter the leaderboard")
    parser.add_argument("--arm", choices=list(ARMS), default=None)
    parser.add_argument("--mode", choices=list(MODES), default="smoke")
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--chain-dir", type=Path, default=None)
    parser.add_argument("--no-gpu-check", action="store_true",
                        help="skip GPU-UUID binding and the readiness "
                             "launch gate (socket-free tests only)")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    contract = load_contract(args.contract)

    def _emit(payload: dict, exit_code: int) -> int:
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(args.output, payload)
        print(json.dumps(payload, default=str), flush=True)
        return exit_code

    if args.preflight:
        payload, exit_code = preflight(contract)
        return _emit(payload, exit_code)
    if args.dispatch_plan:
        plan = smoke_dispatch_commands(contract, mode=args.mode)
        return _emit(plan, EXIT_CLASS["DISPATCH_PLAN"])
    if args.compare_arms:
        payload, exit_code = compare_arms(contract, mode=args.mode)
        return _emit(payload, exit_code)
    if args.acceptance:
        payload, exit_code = smoke_acceptance(contract, mode=args.mode)
        return _emit(payload, exit_code)
    if args.champion:
        if args.chain_dir is None:
            return _emit({"outcome": "REFUSED_BAD_CONTRACT",
                          "reason": "--champion needs --chain-dir"},
                         EXIT_CLASS["REFUSED_BAD_CONTRACT"])
        try:
            payload = champion_of_chain(Path(args.chain_dir))
        except L2AnchorFallbackRefusal as exc:
            return _emit({"outcome": "REFUSED_ANCHOR_FALLBACK",
                          "reason": str(exc), "anchor_fallback": exc.facts},
                         EXIT_CLASS["REFUSED_ANCHOR_FALLBACK"])
        except _l2.L2CurriculumError as exc:
            return _emit({"outcome": "REFUSED_BAD_CONTRACT",
                          "reason": str(exc)},
                         EXIT_CLASS["REFUSED_BAD_CONTRACT"])
        payload["outcome"] = "ARM_COMPLETE"
        return _emit(payload, 0)

    if args.arm is None:
        return _emit({"outcome": "REFUSED_BAD_CONTRACT",
                      "reason": "--worker requires --arm"},
                     EXIT_CLASS["REFUSED_BAD_CONTRACT"])
    try:
        summary = run_worker(args.worker, contract=contract, arm=args.arm,
                             mode=args.mode,
                             enforce_gpu=not args.no_gpu_check,
                             max_candidates=args.max_candidates)
    except Exception as exc:                                # noqa: BLE001
        return _emit({"outcome": "WORKER_FAILED",
                      "worker_id": args.worker, "arm": args.arm,
                      "error": f"{type(exc).__name__}: {exc}"},
                     EXIT_CLASS["WORKER_FAILED"])
    return _emit(summary, EXIT_CLASS.get(summary["outcome"], 1))


if __name__ == "__main__":
    sys.exit(main())
