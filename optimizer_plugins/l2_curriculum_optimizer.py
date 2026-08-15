"""L2 search-curriculum semantics (doc 38 §6/§8) — an EXTENSION of the
existing shared-population DEAP optimizer, never a second optimizer.

``optimizer_plugins.default_optimizer.Plugin`` already owns the typed
DEAP genome, the deterministic shared-population bridge doin-node calls
(``create_shared_population`` / ``evaluate_candidate`` /
``reproduce_shared``) and the stage schedule. This module subclasses it
and adds ONLY what the L2 curriculum comparison needs:

* a **difficulty** attached to every stage, and the typed rule that easy
  and normal scores never share one comparable leaderboard;
* **easy-fitness invalidation** at a difficulty transition plus the
  mandatory re-evaluation of every carried elite under normal-realistic
  conditions before any champion, migration, archive or release path;
* **typed-gene-only inheritance** — a genome may carry parameters and
  nothing else; weights, topology, replay buffers, optimizer moments and
  model artifacts are refused (doc 38 §8 defers Lamarckian inheritance);
* **paired inner/outer generation patience** after a minimum generation
  floor, computed with the ONE reusable comparator
  ``pipeline_plugins._paired_generalization``; and
* **diversity evidence** (unique genomes, allele spread, rejected and
  ineligible share) on every generation so apparent convergence is never
  confused with collapse.

Everything here is pure and dependency-light except the ``Plugin``
subclass at the bottom, so a runner, an aggregator or a test can import
the semantics without importing a training framework.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any, Iterable, Mapping, Sequence

from pipeline_plugins import _paired_generalization as _paired

SCHEMA = "agent_multi.l2_curriculum_optimizer.v1"
GENERATION_RECORD_SCHEMA = "agent_multi.l2_generation_record.v1"
DIVERSITY_SCHEMA = "agent_multi.l2_population_diversity.v1"
INVALIDATION_SCHEMA = "agent_multi.l2_easy_fitness_invalidation.v1"

EASY = "easy_chronological_continuation"
NORMAL = "normal_realistic"
DIFFICULTIES = (EASY, NORMAL)

#: The ONLY difficulty whose scores may enter a leaderboard, champion,
#: migration, archive or release path (doc 38 §6.2).
DECISION_DIFFICULTY = NORMAL

STAGE_KINDS = ("search", "reevaluation")
PROMOTION_ACTIONS = ("champion", "migration", "archive", "release")

#: Requirement 4: L2 inherits TYPED GENES. Any of these keys anywhere in
#: a genome payload means the genome is claiming inherited SAC weights,
#: topology, replay or optimizer state — refused.
WEIGHT_INHERITANCE_KEYS = frozenset({
    "_model_b64", "_best_model_b64", "model_artifact_sha256",
    "model_artifact_bytes", "model_artifact_format", "model_path",
    "model_bytes", "policy", "policy_state_dict", "state_dict",
    "weights", "policy_tensor_sha256", "topology", "net_arch",
    "warm_start_model", "warm_start_model_sha256", "load_model",
    "replay_buffer", "replay_buffer_path", "optimizer_state",
    "optimizer_moments", "terminal_model_path", "best_model_path",
})


class L2CurriculumError(ValueError):
    """Every typed L2 curriculum refusal."""


# ---------------------------------------------------------------------------
# requirement 1 — L1 is FROZEN for every L2 arm
# ---------------------------------------------------------------------------

def assert_l1_recipe_frozen(recipe: Mapping[str, Any]) -> dict:
    """The frozen L1 recipe is normal-realistic phase-1 dynamics at LR
    3e-5 with the phase-2 normal LR also 3e-5 (doc 38 §15/§16 result).
    Anything else refuses BEFORE a candidate is materialized."""
    dynamics = recipe.get("phase1_dynamics")
    if dynamics != NORMAL:
        raise L2CurriculumError(
            f"frozen_l1_recipe.phase1_dynamics {dynamics!r} != {NORMAL!r} "
            "— L1 is FROZEN for every L2 arm (order §2 requirement 1)")
    for key in ("phase1_learning_rate", "phase2_learning_rate"):
        value = recipe.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise L2CurriculumError(
                f"frozen_l1_recipe.{key} must be a number")
        if not math.isclose(float(value), 3e-05, rel_tol=1e-12,
                            abs_tol=0.0):
            raise L2CurriculumError(
                f"frozen_l1_recipe.{key}={value!r} != 3e-05 — the frozen "
                "phase-1 LR region may not move during the L2 comparison")
    return {"phase1_dynamics": NORMAL,
            "phase1_learning_rate": 3e-05,
            "phase2_learning_rate": 3e-05}


def assert_no_l1_gene_in_l2_space(gene_names: Iterable[str],
                                  frozen_names: Iterable[str]) -> None:
    """No L1 curriculum/dynamics field may VARY as an L2 gene while the
    question is whether the L2 search curriculum works."""
    leaked = sorted(set(gene_names) & set(frozen_names))
    if leaked:
        raise L2CurriculumError(
            f"L2 gene space contains frozen L1 curriculum fields {leaked}"
            " — no L1 curriculum gene may vary during the L2 comparison"
            " (order §2 requirement 1)")


def assert_stage_difficulty_binding(config: Mapping[str, Any],
                                    stage: Mapping[str, Any],
                                    binding: Mapping[str, Any]) -> None:
    """The materialized candidate config must actually carry the stage's
    declared difficulty, and — under the default ``evidence_only_v1``
    mechanism — must still train on the frozen L1 phase-1 dynamics."""
    mechanism = binding.get("mechanism")
    allowed = tuple(binding.get("allowed_mechanisms") or ())
    if mechanism not in allowed:
        raise L2CurriculumError(
            f"unknown l2 stage difficulty mechanism {mechanism!r}")
    declared = stage.get("evaluation_difficulty")
    if declared not in DIFFICULTIES:
        raise L2CurriculumError(
            f"stage {stage.get('name')!r} declares unknown difficulty "
            f"{declared!r}")
    if config.get("l2_evaluation_difficulty") != declared:
        raise L2CurriculumError(
            "materialized config l2_evaluation_difficulty "
            f"{config.get('l2_evaluation_difficulty')!r} != the stage's "
            f"declared {declared!r}")
    frozen_training = binding.get("training_difficulty_frozen")
    if mechanism == "evidence_only_v1":
        if config.get("phase1_mode") != frozen_training:
            raise L2CurriculumError(
                "evidence_only_v1 requires the candidate to train on the "
                f"frozen L1 phase-1 dynamics {frozen_training!r}; config "
                f"carries {config.get('phase1_mode')!r}")
        return
    # training_and_evidence_v1 moves an L1 curriculum field and therefore
    # needs an explicit, recorded exception to the L1 freeze.
    if binding.get("l1_freeze_exception_approved") is not True:
        raise L2CurriculumError(
            "training_and_evidence_v1 moves the candidate's phase-1 "
            "dynamics, which is a FROZEN L1 curriculum field; it requires "
            "l1_freeze_exception_approved=true (order §2 requirement 1)")


# ---------------------------------------------------------------------------
# requirement 4 — typed genes only
# ---------------------------------------------------------------------------

def _walk(payload: Any, path: str = "") -> Iterable[tuple]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            here = f"{path}.{key}" if path else str(key)
            yield here, str(key), value
            yield from _walk(value, here)
    elif isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            here = f"{path}[{index}]"
            yield from _walk(value, here)


def assert_typed_genes_only(genome: Mapping[str, Any], *,
                            gene_names: Sequence[str]) -> dict:
    """A genome is a typed parameter vector and NOTHING else.

    Refuses: a missing or non-object ``parameters`` block, an unknown or
    missing gene, a non-numeric allele, and any weight/topology/replay/
    optimizer/artifact key anywhere in the payload (requirement 4).
    """
    if not isinstance(genome, Mapping):
        raise L2CurriculumError("genome must be an object")
    params = genome.get("parameters")
    if not isinstance(params, Mapping):
        raise L2CurriculumError(
            "genome is missing a typed 'parameters' object")
    expected = list(gene_names)
    unknown = sorted(set(params) - set(expected))
    if unknown:
        raise L2CurriculumError(
            f"genome carries genes outside the declared L2 space: {unknown}")
    missing = [name for name in expected if name not in params]
    if missing:
        raise L2CurriculumError(f"genome is missing genes: {missing}")
    for name, value in params.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise L2CurriculumError(
                f"gene {name!r} is not a finite typed number: {value!r}")
    for path, key, _value in _walk(genome):
        if key in WEIGHT_INHERITANCE_KEYS:
            raise L2CurriculumError(
                f"genome claims inherited model state at {path!r} — L2 "
                "inherits TYPED GENES only; SAC weights, topology, replay "
                "buffers and optimizer moments are never inherited "
                "(order §2 requirement 4)")
    return dict(params)


def assert_common_anchor_not_inherited(recipe: Mapping[str, Any],
                                       genome: Mapping[str, Any]) -> None:
    """The frozen common anchor belongs to the L1 recipe, identical for
    every candidate of both arms. A per-genome anchor is inheritance."""
    if recipe.get("candidate_initialization") not in (
            "fresh_build", "frozen_common_anchor"):
        raise L2CurriculumError(
            "frozen_l1_recipe.candidate_initialization must be "
            "'fresh_build' or 'frozen_common_anchor'")
    for path, key, _value in _walk(genome):
        if key in ("common_anchor", "anchor", "anchor_sha256"):
            raise L2CurriculumError(
                f"genome names its own anchor at {path!r} — the anchor is "
                "a frozen recipe fact, never a per-genome inheritance")


# ---------------------------------------------------------------------------
# requirement 2 — easy fitness is invalid at the transition
# ---------------------------------------------------------------------------

def entry_difficulty(entry: Mapping[str, Any]) -> str | None:
    return entry.get("evaluation_difficulty")


def invalidate_easy_fitness(population: Sequence[Mapping[str, Any]], *,
                            from_stage: Mapping[str, Any],
                            to_stage: Mapping[str, Any]) -> dict:
    """At a difficulty transition the easy fitness of EVERY survivor
    becomes invalid. The old value is archived under a deliberately
    NON-comparable key so it can be audited but can never be sorted next
    to a normal-realistic score, and each survivor is flagged
    ``requires_normal_reevaluation`` until it is re-scored.
    """
    old = from_stage.get("evaluation_difficulty")
    new = to_stage.get("evaluation_difficulty")
    if old not in DIFFICULTIES or new not in DIFFICULTIES:
        raise L2CurriculumError(
            "both stages must declare a known evaluation_difficulty")
    if old == new:
        raise L2CurriculumError(
            f"invalidate_easy_fitness called without a difficulty change "
            f"({old!r} -> {new!r})")
    if new != DECISION_DIFFICULTY:
        raise L2CurriculumError(
            "a difficulty transition may only move TOWARDS "
            f"{DECISION_DIFFICULTY!r}; {old!r} -> {new!r} is refused")
    survivors = []
    for entry in population:
        carried = dict(entry)
        archived = {
            "invalidated_difficulty": old,
            "invalidated_fitness": carried.get("fitness"),
            "invalidated_paired": carried.get("paired"),
            "invalidated_at_stage": from_stage.get("name"),
        }
        for key in ("fitness", "paired", "paired_score", "rank",
                    "leaderboard_score"):
            carried.pop(key, None)
        carried["fitness"] = None
        carried["fitness_valid"] = False
        carried["evaluation_difficulty"] = None
        carried["requires_normal_reevaluation"] = True
        carried["promotion_eligible"] = False
        carried["easy_fitness_archive"] = archived
        survivors.append(carried)
    return {
        "schema": INVALIDATION_SCHEMA,
        "from_stage": from_stage.get("name"),
        "to_stage": to_stage.get("name"),
        "from_difficulty": old,
        "to_difficulty": new,
        "survivors_invalidated": len(survivors),
        "reevaluation_required": len(survivors),
        "population": survivors,
    }


def assert_all_elites_reevaluated(
        population: Sequence[Mapping[str, Any]]) -> None:
    """No champion, migration, archive or release may proceed while a
    single carried elite still holds an invalidated easy fitness."""
    pending = [entry.get("candidate_id", index)
               for index, entry in enumerate(population)
               if entry.get("requires_normal_reevaluation")]
    if pending:
        raise L2CurriculumError(
            f"{len(pending)} carried elite(s) still require a "
            f"normal-realistic re-evaluation: {pending} — easy fitness is "
            "invalid at the transition (order §2 requirement 2)")


def assert_normal_leaderboard(
        entries: Sequence[Mapping[str, Any]]) -> None:
    """Hard refusal: an easy-evaluated entry can never appear on a
    comparable normal leaderboard or in a chain objective."""
    offenders = []
    for index, entry in enumerate(entries):
        difficulty = entry_difficulty(entry)
        if difficulty != DECISION_DIFFICULTY:
            offenders.append(
                f"{entry.get('candidate_id', index)}:{difficulty!r}")
        if entry.get("requires_normal_reevaluation"):
            offenders.append(
                f"{entry.get('candidate_id', index)}:pending_reevaluation")
    if offenders:
        raise L2CurriculumError(
            "easy and normal scores may never share one comparable "
            f"leaderboard; offending entries: {sorted(set(offenders))}")


def normal_leaderboard(entries: Sequence[Mapping[str, Any]]) -> list:
    """The ONLY ordering function. It refuses first, then sorts."""
    assert_normal_leaderboard(entries)
    scored = [entry for entry in entries
              if entry.get("fitness") is not None]
    return sorted(scored, key=lambda item: (float(item["fitness"]),
                                            str(item.get("candidate_id"))),
                  reverse=True)


def assert_promotion_eligible(entry: Mapping[str, Any], *,
                              action: str) -> None:
    if action not in PROMOTION_ACTIONS:
        raise L2CurriculumError(f"unknown promotion action {action!r}")
    difficulty = entry_difficulty(entry)
    if difficulty != DECISION_DIFFICULTY:
        raise L2CurriculumError(
            f"{action}: candidate {entry.get('candidate_id')!r} was "
            f"evaluated under {difficulty!r}; only "
            f"{DECISION_DIFFICULTY!r} candidates may become champion, "
            "migrate, archive or seed a release (doc 38 §6.2)")
    if entry.get("requires_normal_reevaluation"):
        raise L2CurriculumError(
            f"{action}: candidate {entry.get('candidate_id')!r} still "
            "carries an invalidated easy fitness")
    if entry.get("candidate_rejected"):
        raise L2CurriculumError(
            f"{action}: candidate {entry.get('candidate_id')!r} is a "
            "rejected candidate")


# ---------------------------------------------------------------------------
# requirement 5 — paired inner/outer patience after a generation floor
# ---------------------------------------------------------------------------

def paired_l2_fitness(inner: Mapping[str, Any], outer: Mapping[str, Any],
                      *, beta: float, min_trades: int = 1,
                      candidate_id: str = "") -> dict:
    """The L2 objective: doc 38 §4.2 paired comparator over the
    inner_validation / outer_validation pair. One implementation, the
    same the L1 program uses; L2 only changes which pair it receives."""
    return _paired.paired_generalization_weekly_v1(
        inner, outer, beta=beta, min_trades_a=min_trades,
        min_trades_b=min_trades, label_a="inner_validation",
        label_b="outer_validation", candidate_id=candidate_id)


def l2_stopping_state(contract_stopping: Mapping[str, Any], *,
                      split_identity: str) -> _paired.PairedStoppingState:
    """Reused stopping bookkeeping; the FLOOR is the minimum generation
    count, so patience cannot fire before it (requirement 5)."""
    return _paired.PairedStoppingState(
        patience=int(contract_stopping["patience"]),
        floor=int(contract_stopping["minimum_generations"]),
        min_delta=float(contract_stopping["min_delta"]),
        split_identity=split_identity)


def l2_generation_stop_decision(state: _paired.PairedStoppingState,
                                best_paired: Mapping[str, Any],
                                generation: int) -> dict:
    """One generation's patience step. A generation whose best candidate
    has no ELIGIBLE paired inner/outer result never consumes patience,
    and no stop is possible before the minimum generation floor."""
    outcome = state.update(best_paired, int(generation))
    stop = bool(outcome["stop"]) and int(generation) >= state.floor
    return {
        "generation": int(generation),
        "improved": bool(outcome["improved"]),
        "stop": stop,
        "reason": outcome["reason"],
        "minimum_generations": state.floor,
        "patience": state.patience,
        "waited": state.waited,
        "best_paired_score": state.best_score,
        "floor_reached": int(generation) >= state.floor,
        "evidence_pair": ["inner_validation", "outer_validation"],
    }


# ---------------------------------------------------------------------------
# requirement 6 — diversity, unique genomes, rejected/ineligible share
# ---------------------------------------------------------------------------

def genome_fingerprint(parameters: Mapping[str, Any], *,
                       gene_names: Sequence[str], decimals: int = 9) -> str:
    payload = [round(float(parameters[name]), decimals)
               for name in gene_names]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode()).hexdigest()[:16]


def population_diversity(entries: Sequence[Mapping[str, Any]], *,
                         gene_names: Sequence[str],
                         gene_bounds: Mapping[str, tuple] | None = None,
                         decimals: int = 9) -> dict:
    """Typed diversity evidence for ONE generation (requirement 6).

    Apparent convergence and collapse look identical in a fitness curve;
    they do not look identical here.
    """
    names = list(gene_names)
    vectors = []
    fingerprints = []
    rejected = 0
    ineligible = 0
    fitnesses = []
    for entry in entries:
        params = entry.get("parameters") or {}
        if all(name in params for name in names):
            vectors.append([float(params[name]) for name in names])
            fingerprints.append(genome_fingerprint(
                params, gene_names=names, decimals=decimals))
        if entry.get("candidate_rejected"):
            rejected += 1
        paired = entry.get("paired") or {}
        if paired and not paired.get("eligible", False):
            ineligible += 1
        value = entry.get("fitness")
        if isinstance(value, (int, float)) and not isinstance(value, bool) \
                and math.isfinite(float(value)):
            fitnesses.append(float(value))

    total = len(entries)
    unique = len(set(fingerprints))
    ranges = {}
    for name in names:
        if gene_bounds and name in gene_bounds:
            low = float(gene_bounds[name][0])
            high = float(gene_bounds[name][1])
        else:
            column = [vector[names.index(name)] for vector in vectors]
            low, high = (min(column), max(column)) if column else (0.0, 0.0)
        ranges[name] = max(high - low, 1e-12)

    distances = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            distances.append(sum(
                abs(vectors[i][k] - vectors[j][k]) / ranges[names[k]]
                for k in range(len(names))) / max(len(names), 1))
    mean_distance = (sum(distances) / len(distances)) if distances else 0.0

    if len(fitnesses) > 1:
        mean_fitness = sum(fitnesses) / len(fitnesses)
        dispersion = math.sqrt(
            sum((value - mean_fitness) ** 2 for value in fitnesses)
            / (len(fitnesses) - 1))
    else:
        dispersion = 0.0

    return {
        "schema": DIVERSITY_SCHEMA,
        "evaluated_count": total,
        "unique_genome_count": unique,
        "unique_genome_fraction": (unique / total) if total else 0.0,
        "mean_pairwise_allele_distance": mean_distance,
        "per_gene_distinct_values": {
            name: len({round(vector[names.index(name)], decimals)
                       for vector in vectors})
            for name in names},
        "rejected_count": rejected,
        "rejected_share": (rejected / total) if total else 0.0,
        "ineligible_count": ineligible,
        "ineligible_share": (ineligible / total) if total else 0.0,
        "fitness_dispersion": dispersion,
        "genome_fingerprints": fingerprints,
    }


def assert_diversity_logged(record: Mapping[str, Any],
                            required_fields: Sequence[str]) -> None:
    diversity = record.get("diversity") or {}
    missing = [field for field in required_fields
               if field not in diversity]
    if missing:
        raise L2CurriculumError(
            f"generation record omits diversity fields {missing} — "
            "apparent convergence must never be publishable without the "
            "collapse evidence (order §2 requirement 6)")


# ---------------------------------------------------------------------------
# stage schedules and budget ledgers
# ---------------------------------------------------------------------------

def expand_stage_schedule(stages: Sequence[Mapping[str, Any]],
                          *, gene_names: Sequence[str] = (),
                          patience: int = 3) -> list:
    """Cumulative generation windows, doc 38 §8 shape plus difficulty.

    The expanded entries carry BOTH the L2 vocabulary (``active_genes``)
    and the ``active_params``/``frozen_params``/``patience`` keys the
    inherited ``default_optimizer`` stage machinery already consumes, so
    the base class needs no change.
    """
    cursor = 0
    expanded = []
    all_names = list(gene_names)
    for index, stage in enumerate(stages):
        kind = stage.get("stage_kind")
        if kind not in STAGE_KINDS:
            raise L2CurriculumError(
                f"stage {stage.get('name')!r} has unknown stage_kind "
                f"{kind!r}")
        difficulty = stage.get("evaluation_difficulty")
        if difficulty not in DIFFICULTIES:
            raise L2CurriculumError(
                f"stage {stage.get('name')!r} declares unknown difficulty "
                f"{difficulty!r}")
        generations = int(stage.get("generations", 0))
        if generations < 1:
            raise L2CurriculumError(
                f"stage {stage.get('name')!r} must declare >= 1 generation")
        if kind == "reevaluation":
            if generations != 1:
                raise L2CurriculumError(
                    "a reevaluation stage spends exactly one "
                    "population-equivalent, so generations must be 1")
            if difficulty != DECISION_DIFFICULTY:
                raise L2CurriculumError(
                    "a reevaluation stage must re-score under "
                    f"{DECISION_DIFFICULTY!r}")
            if stage.get("reevaluates_all_survivors") is not True or \
                    stage.get("invalidates_previous_fitness") is not True:
                raise L2CurriculumError(
                    "a reevaluation stage must declare "
                    "invalidates_previous_fitness and "
                    "reevaluates_all_survivors")
        active = list(stage.get("active_genes") or [])
        expanded.append({
            "name": str(stage["name"]),
            "stage_idx": index,
            "stage_kind": kind,
            "evaluation_difficulty": difficulty,
            "active_genes": active,
            # aliases consumed verbatim by default_optimizer's stage
            # machinery — one schedule, two vocabularies, no fork.
            "active_params": active,
            "frozen_params": [name for name in all_names
                              if name not in active],
            "patience": int(stage.get("patience", patience)),
            "categorical_change_allowed": bool(
                stage.get("categorical_change_allowed", True)),
            "generations": generations,
            "start_gen": cursor,
            "end_gen": cursor + generations,
        })
        cursor += generations
    if not expanded:
        raise L2CurriculumError("an arm needs at least one stage")
    return expanded


def stage_for_generation(schedule: Sequence[Mapping[str, Any]],
                         generation: int) -> dict:
    for stage in schedule:
        if int(stage["start_gen"]) <= int(generation) < int(stage["end_gen"]):
            return dict(stage)
    raise L2CurriculumError(
        f"generation {generation} is outside the declared stage schedule")


def arm_budget_ledger(schedule: Sequence[Mapping[str, Any]], *,
                      population_size: int) -> dict:
    """Every candidate evaluation is charged to ONE shared budget —
    including the mandatory normal re-evaluations, so an easy triage
    stage can never buy L2_EN extra search (requirement 3)."""
    search = sum(stage["generations"] for stage in schedule
                 if stage["stage_kind"] == "search")
    reeval = sum(stage["generations"] for stage in schedule
                 if stage["stage_kind"] == "reevaluation")
    easy = sum(stage["generations"] for stage in schedule
               if stage["evaluation_difficulty"] == EASY)
    return {
        "population_size": int(population_size),
        "search_generations": search,
        "reevaluation_generations": reeval,
        "easy_generations": easy,
        "normal_generations": search + reeval - easy,
        "search_evaluations": search * int(population_size),
        "reevaluation_evaluations": reeval * int(population_size),
        "easy_evaluations": easy * int(population_size),
        "total_candidate_evaluations":
            (search + reeval) * int(population_size),
    }


def assert_arms_identical_budget(ledgers: Mapping[str, Mapping[str, Any]],
                                 *, declared_total: int,
                                 population_size: int) -> dict:
    """Requirement 3: identical TOTAL candidate-evaluation budget and
    population size across arms, and equal to the declared total."""
    arms = sorted(ledgers)
    if len(arms) != 2:
        raise L2CurriculumError(
            f"the L2 comparison needs exactly two arms, got {arms}")
    totals = {arm: int(ledgers[arm]["total_candidate_evaluations"])
              for arm in arms}
    if len(set(totals.values())) != 1:
        raise L2CurriculumError(
            f"arms do not share one candidate-evaluation budget: {totals}"
            " (order §2 requirement 3)")
    if set(totals.values()) != {int(declared_total)}:
        raise L2CurriculumError(
            f"arm budgets {totals} != the declared shared total "
            f"{declared_total}")
    sizes = {arm: int(ledgers[arm]["population_size"]) for arm in arms}
    if set(sizes.values()) != {int(population_size)}:
        raise L2CurriculumError(
            f"arms do not share one population size: {sizes}")
    return {"arms": arms, "total_candidate_evaluations": totals[arms[0]],
            "population_size": int(population_size)}


# ---------------------------------------------------------------------------
# the optimizer extension itself
# ---------------------------------------------------------------------------

from optimizer_plugins import default_optimizer as _default
from optimizer_plugins.default_optimizer import Plugin as _DefaultOptimizer


class Plugin(_DefaultOptimizer):
    """Shared-population bridge with L2 curriculum semantics.

    Overrides exactly three of the four bridge methods doin-node calls
    (``create_shared_population``, ``evaluate_candidate``,
    ``reproduce_shared``); every genetic operator, encode/decode and
    tournament stays the proven implementation in ``default_optimizer``.
    """

    def bind_l2_contract(self, *, gene_space, stage_schedule,
                         frozen_gene_names, population_seed,
                         elitism: int = 1, ga_cxpb: float = 0.5,
                         ga_mutpb: float = 0.2,
                         patience: int = 3) -> None:
        """Bind the L2 gene space and stage schedule.

        This is the L2 analogue of ``setup_shared_mode``: the L2 runner
        owns candidate execution (it must materialize a frozen-L1 config
        and run the two nested role replays), so the shared context is
        bound WITHOUT env/agent/pipeline plugins and
        ``evaluate_candidate`` is only reachable when a caller supplies
        them explicitly through ``setup_shared_mode``.
        """
        names = [str(item[0]) for item in gene_space]
        assert_no_l1_gene_in_l2_space(names, frozen_gene_names)
        # the inherited DEAP creator types, registered exactly once
        _default._ensure_creator()
        self._l2_gene_space = [
            (str(n), float(lo), float(hi), str(kind))
            for n, lo, hi, kind in gene_space]
        self._l2_gene_names = names
        self._l2_stage_schedule = expand_stage_schedule(
            stage_schedule, gene_names=names, patience=int(patience))
        self._l2_population_seed = int(population_seed)
        self._l2_elitism = int(elitism)
        self.params["shared_elitism"] = int(elitism)
        reproduction_config = {
            "higher_is_better": True,
            "shared_elitism": int(elitism),
            "ga_cxpb": float(ga_cxpb),
            "ga_mutpb": float(ga_mutpb),
            "optimization_patience": int(patience),
        }
        if self._shared_context is None:
            self._shared_context = {
                "env_plugin": None, "agent_plugin": None,
                "pipeline_plugin": None,
                "config": reproduction_config,
                "schema": self._l2_gene_space,
            }
        else:
            self._shared_context["schema"] = self._l2_gene_space
            self._shared_context["config"].update(reproduction_config)

    # -- bridge ---------------------------------------------------------
    def create_shared_population(self, population_size: int, *,
                                 seed: int | None = None) -> dict:
        schema = list(self._l2_gene_space)
        rng = random.Random(
            int(seed if seed is not None else self._l2_population_seed))
        stages = self._l2_stage_schedule
        baseline = {name: ((low + high) / 2.0 if kind == "float"
                           else int(round((low + high) / 2.0)))
                    for name, low, high, kind in schema}
        active = set(stages[0]["active_genes"] or self._l2_gene_names)
        population = self._make_shared_stage_population(
            size=int(population_size), baseline=baseline, schema=schema,
            active_params=active, rng=rng)
        for genome in population:
            assert_typed_genes_only(genome, gene_names=self._l2_gene_names)
        payload = [list(item) for item in schema]
        schema_hash = hashlib.sha256(json.dumps(
            payload, separators=(",", ":"),
            sort_keys=True).encode()).hexdigest()
        return {
            "population": population,
            "innovation_tracker": {
                "schema_version": SCHEMA,
                "schema_hash": schema_hash,
                "parameter_names": self._l2_gene_names,
            },
            "stage_schedule": stages,
            "param_defaults": baseline,
            "config_snapshot": {
                "population_size": int(population_size),
                "parameter_schema": payload,
                "schema_hash": schema_hash,
                "population_seed": self._l2_population_seed,
            },
        }

    def evaluate_candidate(self, genome_serialized: dict,
                           generation: int) -> dict:
        """Refuse a non-typed genome BEFORE spending any GPU time."""
        assert_typed_genes_only(genome_serialized,
                                gene_names=self._l2_gene_names)
        return super().evaluate_candidate(genome_serialized, generation)

    def reproduce_shared(self, evaluated_population, generation, seed,
                         innovation_tracker_data, stage_schedule,
                         param_defaults, current_stage_idx: int = 0,
                         no_improve_count: int = 0) -> dict:
        """Deterministic next generation, with a difficulty transition
        handled BEFORE any genetic operator sees a score.

        A stage boundary that changes the evaluation difficulty
        invalidates the easy fitness of every survivor and marks the
        whole carried population for mandatory re-evaluation; the
        carried genomes themselves are untouched typed genes.
        """
        schedule = list(stage_schedule)
        current = schedule[int(current_stage_idx)]
        next_index = min(int(current_stage_idx) + 1, len(schedule) - 1)
        nxt = schedule[next_index]
        crosses = (int(generation) + 1 >= int(current["end_gen"])
                   and next_index != int(current_stage_idx))
        difficulty_changes = (
            crosses and current["evaluation_difficulty"]
            != nxt["evaluation_difficulty"])

        if difficulty_changes:
            invalidation = invalidate_easy_fitness(
                evaluated_population, from_stage=current, to_stage=nxt)
            carried = [{"parameters": dict(entry["parameters"])}
                       for entry in invalidation["population"]]
            for genome in carried:
                assert_typed_genes_only(
                    genome, gene_names=self._l2_gene_names)
            return {
                "population": carried,
                "generation": int(generation) + 1,
                "best_fitness": None,
                "stage_idx": next_index,
                "no_improve_count": 0,
                "stage_advanced": True,
                "difficulty_transition": True,
                "easy_fitness_invalidated": True,
                "requires_normal_reevaluation": len(carried),
                "invalidation": {
                    key: invalidation[key] for key in
                    ("schema", "from_stage", "to_stage",
                     "from_difficulty", "to_difficulty",
                     "survivors_invalidated", "reevaluation_required")},
                "patience": None,
            }

        result = super().reproduce_shared(
            evaluated_population, generation, seed,
            innovation_tracker_data, schedule, param_defaults,
            current_stage_idx=int(current_stage_idx),
            no_improve_count=int(no_improve_count))
        result.setdefault("difficulty_transition", False)
        result.setdefault("easy_fitness_invalidated", False)
        for genome in result["population"]:
            assert_typed_genes_only(genome, gene_names=self._l2_gene_names)
        return result


L2CurriculumOptimizer = Plugin
