"""L2 curriculum arms launch gate (doc 38 §6/§8; order 2026-08-15 §2).

Socket-free, training-free proofs of the ten executable requirements:

 1  L1 stays FROZEN (normal_realistic, phase-1 LR 3e-5) in every
    generation of both arms, and no L1 curriculum field can be a gene;
 2  easy fitness is INVALIDATED at the stage transition, every survivor
    is marked for a mandatory normal-realistic re-evaluation, and an
    easy score can never enter a normal leaderboard, champion,
    migration, archive or release path (hard refusals);
 3  both arms carry an identical total candidate-evaluation budget,
    population size, population seed, initial genome seeds, frozen L1
    recipe, data roles and normal-realistic decision evidence;
 4  L2 inherits TYPED GENES only — weights, topology, replay buffers,
    optimizer moments and per-genome anchors are refused;
 5  L2 patience uses the paired inner/outer comparator and cannot fire
    before the minimum generation floor;
 6  every generation record carries the diversity/collapse evidence;
 7  the data roles are exactly the nested contract's (11509/2190/2190/
    2196 scored rows, 256 forced-hold context rows, sealed 2025 never
    materialized);
 8  a node with different source revisions, plan/domain hashes, dataset
    hash, population seed, genesis or finalized ancestry refuses before
    evaluating;
 9  a second, parallel chain for one arm is a typed refusal, and arms
    dispatch sequentially;
10  the four-GPU smoke plan binds each worker to its contract host+UUID.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from optimizer_plugins import l2_curriculum_optimizer as _l2  # noqa: E402
from tools import l2_curriculum_arms as l2r  # noqa: E402

CLEAN_SOURCES = {
    "agent-multi": {"repo_root": "/repo/agent-multi", "commit": "1" * 40,
                    "dirty": False, "dirty_entries": [],
                    "dirty_untracked_digest": None},
    "gym-fx": {"repo_root": "/repo/gym-fx", "commit": "2" * 40,
               "dirty": False, "dirty_entries": [],
               "dirty_untracked_digest": None},
}
MOVED_SOURCES = copy.deepcopy(CLEAN_SOURCES)
MOVED_SOURCES["agent-multi"]["commit"] = "9" * 40


@pytest.fixture(scope="module")
def bindings() -> dict:
    return l2r.load_bindings()


@pytest.fixture(scope="module")
def contract() -> dict:
    return l2r.load_contract()


@pytest.fixture(autouse=True)
def pinned_sources(monkeypatch):
    monkeypatch.setattr(l2r.ladder, "source_identities",
                        lambda: copy.deepcopy(CLEAN_SOURCES))


def _gene_names(contract: dict) -> list:
    return [str(item[0]) for item in contract["shared"]["gene_space"]]


def _genome(contract: dict, **overrides) -> dict:
    params = {}
    for name, low, high, kind in contract["shared"]["gene_space"]:
        mid = (float(low) + float(high)) / 2.0
        params[name] = int(round(mid)) if kind == "int" else mid
    params.update(overrides)
    return {"parameters": params}


# ---------------------------------------------------------------------------
# socket-free fakes
# ---------------------------------------------------------------------------

def _fake_nested_roles_fn(contract: dict):
    spec = contract["nested_split_contract"]
    pins = spec["role_facts"]

    def fn(contract_arg, bindings_arg, out_dir):
        split_dir = Path(out_dir) / "nested_splits"
        roles = {}
        for role, pin in pins.items():
            entry = dict(pin)
            if pin["status"] == "MATERIALIZED":
                entry["csv"] = str(split_dir / f"{role}.csv")
            roles[role] = entry
        return {
            "binding": {
                "path": str(REPO / spec["path"]),
                "contract_relative_path": spec["path"],
                "sha256": spec["sha256"],
                "mode": "l2",
                "context_bars": spec["context_bars"],
                "source_csv": "fake.csv",
                "source_sha256": "s" * 64,
                "role_facts_pinned": pins,
            },
            "split_dir": str(split_dir),
            "manifest_path": str(split_dir / "nested_split_manifest.json"),
            "manifest_sha256": "d" * 64,
            "roles": roles,
        }
    return fn


class FakePipeline:
    """Stand-in for the solvency-curriculum pipeline. Writes real bytes
    so terminal/best custody is hashed for real."""

    calls: list = []
    inactive_fingerprints: set = set()

    def __init__(self, config: dict):
        self.config = config

    def run_pipeline(self, *, config, env_plugin, agent_plugin, mode):
        assert mode == "train"
        type(self).calls.append(dict(config))
        out = Path(config["save_model"]).parent
        out.mkdir(parents=True, exist_ok=True)
        terminal = out / "terminal.zip"
        terminal.write_bytes(json.dumps(
            {k: config.get(k) for k in
             ("batch_size", "gamma", "tau", "train_freq",
              "gradient_steps", "l2_arm", "l2_stage",
              "l2_evaluation_difficulty")}, sort_keys=True).encode())
        fingerprint = f"{config['batch_size']}"
        result = {"terminal_model_path": str(terminal)}
        if fingerprint in type(self).inactive_fingerprints:
            result["best_model_path"] = None
            result["activity_stopped_without_eligible_checkpoint"] = True
            return result
        best = out / "best.zip"
        best.write_bytes(terminal.read_bytes() + b"-best")
        result["best_model_path"] = str(best)
        return result


def _fake_role_eval(*, config, agent, model_path, role, nested_roles,
                    seed, solvency_mode):
    """Deterministic per-role weekly evidence driven by the genome, so
    the leaderboard has a real ordering and easy evidence is visibly a
    different lens."""
    base = float(config["gamma"]) + float(config["tau"])
    # The acceptance fixture must realize the treatment it claims to test.
    # Distinct fake model bytes prove artifact identity, not that an arm
    # reached behavior, so expose a small deterministic EN response too.
    treatment = 0.01 if config.get("l2_arm") == "L2_EN" else 0.0
    lift = 0.5 if solvency_mode == _l2.EASY else 0.0
    offset = 0.0 if role == "inner_validation" else -0.05
    rap = round(base + lift + offset + treatment, 9)
    return {
        "role": role,
        "solvency_mode": solvency_mode,
        "evaluated_model_path": str(model_path),
        "scored_rows": nested_roles["roles"][role]["scored_rows"],
        "context_rows_forced_hold": 256,
        "trades_total": 7,
        "weekly_return_vector": [round(rap / 2.0 + step * 1e-4, 9)
                                 for step in range(6)],
        "metrics": {"mean_weekly_rap": rap,
                    "mean_weekly_return": rap / 2.0,
                    "max_drawdown_fraction": 0.05,
                    "total_return": rap * 10.0},
    }


def _fake_policy_tensor_sha(path) -> str:
    """Stand-in for ``arm_differentiation.policy_tensor_sha256``.

    The fake pipeline writes plain bytes rather than SB3 containers, so
    the tests inject a byte digest. What the gate proves is unchanged:
    the SELECTED artifact's identity is compared with the anchor's, and
    the runner never falls back to a container digest of its own.
    """
    return "tensor:" + l2r._sha_file(Path(path))


@pytest.fixture()
def runtime(contract, bindings, tmp_path, monkeypatch):
    """A runnable smoke contract: tmp output roots, a tmp hash-bound
    common anchor and the seed-101 host/GPU environment."""
    live = copy.deepcopy(contract)
    live["_contract_sha256"] = contract["_contract_sha256"]
    live["_contract_path"] = contract["_contract_path"]
    live["modes"]["smoke"]["output_root"] = str(tmp_path / "smoke")
    live["modes"]["decision"]["output_root"] = str(tmp_path / "decision")
    anchor = tmp_path / "anchor_seed101.zip"
    anchor.write_bytes(b"anchor-bytes-101")
    live["frozen_l1_recipe"]["common_anchor"] = {
        "path": str(anchor), "sha256": l2r._sha_file(anchor)}
    assigned = live["workers"]["assignments"]["w1"]["gpu_uuid"]
    # The four-worker collaboration proof runs in ONE process, so every
    # worker is pinned to this host while keeping four distinct GPU
    # bindings. That the SHIPPED contract spreads the four workers over
    # the real hosts/UUIDs is proven separately in TestSmokeDispatch.
    for worker in l2r.WORKERS:
        live["workers"]["assignments"][worker]["hostname"] = "omega"
    uuids = [live["workers"]["assignments"][w]["gpu_uuid"]
             for w in l2r.WORKERS]
    monkeypatch.setattr(l2r.socket, "gethostname", lambda: "omega")
    monkeypatch.setattr(l2r, "visible_gpu_uuids", lambda: list(uuids))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", assigned)
    FakePipeline.calls = []
    FakePipeline.inactive_fingerprints = set()
    return SimpleNamespace(
        contract=live, bindings=bindings, tmp_path=tmp_path,
        assigned=assigned,
        kwargs=dict(bindings=bindings, enforce_gpu=False,
                    pipeline_factory=FakePipeline,
                    agent_loader=lambda name: SimpleNamespace(name=name),
                    nested_roles_fn=_fake_nested_roles_fn(live),
                    role_eval_fn=_fake_role_eval,
                    policy_tensor_sha_fn=_fake_policy_tensor_sha))


def _run_arm(runtime, arm: str, workers=("w1", "w2", "w3", "w4"),
             rounds: int = 24, max_candidates: int | None = 1,
             kwargs: dict | None = None) -> dict:
    """Four workers collaborating on ONE chain, driven round-robin —
    one claimed candidate per call, which is how four concurrent hosts
    actually interleave against the flock-backed per-candidate claims."""
    summary = None
    for _ in range(rounds):
        for worker in workers:
            summary = l2r.run_worker(worker, contract=runtime.contract,
                                     arm=arm, mode="smoke",
                                     max_candidates=max_candidates,
                                     **(kwargs or runtime.kwargs))
            if summary.get("outcome") == "ARM_COMPLETE":
                return summary
    return summary


def _chain_path(runtime, arm: str) -> Path:
    exp = l2r.experiment_identity(runtime.contract, runtime.bindings,
                                 mode="smoke")
    chain = l2r.chain_identity(runtime.contract, runtime.bindings,
                               arm=arm, mode="smoke")
    return l2r.chain_dir(runtime.contract, exp, arm, "smoke", chain)


# ---------------------------------------------------------------------------
# requirement 1 — L1 is FROZEN for every L2 arm
# ---------------------------------------------------------------------------

class TestFrozenL1:
    def test_contract_pins_normal_realistic_and_3e5(self, contract):
        recipe = contract["frozen_l1_recipe"]
        assert recipe["phase1_dynamics"] == "normal_realistic"
        assert recipe["phase1_learning_rate"] == 3e-05
        assert recipe["phase2_learning_rate"] == 3e-05
        assert _l2.assert_l1_recipe_frozen(recipe)

    @pytest.mark.parametrize("field,value", [
        ("phase1_dynamics", "easy_chronological_continuation"),
        ("phase1_learning_rate", 1e-4),
        ("phase2_learning_rate", 1e-4),
    ])
    def test_moved_l1_recipe_refuses(self, contract, field, value):
        recipe = dict(contract["frozen_l1_recipe"])
        recipe[field] = value
        with pytest.raises(_l2.L2CurriculumError):
            _l2.assert_l1_recipe_frozen(recipe)

    def test_no_l1_curriculum_field_is_a_gene(self, contract):
        frozen = set(contract["frozen_l1_recipe"]["l1_frozen_gene_names"])
        assert not set(_gene_names(contract)) & frozen
        # the phase-2 learning rate is in the agent's own hparam schema,
        # so the guard must actually fire when someone re-adds it
        with pytest.raises(_l2.L2CurriculumError):
            _l2.assert_no_l1_gene_in_l2_space(
                _gene_names(contract) + ["learning_rate"], frozen)

    def test_every_materialized_candidate_keeps_the_frozen_recipe(
            self, runtime):
        seen = set()
        for arm in l2r.ARMS:
            for stage in l2r.arm_schedule(runtime.contract, arm, "smoke"):
                if stage["stage_kind"] != "search":
                    continue
                config = l2r.materialize_candidate_config(
                    runtime.contract, runtime.bindings, arm=arm,
                    stage=stage, genome=_genome(runtime.contract),
                    out_dir=runtime.tmp_path / arm / stage["name"],
                    mode="smoke")
                seen.add((config["phase1_mode"],
                          config["phase1_learning_rate"],
                          config["easy_learning_rate"],
                          config["learning_rate"]))
                assert config["observation_contract"] == \
                    runtime.contract["observation_contract"]
        assert seen == {("normal_realistic", 3e-05, 3e-05, 3e-05)}

    def test_easy_stage_binds_evidence_not_the_l1_phase(self, runtime):
        stage = next(stage for stage
                     in l2r.arm_schedule(runtime.contract, "L2_EN", "smoke")
                     if stage["evaluation_difficulty"] == _l2.EASY)
        config = l2r.materialize_candidate_config(
            runtime.contract, runtime.bindings, arm="L2_EN", stage=stage,
            genome=_genome(runtime.contract),
            out_dir=runtime.tmp_path / "easy", mode="smoke")
        assert config["l2_evaluation_difficulty"] == _l2.EASY
        assert config["phase1_mode"] == "normal_realistic"

    def test_training_and_evidence_mechanism_needs_an_exception(
            self, runtime):
        live = copy.deepcopy(runtime.contract)
        live["l2_stage_difficulty_binding"]["mechanism"] = \
            "training_and_evidence_v1"
        stage = next(stage for stage
                     in l2r.arm_schedule(live, "L2_EN", "smoke")
                     if stage["evaluation_difficulty"] == _l2.EASY)
        with pytest.raises(_l2.L2CurriculumError, match="FROZEN L1"):
            l2r.materialize_candidate_config(
                live, runtime.bindings, arm="L2_EN", stage=stage,
                genome=_genome(live),
                out_dir=runtime.tmp_path / "exception", mode="smoke")


# ---------------------------------------------------------------------------
# requirement 3 — identical budgets, seeds and population across arms
# ---------------------------------------------------------------------------

class TestArmsAreComparable:
    @pytest.mark.parametrize("mode", list(l2r.MODES))
    def test_identical_budget_population_and_seeds(self, contract, mode):
        summary = l2r.assert_arms_comparable(contract, mode=mode)
        totals = {arm: summary["ledgers"][arm]["total_candidate_evaluations"]
                  for arm in l2r.ARMS}
        assert len(set(totals.values())) == 1
        assert totals["L2_N"] == int(
            l2r.mode_setting(contract, mode, "total_candidate_evaluations"))
        sizes = {summary["ledgers"][arm]["population_size"]
                 for arm in l2r.ARMS}
        assert sizes == {int(l2r.mode_setting(contract, mode,
                                              "population_size"))}

    def test_reevaluation_is_charged_to_the_shared_budget(self, contract):
        ledger = l2r._l2.arm_budget_ledger(
            l2r.arm_schedule(contract, "L2_EN", "decision"),
            population_size=12)
        assert ledger["reevaluation_evaluations"] == 12
        assert (ledger["search_evaluations"]
                + ledger["reevaluation_evaluations"]
                == ledger["total_candidate_evaluations"])

    def test_unequal_budgets_refuse(self, contract):
        live = copy.deepcopy(contract)
        live["stage_schedules"]["L2_N"][0]["generations"] += 1
        with pytest.raises(ValueError, match="budget|candidate-evaluation"):
            l2r.assert_arms_comparable(live, mode="decision")

    def test_both_arms_start_from_the_same_population(self, contract):
        fingerprints = set()
        for arm in l2r.ARMS:
            plugin = l2r._build_optimizer(contract, arm, "decision")
            state = plugin.create_shared_population(
                12, seed=int(contract["shared"]["population_seed"]))
            fingerprints.add(l2r._canonical_sha(
                [genome["parameters"] for genome in state["population"]]))
        assert len(fingerprints) == 1

    def test_l2_n_may_not_declare_an_easy_stage(self, contract):
        live = copy.deepcopy(contract)
        live["stage_schedules"]["L2_N"][0]["evaluation_difficulty"] = \
            _l2.EASY
        with pytest.raises(ValueError, match="L2_N declares easy"):
            l2r.assert_arms_comparable(live, mode="decision")

    def test_l2_en_must_reevaluate_before_any_normal_search(self, contract):
        live = copy.deepcopy(contract)
        live["stage_schedules"]["L2_EN"] = [
            stage for stage in live["stage_schedules"]["L2_EN"]
            if stage["stage_kind"] != "reevaluation"]
        live["stage_schedules"]["L2_EN"][0]["generations"] = 3
        with pytest.raises(ValueError, match="reevaluation"):
            l2r.assert_arms_comparable(live, mode="decision")


# ---------------------------------------------------------------------------
# requirement 2 — easy fitness invalidation and the normal leaderboard
# ---------------------------------------------------------------------------

def _easy_entry(index: int) -> dict:
    return {"candidate_id": f"easy-{index}", "fitness": 1.0 + index,
            "evaluation_difficulty": _l2.EASY,
            "parameters": {"gamma": 0.95}}


def _normal_entry(index: int, fitness: float = 0.5) -> dict:
    return {"candidate_id": f"normal-{index}", "fitness": fitness,
            "evaluation_difficulty": _l2.NORMAL,
            "parameters": {"gamma": 0.95}}


class TestEasyFitnessInvalidation:
    def test_transition_invalidates_and_requires_reevaluation(self):
        result = _l2.invalidate_easy_fitness(
            [_easy_entry(i) for i in range(4)],
            from_stage={"name": "easy", "evaluation_difficulty": _l2.EASY},
            to_stage={"name": "reeval",
                      "evaluation_difficulty": _l2.NORMAL})
        assert result["survivors_invalidated"] == 4
        assert result["reevaluation_required"] == 4
        for entry in result["population"]:
            assert entry["fitness"] is None
            assert entry["fitness_valid"] is False
            assert entry["evaluation_difficulty"] is None
            assert entry["requires_normal_reevaluation"] is True
            assert entry["promotion_eligible"] is False
            # the old value survives ONLY under a non-comparable key
            assert entry["easy_fitness_archive"][
                "invalidated_difficulty"] == _l2.EASY

    def test_pending_reevaluation_blocks_every_promotion(self):
        pending = _l2.invalidate_easy_fitness(
            [_easy_entry(0)],
            from_stage={"name": "e", "evaluation_difficulty": _l2.EASY},
            to_stage={"name": "n", "evaluation_difficulty": _l2.NORMAL},
        )["population"]
        with pytest.raises(_l2.L2CurriculumError, match="re-evaluation"):
            _l2.assert_all_elites_reevaluated(pending)
        for action in _l2.PROMOTION_ACTIONS:
            with pytest.raises(_l2.L2CurriculumError):
                _l2.assert_promotion_eligible(pending[0], action=action)

    def test_easy_score_never_enters_a_normal_leaderboard(self):
        entries = [_normal_entry(0), _easy_entry(1)]
        with pytest.raises(_l2.L2CurriculumError, match="comparable"):
            _l2.assert_normal_leaderboard(entries)
        with pytest.raises(_l2.L2CurriculumError):
            _l2.normal_leaderboard(entries)
        # the easy entry had the HIGHER raw number; without the refusal
        # it would have won the board outright
        assert entries[1]["fitness"] > entries[0]["fitness"]

    def test_pure_normal_leaderboard_orders_and_promotes(self):
        board = _l2.normal_leaderboard(
            [_normal_entry(0, 0.1), _normal_entry(1, 0.9)])
        assert [entry["candidate_id"] for entry in board] == \
            ["normal-1", "normal-0"]
        _l2.assert_promotion_eligible(board[0], action="champion")

    @pytest.mark.parametrize("action", list(_l2.PROMOTION_ACTIONS))
    def test_easy_candidate_refuses_every_promotion_path(self, action):
        with pytest.raises(_l2.L2CurriculumError,
                           match="champion|migrate|archive|release"):
            _l2.assert_promotion_eligible(_easy_entry(0), action=action)

    def test_transition_may_only_move_towards_normal(self):
        with pytest.raises(_l2.L2CurriculumError, match="TOWARDS"):
            _l2.invalidate_easy_fitness(
                [_easy_entry(0)],
                from_stage={"name": "n",
                            "evaluation_difficulty": _l2.NORMAL},
                to_stage={"name": "e",
                          "evaluation_difficulty": _l2.EASY})

    def test_optimizer_invalidates_at_the_stage_boundary(self, contract):
        plugin = l2r._build_optimizer(contract, "L2_EN", "decision")
        schedule = plugin._l2_stage_schedule
        easy_last_gen = schedule[0]["end_gen"] - 1
        evaluated = [{"parameters": _genome(contract)["parameters"],
                      "fitness": 5.0,
                      "evaluation_difficulty": _l2.EASY}
                     for _ in range(4)]
        out = plugin.reproduce_shared(
            evaluated, easy_last_gen, 7,
            {"parameter_names": _gene_names(contract)}, schedule,
            _genome(contract)["parameters"], current_stage_idx=0,
            no_improve_count=0)
        assert out["difficulty_transition"] is True
        assert out["easy_fitness_invalidated"] is True
        assert out["requires_normal_reevaluation"] == 4
        assert out["best_fitness"] is None
        assert all("fitness" not in genome
                   for genome in out["population"])


# ---------------------------------------------------------------------------
# requirement 4 — typed genes only
# ---------------------------------------------------------------------------

class TestTypedGeneInheritance:
    def test_a_clean_typed_genome_passes(self, contract):
        params = _l2.assert_typed_genes_only(
            _genome(contract), gene_names=_gene_names(contract))
        assert set(params) == set(_gene_names(contract))

    @pytest.mark.parametrize("key", [
        "_model_b64", "policy_state_dict", "warm_start_model",
        "replay_buffer", "optimizer_state", "topology",
        "model_artifact_sha256", "best_model_path"])
    def test_inherited_model_state_refuses(self, contract, key):
        genome = _genome(contract)
        genome[key] = "inherited"
        with pytest.raises(_l2.L2CurriculumError, match="TYPED GENES"):
            _l2.assert_typed_genes_only(
                genome, gene_names=_gene_names(contract))

    def test_nested_inherited_state_refuses(self, contract):
        genome = _genome(contract)
        genome["lineage"] = {"parent": {"weights": [0.1, 0.2]}}
        with pytest.raises(_l2.L2CurriculumError, match="TYPED GENES"):
            _l2.assert_typed_genes_only(
                genome, gene_names=_gene_names(contract))

    def test_unknown_or_missing_gene_refuses(self, contract):
        extra = _genome(contract)
        extra["parameters"]["ent_coef"] = 0.2
        with pytest.raises(_l2.L2CurriculumError, match="outside"):
            _l2.assert_typed_genes_only(
                extra, gene_names=_gene_names(contract))
        short = _genome(contract)
        short["parameters"].pop("gamma")
        with pytest.raises(_l2.L2CurriculumError, match="missing genes"):
            _l2.assert_typed_genes_only(
                short, gene_names=_gene_names(contract))

    def test_per_genome_anchor_refuses(self, contract):
        genome = _genome(contract)
        genome["anchor"] = {"path": "/x.zip"}
        with pytest.raises(_l2.L2CurriculumError, match="anchor"):
            _l2.assert_common_anchor_not_inherited(
                contract["frozen_l1_recipe"], genome)

    def test_materialization_refuses_an_inheriting_genome(self, runtime):
        stage = l2r.arm_schedule(runtime.contract, "L2_N", "smoke")[0]
        genome = _genome(runtime.contract)
        genome["warm_start_model"] = "/tmp/champion.zip"
        with pytest.raises(_l2.L2CurriculumError):
            l2r.materialize_candidate_config(
                runtime.contract, runtime.bindings, arm="L2_N",
                stage=stage, genome=genome,
                out_dir=runtime.tmp_path / "inherit", mode="smoke")

    def test_the_common_anchor_is_identical_for_every_candidate(
            self, runtime):
        anchors = set()
        for arm in l2r.ARMS:
            stage = next(s for s in l2r.arm_schedule(
                runtime.contract, arm, "smoke") if s["stage_kind"]
                == "search")
            config = l2r.materialize_candidate_config(
                runtime.contract, runtime.bindings, arm=arm, stage=stage,
                genome=_genome(runtime.contract),
                out_dir=runtime.tmp_path / f"anchor-{arm}", mode="smoke")
            anchors.add((config["warm_start_model"],
                         config["warm_start_model_sha256"]))
        assert len(anchors) == 1


# ---------------------------------------------------------------------------
# requirement 5 — paired inner/outer patience after a generation floor
# ---------------------------------------------------------------------------

def _summary(rap: float, trades: int = 7) -> dict:
    return {"mean_weekly_rap": rap, "mean_weekly_return": rap / 2,
            "max_drawdown_fraction": 0.05, "trades_total": trades}


class TestPairedL2Patience:
    def test_fitness_uses_the_inner_outer_pair(self):
        paired = _l2.paired_l2_fitness(_summary(1.0), _summary(0.5),
                                       beta=0.25, candidate_id="c")
        assert paired["labels"] == ["inner_validation", "outer_validation"]
        assert paired["eligible"] is True
        assert paired["paired_score"] == pytest.approx(
            0.5 * 1.5 - 0.25 * 0.5)

    def test_excellent_inner_and_collapsed_outer_cannot_win(self):
        honest = _l2.paired_l2_fitness(_summary(0.6), _summary(0.6),
                                       beta=0.25, candidate_id="honest")
        overfit = _l2.paired_l2_fitness(_summary(2.0), _summary(-1.0),
                                        beta=0.25, candidate_id="overfit")
        assert overfit["paired_score"] < honest["paired_score"]

    def test_inactive_member_makes_the_pair_ineligible(self):
        paired = _l2.paired_l2_fitness(_summary(1.0), _summary(1.0, 0),
                                       beta=0.25)
        assert paired["eligible"] is False
        assert paired["paired_score"] is None

    def test_patience_cannot_fire_before_the_floor(self, contract):
        spec = dict(contract["l2_stopping"])
        state = _l2.l2_stopping_state(spec, split_identity="s")
        flat = _l2.paired_l2_fitness(_summary(0.4), _summary(0.4),
                                     beta=0.25)
        decisions = [_l2.l2_generation_stop_decision(state, flat, gen)
                     for gen in range(int(spec["minimum_generations"]))]
        assert not any(d["stop"] for d in decisions)
        assert all(d["floor_reached"] is False for d in decisions[1:])

    def test_patience_fires_after_the_floor_without_improvement(
            self, contract):
        spec = dict(contract["l2_stopping"])
        state = _l2.l2_stopping_state(spec, split_identity="s")
        flat = _l2.paired_l2_fitness(_summary(0.4), _summary(0.4),
                                     beta=0.25)
        stopped = False
        for gen in range(int(spec["minimum_generations"])
                         + int(spec["patience"]) + 2):
            stopped = _l2.l2_generation_stop_decision(
                state, flat, gen)["stop"] or stopped
        assert stopped is True

    def test_ineligible_generations_never_consume_patience(self, contract):
        spec = dict(contract["l2_stopping"])
        state = _l2.l2_stopping_state(spec, split_identity="s")
        dead = _l2.paired_l2_fitness(_summary(0.4), _summary(0.4, 0),
                                     beta=0.25)
        for gen in range(20):
            assert _l2.l2_generation_stop_decision(
                state, dead, gen)["stop"] is False
        assert state.waited == 0

    def test_stopping_state_refuses_a_foreign_split_identity(self,
                                                             contract):
        state = _l2.l2_stopping_state(dict(contract["l2_stopping"]),
                                      split_identity="a")
        with pytest.raises(Exception, match="split identity"):
            l2r._paired.PairedStoppingState.from_state(
                state.to_state(), split_identity="b")


# ---------------------------------------------------------------------------
# requirement 6 — diversity evidence
# ---------------------------------------------------------------------------

class TestDiversityEvidence:
    def test_collapse_and_spread_are_distinguishable(self, contract):
        names = _gene_names(contract)
        bounds = {str(n): (lo, hi)
                  for n, lo, hi, _k in contract["shared"]["gene_space"]}
        clone = _genome(contract)["parameters"]
        collapsed = _l2.population_diversity(
            [{"parameters": dict(clone), "fitness": 1.0}
             for _ in range(6)], gene_names=names, gene_bounds=bounds)
        varied = _l2.population_diversity(
            [{"parameters": {**clone, "gamma": 0.9 + 0.01 * i},
              "fitness": float(i)} for i in range(6)],
            gene_names=names, gene_bounds=bounds)
        assert collapsed["unique_genome_count"] == 1
        assert collapsed["unique_genome_fraction"] == pytest.approx(1 / 6)
        assert collapsed["mean_pairwise_allele_distance"] == 0.0
        assert varied["unique_genome_count"] == 6
        assert varied["mean_pairwise_allele_distance"] > 0.0
        assert varied["fitness_dispersion"] > 0.0

    def test_rejected_and_ineligible_shares_are_reported(self, contract):
        names = _gene_names(contract)
        entries = [
            {"parameters": _genome(contract)["parameters"],
             "candidate_rejected": True, "fitness": None,
             "paired": {"eligible": False}},
            {"parameters": _genome(contract)["parameters"],
             "fitness": 1.0, "paired": {"eligible": True}}]
        diversity = _l2.population_diversity(entries, gene_names=names)
        assert diversity["rejected_share"] == pytest.approx(0.5)
        assert diversity["ineligible_share"] == pytest.approx(0.5)

    def test_a_record_without_diversity_refuses(self, contract):
        with pytest.raises(_l2.L2CurriculumError, match="diversity"):
            _l2.assert_diversity_logged(
                {"diversity": {"unique_genome_count": 3}},
                contract["diversity"]["required_fields"])


# ---------------------------------------------------------------------------
# requirement 7 — the nested data roles and the sealed 2025 year
# ---------------------------------------------------------------------------

class TestNestedRoles:
    def test_contract_pins_the_exact_role_counts(self, contract):
        pins = contract["nested_split_contract"]["role_facts"]
        assert pins["fit_train"]["scored_rows"] == 11509
        assert pins["train_monitor"]["scored_rows"] == 2190
        assert pins["inner_validation"]["scored_rows"] == 2190
        assert pins["outer_validation"]["scored_rows"] == 2196
        for role in ("train_monitor", "inner_validation",
                     "outer_validation"):
            assert pins[role]["context_rows"] == 256
        assert pins["sealed_test"]["status"] == "SEALED"

    def test_real_materialization_matches_the_pins_and_seals_2025(
            self, contract, bindings, tmp_path):
        roles = l2r.materialize_l2_nested_roles(contract, bindings,
                                                tmp_path)
        assert roles["binding"]["mode"] == "l2"
        pins = contract["nested_split_contract"]["role_facts"]
        for role, pin in pins.items():
            if pin["status"] != "MATERIALIZED":
                continue
            for key in ("csv_sha256", "scored_rows", "context_rows",
                        "score_start", "score_end"):
                assert roles["roles"][role][key] == pin[key]
        assert roles["roles"]["sealed_test"]["status"] == "SEALED"
        assert roles["roles"]["sealed_test"].get("csv") is None
        assert not (tmp_path / "nested_splits" / "sealed_test.csv").exists()

    def test_a_sealed_test_materialization_refuses(self, contract):
        manifest_roles = {
            role: dict(pin) for role, pin
            in contract["nested_split_contract"]["role_facts"].items()}
        manifest_roles["sealed_test"] = {
            "status": "MATERIALIZED", "csv": "/tmp/sealed_test.csv",
            "csv_sha256": "x" * 64, "scored_rows": 2190,
            "context_rows": 256}
        refusals = l2r.p1.verify_role_facts(
            manifest_roles,
            contract["nested_split_contract"]["role_facts"])
        assert any("sealed" in text.lower() for text in refusals)

    def test_executable_config_can_never_open_the_sealed_year(self,
                                                              runtime):
        stage = l2r.arm_schedule(runtime.contract, "L2_N", "smoke")[0]
        config = l2r.materialize_candidate_config(
            runtime.contract, runtime.bindings, arm="L2_N", stage=stage,
            genome=_genome(runtime.contract),
            out_dir=runtime.tmp_path / "sealed", mode="smoke")
        assert config["evaluate_test_split"] is False
        assert config["nested_split_mode"] == "l2"
        assert config["selection_metric"] == l2r.SELECTION_METRIC
        for key in ("test_start", "test_end", "test_years"):
            assert config.get(key) is None
        config["evaluate_test_split"] = True
        with pytest.raises(RuntimeError, match="release-only"):
            l2r.assert_sealed_test_inaccessible(config)

    def test_the_replay_helper_refuses_the_sealed_role(self):
        with pytest.raises(RuntimeError, match="sealed"):
            l2r.p1._outer_validation_final_eval(
                config={}, agent=None, model_path="x",
                artifact_role="best_checkpoint", nested_roles={},
                seed=101, role="sealed_test")


# ---------------------------------------------------------------------------
# requirement 8 — node identity, requirement 9 — one chain per arm
# ---------------------------------------------------------------------------

class TestNodeIdentityAndChains:
    def test_identity_facts_cover_every_required_proof(self, contract,
                                                       bindings):
        facts = l2r.node_identity_facts(contract, bindings, arm="L2_N",
                                        mode="smoke")
        for key in l2r.NODE_IDENTITY_KEYS:
            assert facts.get(key)
        assert facts["plan_hash"] == contract["_contract_sha256"]

    @pytest.mark.parametrize("field", [
        "plan_hash", "domain_hash", "dataset_sha256", "population_seed",
        "genesis", "experiment_identity"])
    def test_a_mismatched_fact_refuses(self, contract, bindings, field):
        facts = l2r.node_identity_facts(contract, bindings, arm="L2_N",
                                        mode="smoke")
        other = dict(facts)
        other[field] = "tampered"
        assert l2r.verify_node_identity(facts, other)

    def test_a_moved_source_revision_refuses(self, contract, bindings,
                                             monkeypatch):
        mine = l2r.node_identity_facts(contract, bindings, arm="L2_N",
                                       mode="smoke")
        theirs = l2r.node_identity_facts(
            contract, bindings, arm="L2_N", mode="smoke",
            sources=copy.deepcopy(MOVED_SOURCES))
        refusals = l2r.verify_node_identity(mine, theirs)
        assert any("source_revisions" in text for text in refusals)
        assert any("genesis" in text for text in refusals)

    def test_a_different_finalized_ancestry_refuses(self, contract,
                                                    bindings):
        facts = l2r.node_identity_facts(contract, bindings, arm="L2_N",
                                        mode="smoke")
        refusals = l2r.verify_node_identity(
            facts, facts, finalized_ancestry="aaa",
            registered_ancestry="bbb")
        assert any("finalized ancestry" in text for text in refusals)

    def test_worker_refuses_a_node_identity_mismatch(self, runtime,
                                                     monkeypatch):
        chain = l2r.open_or_join_chain(
            runtime.contract, runtime.bindings, arm="L2_N", mode="smoke",
            worker_id="w1")
        registry_path = Path(chain["_chain_dir"]) / "chain.json"
        registry = json.loads(registry_path.read_text())
        registry["node_identity"]["domain_hash"] = "0" * 64
        registry_path.write_text(json.dumps(registry))
        summary = l2r.run_worker("w2", contract=runtime.contract,
                                 arm="L2_N", mode="smoke",
                                 **runtime.kwargs)
        assert summary["outcome"] == "REFUSED_NODE_IDENTITY_MISMATCH"
        assert "domain_hash" in summary["reason"]

    def test_a_second_parallel_chain_for_one_arm_refuses(self, runtime):
        chain = l2r.open_or_join_chain(
            runtime.contract, runtime.bindings, arm="L2_N", mode="smoke",
            worker_id="w1")
        arm_path = Path(chain["_chain_dir"]).parent
        (arm_path / "chain_deadbeefdeadbeef").mkdir(parents=True)
        with pytest.raises(RuntimeError, match="REFUSED_PARALLEL_CHAIN"):
            l2r.assert_single_chain_per_arm(arm_path,
                                            chain["chain_identity"])
        summary = l2r.run_worker("w2", contract=runtime.contract,
                                 arm="L2_N", mode="smoke",
                                 **runtime.kwargs)
        assert summary["outcome"] == "REFUSED_PARALLEL_CHAIN"

    def test_four_workers_join_the_same_chain(self, runtime):
        identities = set()
        for worker in l2r.WORKERS:
            chain = l2r.open_or_join_chain(
                runtime.contract, runtime.bindings, arm="L2_N",
                mode="smoke", worker_id=worker)
            identities.add((chain["chain_identity"], chain["_chain_dir"]))
        assert len(identities) == 1

    def test_arms_dispatch_sequentially(self, runtime):
        summary = l2r.run_worker("w1", contract=runtime.contract,
                                 arm="L2_EN", mode="smoke",
                                 **runtime.kwargs)
        assert summary["outcome"] == "REFUSED_OUT_OF_ARM_ORDER"

    def test_each_arm_derives_its_own_chain(self, contract, bindings):
        chains = {l2r.chain_identity(contract, bindings, arm=arm,
                                     mode="smoke") for arm in l2r.ARMS}
        assert len(chains) == 2


# ---------------------------------------------------------------------------
# end-to-end: four workers, one chain, both arms, fake compute
# ---------------------------------------------------------------------------

class TestFourWorkerSmokeMechanics:
    def test_l2_n_completes_on_one_chain_with_typed_records(self, runtime):
        summary = _run_arm(runtime, "L2_N")
        assert summary["outcome"] == "ARM_COMPLETE"
        chain = _chain_path(runtime, "L2_N")
        assert (chain / "arm_complete.json").is_file()
        registry = json.loads((chain / "chain.json").read_text())
        assert registry["finalized_generations"] == 3
        assert len(list(chain.glob("chain_*"))) == 0
        workers = set()
        for generation in range(3):
            record = json.loads(
                (chain / f"gen{generation:03d}" /
                 "generation_record.json").read_text())
            assert record["evaluation_difficulty"] == _l2.NORMAL
            assert record["schema"] == l2r.GENERATION_RECORD_SCHEMA
            for field in runtime.contract["diversity"]["required_fields"]:
                assert field in record["diversity"]
            for index in range(4):
                candidate = json.loads(
                    (chain / f"gen{generation:03d}" /
                     f"candidate{index:03d}" /
                     "candidate_record.json").read_text())
                assert candidate["evaluation_difficulty"] == _l2.NORMAL
                assert candidate["decision_evidence"] is True
                assert candidate["terminal_model_sha256"]
                assert candidate["nested_role_facts"][
                    "outer_validation"]["scored_rows"] == 2196
                assert candidate["nested_role_facts"][
                    "sealed_test"]["status"] == "SEALED"
                assert set(candidate["evidence"]) == {"inner_validation",
                                                      "outer_validation"}
                workers.add(candidate["worker_id"])
        assert len(workers) > 1, "the chain was not collaborative"

    def test_l2_en_invalidates_and_reevaluates_before_any_champion(
            self, runtime):
        _run_arm(runtime, "L2_N")
        summary = _run_arm(runtime, "L2_EN")
        assert summary["outcome"] == "ARM_COMPLETE"
        chain = _chain_path(runtime, "L2_EN")
        gen0 = json.loads((chain / "gen000" /
                           "generation_record.json").read_text())
        gen1 = json.loads((chain / "gen001" /
                           "generation_record.json").read_text())
        gen2 = json.loads((chain / "gen002" /
                           "generation_record.json").read_text())
        assert gen0["evaluation_difficulty"] == _l2.EASY
        assert gen0["difficulty_transition"] is True
        assert gen0["easy_fitness_invalidated"] is True
        assert gen0["requires_normal_reevaluation"] == 4
        assert gen0["best_candidate"] is None, \
            "an easy generation may not publish a best candidate"
        assert gen1["stage"]["stage_kind"] == "reevaluation"
        assert gen1["evaluation_difficulty"] == _l2.NORMAL
        assert gen2["evaluation_difficulty"] == _l2.NORMAL
        # the re-evaluated survivors are exactly the easy generation's
        # genomes, re-scored under normal-realistic evidence
        easy_genomes = {
            json.loads((chain / "gen000" / f"candidate{i:03d}" /
                        "candidate_record.json").read_text())[
                "genome_fingerprint"] for i in range(4)}
        reeval_genomes = {
            json.loads((chain / "gen001" / f"candidate{i:03d}" /
                        "candidate_record.json").read_text())[
                "genome_fingerprint"] for i in range(4)}
        assert easy_genomes == reeval_genomes

    def test_the_champion_never_comes_from_an_easy_generation(self,
                                                              runtime):
        _run_arm(runtime, "L2_N")
        _run_arm(runtime, "L2_EN")
        chain = _chain_path(runtime, "L2_EN")
        resolved = l2r.champion_of_chain(chain)
        assert resolved["champion"] is not None
        assert resolved["champion"]["evaluation_difficulty"] == _l2.NORMAL
        assert resolved["champion"]["promotion_eligible"] is True
        easy_ids = {
            json.loads((chain / "gen000" / f"candidate{i:03d}" /
                        "candidate_record.json").read_text())[
                "candidate_identity"] for i in range(4)}
        assert resolved["champion"]["candidate_identity"] not in easy_ids

    def test_the_easy_lens_really_scored_higher_and_still_lost(self,
                                                              runtime):
        """The easy evidence is deliberately generous in the fake, so the
        refusal is doing real work: the highest raw number in the whole
        arm belongs to an easy candidate that never becomes champion."""
        _run_arm(runtime, "L2_N")
        _run_arm(runtime, "L2_EN")
        chain = _chain_path(runtime, "L2_EN")
        easy = [json.loads((chain / "gen000" / f"candidate{i:03d}" /
                            "candidate_record.json").read_text())
                for i in range(4)]
        champion = l2r.champion_of_chain(chain)["champion"]
        assert max(record["fitness"] for record in easy) > \
            champion["fitness"]

    def test_a_mixed_difficulty_generation_refuses_to_close(self, runtime):
        _run_arm(runtime, "L2_N", workers=("w1",), rounds=4,
                 max_candidates=None)
        chain = _chain_path(runtime, "L2_N")
        path = chain / "gen000" / "candidate000" / "candidate_record.json"
        record = json.loads(path.read_text())
        record["evaluation_difficulty"] = _l2.EASY
        path.write_text(json.dumps(record))
        (chain / "gen000" / "generation_record.json").unlink()
        with pytest.raises(RuntimeError, match="mixes evaluation"):
            l2r.finalize_generation(runtime.contract, chain, 0)

    def test_an_incomplete_generation_refuses_to_close(self, runtime):
        l2r.run_worker("w1", contract=runtime.contract, arm="L2_N",
                       mode="smoke", max_candidates=2, **runtime.kwargs)
        chain = _chain_path(runtime, "L2_N")
        with pytest.raises(RuntimeError, match="not complete"):
            l2r.finalize_generation(runtime.contract, chain, 0)

    def test_finalized_ancestry_grows_with_each_generation(self, runtime):
        _run_arm(runtime, "L2_N")
        chain = _chain_path(runtime, "L2_N")
        registry = json.loads((chain / "chain.json").read_text())
        assert registry["finalized_ancestry"] == \
            l2r.compute_finalized_ancestry(chain)
        assert registry["finalized_ancestry"] != registry["genesis"][:32]


# ---------------------------------------------------------------------------
# requirement 10 — the four-GPU smoke dispatch plan
# ---------------------------------------------------------------------------

class TestSmokeDispatch:
    def test_plan_binds_four_distinct_gpus_and_one_chain_per_arm(
            self, contract, bindings):
        plan = l2r.smoke_dispatch_commands(contract, bindings, mode="smoke")
        assert plan["arm_order"] == ["L2_N", "L2_EN"]
        uuids = set()
        for arm, block in plan["arms"].items():
            assert len({block["chain_identity"]}) == 1
            assert set(block["workers"]) == set(l2r.WORKERS)
            for worker, command in block["workers"].items():
                uuids.add(command["gpu_uuid"])
                assert f"--worker {worker}" in command["command"]
                assert f"--arm {arm}" in command["command"]
                assert "--mode smoke" in command["command"]
                assert command["command"].startswith(
                    f"CUDA_VISIBLE_DEVICES={command['gpu_uuid']}")
        assert len(uuids) == 4

    def test_smoke_and_decision_never_share_an_identity_or_root(
            self, contract, bindings):
        smoke = l2r.smoke_dispatch_commands(contract, bindings,
                                            mode="smoke")
        decision = l2r.smoke_dispatch_commands(contract, bindings,
                                               mode="decision")
        assert smoke["experiment_identity"] != \
            decision["experiment_identity"]
        assert smoke["output_root"] != decision["output_root"]

    def test_the_smoke_budget_is_bounded(self, contract):
        knobs = l2r.mode_setting(contract, "smoke",
                                 "candidate_budget_knobs")
        assert knobs["epoch_timesteps"] <= 2000
        assert knobs["phase1_epochs"] + knobs["phase2_max_epochs"] <= \
            knobs["total_max_passes"]
        assert int(l2r.mode_setting(contract, "smoke",
                                    "total_candidate_evaluations")) == 12


class TestGpuAndHostGates:
    def test_wrong_host_refuses(self, runtime, monkeypatch):
        monkeypatch.setattr(l2r.socket, "gethostname", lambda: "laptop")
        refusal = l2r.check_gpu_binding(runtime.contract, "w1")
        assert refusal["outcome"] == "REFUSED_WRONG_HOST"

    def test_unbound_cuda_refuses(self, runtime, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        refusal = l2r.check_gpu_binding(runtime.contract, "w1")
        assert refusal["outcome"] == "REFUSED_GPU_UNBOUND"

    def test_mismatched_cuda_refuses(self, runtime, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-other")
        refusal = l2r.check_gpu_binding(runtime.contract, "w1")
        assert refusal["outcome"] == "REFUSED_GPU_UNBOUND"

    def test_bound_worker_passes(self, runtime):
        assert l2r.check_gpu_binding(runtime.contract, "w1") is None


class TestContractRefusals:
    def test_lexicographic_metric_refuses(self, contract, tmp_path):
        live = {k: v for k, v in contract.items()
                if not k.startswith("_")}
        live["selection_metric"] = "lexicographic_weekly_v1"
        path = tmp_path / "c.json"
        path.write_text(json.dumps(live))
        with pytest.raises(ValueError, match="paired comparator"):
            l2r.load_contract(path)

    def test_l1_gene_leak_refuses_at_load(self, contract, tmp_path):
        live = copy.deepcopy({k: v for k, v in contract.items()
                              if not k.startswith("_")})
        live["shared"]["gene_space"].append(
            ["learning_rate", 1e-5, 1e-3, "float"])
        path = tmp_path / "c.json"
        path.write_text(json.dumps(live))
        with pytest.raises(_l2.L2CurriculumError, match="frozen L1"):
            l2r.load_contract(path)

    def test_l1_split_mode_refuses(self, contract, tmp_path):
        live = copy.deepcopy({k: v for k, v in contract.items()
                              if not k.startswith("_")})
        live["nested_split_contract"]["mode"] = "l1"
        path = tmp_path / "c.json"
        path.write_text(json.dumps(live))
        with pytest.raises(ValueError, match="nested_split_mode 'l2'"):
            l2r.load_contract(path)

    def test_unsealed_test_pin_refuses(self, contract, tmp_path):
        live = copy.deepcopy({k: v for k, v in contract.items()
                              if not k.startswith("_")})
        live["nested_split_contract"]["role_facts"]["sealed_test"][
            "status"] = "MATERIALIZED"
        path = tmp_path / "c.json"
        path.write_text(json.dumps(live))
        with pytest.raises(ValueError, match="SEALED"):
            l2r.load_contract(path)

    def test_the_invalid_paused_campaign_is_named_forbidden(self,
                                                            contract):
        assert any("eth-4h-anchored-full-sac-shared-v2" in item
                   for item in contract["forbidden"])


# ---------------------------------------------------------------------------
# gate 11 — the anchor-fallback gate (finding D8-20260815)
#
# The failure that voided three prior experiments was NOT a measurement
# bug: when no checkpoint passed the trade gate, selection fell back to
# the untouched warm-start anchor, so every arm scored the SAME object.
# In eth_curriculum_decision_20260807_v2 the arms E4, EN4_10 and N14 all
# selected checkpoints whose POLICY TENSOR equals anchor_seed101.zip's
# (8137b224…), so paired_differences_EN_minus_N subtracted the anchor
# from itself. These proofs make that state unreachable here.
# ---------------------------------------------------------------------------

def _with(runtime, **overrides) -> dict:
    kwargs = dict(runtime.kwargs)
    kwargs.update(overrides)
    return kwargs


def _anchor_everything(runtime) -> dict:
    """Every selected checkpoint hashes to the anchor's policy tensor —
    the exact eth_curriculum_decision_20260807_v2 fallback state."""
    anchor = runtime.contract["frozen_l1_recipe"]["common_anchor"]["path"]
    return _with(runtime,
                 policy_tensor_sha_fn=lambda _path:
                 _fake_policy_tensor_sha(anchor))


def _all_records(chain: Path) -> list:
    return [json.loads(path.read_text())
            for path in sorted(chain.glob("gen*/candidate*/"
                                          "candidate_record.json"))]


def _candidate_path(chain: Path, record: dict) -> Path:
    return (chain / f"gen{int(record['generation']):03d}"
            / f"candidate{int(record['index']):03d}"
            / "candidate_record.json")


def _rewrite(path: Path, mutate) -> dict:
    record = json.loads(path.read_text())
    mutate(record)
    path.write_text(json.dumps(record))
    return record


class TestAnchorFallbackGate:
    def test_every_candidate_records_both_tensor_shas(self, runtime):
        _run_arm(runtime, "L2_N")
        chain = _chain_path(runtime, "L2_N")
        anchor_sha = _fake_policy_tensor_sha(
            runtime.contract["frozen_l1_recipe"]["common_anchor"]["path"])
        records = _all_records(chain)
        assert len(records) == 12
        for record in records:
            gate = record["anchor_fallback_gate"]
            assert gate["schema"] == l2r.ANCHOR_GATE_SCHEMA
            assert gate["identity"] == "policy_tensor_sha256"
            assert gate["checked"] is True
            assert record["anchor_policy_tensor_sha256"] == anchor_sha
            assert record["scored_policy_tensor_sha256"]
            assert record["selected_equals_anchor"] is False
            assert record["scored_policy_tensor_sha256"] != anchor_sha

    def test_the_scored_identity_is_the_tensor_not_the_container(
            self, runtime):
        """The discriminator may never be the .zip digest: SB3 embeds
        member mtimes, so identical weights hash differently there."""
        _run_arm(runtime, "L2_N")
        record = _all_records(_chain_path(runtime, "L2_N"))[0]
        gate = record["anchor_fallback_gate"]
        assert gate["selected_container_sha256"] == \
            record["best_model_sha256"]
        assert gate["selected_policy_tensor_sha256"] != \
            gate["selected_container_sha256"]
        assert gate["anchor_policy_tensor_sha256"] != \
            gate["anchor_container_sha256"]

    def test_a_selected_checkpoint_that_is_the_anchor_is_never_scored(
            self, runtime):
        """The decisive D8 shape: DIFFERENT containers, IDENTICAL policy
        tensors. A container rule calls this healthy; the tensor rule
        names it for what it is."""
        summary = _run_arm(runtime, "L2_N",
                           kwargs=_anchor_everything(runtime))
        assert summary["outcome"] == "ARM_COMPLETE"
        chain = _chain_path(runtime, "L2_N")
        records = _all_records(chain)
        assert records
        for record in records:
            assert record["selected_equals_anchor"] is True
            assert record["candidate_rejected"] is True
            assert record["candidate_rejected_reason"] == \
                "selected_checkpoint_is_the_frozen_warm_start_anchor"
            assert record["fitness"] is None
            assert record["paired"]["eligible"] is False
            assert record["promotion_eligible"] is False
            assert record["evidence"] == {}, \
                "an anchor-identical checkpoint must never be scored"
            # the containers really do differ; only the tensors match
            assert record["best_model_sha256"] != \
                record["anchor_fallback_gate"]["anchor_container_sha256"]

    def test_an_all_anchor_arm_refuses_instead_of_reporting_a_null(
            self, runtime):
        _run_arm(runtime, "L2_N", kwargs=_anchor_everything(runtime))
        chain = _chain_path(runtime, "L2_N")
        facts = l2r.arm_anchor_fallback_facts(chain)
        assert facts["all_selected_equal_anchor"] is True
        assert facts["verdict"] == "ANCHOR_FALLBACK"
        assert facts["distinct_scored_policy_tensors"] == 1
        with pytest.raises(l2r.L2AnchorFallbackRefusal,
                           match="REFUSED_ANCHOR_FALLBACK"):
            l2r.champion_of_chain(chain)

    def test_a_partial_anchor_fallback_still_publishes_a_champion(
            self, runtime):
        """Only a WHOLLY anchor-bound arm refuses. Individual
        anchor-identical candidates are typed rejected outcomes and the
        arm keeps its remaining, real evidence."""
        anchor = runtime.contract["frozen_l1_recipe"]["common_anchor"]

        def sha(path):
            if Path(path).parent.name == "candidate000":
                return _fake_policy_tensor_sha(anchor["path"])
            return _fake_policy_tensor_sha(path)

        _run_arm(runtime, "L2_N",
                 kwargs=_with(runtime, policy_tensor_sha_fn=sha))
        chain = _chain_path(runtime, "L2_N")
        facts = l2r.arm_anchor_fallback_facts(chain)
        assert facts["verdict"] == "PARTIAL_ANCHOR_FALLBACK"
        assert facts["selected_equals_anchor_count"] == 3
        resolved = l2r.champion_of_chain(chain)
        assert resolved["champion"] is not None
        assert resolved["champion"]["selected_equals_anchor"] is False

    def test_an_unreadable_anchor_refuses_before_any_candidate(
            self, runtime):
        def sha(path):
            raise RuntimeError("policy.pth member missing")

        summary = l2r.run_worker(
            "w1", contract=runtime.contract, arm="L2_N", mode="smoke",
            max_candidates=1, **_with(runtime, policy_tensor_sha_fn=sha))
        assert summary["outcome"] == "REFUSED_ANCHOR_UNVERIFIED"
        assert "policy tensors" in summary["reason"]
        assert not _chain_path(runtime, "L2_N").exists()

    def test_a_missing_anchor_refuses_before_any_candidate(self, runtime):
        Path(runtime.contract["frozen_l1_recipe"]["common_anchor"][
            "path"]).unlink()
        summary = l2r.run_worker(
            "w1", contract=runtime.contract, arm="L2_N", mode="smoke",
            max_candidates=1, **runtime.kwargs)
        assert summary["outcome"] == "REFUSED_ANCHOR_UNVERIFIED"
        assert "does not exist" in summary["reason"]

    def test_a_drifted_anchor_refuses(self, runtime):
        Path(runtime.contract["frozen_l1_recipe"]["common_anchor"][
            "path"]).write_bytes(b"a different anchor entirely")
        summary = l2r.run_worker(
            "w1", contract=runtime.contract, arm="L2_N", mode="smoke",
            max_candidates=1, **runtime.kwargs)
        assert summary["outcome"] == "REFUSED_ANCHOR_UNVERIFIED"
        assert "contract pins" in summary["reason"]

    def test_behaviour_degeneracy_is_proven_from_the_evidence(self):
        live = l2r.evidence_behavior_facts({
            role: {"trades_total": 7,
                   "weekly_return_vector": [0.01, 0.02, -0.03]}
            for role in l2r.EVIDENCE_PAIR})
        assert live["behavior_degenerate"] is False
        assert live["behavior_fingerprint"]
        dead = l2r.evidence_behavior_facts({
            role: {"trades_total": 0,
                   "weekly_return_vector": [0.0, 0.0, 0.0]}
            for role in l2r.EVIDENCE_PAIR})
        assert dead["behavior_degenerate"] is True
        assert dead["behavior_fingerprint"] != live["behavior_fingerprint"]
        assert l2r.evidence_behavior_facts({})["behavior_degenerate"] is True


# ---------------------------------------------------------------------------
# gates 12 and 13 — arm differentiation and the campaign verdict
# ---------------------------------------------------------------------------

def _both_arms(runtime) -> tuple:
    _run_arm(runtime, "L2_N")
    _run_arm(runtime, "L2_EN")
    return _chain_path(runtime, "L2_N"), _chain_path(runtime, "L2_EN")


def _force_champion_facts(runtime, chain: Path, *, tensor: str,
                          trades: int = 7, rap: float = 0.4,
                          fingerprint: str = "fp") -> dict:
    champion = l2r.champion_of_chain(chain)["champion"]

    def mutate(record):
        record["scored_policy_tensor_sha256"] = tensor
        record["anchor_fallback_gate"][
            "selected_policy_tensor_sha256"] = tensor
        record["paired"]["paired_score"] = rap
        record["behavior"] = {"schema": l2r.BEHAVIOR_FACTS_SCHEMA,
                              "behavior_fingerprint": fingerprint,
                              "behavior_degenerate": trades == 0,
                              "trades_total": trades}
        record["behavior_fingerprint"] = fingerprint
        record["behavior_degenerate"] = trades == 0
        for role in l2r.EVIDENCE_PAIR:
            record["evidence"][role]["trades_total"] = trades
            record["evidence"][role]["metrics"] = {
                "mean_weekly_rap": rap, "mean_weekly_return": rap / 2.0,
                "max_drawdown_fraction": 0.05, "total_return": rap * 10.0}

    return _rewrite(_candidate_path(chain, champion), mutate)


class TestArmDifferentiationBoundary:
    def test_a_completed_smoke_separates_its_two_arms(self, runtime):
        _both_arms(runtime)
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "CAMPAIGN_VERDICT"
        assert code == 0
        assert payload["require_informative"] is True
        assert payload["checkpoint_identity"] == "policy_tensor_sha256"
        report = payload["differentiation"]
        assert report["differentiated"] is True
        assert report["informative"] is True
        assert [pair["verdict"] for pair in report["pairs"]] == ["OK"]
        for arm in l2r.ARMS:
            facts = payload["arms"][arm]
            assert facts["state"] == "COMPLETE"
            assert facts["scored_policy_tensor_sha256"]
            assert facts["behavior_fingerprint"]
        assert payload["arms"]["L2_N"]["scored_policy_tensor_sha256"] != \
            payload["arms"]["L2_EN"]["scored_policy_tensor_sha256"]

    def test_two_arms_that_scored_one_policy_refuse(self, runtime):
        """The exact eth_curriculum_decision_20260807_v2 signature."""
        n_chain, en_chain = _both_arms(runtime)
        for chain in (n_chain, en_chain):
            _force_champion_facts(runtime, chain, tensor="tensor:8137b224",
                                  fingerprint="fp-shared")
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "REFUSED_ARMS_NOT_DIFFERENTIATED"
        assert code == 4
        verdicts = {pair["verdict"]
                    for pair in payload["differentiation"]["pairs"]}
        assert "SHARED_SCORED_POLICY" in verdicts

    def test_distinct_policies_with_a_lost_metric_refuse(self, runtime):
        n_chain, en_chain = _both_arms(runtime)
        _force_champion_facts(runtime, n_chain, tensor="tensor:aaaa",
                              fingerprint="fp-n")
        _force_champion_facts(runtime, en_chain, tensor="tensor:bbbb",
                              fingerprint="fp-en")
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "REFUSED_ARMS_NOT_DIFFERENTIATED"
        assert "METRIC_COLLAPSE" in {
            pair["verdict"] for pair in payload["differentiation"]["pairs"]}

    def test_identical_behaviour_under_different_treatments_refuses(
            self, runtime):
        n_chain, en_chain = _both_arms(runtime)
        for chain in (n_chain, en_chain):
            _force_champion_facts(
                runtime, chain,
                tensor=f"tensor:{chain.name}", fingerprint="fp-identical")
        payload, _ = l2r.compare_arms(runtime.contract, runtime.bindings,
                                      mode="smoke")
        assert payload["outcome"] == "REFUSED_ARMS_NOT_DIFFERENTIATED"
        assert "TREATMENT_NOT_REALIZED" in {
            pair["verdict"] for pair in payload["differentiation"]["pairs"]}

    def test_two_dead_arms_are_unanswerable_not_a_defect(self, runtime):
        """Identical metrics from two DEGENERATE arms are real and must
        not be flagged as a measurement bug — but they also cannot
        answer the question, so require_informative types the campaign
        UNANSWERABLE instead of publishing a null."""
        n_chain, en_chain = _both_arms(runtime)
        _force_champion_facts(runtime, n_chain, tensor="tensor:dead-n",
                              trades=0, rap=0.0, fingerprint="fp-dead-n")
        _force_champion_facts(runtime, en_chain, tensor="tensor:dead-en",
                              trades=0, rap=0.0, fingerprint="fp-dead-en")
        permissive, code = l2r.compare_arms(
            runtime.contract, runtime.bindings, mode="smoke",
            require_informative=False)
        assert permissive["outcome"] == "CAMPAIGN_VERDICT"
        assert code == 0
        assert [pair["verdict"]
                for pair in permissive["differentiation"]["pairs"]] == \
            ["DEGENERATE_IDENTICAL"], \
            "a legitimately dead pair is NOT a differentiation defect"
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "CAMPAIGN_UNANSWERABLE"
        assert code == 4
        assert payload["differentiation"]["informative"] is False
        assert "NO_INFORMATIVE_CONTRAST" in {
            pair["verdict"] for pair in payload["differentiation"]["pairs"]}

    def test_an_anchor_bound_arm_refuses_at_the_comparison_boundary(
            self, runtime):
        _run_arm(runtime, "L2_N")
        _run_arm(runtime, "L2_EN", kwargs=_anchor_everything(runtime))
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "REFUSED_ANCHOR_FALLBACK"
        assert code == 4
        assert payload["arms"]["L2_EN"]["state"] == "ANCHOR_FALLBACK"

    def test_an_unfinished_campaign_cannot_be_compared(self, runtime):
        _run_arm(runtime, "L2_N")
        payload, code = l2r.compare_arms(runtime.contract, runtime.bindings,
                                         mode="smoke")
        assert payload["outcome"] == "CAMPAIGN_INCOMPLETE"
        assert code == 4
        assert payload["arms"]["L2_EN"]["state"] == "MISSING"

    def test_the_two_arms_declare_different_treatments(self, contract):
        treatments = {arm: l2r.arm_treatment(contract, arm, "smoke")
                      for arm in l2r.ARMS}
        assert treatments["L2_N"] != treatments["L2_EN"]
        assert treatments["L2_N"]["declares_easy_triage"] is False
        assert treatments["L2_EN"]["declares_easy_triage"] is True


class TestSmokeAcceptanceOutput:
    def test_acceptance_proves_the_arms_are_distinguishable(self, runtime):
        _both_arms(runtime)
        payload, code = l2r.smoke_acceptance(runtime.contract,
                                             runtime.bindings, mode="smoke")
        assert payload["schema"] == l2r.ACCEPTANCE_SCHEMA
        assert payload["outcome"] == "ACCEPTANCE_PASS"
        assert code == 0
        assert payload["proved_arms_are_distinguishable"] is True
        assert payload["preflight"]["outcome"] == "PREFLIGHT_PASS"
        anchor_gate = payload["gates"]["anchor_fallback"]
        assert anchor_gate["status"] == "PASS"
        assert anchor_gate["identity"] == "policy_tensor_sha256"
        assert set(anchor_gate["arms"]) == set(l2r.ARMS)
        for arm in l2r.ARMS:
            assert anchor_gate["arms"][arm]["selected_equals_anchor_count"] \
                == 0
            assert anchor_gate["arms"][arm]["checked_candidates"] == 12
        diff_gate = payload["gates"]["arm_differentiation"]
        assert diff_gate["status"] == "PASS"
        assert diff_gate["require_informative"] is True
        assert diff_gate["pairs"][0]["verdict"] == "OK"
        assert "PROVED" in payload["statement"]

    def test_acceptance_says_plainly_when_it_cannot_distinguish(
            self, runtime):
        n_chain, en_chain = _both_arms(runtime)
        for chain in (n_chain, en_chain):
            _force_champion_facts(runtime, chain, tensor="tensor:8137b224",
                                  fingerprint="fp-shared")
        payload, code = l2r.smoke_acceptance(runtime.contract,
                                             runtime.bindings, mode="smoke")
        assert payload["outcome"] == "ACCEPTANCE_REFUSED"
        assert code == 4
        assert payload["proved_arms_are_distinguishable"] is False
        assert payload["gates"]["arm_differentiation"]["status"] == "REFUSED"
        assert "CANNOT" in payload["statement"]
        assert any("SHARED_SCORED_POLICY" in reason
                   for reason in payload["refusals"])

    def test_acceptance_types_a_dead_campaign_unanswerable(self, runtime):
        n_chain, en_chain = _both_arms(runtime)
        _force_champion_facts(runtime, n_chain, tensor="tensor:dead-n",
                              trades=0, rap=0.0, fingerprint="fp-dead-n")
        _force_champion_facts(runtime, en_chain, tensor="tensor:dead-en",
                              trades=0, rap=0.0, fingerprint="fp-dead-en")
        payload, code = l2r.smoke_acceptance(runtime.contract,
                                             runtime.bindings, mode="smoke")
        assert payload["outcome"] == "ACCEPTANCE_REFUSED"
        assert code == 4
        assert payload["gates"]["arm_differentiation"]["status"] == \
            "UNANSWERABLE"
        assert payload["gates"]["anchor_fallback"]["status"] == "PASS"

    def test_preflight_declares_the_gates_before_any_gpu_is_touched(
            self, contract):
        payload, code = l2r.preflight(contract)
        assert payload["outcome"] == "PREFLIGHT_PASS"
        assert code == 0
        precondition = payload["differentiation_precondition"]
        assert precondition["treatments_distinguishable"] is True
        assert precondition["checkpoint_identity"] == "policy_tensor_sha256"
        assert precondition["gates_enforced"]["require_informative"] is True
        assert payload["anchor_declared"] is True
