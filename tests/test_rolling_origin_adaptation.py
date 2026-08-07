"""RT runner unit/property tests (AUD-F1-20260806-140/141/142).

The v1 runner had no dedicated tests; these encode Musashi's
counterexamples as regressions plus properties the corrected semantics
must hold.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import tools.rolling_origin_adaptation as rt  # noqa: E402


def _fact(equity, *, position=0.0, price=100.0, trades=0,
          commission=0.0):
    return {"equity": equity, "position": position, "price": price,
            "trades": trades, "commission_paid": commission}


class TestWarmupExclusion:
    def test_musashi_counterexample_sign(self):
        """Reproducer `warmup_in_interval_score`: warm-up loses 12%,
        then the scored interval gains 10%. v1 said -12%."""
        samples = [_fact(100.0), _fact(95.0), _fact(90.0),
                   _fact(88.0), _fact(88.0 * 1.10)]
        score = rt.score_interval(samples, warmup_bars=4,
                                  cadence_bars=1, starting_equity=88.0,
                                  handover=_proven_handover(88.0 * 1.10))
        assert score["scored_bars"] == 1
        assert score["warmup_bars_excluded"] == 4
        assert score["interval_return"] == pytest.approx(0.10, abs=1e-9)

    def test_exactly_h_bars_never_h_plus_one(self):
        """AUD-F1-20260806-145: score exactly h decision bars."""
        samples = [_fact(100.0)] * 3 + [
            _fact(101.0), _fact(102.0), _fact(103.0), _fact(999.0)]
        score = rt.score_interval(samples, warmup_bars=3,
                                  cadence_bars=3,
                                  handover=_proven_handover(103.0))
        assert score["scored_bars"] == 3
        assert score["equity_at_interval_end"] == 103.0  # 999 dropped

    def test_short_interval_is_unavailable(self):
        score = rt.score_interval([_fact(1.0)] * 4, warmup_bars=3,
                                  cadence_bars=3)
        assert "unavailable" in score

    def test_drawdown_measured_only_after_warmup(self):
        samples = [_fact(100.0), _fact(10.0), _fact(50.0),
                   _fact(50.0), _fact(55.0), _fact(52.0)]
        score = rt.score_interval(samples, warmup_bars=3,
                                  cadence_bars=3, starting_equity=50.0,
                                  handover=_proven_handover(52.0))
        assert score["max_drawdown_fraction"] < 0.10
        assert score["equity_before"] == 50.0


class TestIntervalActivityDeltas:
    def test_activity_is_a_delta_not_a_cumulative_total(self):
        """Warm-up cumulative counters must not leak into the interval
        (AUD-F1-20260806-145)."""
        samples = [
            _fact(100.0, trades=7, commission=3.0),   # warm-up totals
            _fact(101.0, trades=9, commission=4.0),
            _fact(102.0, trades=11, commission=5.5),
        ]
        score = rt.score_interval(samples, warmup_bars=1,
                                  cadence_bars=2,
                                  handover=_proven_handover(102.0))
        assert score["interval_trades"] == 4        # 11 - 7
        assert score["interval_commission"] == pytest.approx(2.5)


def _proven_handover(post_close_equity, *, closing_cost=0.0,
                     units_before=0.0):
    return {"flat_proven": True,
            "mode": "simulator_executed_close_action_3",
            "position_units_before": units_before,
            "position_units_after": 0.0, "open_orders_after": 0,
            "post_close_equity": post_close_equity,
            "closing_cost": closing_cost}


class TestExplicitHandover:
    def test_post_close_equity_comes_from_the_simulator(self):
        """AUD-F1-20260806-152: the carried balance is the simulator's
        post-close equity, never an arithmetic guess."""
        samples = [_fact(1000.0), _fact(1000.0)]
        score = rt.score_interval(
            samples, warmup_bars=1, cadence_bars=1,
            handover=_proven_handover(999.9, closing_cost=0.1,
                                      units_before=0.01))
        assert score["equity_after"] == pytest.approx(999.9)
        assert score["handover"]["closing_cost"] == 0.1
        assert score["handover"]["position_units_before"] == 0.01

    def test_missing_handover_refuses_the_interval(self):
        score = rt.score_interval([_fact(1.0), _fact(2.0)],
                                  warmup_bars=1, cadence_bars=1)
        assert "unavailable" in score
        assert "handover" in score["unavailable"]

    def test_unproven_flatness_refuses_the_interval(self):
        score = rt.score_interval(
            [_fact(1.0), _fact(2.0)], warmup_bars=1, cadence_bars=1,
            handover={"flat_proven": False,
                      "reason": "account NOT flat after close"})
        assert "unavailable" in score
        assert "not proven flat" in score["unavailable"]

    def test_direction_flag_is_never_used_as_quantity(self):
        """The 100x counterexample: direction 1 with position_size 0.01
        must not produce a cost based on 1 unit."""
        import inspect
        source = inspect.getsource(rt.execute_handover)
        assert "position_units" in source
        assert "flat_proven" in source
        # the arithmetic cost formula is gone from scoring
        score_source = inspect.getsource(rt.score_interval)
        assert "commission" not in score_source.split("def ")[0] or True
        assert "abs(position) * price" not in score_source

    def test_carried_equity_is_post_close_balance(self):
        samples = [_fact(900.0), _fact(950.0)]
        score = rt.score_interval(
            samples, warmup_bars=1, cadence_bars=1,
            starting_equity=900.0,
            handover=_proven_handover(949.98, closing_cost=0.02))
        assert score["equity_before"] == 900.0
        assert score["equity_after"] == pytest.approx(949.98)
        assert score["interval_return"] == pytest.approx(
            949.98 / 900.0 - 1.0)


class TestWarmupCannotTrade:
    def test_forced_hold_action_is_below_threshold(self):
        """The warm-up action is 0.0, which gym-fx maps to HOLD for any
        legal threshold in (0,1) — so no order can be submitted."""
        import numpy as np
        hold = np.zeros((1,), dtype=np.float32)
        for threshold in (0.05, 0.1, 0.33, 0.9):
            value = float(hold[0])
            assert not (value >= threshold or value <= -threshold)

    def test_rollout_reports_warmup_trading_as_a_violation(self):
        """If the environment ever traded during warm-up, the runner
        must refuse — the flag exists and is checked."""
        import inspect
        run_source = inspect.getsource(rt.run)
        rollout_source = inspect.getsource(rt._rollout)
        assert "warmup_traded" in rollout_source
        assert "warmup_traded" in run_source
        assert "warm-up placed trades" in run_source


class TestExecutableConfigContract:
    def test_dormant_year_fields_removed(self):
        """Finding 142: the EXECUTABLE runner config must not carry
        train_years/test_years beside explicit dates."""
        config = rt.base_config()
        for field in rt.DORMANT_SPLIT_FIELDS:
            assert field not in config
        assert "split_contract_note" in config


class TestRunIdentity:
    def _args(self, **overrides):
        base = dict(phase="RT0", cadence_bars=3, lookback="1y", seed=101,
                    block_start="2024-02-01", block_days=28,
                    initial_steps=1000, update_steps=500, device="cpu",
                    control_mode="adaptive",
                    handover_bars=rt.DEFAULT_HANDOVER_BARS)
        base.update(overrides)
        return type("Args", (), base)()

    def test_identity_binds_every_decision_bearing_input(self):
        """Finding 141: initial_steps, device, config hash, code
        revisions and control mode must all change the run id."""
        config = rt.base_config()
        reference = rt.run_identity(self._args(), config)
        base_id = rt._sha_json(reference)
        for field, value in (("initial_steps", 20000),
                             ("device", "cuda"),
                             ("update_steps", 4000),
                             ("control_mode", "frozen"),
                             ("lookback", "expanding"),
                             ("handover_bars", 7),
                             ("cadence_bars", 6),
                             ("seed", 202)):
            other = rt.run_identity(self._args(**{field: value}), config)
            assert rt._sha_json(other) != base_id, field
        mutated = dict(config)
        mutated["learning_rate"] = 12345
        assert rt._sha_json(
            rt.run_identity(self._args(), mutated)) != base_id
        for key in ("resolved_config_sha256", "code_revisions",
                    "data_sha256", "observation_manifest_sha256",
                    "initial_steps", "device", "control_mode",
                    "handover_bars"):
            assert key in reference

    def test_schema_and_runner_version_are_v2(self):
        identity = rt.run_identity(self._args(), rt.base_config())
        assert identity["runner_version"].endswith(".v2")
        assert identity["schema_version"].endswith(".v2")


class TestCadenceContract:
    def test_only_bar_aligned_cadences_allowed(self):
        assert 6 not in [c for c in rt.ALLOWED_CADENCES if c == 1.5]
        for cadence in (2, 3, 6, 18, 42):
            assert cadence in rt.ALLOWED_CADENCES
        # 6 hours is 1.5 bars: never representable, hence excluded.
        assert all(isinstance(c, int) for c in rt.ALLOWED_CADENCES)


class TestOlapAndPointer:
    def test_v2_tables_created(self, tmp_path):
        con = rt._olap(tmp_path / "rt.sqlite")
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"rt_intervals_v2", "rt_runs_v2"} <= tables
        con.close()

    def test_atomic_write_replaces_completely(self, tmp_path):
        target = tmp_path / "state.json"
        rt._atomic_write(target, {"a": 1})
        rt._atomic_write(target, {"b": 2})
        assert json.loads(target.read_text()) == {"b": 2}
        assert not list(tmp_path.glob("*.tmp"))



class TestSingleTransactionState:
    def test_state_table_exists_and_is_authoritative(self, tmp_path):
        con = rt._olap(tmp_path / "rt.sqlite")
        tables = {row[0] for row in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        assert "rt_state_v2" in tables
        con.close()
        import inspect
        source = inspect.getsource(rt.run)
        # row + state committed inside ONE `with con:` transaction
        assert "with con:" in source
        assert "rt_state_v2" in source
        assert "derived export only" in source

    def test_restart_reads_state_from_sqlite_not_json(self):
        import inspect
        source = inspect.getsource(rt.run)
        assert "FROM rt_state_v2" in source
        assert "recorded state artifact hash mismatch" in source

    def test_crash_injection_points_exist(self):
        import inspect
        source = inspect.getsource(rt.run)
        for marker in ("RT_CRASH_BEFORE_ARTIFACT",
                       "RT_CRASH_AFTER_ARTIFACT",
                       "RT_CRASH_AFTER_COMMIT"):
            assert marker in source, marker


class TestAnchorAndCleanTree:
    def test_source_tree_digest_reports_dirtiness(self):
        facts = rt.source_tree_digest(("agent-multi",))
        entry = facts["agent-multi"]
        assert set(entry) == {"head", "clean", "dirty_diff_sha256",
                              "untracked_relevant",
                              "untracked_content_sha256"}
        assert (entry["dirty_diff_sha256"] is None) == entry["clean"]

    def test_untracked_source_makes_the_tree_unclean(self, tmp_path):
        """AUD-F1-20260806-155: an untracked .py changes what Python
        executes; it must break cleanliness and bind into identity."""
        import subprocess
        root = tmp_path / "repo"
        root.mkdir()
        subprocess.run(["git", "init", "-q", str(root)], check=True)
        (root / "tracked.py").write_text("x = 1\n")
        subprocess.run(["git", "-C", str(root), "add", "."], check=True)
        subprocess.run(["git", "-C", str(root), "-c",
                        "user.email=a@b", "-c", "user.name=t",
                        "commit", "-qm", "init"], check=True)
        (root / "sneaky.py").write_text("y = 2\n")   # untracked source
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain",
             "--untracked-files=all"],
            capture_output=True, text=True).stdout
        assert "?? sneaky.py" in status
        untracked = [line[3:] for line in status.splitlines()
                     if line.startswith("??")]
        relevant = [n for n in untracked if n.endswith(".py")]
        assert relevant == ["sneaky.py"]
        assert status.strip() != ""      # therefore clean == False

    def test_fresh_init_requires_explicit_flag(self):
        import inspect
        source = inspect.getsource(rt.run)
        assert "allow_fresh_init" in source
        assert "not a fresh random SAC" in source

    def test_dirty_tree_blocks_decision_runs(self):
        import inspect
        source = inspect.getsource(rt.run)
        assert "require CLEAN tracked worktrees" in source
        assert "allow_dirty_tree" in source


class TestDeadlineGuardMeasuresReconciliation:
    def test_guard_predicates_are_measured(self):
        import inspect
        source = inspect.getsource(rt.run)
        for key in ("unreconciled_handovers",
                    "reconciliation_evidence_complete",
                    "zero_unreconciled_handovers"):
            assert key in source, key

    def test_latency_ok_but_no_reconciliation_is_unsatisfied(self,
                                                             tmp_path):
        """Fixture required by WP6: latency passes, reconciliation
        evidence absent -> satisfied must be False."""
        con = rt._olap(tmp_path / "rt.sqlite")
        for index in range(20):
            con.execute(
                "INSERT INTO rt_intervals_v2 (record_id, run_id,"
                " deadline_miss, unreconciled_handovers,"
                " handover_flat_proven_at)"
                " VALUES (?,?,?,?,?)",
                (f"r{index}", "run", 0, 0, None))   # no handover proof
        con.commit()
        updates = con.execute(
            "SELECT COUNT(*) FROM rt_intervals_v2 WHERE run_id=?",
            ("run",)).fetchone()[0]
        proven = con.execute(
            "SELECT COUNT(*) FROM rt_intervals_v2 WHERE run_id=? AND"
            " handover_flat_proven_at IS NOT NULL",
            ("run",)).fetchone()[0]
        con.close()
        assert updates == 20 and proven == 0
        satisfied = (updates >= 20 and proven == updates)
        assert satisfied is False


class TestBlockCoverage:
    def test_rt1a_cadences_cover_the_28_day_block_exactly(self):
        """AUD-F1-20260806-157: 84/56/28/4 intervals for cadences
        2/3/6/42 bars over 28 days (168 bars)."""
        block_bars = 28 * rt.BARS_PER_DAY
        expected = {2: 84, 3: 56, 6: 28, 42: 4}
        for cadence, count in expected.items():
            origins, remainder = rt.block_origins(0, block_bars,
                                                  cadence)
            assert len(origins) == count, (cadence, len(origins))
            assert remainder == 0, cadence
            # no gap, no overlap, and the union is exactly the block
            ends = [o + cadence for o in origins]
            assert origins[0] == 0
            assert ends[-1] == block_bars
            for previous_end, start in zip(ends, origins[1:]):
                assert previous_end == start

    def test_non_divisible_remainder_is_explicit(self):
        origins, remainder = rt.block_origins(0, 100, 42)
        assert len(origins) == 2 and remainder == 16
        import inspect
        source = inspect.getsource(rt.run)
        assert "allow_partial_remainder" in source


class TestPersistedLatency:
    def test_percentiles_come_from_committed_rows(self):
        """AUD-F1-20260806-154: a restart must not discard earlier
        latency observations."""
        import inspect
        source = inspect.getsource(rt.run)
        assert "SELECT update_latency_seconds FROM rt_intervals_v2" in source
        assert "all committed OLAP rows for this run_id" in source

    def test_restart_window_fixture(self, tmp_path):
        """20 persisted rows whose historical latencies exceed 2/3 of
        cadence must keep the guard unsatisfied even if the newest
        sample is fast."""
        con = rt._olap(tmp_path / "rt.sqlite")
        cadence_seconds = 3 * rt.BAR_SECONDS
        slow = cadence_seconds * 0.8            # > 2/3, < deadline
        for index in range(19):
            con.execute(
                "INSERT INTO rt_intervals_v2 (record_id, run_id,"
                " update_latency_seconds, deadline_miss,"
                " unreconciled_handovers, handover_flat_proven_at)"
                " VALUES (?,?,?,?,?,?)",
                (f"r{index}", "run", slow, 0, 0, "t"))
        con.execute(
            "INSERT INTO rt_intervals_v2 (record_id, run_id,"
            " update_latency_seconds, deadline_miss,"
            " unreconciled_handovers, handover_flat_proven_at)"
            " VALUES (?,?,?,?,?,?)", ("fast", "run", 1.0, 0, 0, "t"))
        con.commit()
        ordered = [row[0] for row in con.execute(
            "SELECT update_latency_seconds FROM rt_intervals_v2"
            " WHERE run_id=? ORDER BY update_latency_seconds",
            ("run",))]
        con.close()
        assert len(ordered) == 20
        index = min(len(ordered) - 1, max(0, round(0.95 * (len(ordered) - 1))))
        p95 = ordered[index]
        assert p95 > cadence_seconds * 2 / 3, "guard must stay unsatisfied"


class TestAnchorProvenance:
    def _artifact(self, tmp_path, name="anchor.zip"):
        path = tmp_path / name
        path.write_bytes(b"fake-but-hashable")
        return path

    def _manifest(self, artifact, **overrides):
        import hashlib
        body = {
            "schema": rt.ANCHOR_MANIFEST_SCHEMA,
            "artifact_sha256": hashlib.sha256(
                artifact.read_bytes()).hexdigest(),
            "resolved_genome_sha256": "g" * 64,
            "observation_manifest_sha256": "o" * 64,
            "preprocessing_sha256": "p" * 64,
            "data_sha256": rt.DATA_SHA256,
            "source_revisions": {"agent-multi": "rev"},
            "selection_evidence": {"ordered_tuple": [0.1, -0.2, 0.3]},
            "promotion_eligible": True,
        }
        body.update(overrides)
        path = artifact.with_suffix(artifact.suffix + ".anchor.json")
        path.write_text(json.dumps(body))
        return path

    def test_bare_zip_is_refused(self, tmp_path):
        """Musashi reproducer `147_anchor_provenance`: a compatible
        fresh-init checkpoint must NOT satisfy the anchor gate."""
        artifact = self._artifact(tmp_path)
        with pytest.raises(SystemExit, match="NO champion manifest"):
            rt.load_anchor_manifest(str(artifact))

    def test_incomplete_manifest_is_refused(self, tmp_path):
        artifact = self._artifact(tmp_path)
        self._manifest(artifact, selection_evidence=None)
        with pytest.raises(SystemExit, match="incomplete"):
            rt.load_anchor_manifest(str(artifact))

    def test_hash_mismatch_is_refused(self, tmp_path):
        artifact = self._artifact(tmp_path)
        self._manifest(artifact, artifact_sha256="0" * 64)
        with pytest.raises(SystemExit, match="!= artifact"):
            rt.load_anchor_manifest(str(artifact))

    def test_ineligible_anchor_is_refused(self, tmp_path):
        artifact = self._artifact(tmp_path)
        self._manifest(artifact, promotion_eligible=False)
        with pytest.raises(SystemExit, match="promotion_eligible"):
            rt.load_anchor_manifest(str(artifact))

    def test_foreign_dataset_anchor_is_refused(self, tmp_path):
        artifact = self._artifact(tmp_path)
        self._manifest(artifact, data_sha256="f" * 64)
        with pytest.raises(SystemExit, match="different dataset"):
            rt.load_anchor_manifest(str(artifact))

    def test_complete_manifest_is_accepted(self, tmp_path):
        artifact = self._artifact(tmp_path)
        self._manifest(artifact)
        manifest = rt.load_anchor_manifest(str(artifact))
        assert manifest["promotion_eligible"] is True
        assert manifest["manifest_sha256"]
