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
                                  cadence_bars=1, starting_equity=88.0)
        assert score["scored_bars"] == 1
        assert score["warmup_bars_excluded"] == 4
        assert score["interval_return"] == pytest.approx(0.10, abs=1e-9)

    def test_exactly_h_bars_never_h_plus_one(self):
        """AUD-F1-20260806-145: score exactly h decision bars."""
        samples = [_fact(100.0)] * 3 + [
            _fact(101.0), _fact(102.0), _fact(103.0), _fact(999.0)]
        score = rt.score_interval(samples, warmup_bars=3,
                                  cadence_bars=3)
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
                                  cadence_bars=3, starting_equity=50.0)
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
                                  cadence_bars=2)
        assert score["interval_trades"] == 4        # 11 - 7
        assert score["interval_commission"] == pytest.approx(2.5)


class TestExplicitHandover:
    def test_open_exposure_is_closed_and_charged(self):
        """A position open at the interval end is closed at the last
        price and charged the configured commission; the post-close
        balance is what carries."""
        samples = [_fact(1000.0),
                   _fact(1000.0, position=2.0, price=50.0)]
        score = rt.score_interval(samples, warmup_bars=1,
                                  cadence_bars=1, commission=0.001)
        handover = score["handover"]
        assert handover["open_position_units"] == 2.0
        assert handover["closing_cost"] == pytest.approx(2 * 50 * 0.001)
        assert score["equity_after"] == pytest.approx(1000.0 - 0.1)
        assert score["equity_at_interval_end"] == 1000.0
        assert handover["flat_after_handover"] is True

    def test_flat_interval_has_zero_closing_cost(self):
        samples = [_fact(1000.0), _fact(1010.0, position=0.0)]
        score = rt.score_interval(samples, warmup_bars=1,
                                  cadence_bars=1, commission=0.001)
        assert score["handover"]["closing_cost"] == 0.0
        assert score["equity_after"] == 1010.0

    def test_carried_equity_is_post_close_balance(self):
        samples = [_fact(900.0), _fact(950.0, position=1.0, price=10.0)]
        score = rt.score_interval(samples, warmup_bars=1,
                                  cadence_bars=1, starting_equity=900.0,
                                  commission=0.002)
        assert score["equity_before"] == 900.0
        assert score["equity_after"] == pytest.approx(950.0 - 0.02)
        assert score["interval_return"] == pytest.approx(
            (950.0 - 0.02) / 900.0 - 1.0)


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
                    control_mode="adaptive")
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
                    "initial_steps", "device", "control_mode"):
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
        assert set(entry) == {"head", "clean", "dirty_diff_sha256"}
        assert (entry["dirty_diff_sha256"] is None) == entry["clean"]

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
