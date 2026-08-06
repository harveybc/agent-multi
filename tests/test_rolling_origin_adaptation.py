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


class TestWarmupExclusion:
    def test_musashi_counterexample_sign(self):
        """Reproducer `warmup_in_interval_score`: warm-up history loses
        12% then the scored interval gains 10%. v1 reported -12%; v2
        must report +10% and score exactly one bar."""
        warmup = [100.0, 95.0, 90.0, 88.0]     # 4 warm-up bars
        interval = [88.0 * 1.10]               # +10% deployment bar
        score = rt.score_interval(warmup + interval, warmup_bars=4,
                                  starting_equity=88.0)
        assert score["scored_bars"] == 1
        assert score["warmup_bars_excluded"] == 4
        assert score["interval_return"] == pytest.approx(0.10, abs=1e-9)

    def test_no_scored_bars_is_unavailable_never_zero(self):
        score = rt.score_interval([1.0, 2.0], warmup_bars=5)
        assert "unavailable" in score
        assert score["scored_bars"] == 0

    def test_drawdown_measured_only_after_warmup(self):
        # A catastrophic warm-up drawdown must not pollute the interval.
        series = [100.0, 10.0, 50.0] + [50.0, 55.0, 52.0]
        score = rt.score_interval(series, warmup_bars=3,
                                  starting_equity=50.0)
        assert score["max_drawdown_fraction"] < 0.10
        assert score["equity_before"] == 50.0
        assert score["equity_after"] == 52.0


class TestAccountContinuity:
    def test_carried_equity_is_the_baseline_not_config_cash(self):
        """Finding 140: the next interval opens on the previous close."""
        score = rt.score_interval([9000.0, 9500.0], warmup_bars=1,
                                  starting_equity=9000.0)
        assert score["equity_before"] == 9000.0
        assert score["interval_return"] == pytest.approx(
            9500.0 / 9000.0 - 1.0)

    def test_without_carry_uses_first_scored_bar(self):
        score = rt.score_interval([1.0, 200.0, 210.0], warmup_bars=1)
        assert score["equity_before"] == 200.0
        assert score["interval_return"] == pytest.approx(0.05)


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
