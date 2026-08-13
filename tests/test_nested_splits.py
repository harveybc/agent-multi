"""WP1 nested-split contract — the eight §5.4 test classes.

Exact ETH row/date assertions; boundary mutation fails; overlap/gap/
reordering fail; context rows produce no actions/trades/equity/metrics;
first scored row has complete context; full year intervals intact; the
sealed test path cannot open in train/L1/L2 modes; window/scaling
changes recompute required context.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_plugins import _nested_splits as ns

REPO = Path(__file__).resolve().parent.parent
CONTRACT_PATH = (REPO / "examples/config/phase_3_eth_sac_dynamics/"
                 "splits/eth_nested_split_contract_v1.json")
ETH_SOURCE = Path("/home/harveybc/Documents/GitHub/predictor/examples/"
                  "data/project3/ethusdt_4h_tech_stat_full_model_ready.csv")

needs_eth = pytest.mark.skipif(
    not ETH_SOURCE.is_file(), reason="pinned ETH csv absent on this host")


def _synthetic(tmp_path: Path, rows: int = 700, freq: str = "4h",
               start: str = "2020-01-01") -> Path:
    dates = pd.date_range(start, periods=rows, freq=freq)
    frame = pd.DataFrame({"DATE_TIME": dates, "CLOSE": range(rows)})
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    return path


def _synthetic_contract(tmp_path: Path, source: Path, **overrides) -> dict:
    import hashlib

    total = len(pd.read_csv(source))
    dates = pd.read_csv(source)["DATE_TIME"]
    contract = {
        "schema": ns.SCHEMA,
        "source_csv": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "date_column": "DATE_TIME",
        "window_size": 8,
        "scaling_window": 16,
        "max_feature_lookback": 0,
        "roles": {
            "fit_train": {"start": dates.iloc[0], "end": dates.iloc[400]},
            "train_monitor": {"start": dates.iloc[300],
                              "end": dates.iloc[400]},
            "inner_validation": {"start": dates.iloc[400],
                                 "end": dates.iloc[500]},
            "outer_validation": {"start": dates.iloc[500],
                                 "end": dates.iloc[600]},
            "sealed_test": {"start": dates.iloc[600],
                            "end": str(pd.Timestamp(dates.iloc[-1])
                                       + pd.Timedelta(hours=4))},
        },
    }
    contract.update(overrides)
    return contract


class TestExactEthAssertions:
    @needs_eth
    def test_eth_rows_and_dates_derive_exactly(self, tmp_path):
        contract = ns.load_contract(CONTRACT_PATH)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path, mode="l1")
        roles = manifest["roles"]
        assert roles["fit_train"]["scored_rows"] == 11509
        assert roles["train_monitor"]["scored_rows"] == 2190
        assert roles["inner_validation"]["scored_rows"] == 2190
        assert roles["outer_validation"]["scored_rows"] == 2196
        assert roles["sealed_test"]["status"] == "SEALED"
        assert manifest["context_bars"] == 256
        # full calendar-year score intervals intact (class 6)
        assert roles["inner_validation"]["score_start"].startswith("2023-01-01")
        assert roles["inner_validation"]["score_end"].startswith("2023-12-31")
        assert roles["outer_validation"]["score_start"].startswith("2024-01-01")
        assert roles["outer_validation"]["score_end"].startswith("2024-12-31")
        # every evaluation split carries the full causal prefix (class 5)
        for role in ("train_monitor", "inner_validation",
                     "outer_validation"):
            assert roles[role]["context_rows"] == 256

    @needs_eth
    def test_one_row_boundary_mutation_fails(self, tmp_path):
        contract = ns.load_contract(CONTRACT_PATH)
        contract["expected_rows"] = dict(contract["expected_rows"])
        contract["expected_rows"]["inner_validation"] = 2189
        with pytest.raises(ns.NestedSplitError, match="derived 2190"):
            ns.materialize_nested_splits(contract, tmp_path, mode="l1")


class TestStructuralRefusals:
    def test_overlap_fails(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        dates = pd.read_csv(source)["DATE_TIME"]
        contract["roles"]["inner_validation"]["start"] = dates.iloc[390]
        with pytest.raises(ns.NestedSplitError, match="gap or overlap"):
            ns.materialize_nested_splits(contract, tmp_path / "o",
                                         mode="l1")

    def test_gap_fails(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        dates = pd.read_csv(source)["DATE_TIME"]
        contract["roles"]["inner_validation"]["start"] = dates.iloc[410]
        with pytest.raises(ns.NestedSplitError, match="gap or overlap"):
            ns.materialize_nested_splits(contract, tmp_path / "g",
                                         mode="l1")

    def test_reordered_source_fails(self, tmp_path):
        source = _synthetic(tmp_path)
        frame = pd.read_csv(source)
        frame = frame.iloc[::-1]
        frame.to_csv(source, index=False)
        contract = _synthetic_contract(tmp_path, source)
        with pytest.raises(ns.NestedSplitError, match="chronologically"):
            ns.materialize_nested_splits(contract, tmp_path / "r",
                                         mode="l1")

    def test_monitor_outside_fit_fails(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        dates = pd.read_csv(source)["DATE_TIME"]
        contract["roles"]["train_monitor"]["end"] = dates.iloc[450]
        with pytest.raises(ns.NestedSplitError, match="subset"):
            ns.materialize_nested_splits(contract, tmp_path / "m",
                                         mode="l1")

    def test_source_hash_mismatch_fails(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source,
                                       source_sha256="0" * 64)
        with pytest.raises(ns.NestedSplitError, match="source sha"):
            ns.materialize_nested_splits(contract, tmp_path / "h",
                                         mode="l1")


class TestSealedTest:
    @pytest.mark.parametrize("mode", ["l1", "l2", "train"])
    def test_sealed_in_every_development_mode(self, tmp_path, mode):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path / mode, mode=mode)
        entry = manifest["roles"]["sealed_test"]
        assert entry["status"] == "SEALED"
        assert not (tmp_path / mode / "sealed_test.csv").exists()

    def test_release_mode_materializes_test(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path / "rel", mode="release")
        assert manifest["roles"]["sealed_test"]["status"] == "MATERIALIZED"


class TestContextSemantics:
    def test_context_recomputes_with_windows(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source, scaling_window=32)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path / "w", mode="l1")
        assert manifest["context_bars"] == 32
        contract = _synthetic_contract(tmp_path, source, window_size=64,
                                       scaling_window=16)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path / "w2", mode="l1")
        assert manifest["context_bars"] == 64

    def test_insufficient_context_refuses(self, tmp_path):
        source = _synthetic(tmp_path, rows=700)
        contract = _synthetic_contract(tmp_path, source,
                                       scaling_window=1000)
        with pytest.raises(ns.NestedSplitError, match="context rows"):
            ns.materialize_nested_splits(contract, tmp_path / "i",
                                         mode="l1")

    def test_prefix_wrapper_forces_holds_and_tags(self):
        class FakeEnv:
            def __init__(self):
                self.actions = []
                self.equity = 1000.0

            def reset(self, **kwargs):
                return "obs", {"equity": self.equity}

            def step(self, action):
                self.actions.append(action)
                return "obs", 0.0, False, False, {
                    "equity": self.equity, "trades": 0}

        env = FakeEnv()
        wrapped = ns.ContextPrefixWrapper(env, context_rows=3)
        _, info = wrapped.reset()
        assert info["is_context_prefix"] is True
        for i in range(5):
            _, _, _, _, info = wrapped.step(0.9)
            assert info["is_context_prefix"] == (i < 3)
        assert env.actions[:3] == [0.0, 0.0, 0.0]     # forced holds
        assert env.actions[3:] == [0.9, 0.9]          # agent acts after

    def test_prefix_wrapper_is_accepted_as_gymnasium_env(self):
        """The selected model is loaded against this wrapper only after
        hours of training, so Stable-Baselines must recognize the wrapper
        as Gymnasium before a decision run starts."""
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3.common.vec_env.patch_gym import _patch_env

        class GymEnv(gym.Env):
            observation_space = spaces.Box(-1.0, 1.0, shape=(1,))
            action_space = spaces.Box(-1.0, 1.0, shape=(1,))

            def reset(self, **kwargs):
                return self.observation_space.sample(), {
                    "equity": 1000.0, "trades": 0}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {
                    "equity": 1000.0, "trades": 0}

        wrapped = ns.ContextPrefixWrapper(GymEnv(), context_rows=1)
        assert isinstance(wrapped, gym.Env)
        assert _patch_env(wrapped) is wrapped

    def test_prefix_wrapper_refuses_account_mutation(self):
        class MutatingEnv:
            def reset(self, **kwargs):
                return "obs", {"equity": 1000.0}

            def step(self, action):
                return "obs", 0.0, False, False, {
                    "equity": 999.0, "trades": 0}

        wrapped = ns.ContextPrefixWrapper(MutatingEnv(), context_rows=2)
        wrapped.reset()
        with pytest.raises(ns.NestedSplitError, match="equity changed"):
            wrapped.step(0.5)

    def test_prefix_wrapper_refuses_prefix_trades(self):
        class TradingEnv:
            def reset(self, **kwargs):
                return "obs", {"equity": 1000.0}

            def step(self, action):
                return "obs", 0.0, False, False, {
                    "equity": 1000.0, "trades": 1}

        wrapped = ns.ContextPrefixWrapper(TradingEnv(), context_rows=2)
        wrapped.reset()
        with pytest.raises(ns.NestedSplitError, match="trades occurred"):
            wrapped.step(0.5)

    def test_prefix_wrapper_counts_both_populations(self):
        """Finding 231 §5: the wrapper reports what it separated so a
        caller can PROVE the score boundary was reached."""
        class FlatEnv:
            def reset(self, **kwargs):
                return "obs", {"equity": 1000.0, "position": 0,
                               "trades": 0}

            def step(self, action):
                return "obs", 1.0, False, False, {
                    "equity": 1000.0, "position": 0, "trades": 0}

        wrapped = ns.ContextPrefixWrapper(FlatEnv(), context_rows=3)
        wrapped.reset()
        for _ in range(5):
            wrapped.step(0.9)
        assert wrapped.context_prefix_steps == 3
        assert wrapped.scored_steps == 2
        wrapped.reset()
        assert (wrapped.context_prefix_steps, wrapped.scored_steps) == (0, 0)

    @pytest.mark.parametrize("key,value", [
        ("position", 1),
        ("position_units", 0.5),
        ("open_order_count", 1),
        ("commission_paid", 0.02),
    ])
    def test_prefix_wrapper_refuses_orders_and_positions(self, key, value):
        """A prefix row may not open an order or a position even when
        equity has not settled yet and no trade has closed."""
        class MutatingEnv:
            def reset(self, **kwargs):
                return "obs", {"equity": 1000.0, "position": 0,
                               "position_units": 0.0,
                               "open_order_count": 0,
                               "commission_paid": 0.0, "trades": 0}

            def step(self, action):
                info = {"equity": 1000.0, "position": 0,
                        "position_units": 0.0, "open_order_count": 0,
                        "commission_paid": 0.0, "trades": 0}
                info[key] = value
                return "obs", 0.0, False, False, info

        wrapped = ns.ContextPrefixWrapper(MutatingEnv(), context_rows=2)
        wrapped.reset()
        with pytest.raises(ns.NestedSplitError,
                           match="account state changed"):
            wrapped.step(0.5)

    def test_prefix_wrapper_allows_unchanged_account_facts(self):
        class SteadyEnv:
            def reset(self, **kwargs):
                return "obs", {"equity": 1000.0, "position": 1,
                               "position_units": 2.0,
                               "open_order_count": 0,
                               "commission_paid": 3.0, "trades": 0}

            def step(self, action):
                return "obs", 0.0, False, False, {
                    "equity": 1000.0, "position": 1,
                    "position_units": 2.0, "open_order_count": 0,
                    "commission_paid": 3.0, "trades": 0}

        wrapped = ns.ContextPrefixWrapper(SteadyEnv(), context_rows=2)
        wrapped.reset()
        wrapped.step(0.5)
        wrapped.step(0.5)
        assert wrapped.context_prefix_steps == 2


class TestManifestFailClosed:
    def test_manifest_verifies_and_detects_drift(self, tmp_path):
        source = _synthetic(tmp_path)
        contract = _synthetic_contract(tmp_path, source)
        manifest = ns.materialize_nested_splits(
            contract, tmp_path / "v", mode="l1")
        ns.verify_split_manifest(manifest["manifest_path"])
        drifted = tmp_path / "v" / "inner_validation.csv"
        drifted.write_text(drifted.read_text() + "tamper\n")
        with pytest.raises(ns.NestedSplitError, match="hash drift"):
            ns.verify_split_manifest(manifest["manifest_path"])

    def test_cross_contract_identity_refuses(self, tmp_path):
        source = _synthetic(tmp_path)
        a = ns.materialize_nested_splits(
            _synthetic_contract(tmp_path, source), tmp_path / "a",
            mode="l1")
        b = ns.materialize_nested_splits(
            _synthetic_contract(tmp_path, source, scaling_window=32),
            tmp_path / "b", mode="l1")
        with pytest.raises(ns.NestedSplitError, match="identity mismatch"):
            ns.assert_same_split_identity(a, b)


class TestPipelineHook:
    @needs_eth
    def test_pipeline_split_csv_delegates_to_nested_contract(self, tmp_path):
        from pipeline_plugins.rl_pipeline_with_validation import (
            PipelinePlugin,
        )

        pipeline = PipelinePlugin({})
        config = {
            "nested_split_contract": str(CONTRACT_PATH),
            "nested_split_dir": str(tmp_path / "splits"),
            "save_model": str(tmp_path / "model.zip"),
            "selection_metric": "paired_generalization_weekly_v1",
        }
        paths = pipeline._split_csv(config)
        assert set(paths) == {"train", "train_monitor", "validation",
                              "outer_validation", "train_tail", "val"}
        assert "test" not in paths          # sealed outside release mode
        # loop-compat aliases pair monitor (in-sample) with inner (val)
        assert paths["train_tail"] == paths["train_monitor"]
        assert paths["val"] == paths["validation"]
        assert Path(paths["validation"]).is_file()
        assert config["nested_split_manifest"]

    @needs_eth
    def test_pipeline_resolves_eth_roles_from_the_verified_manifest(
        self, tmp_path
    ):
        """AUD-F1-20260812-231 §1: the executing selector learns the
        role and its context_rows from the verified manifest — 256
        context rows before 2,190 scored rows on BOTH internal
        evaluation roles — never from a file name or a row position."""
        from pipeline_plugins.rl_pipeline_with_validation import (
            PipelinePlugin,
        )

        pipeline = PipelinePlugin({})
        config = {
            "nested_split_contract": str(CONTRACT_PATH),
            "nested_split_dir": str(tmp_path / "splits"),
            "save_model": str(tmp_path / "model.zip"),
            "selection_metric": "paired_generalization_weekly_v1",
        }
        paths = pipeline._split_csv(config)
        for alias, expected_role in (("train_tail", "train_monitor"),
                                     ("val", "inner_validation")):
            role, entry = pipeline._resolve_nested_role(
                config, paths[alias])
            assert role == expected_role
            assert entry["context_rows"] == 256
            assert entry["scored_rows"] == 2190
        role, entry = pipeline._resolve_nested_role(config, paths["train"])
        assert (role, entry["context_rows"]) == ("fit_train", 0)

    @needs_eth
    def test_nested_config_refuses_legacy_validation_only_metric(
        self, tmp_path
    ):
        from pipeline_plugins.rl_pipeline_with_validation import (
            PipelinePlugin,
        )

        pipeline = PipelinePlugin({})
        config = {
            "nested_split_contract": str(CONTRACT_PATH),
            "nested_split_dir": str(tmp_path / "splits"),
            "save_model": str(tmp_path / "model.zip"),
            "selection_metric": "lexicographic_weekly_v1",
        }
        with pytest.raises(ValueError, match="validation-only branch"):
            pipeline._split_csv(config)
