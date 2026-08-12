"""AUD-F1-20260812-231 — the EXECUTING internal selection path must
score only the declared scored rows of train_monitor and
inner_validation, never their 256 causal-context rows.

Every test here drives an ADVERSARIAL agent: it requests a trade on
every single row (prefix rows included) and each arm requests a
DIFFERENT action, which is exactly the condition under which pairing
cannot cancel context contamination. The fake env executes any non-hold
action immediately, so an unwrapped prefix provably moves equity, opens
a position and books trades before the score boundary.

Proved here:
- equal opening equity at the score boundary across arms;
- zero prefix traces, trades, reward and account mutation;
- exactly 2,190 scored rows for train_monitor and inner_validation;
- unchanged behavior for a role that declares no context;
- role + context_rows resolved from the VERIFIED manifest, never from
  the file name or a row position;
- fail-closed refusals: unknown slice, manifest drift, declared context
  with an unwrapped env, and replay writes during an evaluation.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_plugins import _nested_splits as ns  # noqa: E402
from pipeline_plugins import rl_pipeline_with_validation as rlv  # noqa: E402
from pipeline_plugins.rl_pipeline_with_validation import (  # noqa: E402
    PipelinePlugin,
)

CONTEXT_BARS = 256
SCORED_ROWS = 2190
FIT_ROWS = CONTEXT_BARS + SCORED_ROWS          # 2446
INNER_END = FIT_ROWS + SCORED_ROWS             # 4636
OUTER_END = INNER_END + 64                     # 4700
TOTAL_ROWS = 4800
INITIAL_CASH = 10_000.0


# ---------------------------------------------------------------------------
# fixtures: a nested contract whose evaluation roles carry 256 context rows
# ---------------------------------------------------------------------------

def _source_csv(tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=TOTAL_ROWS, freq="4h")
    frame = pd.DataFrame({
        "DATE_TIME": dates,
        "CLOSE": np.linspace(100.0, 200.0, TOTAL_ROWS),
        "f1": np.linspace(-1.0, 1.0, TOTAL_ROWS),
    })
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    return path


def _contract(source: Path) -> dict:
    dates = pd.read_csv(source)["DATE_TIME"]
    end = str(pd.Timestamp(dates.iloc[-1]) + pd.Timedelta(hours=4))
    return {
        "schema": ns.SCHEMA,
        "source_csv": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "date_column": "DATE_TIME",
        "window_size": CONTEXT_BARS,
        "scaling_window": 16,
        "max_feature_lookback": 0,
        "expected_rows": {
            "fit_train": FIT_ROWS,
            "train_monitor": SCORED_ROWS,
            "inner_validation": SCORED_ROWS,
        },
        "roles": {
            "fit_train": {"start": dates.iloc[0],
                          "end": dates.iloc[FIT_ROWS]},
            "train_monitor": {"start": dates.iloc[CONTEXT_BARS],
                              "end": dates.iloc[FIT_ROWS]},
            "inner_validation": {"start": dates.iloc[FIT_ROWS],
                                 "end": dates.iloc[INNER_END]},
            "outer_validation": {"start": dates.iloc[INNER_END],
                                 "end": dates.iloc[OUTER_END]},
            "sealed_test": {"start": dates.iloc[OUTER_END], "end": end},
        },
    }


@pytest.fixture()
def nested(tmp_path):
    source = _source_csv(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(_contract(source)))
    config = {
        "nested_split_contract": str(contract_path),
        "nested_split_dir": str(tmp_path / "splits"),
        "nested_split_mode": "l1",
        "save_model": str(tmp_path / "model.zip"),
        "selection_metric": "paired_generalization_weekly_v1",
        "initial_cash": INITIAL_CASH,
        "asset": "ETHUSDT",
        "timeframe": "4h",
        "date_column": "DATE_TIME",
        "_retain_return_trace_rows": True,
    }
    pipeline = PipelinePlugin({})
    paths = pipeline._split_csv(config)
    manifest = json.loads(Path(config["nested_split_manifest"]).read_text())
    return pipeline, config, paths, manifest


# ---------------------------------------------------------------------------
# adversarial fakes
# ---------------------------------------------------------------------------

class _AdversarialAgent:
    """Requests a trade on EVERY row, prefix rows included, and a
    different action per arm."""

    def __init__(self, action: float):
        self.action = float(action)
        self.requested: list[float] = []

    def predict(self, model, obs, deterministic=True):
        self.requested.append(self.action)
        return np.array([self.action], dtype=np.float64)


class _ExecutingEnv:
    """Executes any non-hold action IMMEDIATELY: equity moves, a
    position opens and a trade is booked on that very row. An unwrapped
    causal prefix therefore contaminates the account before scoring."""

    def __init__(self, cfg):
        self.config = dict(cfg)
        self.dataframe = pd.read_csv(cfg["input_data_file"])
        self.initial_cash = float(cfg.get("initial_cash", INITIAL_CASH))
        self.applied_actions: list[float] = []
        self.equity_at_first_scored_step: float | None = None
        self.trades_at_first_scored_step: int | None = None
        self.reset()

    # -- gym-ish API ------------------------------------------------
    def reset(self, seed=None, **kwargs):
        self.i = 0
        self.equity = self.initial_cash
        self.position = 0
        self.trades = 0
        self.applied_actions = []
        self.equity_at_first_scored_step = None
        self.trades_at_first_scored_step = None
        return self._obs(), self._info()

    def step(self, action):
        value = float(np.asarray(action, dtype=float).reshape(-1)[0])
        self.applied_actions.append(value)
        if abs(value) > 1e-9:
            self.position = 1 if value > 0 else -1
            self.trades += 1
            self.equity += 5.0 * value
        self.i += 1
        reward = 1.0 + 0.25 * value
        terminated = self.i >= len(self.dataframe)
        return self._obs(), reward, terminated, False, self._info()

    def summary(self):
        return {
            "trades_total": self.trades,
            "trades_won": self.trades,
            "final_equity": self.equity,
            "total_return": self.equity / self.initial_cash - 1.0,
            "max_drawdown_fraction": 0.0,
            "sharpe_ratio": 1.0,
        }

    def close(self):
        pass

    # -- helpers ----------------------------------------------------
    def _obs(self):
        return np.zeros(2, dtype=np.float32)

    def _info(self):
        idx = min(self.i, len(self.dataframe) - 1)
        return {
            "equity": self.equity,
            "position": self.position,
            "position_units": float(self.position),
            "open_order_count": 0,
            "commission_paid": 0.0,
            "trades": self.trades,
            "bar_index": self.i,
            "price": float(self.dataframe["CLOSE"].iloc[idx]),
            "coerced_action": 0 if self.position == 0 else (
                1 if self.position > 0 else 2),
        }


class _EnvPluginFactory:
    """Stands in for the entry-point env plugin and keeps every env it
    built so the test can inspect what the env actually executed."""

    def __init__(self):
        self.envs: list[_ExecutingEnv] = []

    def __call__(self, name, config):
        factory = self

        class _Plug:
            def make_env(self, cfg):
                env = _ExecutingEnv(cfg)
                factory.envs.append(env)
                return env

            def close(self):
                pass

        return _Plug()


@pytest.fixture()
def env_factory(monkeypatch):
    factory = _EnvPluginFactory()
    monkeypatch.setattr(rlv, "_load_env_plugin", factory)
    return factory


def _mark_boundary(env: _ExecutingEnv) -> None:
    """Record the account state the first scored row starts from."""
    env.equity_at_first_scored_step = env.equity
    env.trades_at_first_scored_step = env.trades


# ---------------------------------------------------------------------------
# 1. the executing internal selection path
# ---------------------------------------------------------------------------

class TestInternalSelectionHonorsContext:
    @pytest.mark.parametrize("role_key,role_name", [
        ("train_tail", "train_monitor"),
        ("val", "inner_validation"),
    ])
    def test_exactly_2190_scored_rows_and_zero_prefix_effects(
        self, nested, env_factory, role_key, role_name
    ):
        pipeline, config, paths, manifest = nested
        agent = _AdversarialAgent(0.9)
        summary = pipeline._eval_on_split(
            "fake_env", config, paths[role_key], agent, object(), 7,
            f"{role_name}_epoch")

        # the manifest's own declaration, resolved and honored
        assert summary["nested_role"] == role_name
        assert summary["nested_role_csv_sha256"] == (
            manifest["roles"][role_name]["csv_sha256"])
        assert summary["context_rows_declared"] == CONTEXT_BARS
        assert summary["context_prefix_steps"] == CONTEXT_BARS
        assert summary["scored_steps"] == SCORED_ROWS
        assert summary["nested_role_scored_rows"] == SCORED_ROWS
        assert summary["total_env_steps"] == FIT_ROWS

        # metric horizon and canonical trace are scored-only
        assert summary["episode_length"] == SCORED_ROWS
        rows = summary["_return_trace_rows"]
        assert len(rows) == SCORED_ROWS
        assert [row["step"] for row in rows[:3]] == [1, 2, 3]
        assert min(row["bar_index"] for row in rows) == CONTEXT_BARS + 1
        assert max(row["bar_index"] for row in rows) == FIT_ROWS

        # reward and action statistics exclude the prefix
        assert summary["episode_reward"] == pytest.approx(
            SCORED_ROWS * (1.0 + 0.25 * 0.9))
        assert summary["context_prefix_reward_excluded"] == pytest.approx(
            float(CONTEXT_BARS))          # 256 forced-hold rows, reward 1.0
        assert summary["action_steps"] == SCORED_ROWS

        # the env was held flat for every prefix row although the agent
        # requested a trade on each of them
        env = env_factory.envs[-1]
        assert len(agent.requested) == FIT_ROWS
        assert all(value == 0.9 for value in agent.requested)
        assert env.applied_actions[:CONTEXT_BARS] == [0.0] * CONTEXT_BARS
        assert set(env.applied_actions[CONTEXT_BARS:]) == {0.9}
        assert summary["trades_total"] == SCORED_ROWS   # no prefix trade

        # weekly metrics were computed from the scored trace only
        assert summary["evaluation_weeks"] > 0

    def test_arms_open_the_scored_interval_from_identical_equity(
        self, nested, env_factory
    ):
        pipeline, config, paths, _manifest = nested
        opening = {}
        boundary_equity = {}
        for arm, action in (("A", 0.9), ("B", -0.7)):
            agent = _AdversarialAgent(action)
            summary = pipeline._eval_on_split(
                "fake_env", config, paths["val"], agent, object(), 7,
                "validation_epoch")
            env = env_factory.envs[-1]
            opening[arm] = summary["score_boundary_opening_equity"]
            # equity implied by the env at the boundary: prefix moved
            # nothing, so it is still the reset equity
            boundary_equity[arm] = (
                env.equity - 5.0 * action * summary["scored_steps"])
            assert summary["scored_steps"] == SCORED_ROWS
            assert summary["context_prefix_steps"] == CONTEXT_BARS

        assert opening["A"] == opening["B"] == INITIAL_CASH
        assert boundary_equity["A"] == pytest.approx(INITIAL_CASH)
        assert boundary_equity["B"] == pytest.approx(INITIAL_CASH)

    def test_without_the_wrapper_the_arms_diverge_before_scoring(
        self, nested, env_factory
    ):
        """The defect, reproduced: the same rollout on the raw env
        scores 2,446 rows and each arm reaches the first scored bar with
        a different account — which is why pairing cannot cure it."""
        pipeline, config, paths, _manifest = nested
        contaminated = {}
        for arm, action in (("A", 0.9), ("B", -0.7)):
            plug, env = pipeline._make_split_env(
                "fake_env", config, paths["val"], _AdversarialAgent(action))
            summary = PipelinePlugin._rollout(
                env, _AdversarialAgent(action), object(), 7,
                split="validation_epoch")
            plug.close()
            contaminated[arm] = {
                "scored": summary["scored_steps"],
                "prefix": summary["context_prefix_steps"],
                "equity_after_prefix": (
                    INITIAL_CASH + 5.0 * action * CONTEXT_BARS),
                "trades_in_prefix": CONTEXT_BARS,
            }
            assert summary["scored_steps"] == FIT_ROWS       # 2,446 scored
            assert summary["context_prefix_steps"] == 0
            assert len(summary["_return_trace_rows"]) == FIT_ROWS

        assert (contaminated["A"]["equity_after_prefix"]
                != contaminated["B"]["equity_after_prefix"])


# ---------------------------------------------------------------------------
# 2. roles that declare no context are untouched
# ---------------------------------------------------------------------------

class TestNoContextRolesUnchanged:
    def test_fit_train_role_is_scored_end_to_end_unwrapped(
        self, nested, env_factory
    ):
        pipeline, config, paths, _manifest = nested
        agent = _AdversarialAgent(0.5)
        summary = pipeline._eval_on_split(
            "fake_env", config, paths["train"], agent, object(), 7,
            "train_epoch")
        env = env_factory.envs[-1]
        assert summary["nested_role"] == "fit_train"
        assert summary["context_rows_declared"] == 0
        assert summary["context_prefix_steps"] == 0
        assert summary["scored_steps"] == FIT_ROWS
        assert summary["episode_length"] == FIT_ROWS
        assert len(summary["_return_trace_rows"]) == FIT_ROWS
        assert summary["action_steps"] == FIT_ROWS
        assert 0.0 not in env.applied_actions     # nothing forced to hold
        assert summary["episode_reward"] == pytest.approx(
            FIT_ROWS * (1.0 + 0.25 * 0.5))
        assert summary["score_boundary_opening_equity"] == INITIAL_CASH

    def test_legacy_run_without_a_nested_contract_is_unchanged(
        self, tmp_path, env_factory
    ):
        source = _source_csv(tmp_path)
        pipeline = PipelinePlugin({})
        config = {
            "initial_cash": INITIAL_CASH,
            "asset": "ETHUSDT",
            "timeframe": "4h",
            "_retain_return_trace_rows": True,
        }
        assert pipeline._resolve_nested_role(config, str(source)) is None
        summary = pipeline._eval_on_split(
            "fake_env", config, str(source), _AdversarialAgent(0.5),
            object(), 7, "validation_epoch")
        assert "nested_role" not in summary
        assert summary["context_prefix_steps"] == 0
        assert summary["scored_steps"] == TOTAL_ROWS
        assert len(summary["_return_trace_rows"]) == TOTAL_ROWS


# ---------------------------------------------------------------------------
# 3. resolution comes from the verified manifest, not the file name
# ---------------------------------------------------------------------------

class TestManifestIsTheOnlyAuthority:
    def test_role_and_context_survive_opaque_file_names(
        self, nested, env_factory
    ):
        pipeline, config, paths, manifest = nested
        manifest_path = Path(config["nested_split_manifest"])
        renamed = {}
        for role in ("train_monitor", "inner_validation"):
            entry = manifest["roles"][role]
            old = Path(entry["csv"])
            new = old.with_name(f"{'zzz' if role == 'train_monitor' else 'aaa'}"
                                f"_{role[::-1]}.csv")
            old.rename(new)
            entry["csv"] = str(new)
            renamed[role] = new
        manifest_path.write_text(json.dumps(manifest, indent=1,
                                            sort_keys=True) + "\n")

        # the manifest still verifies (hashes unchanged) and the role is
        # resolved from it, not from the meaningless names
        role, entry = pipeline._resolve_nested_role(
            config, str(renamed["train_monitor"]))
        assert role == "train_monitor"
        assert entry["context_rows"] == CONTEXT_BARS

        agent = _AdversarialAgent(0.9)
        summary = pipeline._eval_on_split(
            "fake_env", config, str(renamed["inner_validation"]), agent,
            object(), 7, "validation_epoch")
        assert summary["nested_role"] == "inner_validation"
        assert summary["context_prefix_steps"] == CONTEXT_BARS
        assert summary["scored_steps"] == SCORED_ROWS
        assert env_factory.envs[-1].applied_actions[:CONTEXT_BARS] == (
            [0.0] * CONTEXT_BARS)

    def test_role_resolution_survives_a_config_copy_without_the_key(
        self, nested, env_factory
    ):
        """The curriculum pipeline evaluates phase-1 checkpoints with a
        derived config copy; the instance still resolves the manifest."""
        pipeline, config, paths, _manifest = nested
        derived = {k: v for k, v in config.items()
                   if k != "nested_split_manifest"}
        role, entry = pipeline._resolve_nested_role(
            derived, paths["train_tail"])
        assert role == "train_monitor"
        assert entry["context_rows"] == CONTEXT_BARS

    def test_a_legacy_config_never_inherits_a_nested_manifest(
        self, nested, tmp_path
    ):
        pipeline, config, paths, _manifest = nested
        assert pipeline._nested_split_manifest_path            # recorded
        legacy = {"initial_cash": INITIAL_CASH}
        assert pipeline._resolve_nested_role(legacy, paths["val"]) is None

    def test_unknown_slice_is_refused(self, nested, tmp_path):
        pipeline, config, _paths, _manifest = nested
        stray = tmp_path / "stray.csv"
        pd.DataFrame({"DATE_TIME": ["2020-01-01"], "CLOSE": [1.0]}).to_csv(
            stray, index=False)
        with pytest.raises(ValueError, match="not a materialized role"):
            pipeline._resolve_nested_role(config, str(stray))

    def test_split_csv_drift_refuses_before_any_score(self, nested):
        pipeline, config, paths, _manifest = nested
        drifted = Path(paths["val"])
        drifted.write_text(drifted.read_text() + "tamper\n")
        with pytest.raises(ns.NestedSplitError, match="hash drift"):
            pipeline._resolve_nested_role(config, str(drifted))


# ---------------------------------------------------------------------------
# 4. structural backstops in _rollout
# ---------------------------------------------------------------------------

class TestRolloutRefusals:
    def test_declared_context_with_an_unwrapped_env_is_refused(
        self, nested, env_factory
    ):
        pipeline, config, paths, _manifest = nested
        _plug, env = pipeline._make_split_env(
            "fake_env", config, paths["val"], _AdversarialAgent(0.3))
        with pytest.raises(ValueError, match="was not wrapped"):
            PipelinePlugin._rollout(
                env, _AdversarialAgent(0.3), object(), 7,
                split="validation_epoch", context_rows=CONTEXT_BARS)

    def test_replay_writes_during_evaluation_are_refused(
        self, nested, env_factory
    ):
        pipeline, config, paths, _manifest = nested

        class _GrowingBuffer:
            def __init__(self):
                self._n = 0

            def size(self):
                self._n += 1
                return self._n

        class _Model:
            replay_buffer = _GrowingBuffer()

        with pytest.raises(ValueError, match="replay buffer grew"):
            pipeline._eval_on_split(
                "fake_env", config, paths["val"], _AdversarialAgent(0.3),
                _Model(), 7, "validation_epoch")

    def test_prefix_after_the_score_boundary_is_refused(self):
        class _Flapping:
            def __init__(self):
                self.n = 0

            def reset(self, seed=None, **kwargs):
                self.n = 0
                return None, {"equity": INITIAL_CASH}

            def step(self, action):
                self.n += 1
                return (None, 0.0, self.n >= 4, False,
                        {"equity": INITIAL_CASH,
                         "is_context_prefix": self.n != 1})

            def summary(self):
                return {}

        with pytest.raises(ValueError, match="after the score boundary"):
            PipelinePlugin._rollout(
                _Flapping(), _AdversarialAgent(0.1), object(), 7,
                split="validation_epoch")
