"""RL08: warehouse results bind task, data, representation, seed, algorithm,
actions, reward, evaluation population, costs and measured resource use."""
from __future__ import annotations

import json

import pytest

from ._mechanisms import assert_checkout_resolution, require


def _full_record():
    return {
        "arm": "RL-D1", "algorithm": "DQN", "representation": "modular_temporal",
        "task": {"dataset_id": "fixture", "selected_feature_manifest_sha256": "a" * 64,
                 "manifest_status": "FROZEN"},
        "data": {"model_ready_view_sha256": "b" * 64, "train_rows": [0, 300],
                 "validation_rows": [300, 350], "test_rows": None},
        "representation_identity": {"modular_config_sha256": "c" * 64, "layout_digest": "d" * 64,
                                    "regimes": {"branch_0": "R0", "core": "R0"},
                                    "donor_identity": None, "parameter_count": 1234},
        "seed": 0, "paired_seed_group": "s0",
        "actions": {"action_space_mode": "discrete", "mapping": {"0": "hold", "1": "long", "2": "short"}},
        "reward": {"plugin": "pnl_reward", "frozen_before_fit": True},
        "evaluation_population": {"split": "validation", "episodes": 1, "rows": [300, 350],
                                  "selection_metric": "net_return"},
        "costs": {"commission": 0.001, "slippage": 0.0, "financing_enabled": False},
        "metrics": {"net_return": 0.0, "max_drawdown_fraction": 0.0,
                    "sharpe": {"value": None, "convention": "per-bar, not annualized", "undefined_reason": "zero variance"},
                    "turnover_units": 0.0, "trades_closed": 0, "exposure_fraction": 0.0},
        "baselines": {"no_trade": {"net_return": 0.0}, "heuristic": "UNAVAILABLE"},
        "resources": {"host_alias": "worker_b", "device": "cpu", "wall_s": 1.0,
                      "peak_rss_bytes": 1, "gradient_updates": 1, "pretraining_cost": {"state": "NONE"}},
        "versions": {"stable_baselines3": "2.9.0", "torch": "x", "engine_pin": "3ecdb256"},
        "status": "FIXTURE_NOT_A_RESULT",
    }


def test_full_record_validates_and_digests_stably():
    assert_checkout_resolution("RL08")
    validate = require("RL08", "rl_temporal.warehouse_binding", "validate_result_record",
                       "result record validator")
    digest = require("RL08", "rl_temporal.warehouse_binding", "record_digest", "record digest")
    record = _full_record()
    doc = validate(record)
    assert doc["schema"] == "rl_temporal_result.v1"
    assert digest(record) == digest(json.loads(json.dumps(record)))


@pytest.mark.parametrize("missing", ["seed", "actions", "reward", "evaluation_population",
                                     "costs", "resources", "representation_identity", "task", "data"])
def test_missing_binding_is_refused_by_name(missing):
    validate = require("RL08", "rl_temporal.warehouse_binding", "validate_result_record",
                       "result record validator")
    record = _full_record()
    record.pop(missing)
    with pytest.raises(ValueError, match=missing):
        validate(record)


def test_draft_manifest_cannot_label_a_result():
    validate = require("RL08", "rl_temporal.warehouse_binding", "validate_result_record",
                       "result record validator")
    record = _full_record()
    record["task"]["manifest_status"] = "DRAFT_NOT_FROZEN"
    record["status"] = "RESULT"
    with pytest.raises(ValueError, match="DRAFT_NOT_FROZEN"):
        validate(record)


def test_heuristic_baseline_without_gate_pass_is_unavailable():
    validate = require("RL08", "rl_temporal.warehouse_binding", "validate_result_record",
                       "result record validator")
    record = _full_record()
    record["baselines"]["heuristic"] = {"net_return": 0.1}
    with pytest.raises(ValueError, match="naive_gate"):
        validate(record)
