from __future__ import annotations

import pytest

from optimizer_plugins.project3_full_genome_optimizer import Plugin


def _config():
    return {
        "mixed_genome_schema": [
            {
                "name": "scaling",
                "kind": "categorical",
                "choices": ["none", "rolling_zscore"],
                "target": "feature_scaling",
            },
            {
                "name": "learning_rate_gene",
                "kind": "log_float",
                "low": 1e-5,
                "high": 1e-3,
                "target": "learning_rate",
            },
            {
                "name": "window",
                "kind": "int",
                "low": 24,
                "high": 168,
                "target": "window_size",
            },
            {
                "name": "architecture",
                "kind": "categorical",
                "choices": ["small", "large"],
                "choice_patches": {
                    "small": {"net_arch": [128, 128]},
                    "large": {"net_arch": [512, 256]},
                },
            },
            {
                "name": "feature_group__price",
                "kind": "boolean",
                "target": "_feature_group_price",
            },
            {
                "name": "feature_group__volume",
                "kind": "boolean",
                "target": "_feature_group_volume",
            },
        ],
        "mixed_genome_feature_groups": {
            "price": ["return_1", "atr_14"],
            "volume": ["volume_ratio_20"],
        },
        "mixed_genome_required_feature_group": "price",
        "mixed_genome_max_observation_elements": 200,
        "mixed_genome_max_replay_observation_values": 50_688,
        "buffer_size": 1_000,
        "batch_size": 128,
        "learning_starts": 500,
        "l1_min_checkpoint_timesteps": 10,
        "mixed_genome_repair_rules": [
            {
                "if": {"architecture": "large"},
                "set": {"batch_size": 256},
            }
        ],
        "initial_candidate_decoded": {
            "scaling": "rolling_zscore",
            "learning_rate_gene": 0.0001,
            "window": 72,
            "architecture": "large",
            "feature_group__price": True,
            "feature_group__volume": False,
        },
    }


def test_mixed_genome_decodes_categories_log_values_and_feature_groups():
    plugin = Plugin(_config())
    config = _config()
    schema = plugin._effective_schema([], config)
    encoded = plugin._initial_params(schema, config)
    run = plugin._candidate_run_config(encoded, config)

    assert run["feature_scaling"] == "rolling_zscore"
    assert run["learning_rate"] == pytest.approx(0.0001)
    assert run["window_size"] == 72
    assert run["net_arch"] == [512, 256]
    assert run["batch_size"] == 256
    assert run["feature_columns"] == ["return_1", "atr_14"]
    assert run["feature_list"] == ["return_1", "atr_14"]
    assert run["_mixed_genome_decoded"]["feature_group__volume"] is False


def test_mixed_genome_repairs_all_feature_groups_disabled():
    plugin = Plugin(_config())
    config = _config()
    schema = plugin._effective_schema([], config)
    encoded = plugin._initial_params(schema, config)
    encoded["feature_group__price"] = 0
    encoded["feature_group__volume"] = 0

    run = plugin._candidate_run_config(encoded, config)

    assert run["feature_columns"] == ["return_1", "atr_14"]
    assert run["_mixed_genome_decoded"]["feature_group__price"] is True


def test_mixed_genome_repairs_observation_and_replay_memory_limits():
    plugin = Plugin(_config())
    config = _config()
    schema = plugin._effective_schema([], config)
    encoded = plugin._initial_params(schema, config)
    encoded["window"] = 168
    encoded["feature_group__volume"] = 1
    run = plugin._candidate_run_config(encoded, config)

    assert len(run["feature_columns"]) == 3
    assert run["window_size"] == 66
    assert run["buffer_size"] == 256
    assert [item["field"] for item in run["_mixed_genome_decoded"]["_repairs"]] == [
        "l1_min_checkpoint_timesteps",
        "window_size",
        "buffer_size",
    ]


def test_shared_population_uses_mixed_schema_instead_of_agent_schema():
    class Agent:
        @staticmethod
        def hparam_schema():
            return [("ignored", 0.0, 1.0, "float")]

    plugin = Plugin(_config())
    plugin.setup_shared_mode(
        env_plugin=object(),
        agent_plugin=Agent(),
        pipeline_plugin=object(),
        config=_config(),
    )
    state = plugin.create_shared_population(3, seed=7)

    assert len(state["population"]) == 3
    assert state["innovation_tracker"]["parameter_names"] == [
        "scaling",
        "learning_rate_gene",
        "window",
        "architecture",
        "feature_group__price",
        "feature_group__volume",
    ]


def test_unknown_initial_decoded_gene_fails_closed():
    config = _config()
    config["initial_candidate_decoded"]["unknown"] = 1
    plugin = Plugin(config)
    schema = plugin._effective_schema([], config)

    with pytest.raises(ValueError, match="unknown genes"):
        plugin._initial_params(schema, config)


def test_resolve_best_config_decodes_the_winning_chromosome():
    config = _config()
    plugin = Plugin(config)
    schema = plugin._effective_schema([], config)
    encoded = plugin._initial_params(schema, config)

    resolved = plugin.resolve_best_config(
        {**encoded, "_best_fitness": 0.5},
        config,
    )

    assert resolved["feature_scaling"] == "rolling_zscore"
    assert resolved["window_size"] == 72
    assert resolved["feature_columns"] == ["return_1", "atr_14"]
