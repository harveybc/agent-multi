"""Mixed categorical/numeric Project 3 optimizer built on the proven DEAP bridge.

The chromosome remains numeric so the existing local and DOIN shared-population
protocols can reproduce, hash, migrate, and deduplicate it unchanged. This
plugin decodes those genes into explicit runtime config patches before each
candidate is trained.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Dict, Iterable

from optimizer_plugins.default_optimizer import (
    Plugin as DefaultOptimizer,
    _ensure_creator,
)


SCHEMA_VERSION = "agent_multi.project3_full_genome.v1"
_KINDS = {"int", "float", "log_float", "categorical", "boolean"}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _schema_hash(schema: Iterable[dict[str, Any]]) -> str:
    payload = _canonical_json(list(schema)).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _set_path(document: dict[str, Any], path: str, value: Any) -> None:
    """Set an existing-or-new dotted runtime path without evaluating code."""
    tokens = [token for token in str(path).split(".") if token]
    if not tokens:
        raise ValueError("mixed genome target path cannot be empty")
    current: dict[str, Any] = document
    for token in tokens[:-1]:
        child = current.get(token)
        if child is None:
            child = {}
            current[token] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"mixed genome target {path!r} crosses non-object key {token!r}"
            )
        current = child
    current[tokens[-1]] = copy.deepcopy(value)


class Plugin(DefaultOptimizer):
    """Decode a versioned mixed genome and reuse the established evaluator."""

    plugin_params = {
        **DefaultOptimizer.plugin_params,
        "mixed_genome_schema": [],
        "mixed_genome_feature_groups": {},
        "mixed_genome_repair_rules": [],
    }

    plugin_debug_vars = [
        *DefaultOptimizer.plugin_debug_vars,
        "mixed_genome_schema",
    ]

    @staticmethod
    def _declared_schema(config: Dict[str, Any]) -> list[dict[str, Any]]:
        raw = config.get("mixed_genome_schema")
        if not isinstance(raw, list) or not raw:
            raise ValueError("mixed_genome_schema must be a non-empty list")
        names: set[str] = set()
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ValueError(f"mixed_genome_schema[{index}] must be an object")
            name = str(item.get("name") or "").strip()
            kind = str(item.get("kind") or "").strip().lower()
            if not name or name in names:
                raise ValueError(f"invalid or duplicate mixed genome name {name!r}")
            if kind not in _KINDS:
                raise ValueError(f"mixed genome gene {name!r} has invalid kind {kind!r}")
            target = str(item.get("target") or "").strip()
            choices = item.get("choices")
            if kind == "categorical":
                if not isinstance(choices, list) or not choices:
                    raise ValueError(
                        f"categorical mixed genome gene {name!r} needs choices"
                    )
            elif kind == "boolean":
                choices = [False, True]
            else:
                low = float(item.get("low"))
                high = float(item.get("high"))
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    raise ValueError(f"mixed genome gene {name!r} has invalid bounds")
                if kind == "log_float" and (low <= 0.0 or high <= 0.0):
                    raise ValueError(f"log_float gene {name!r} bounds must be positive")
            if not target and not item.get("choice_patches"):
                raise ValueError(
                    f"mixed genome gene {name!r} needs target or choice_patches"
                )
            normalized_item = copy.deepcopy(item)
            normalized_item["name"] = name
            normalized_item["kind"] = kind
            if choices is not None:
                normalized_item["choices"] = copy.deepcopy(choices)
            normalized.append(normalized_item)
            names.add(name)
        return normalized

    def _effective_schema(self, raw_schema, config: Dict[str, Any]):
        del raw_schema
        result = []
        for gene in self._declared_schema(config):
            kind = gene["kind"]
            name = gene["name"]
            if kind in {"categorical", "boolean"}:
                result.append((name, 0.0, float(len(gene["choices"]) - 1), "int"))
            elif kind == "int":
                result.append(
                    (name, float(gene["low"]), float(gene["high"]), "int")
                )
            elif kind == "log_float":
                result.append(
                    (
                        name,
                        math.log10(float(gene["low"])),
                        math.log10(float(gene["high"])),
                        "float",
                    )
                )
            else:
                result.append(
                    (name, float(gene["low"]), float(gene["high"]), "float")
                )
        return result

    def _initial_params(self, schema, config: Dict[str, Any]) -> dict[str, Any]:
        _ensure_creator()
        # Mixed-gene names are labels, not legacy flat runtime keys. Starting
        # from explicit numeric midpoints avoids collisions such as a gene
        # named ``mode`` inheriting the runtime string ``"inference"``.
        params = {
            name: (
                int(round((float(low) + float(high)) / 2.0))
                if kind == "int"
                else (float(low) + float(high)) / 2.0
            )
            for name, low, high, kind in schema
        }
        encoded_initial = config.get("initial_candidate_params")
        if isinstance(encoded_initial, dict):
            params.update({
                key: value
                for key, value in encoded_initial.items()
                if key in params
            })
        declared = {
            item["name"]: item for item in self._declared_schema(config)
        }
        decoded = config.get("initial_candidate_decoded")
        if decoded is None:
            return params
        if not isinstance(decoded, dict):
            raise ValueError("initial_candidate_decoded must be an object")
        unknown = sorted(set(decoded) - set(declared))
        if unknown:
            raise ValueError(
                f"initial_candidate_decoded has unknown genes: {unknown}"
            )
        for name, value in decoded.items():
            gene = declared[name]
            kind = gene["kind"]
            if kind in {"categorical", "boolean"}:
                try:
                    params[name] = gene["choices"].index(value)
                except ValueError as exc:
                    raise ValueError(
                        f"initial value {value!r} is not a choice for {name!r}"
                    ) from exc
            elif kind == "log_float":
                numeric = float(value)
                if numeric <= 0.0:
                    raise ValueError(f"initial log_float {name!r} must be positive")
                params[name] = math.log10(numeric)
            elif kind == "int":
                params[name] = int(value)
            else:
                params[name] = float(value)
        return self._decode(self._encode(params, schema), schema)

    @staticmethod
    def _condition_matches(
        condition: Any,
        decoded: dict[str, Any],
    ) -> bool:
        if condition is None:
            return True
        if not isinstance(condition, dict):
            raise ValueError("enabled_if must be an object")
        for name, expected in condition.items():
            actual = decoded.get(str(name))
            if isinstance(expected, list):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False
        return True

    def _decode_typed(
        self,
        candidate_params: Dict[str, Any],
        config: Dict[str, Any],
    ) -> dict[str, Any]:
        decoded: dict[str, Any] = {}
        for gene in self._declared_schema(config):
            name = gene["name"]
            kind = gene["kind"]
            raw = candidate_params[name]
            if kind in {"categorical", "boolean"}:
                choices = gene["choices"]
                index = max(0, min(len(choices) - 1, int(round(float(raw)))))
                value = copy.deepcopy(choices[index])
            elif kind == "int":
                value = int(round(float(raw)))
            elif kind == "log_float":
                value = 10.0 ** float(raw)
            else:
                value = float(raw)
            decoded[name] = value
        return decoded

    @staticmethod
    def _apply_patch(
        run_config: dict[str, Any],
        patch: dict[str, Any],
    ) -> None:
        for path, value in patch.items():
            _set_path(run_config, str(path), value)

    def _candidate_run_config(
        self,
        candidate_params: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        run_config = copy.deepcopy(config)
        decoded = self._decode_typed(candidate_params, config)
        active: dict[str, Any] = {}
        schema = self._declared_schema(config)

        for gene in schema:
            name = gene["name"]
            if not self._condition_matches(gene.get("enabled_if"), decoded):
                continue
            value = decoded[name]
            active[name] = value
            target = str(gene.get("target") or "").strip()
            if target:
                _set_path(run_config, target, value)
            patches = gene.get("choice_patches")
            if patches is not None:
                if not isinstance(patches, dict):
                    raise ValueError(f"choice_patches for {name!r} must be an object")
                patch = patches.get(str(value))
                if patch is None and isinstance(value, bool):
                    patch = patches.get(str(value).lower())
                if patch is not None:
                    if not isinstance(patch, dict):
                        raise ValueError(
                            f"choice patch for {name!r}={value!r} must be an object"
                        )
                    self._apply_patch(run_config, patch)

        self._apply_feature_groups(run_config, active, config)
        self._apply_repair_rules(run_config, active, config)
        self._apply_resource_repairs(run_config, active, config)
        run_config["_mixed_genome_encoded"] = copy.deepcopy(candidate_params)
        run_config["_mixed_genome_decoded"] = active
        run_config["_mixed_genome_schema_hash"] = _schema_hash(schema)
        return run_config

    @staticmethod
    def _apply_feature_groups(
        run_config: dict[str, Any],
        decoded: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        groups = config.get("mixed_genome_feature_groups")
        if not groups:
            return
        if not isinstance(groups, dict):
            raise ValueError("mixed_genome_feature_groups must be an object")
        columns: list[str] = []
        for group_name, members in groups.items():
            gene_name = f"feature_group__{group_name}"
            if not bool(decoded.get(gene_name, False)):
                continue
            if not isinstance(members, list):
                raise ValueError(f"feature group {group_name!r} must be a list")
            for column in members:
                value = str(column)
                if value not in columns:
                    columns.append(value)
        if not columns:
            required = str(
                config.get("mixed_genome_required_feature_group") or ""
            ).strip()
            members = groups.get(required)
            if not required or not isinstance(members, list) or not members:
                raise ValueError("mixed genome disabled every feature group")
            decoded[f"feature_group__{required}"] = True
            columns = [str(value) for value in members]
        run_config["feature_columns"] = columns
        run_config["feature_list"] = columns

    @staticmethod
    def _apply_repair_rules(
        run_config: dict[str, Any],
        decoded: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        rules = config.get("mixed_genome_repair_rules") or []
        if not isinstance(rules, list):
            raise ValueError("mixed_genome_repair_rules must be a list")
        for index, rule in enumerate(rules):
            if not isinstance(rule, dict):
                raise ValueError(f"mixed_genome_repair_rules[{index}] must be an object")
            condition = rule.get("if")
            if not Plugin._condition_matches(condition, decoded):
                continue
            patch = rule.get("set") or {}
            if not isinstance(patch, dict):
                raise ValueError(f"mixed genome repair rule {index} set must be an object")
            Plugin._apply_patch(run_config, patch)

    @staticmethod
    def _apply_resource_repairs(
        run_config: dict[str, Any],
        decoded: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        features = list(run_config.get("feature_columns") or [])
        window = max(1, int(run_config.get("window_size") or 1))
        repairs: list[dict[str, Any]] = []
        learning_starts = max(0, int(run_config.get("learning_starts") or 0))
        required_checkpoint_step = learning_starts + 1 if learning_starts else 0
        if (
            int(run_config.get("l1_min_checkpoint_timesteps") or 0)
            != required_checkpoint_step
        ):
            repairs.append(
                {
                    "field": "l1_min_checkpoint_timesteps",
                    "requested": run_config.get("l1_min_checkpoint_timesteps"),
                    "resolved": required_checkpoint_step,
                    "reason": "post_learning_checkpoint_barrier",
                }
            )
            run_config["l1_min_checkpoint_timesteps"] = required_checkpoint_step
        max_observation_elements = int(
            config.get("mixed_genome_max_observation_elements") or 0
        )
        if (
            features
            and max_observation_elements > 0
            and len(features) * window > max_observation_elements
        ):
            repaired_window = max(
                1,
                max_observation_elements // len(features),
            )
            repairs.append(
                {
                    "field": "window_size",
                    "requested": window,
                    "resolved": repaired_window,
                    "reason": "observation_element_limit",
                }
            )
            run_config["window_size"] = repaired_window
            window = repaired_window

        max_replay_values = int(
            config.get("mixed_genome_max_replay_observation_values") or 0
        )
        buffer_size = max(1, int(run_config.get("buffer_size") or 1))
        observation_elements = max(1, len(features) * window)
        if (
            max_replay_values > 0
            and buffer_size * observation_elements > max_replay_values
        ):
            repaired_buffer = max(
                int(run_config.get("batch_size") or 1),
                max_replay_values // observation_elements,
            )
            repairs.append(
                {
                    "field": "buffer_size",
                    "requested": buffer_size,
                    "resolved": repaired_buffer,
                    "reason": "replay_observation_value_limit",
                }
            )
            run_config["buffer_size"] = repaired_buffer

        if repairs:
            decoded["_repairs"] = repairs

    def _candidate_metric_evidence(
        self,
        candidate_params: Dict[str, Any],
        run_config: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema = self._declared_schema(config)
        decoded = dict(run_config.get("_mixed_genome_decoded") or {})
        return {
            "mixed_genome_schema_version": SCHEMA_VERSION,
            "mixed_genome_schema_hash": _schema_hash(schema),
            "mixed_genome_encoded": copy.deepcopy(candidate_params),
            "mixed_genome_decoded": decoded,
            "mixed_genome_resolved": {
                "input_data_file": run_config.get("input_data_file"),
                "feature_columns": copy.deepcopy(run_config.get("feature_columns")),
                "feature_scaling": run_config.get("feature_scaling"),
                "feature_scaling_window": run_config.get("feature_scaling_window"),
                "feature_clip": run_config.get("feature_clip"),
                "window_size": run_config.get("window_size"),
                "learning_rate": run_config.get("learning_rate"),
                "batch_size": run_config.get("batch_size"),
                "buffer_size": run_config.get("buffer_size"),
                "learning_starts": run_config.get("learning_starts"),
                "gamma": run_config.get("gamma"),
                "tau": run_config.get("tau"),
                "train_freq": run_config.get("train_freq"),
                "gradient_steps": run_config.get("gradient_steps"),
                "ent_coef": run_config.get("ent_coef"),
                "net_arch": copy.deepcopy(run_config.get("net_arch")),
                "continuous_action_threshold": run_config.get(
                    "continuous_action_threshold"
                ),
                "rel_volume": run_config.get("rel_volume"),
                "k_sl": run_config.get("k_sl"),
                "k_tp": run_config.get("k_tp"),
            },
        }

    def resolve_best_config(
        self,
        optimal: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Resolve the winning chromosome for an optional final reproduction."""
        encoded = {
            key: value
            for key, value in optimal.items()
            if not key.startswith("_")
        }
        return self._candidate_run_config(encoded, config)

    def shared_domain_contract(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return the encoded numeric contract consumed by doin-node.

        DOIN's shared-population protocol intentionally transports numeric
        chromosomes. Categorical, boolean, and logarithmic genes are encoded
        here by the same implementation that later decodes candidates, which
        prevents node materialization from developing a second interpretation
        of the genome.
        """
        schema = self._effective_schema([], config)
        initial = self._initial_params(schema, config)
        schema_payload = [list(item) for item in schema]
        return {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": hashlib.sha256(
                _canonical_json(schema_payload).encode("utf-8")
            ).hexdigest(),
            "param_bounds": {
                name: [low, high] for name, low, high, _kind in schema
            },
            "initial_candidate_params": initial,
            "parameter_schema": schema_payload,
        }
