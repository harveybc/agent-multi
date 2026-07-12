from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from trading_contracts import (
    CandidateGenomePatch,
    TradingExperimentConfig,
    content_hash,
)


class ConfigResolutionError(ValueError):
    pass


CANONICAL_SCHEMA = "trading_experiment.v1"

_CONTROL_KEYS = {"load_config", "base_config", "candidate_patch", "runtime_overlay"}

_SECTION_KEYS = {
    "mode": ("experiment", "mode"),
    "use_optimizer": ("optimization", "enabled"),
    "quiet_mode": ("experiment", "quiet_mode"),
    "env_plugin": ("environment", "plugin"),
    "agent_plugin": ("asset_policy", "plugin"),
    "pipeline_plugin": ("training", "pipeline_plugin"),
    "optimizer_plugin": ("optimization", "plugin"),
    "optimization_resume": ("optimization", "optimization_resume"),
    "optimization_pause_on_resume": (
        "optimization",
        "optimization_pause_on_resume",
    ),
    "input_data_file": ("data", "input_data_file"),
    "asset": ("data", "asset"),
    "timeframe": ("data", "timeframe"),
    "features_preset": ("data", "features_preset"),
    "feature_list": ("data", "feature_list"),
    "date_column": ("data", "date_column"),
    "price_column": ("data", "price_column"),
    "save_model": ("artifacts", "save_model"),
    "load_model": ("artifacts", "load_model"),
    "results_file": ("artifacts", "results_file"),
    "save_config": ("artifacts", "save_config"),
    "resolved_config_file": ("artifacts", "resolved_config_file"),
    "config_manifest_file": ("artifacts", "config_manifest_file"),
    "save_log": ("artifacts", "save_log"),
    "optimizer_output_file": ("artifacts", "optimizer_output_file"),
}

_ENVIRONMENT_KEYS = {
    "env_mode",
    "headers",
    "max_rows",
    "window_size",
    "initial_cash",
    "position_size",
    "commission",
    "slippage",
    "data_feed_plugin",
    "broker_plugin",
    "strategy_plugin",
    "preprocessor_plugin",
    "reward_plugin",
    "metrics_plugin",
}

_TRAINING_KEYS = {
    "total_timesteps",
    "eval_episodes",
    "eval_seed",
    "train_seed",
    "device",
    "agent_verbose",
}


@dataclass(frozen=True)
class ConfigResolution:
    canonical: TradingExperimentConfig
    runtime: dict[str, Any]
    manifest: dict[str, Any]


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigResolutionError(f"config file not found: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigResolutionError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigResolutionError(f"config root must be an object: {source}")
    return value


def write_json_file(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _put_section_value(
    canonical: dict[str, Any],
    section: str,
    key: str,
    value: Any,
) -> None:
    canonical.setdefault(section, {})[key] = copy.deepcopy(value)


def _apply_legacy_overlay(
    canonical: dict[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(canonical)
    experiment = result.setdefault("experiment", {})
    legacy_flat = experiment.setdefault("legacy_flat", {})

    for key, value in overlay.items():
        if key in _CONTROL_KEYS:
            continue
        legacy_flat[key] = copy.deepcopy(value)
        if key in _SECTION_KEYS:
            section, section_key = _SECTION_KEYS[key]
            _put_section_value(result, section, section_key, value)
        elif key in _ENVIRONMENT_KEYS:
            _put_section_value(result, "environment", key, value)
        elif key in _TRAINING_KEYS:
            _put_section_value(result, "training", key, value)
        elif key.startswith("ga_"):
            _put_section_value(result, "optimization", key, value)
    return result


def translate_legacy_config(
    legacy: Mapping[str, Any],
    *,
    name: str = "legacy-agent-multi-config",
) -> dict[str, Any]:
    canonical = TradingExperimentConfig(
        experiment={"name": name, "legacy_flat": {}},
    ).model_dump(mode="python")
    return _apply_legacy_overlay(canonical, legacy)


def _merge_source(
    canonical: dict[str, Any],
    source: Mapping[str, Any],
    *,
    source_name: str,
) -> tuple[dict[str, Any], str]:
    schema_version = source.get("schema_version")
    if schema_version is None:
        return _apply_legacy_overlay(canonical, source), "legacy"
    if schema_version != CANONICAL_SCHEMA:
        raise ConfigResolutionError(
            f"unsupported config schema in {source_name}: {schema_version!r}"
        )
    return _deep_merge(canonical, source), "canonical"


def _decode_pointer_token(token: str) -> str:
    return token.replace("~1", "/").replace("~0", "~")


def _apply_existing_json_pointer(document: Any, pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        raise ConfigResolutionError(f"candidate gene is not a JSON pointer: {pointer}")
    tokens = [_decode_pointer_token(token) for token in pointer[1:].split("/")]
    if not tokens or tokens == [""]:
        raise ConfigResolutionError("candidate patch cannot replace the document root")

    current = document
    for token in tokens[:-1]:
        if isinstance(current, dict) and token in current:
            current = current[token]
        elif isinstance(current, list) and token.isdigit() and int(token) < len(current):
            current = current[int(token)]
        else:
            raise ConfigResolutionError(f"candidate path does not exist: {pointer}")

    final = tokens[-1]
    if isinstance(current, dict) and final in current:
        current[final] = copy.deepcopy(value)
    elif isinstance(current, list) and final.isdigit() and int(final) < len(current):
        current[int(final)] = copy.deepcopy(value)
    else:
        raise ConfigResolutionError(f"candidate path does not exist: {pointer}")


def _set_runtime(
    runtime: dict[str, Any],
    owners: dict[str, str],
    key: str,
    value: Any,
    source: str,
) -> None:
    if key in owners and runtime[key] != value:
        raise ConfigResolutionError(
            f"canonical runtime key collision for {key!r} from {source}: "
            f"{owners[key]} set {runtime[key]!r}, new value is {value!r}"
        )
    runtime[key] = copy.deepcopy(value)
    owners[key] = source


def canonical_to_runtime(config: TradingExperimentConfig) -> dict[str, Any]:
    data = config.model_dump(mode="python")
    experiment = data["experiment"]
    runtime = copy.deepcopy(experiment.get("legacy_flat", {}))
    canonical_owners: dict[str, str] = {}

    aliases = {
        ("environment", "plugin"): "env_plugin",
        ("asset_policy", "plugin"): "agent_plugin",
        ("training", "pipeline_plugin"): "pipeline_plugin",
        ("optimization", "plugin"): "optimizer_plugin",
        ("optimization", "enabled"): "use_optimizer",
        ("optimization", "metric"): "optimization_metric",
        ("lifecycle_policy", "plugin"): "strategy_plugin",
        ("experiment", "name"): "experiment_name",
    }
    # These values historically lived in the environment's flat config, but
    # the canonical schema assigns ownership to the risk section. Ignoring
    # the legacy/default copies prevents a default environment value from
    # colliding with the explicit risk policy.
    ignored = {
        ("experiment", "legacy_flat"),
        ("environment", "initial_cash"),
        ("environment", "position_size"),
        ("environment", "commission"),
        ("environment", "slippage"),
        ("environment", "leverage"),
        ("environment", "rel_volume"),
        ("environment", "min_order_volume"),
        ("environment", "max_order_volume"),
        ("environment", "size_mode"),
        ("environment", "atr_period"),
        ("environment", "k_sl"),
        ("environment", "k_tp"),
    }
    flatten_sections = {
        "data",
        "environment",
        "training",
        "optimization",
        "asset_policy",
        "lifecycle_policy",
        "risk",
        "walk_forward",
        "artifacts",
    }
    experiment_runtime_keys = {"mode", "quiet_mode"}

    for section, values in data.items():
        if section == "schema_version" or not isinstance(values, dict):
            continue
        runtime[f"{section}_config"] = copy.deepcopy(values)
        canonical_owners[f"{section}_config"] = f"/{section}"
        for key, value in values.items():
            if (section, key) in ignored:
                continue
            runtime_key = aliases.get((section, key), key)
            should_flatten = (
                section in flatten_sections
                or (section, key) in aliases
                or (section == "experiment" and key in experiment_runtime_keys)
            )
            if not should_flatten:
                continue
            if runtime_key.endswith("_config") and isinstance(value, dict):
                _set_runtime(
                    runtime,
                    canonical_owners,
                    runtime_key,
                    value,
                    f"/{section}/{key}",
                )
            else:
                _set_runtime(
                    runtime,
                    canonical_owners,
                    runtime_key,
                    value,
                    f"/{section}/{key}",
                )

    runtime["canonical_schema_version"] = config.schema_version
    runtime["canonical_config_hash"] = config.canonical_hash
    return runtime


def resolve_config(
    defaults: Mapping[str, Any],
    *,
    base_profile: Mapping[str, Any] | None = None,
    file_config: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    candidate_patch: Mapping[str, Any] | None = None,
    source_descriptors: Mapping[str, str] | None = None,
) -> ConfigResolution:
    canonical = translate_legacy_config(defaults, name="agent-multi-defaults")
    source_kinds: dict[str, str] = {"defaults": "legacy"}
    source_hashes: dict[str, str] = {"defaults": content_hash(dict(defaults))}

    for name, source in (
        ("base_profile", base_profile),
        ("file_config", file_config),
        ("cli_overrides", cli_overrides),
    ):
        if not source:
            continue
        canonical, kind = _merge_source(canonical, source, source_name=name)
        source_kinds[name] = kind
        source_hashes[name] = content_hash(dict(source))

    try:
        pre_candidate = TradingExperimentConfig.model_validate(canonical)
    except Exception as exc:
        raise ConfigResolutionError(f"invalid resolved experiment config: {exc}") from exc

    pre_candidate_hash = pre_candidate.canonical_hash
    patch_hash: str | None = None
    if candidate_patch:
        try:
            patch = CandidateGenomePatch.model_validate(candidate_patch)
        except Exception as exc:
            raise ConfigResolutionError(f"invalid candidate patch: {exc}") from exc
        if patch.base_config_hash != pre_candidate_hash:
            raise ConfigResolutionError(
                "candidate base_config_hash mismatch: "
                f"expected {pre_candidate_hash}, got {patch.base_config_hash}"
            )
        candidate_document = pre_candidate.model_dump(mode="python")
        for pointer, value in sorted(patch.genes.items()):
            _apply_existing_json_pointer(candidate_document, pointer, value)
        try:
            resolved = TradingExperimentConfig.model_validate(candidate_document)
        except Exception as exc:
            raise ConfigResolutionError(f"candidate patch produced invalid config: {exc}") from exc
        patch_hash = content_hash(patch)
    else:
        resolved = pre_candidate

    runtime = canonical_to_runtime(resolved)
    manifest = {
        "schema_version": "config_resolution_manifest.v1",
        "canonical_schema_version": resolved.schema_version,
        "resolved_config_hash": resolved.canonical_hash,
        "pre_candidate_config_hash": pre_candidate_hash,
        "candidate_patch_hash": patch_hash,
        "source_kinds": source_kinds,
        "source_hashes": source_hashes,
        "source_descriptors": dict(source_descriptors or {}),
    }
    return ConfigResolution(canonical=resolved, runtime=runtime, manifest=manifest)
