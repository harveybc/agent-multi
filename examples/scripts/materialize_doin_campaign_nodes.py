#!/usr/bin/env python3
"""Materialize one shared DOIN node config per worker from a canonical job config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.canonical_config import resolve_config
from app.config import DEFAULT_VALUES
from optimizer_plugins.project3_full_genome_optimizer import (
    Plugin as Project3FullGenomeOptimizer,
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_defaults(canonical: dict[str, Any], bounds: dict[str, Any]) -> dict[str, Any]:
    sources = [
        canonical.get("training") or {},
        canonical.get("asset_policy") or {},
        canonical.get("risk") or {},
    ]
    result: dict[str, Any] = {}
    for name in bounds:
        for source in sources:
            if name in source:
                result[name] = source[name]
                break
        else:
            raise ValueError(f"canonical config has no initial value for bounded parameter {name}")
    return result


def _optimization_contract(
    canonical: dict[str, Any],
    optimization: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Resolve the numeric domain contract from the canonical optimizer."""
    if optimization.get("mixed_genome_schema"):
        runtime = resolve_config(
            DEFAULT_VALUES,
            file_config=canonical,
        ).runtime
        # Legacy-shaped test and migration configs can carry optimizer extras
        # that the canonical translator does not yet own. The optimization
        # section is authoritative for this materialization step.
        runtime.update(optimization)
        contract = Project3FullGenomeOptimizer().shared_domain_contract(runtime)
        return (
            dict(contract["param_bounds"]),
            dict(contract["initial_candidate_params"]),
            {
                "mixed_genome_schema_version": contract["schema_version"],
                "mixed_genome_schema_hash": contract["schema_hash"],
                "mixed_genome_parameter_schema": contract["parameter_schema"],
            },
        )

    bounds = optimization.get("hyperparameter_bounds") or {}
    if not bounds:
        raise ValueError("canonical optimization config has no hyperparameter bounds")
    return dict(bounds), _candidate_defaults(canonical, bounds), {}


def _declared_max_batch_size(canonical: dict[str, Any]) -> int:
    optimization = canonical.get("optimization") or {}
    for gene in optimization.get("mixed_genome_schema") or []:
        if not isinstance(gene, dict) or gene.get("target") != "batch_size":
            continue
        choices = gene.get("choices") or []
        numeric = [int(value) for value in choices if isinstance(value, (int, float))]
        if numeric:
            return max(numeric)
    bounds = optimization.get("hyperparameter_bounds") or {}
    batch_bounds = bounds.get("batch_size")
    if isinstance(batch_bounds, (list, tuple)) and len(batch_bounds) == 2:
        return int(batch_bounds[1])
    return int((canonical.get("training") or {}).get("batch_size") or 512)


def materialize(
    *,
    template_dir: Path,
    output_dir: Path,
    canonical_config: Path,
    load_config: str,
    domain_id: str,
    campaign_slug: str,
) -> list[Path]:
    canonical = _load(canonical_config)
    optimization = canonical.get("optimization") or {}
    bounds, initial, contract_metadata = _optimization_contract(
        canonical,
        optimization,
    )
    stages = optimization.get("optimization_stages") or []
    if not stages:
        raise ValueError("canonical optimization config has no staged schedule")
    population = int(optimization.get("ga_population", 0))
    if population < 1:
        raise ValueError("canonical optimization population must be positive")
    metric = str(optimization.get("metric") or canonical.get("objectives", {}).get("selection_metric"))
    metric_schema = str(optimization.get("metric_schema") or "trading.metrics.v1")
    higher_is_better = bool(optimization.get("higher_is_better", True))
    plugin_contract = {
        "env_plugin": (canonical.get("environment") or {}).get("plugin"),
        "preprocessor_plugin": (canonical.get("environment") or {}).get(
            "preprocessor_plugin"
        ),
        "agent_plugin": (canonical.get("asset_policy") or {}).get("plugin"),
        "pipeline_plugin": (canonical.get("training") or {}).get("pipeline_plugin"),
        "optimizer_plugin": optimization.get("plugin"),
    }
    created: list[Path] = []
    for template_path in sorted(template_dir.glob("*_node.json")):
        node = _load(template_path)
        label = str(node.get("node_label") or template_path.stem.replace("_node", ""))
        state_name = f"doin-data-{campaign_slug}-{label}"
        node["$doc"] = (
            f"Generated shared-population worker {label} for {domain_id}. "
            "All workers share one semantic domain and use isolated runtime state."
        )
        node["data_dir"] = f"./{state_name}"
        node["identity_file"] = f"./{state_name}/identity.pem"
        node["experiment_stats_file"] = f"./{state_name}/experiment_stats.csv"
        node["olap_db_path"] = f"./{state_name}/olap.db"
        node["reset_chain"] = False
        # Ordered startup verifies each worker's shared lineage before launching
        # the next worker. Population initialization/recovery therefore happens
        # before the full peer barrier, while candidate claiming remains behind
        # the template's shared_min_peers requirement.
        node["shared_initialize_before_peers"] = True
        domains = node.get("domains") or []
        if len(domains) != 1:
            raise ValueError(f"{template_path} must contain exactly one domain")
        domain = domains[0]
        domain["domain_id"] = domain_id
        domain["higher_is_better"] = higher_is_better
        domain["metric_type"] = metric
        domain["param_bounds"] = bounds
        limits = domain.setdefault("resource_limits", {})
        limits.update({
            "max_training_seconds": max(
                int(limits.get("max_training_seconds") or 0),
                7 * 24 * 60 * 60,
            ),
            "max_epochs": int(
                (canonical.get("training") or {}).get("max_epochs") or 2_000
            ),
            "max_batch_size": _declared_max_batch_size(canonical),
        })
        opt = domain.setdefault("optimization_config", {})
        opt.update({
            "load_config": load_config,
            "metric_type": metric,
            "optimization_metric": metric,
            "metric_schema": metric_schema,
            "higher_is_better": higher_is_better,
            "shared_population": True,
            "shared_population_size": population,
            "population_size": population,
            "initial_candidate_params": initial,
            "ga_seed": int(optimization.get("ga_seed", 0)),
            "ga_population": population,
            "optimization_patience": int(optimization.get("optimization_patience", 1)),
            "optimization_stages": stages,
            "hyperparameter_bounds": bounds,
            "optimization_resume": False,
            "optimization_pause_on_resume": False,
        })
        opt.update(contract_metadata)
        opt.update({key: value for key, value in plugin_contract.items() if value})
        output_path = output_dir / template_path.name
        _write(output_path, node)
        created.append(output_path)
    if not created:
        raise ValueError(f"no *_node.json templates found under {template_dir}")
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--canonical-config", type=Path, required=True)
    parser.add_argument("--load-config", required=True)
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--campaign-slug", required=True)
    args = parser.parse_args()
    paths = materialize(
        template_dir=args.template_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        canonical_config=args.canonical_config.resolve(),
        load_config=args.load_config,
        domain_id=args.domain_id,
        campaign_slug=args.campaign_slug,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
