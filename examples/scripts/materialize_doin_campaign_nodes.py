#!/usr/bin/env python3
"""Materialize one shared DOIN node config per worker from a canonical job config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


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
    bounds = optimization.get("hyperparameter_bounds") or {}
    if not bounds:
        raise ValueError("canonical optimization config has no hyperparameter bounds")
    stages = optimization.get("optimization_stages") or []
    if not stages:
        raise ValueError("canonical optimization config has no staged schedule")
    population = int(optimization.get("ga_population", 0))
    if population < 1:
        raise ValueError("canonical optimization population must be positive")
    initial = _candidate_defaults(canonical, bounds)
    metric = str(optimization.get("metric") or canonical.get("objectives", {}).get("selection_metric"))
    metric_schema = str(optimization.get("metric_schema") or "trading.metrics.v1")
    higher_is_better = bool(optimization.get("higher_is_better", True))
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
        domains = node.get("domains") or []
        if len(domains) != 1:
            raise ValueError(f"{template_path} must contain exactly one domain")
        domain = domains[0]
        domain["domain_id"] = domain_id
        domain["higher_is_better"] = higher_is_better
        domain["metric_type"] = metric
        domain["param_bounds"] = bounds
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
