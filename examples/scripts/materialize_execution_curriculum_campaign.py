#!/usr/bin/env python3
"""Materialize the launchable curriculum config and all shared DOIN workers."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from examples.scripts.materialize_doin_campaign_nodes import (
    materialize as materialize_nodes,
)
from examples.scripts.materialize_execution_curriculum_followup import (
    materialize as materialize_followup,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def materialize_campaign(
    *,
    agent_root: Path,
    doin_root: Path,
    source_model_file: Path,
    source_parameters_file: Path,
    output_root: Path,
    domain_id: str,
    campaign_slug: str,
    base_config: Path | None = None,
    node_template_dir: Path | None = None,
    source_model_runtime_path: str | None = None,
) -> dict:
    agent_root = agent_root.resolve()
    doin_root = doin_root.resolve()
    output_root = output_root.expanduser().resolve()
    canonical_path = output_root / "canonical_config.json"
    node_output_dir = output_root / "nodes"
    source_runtime_path = source_model_runtime_path or (
        "${ARTIFACT_ROOT}/full_genome/usdcad_4h/champion_policy.zip"
    )

    materialize_followup(
        base_config=(base_config.expanduser().resolve() if base_config else (
            agent_root
            / "examples/config/phase_1_asset_policy/optimization"
            / "phase_1_asset_policy_usdcad_4h_execution_curriculum_template_v1.json"
        )),
        curriculum_config=(
            agent_root
            / "examples/config/execution_curriculum"
            / "project3_execution_cost_curriculum_v1.json"
        ),
        output_config=canonical_path,
        source_model_runtime_path=source_runtime_path,
        source_model_file=source_model_file.expanduser().resolve(),
        source_parameters_file=source_parameters_file.expanduser().resolve(),
        template=False,
    )
    node_paths = materialize_nodes(
        template_dir=(
            node_template_dir.expanduser().resolve()
            if node_template_dir
            else
            doin_root
            / "examples/trading/phase_1_asset_policy_usdcad_4h_full_genome_v1"
        ),
        output_dir=node_output_dir,
        canonical_config=canonical_path,
        load_config=str(canonical_path),
        domain_id=domain_id,
        campaign_slug=campaign_slug,
    )
    manifest = {
        "schema_version": "agent_multi.curriculum_campaign_materialization.v1",
        "domain_id": domain_id,
        "campaign_slug": campaign_slug,
        "source_model_file": str(source_model_file.expanduser().resolve()),
        "source_model_sha256": _sha256(source_model_file.expanduser().resolve()),
        "source_parameters_file": str(
            source_parameters_file.expanduser().resolve()
        ),
        "source_parameters_sha256": _sha256(
            source_parameters_file.expanduser().resolve()
        ),
        "canonical_config": str(canonical_path),
        "canonical_config_sha256": _sha256(canonical_path),
        "node_configs": {
            path.name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in node_paths
        },
    }
    manifest_path = output_root / "materialization_manifest.json"
    _write(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--doin-root", type=Path, required=True)
    parser.add_argument("--source-model-file", type=Path, required=True)
    parser.add_argument("--source-parameters-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--domain-id", required=True)
    parser.add_argument("--campaign-slug", required=True)
    parser.add_argument("--base-config", type=Path)
    parser.add_argument("--node-template-dir", type=Path)
    parser.add_argument("--source-model-runtime-path")
    args = parser.parse_args()
    result = materialize_campaign(
        agent_root=args.agent_root,
        doin_root=args.doin_root,
        source_model_file=args.source_model_file,
        source_parameters_file=args.source_parameters_file,
        output_root=args.output_root,
        domain_id=args.domain_id,
        campaign_slug=args.campaign_slug,
        base_config=args.base_config,
        node_template_dir=args.node_template_dir,
        source_model_runtime_path=args.source_model_runtime_path,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
