#!/usr/bin/env python3
"""Materialize a grouped-extractor experiment from an existing flat config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent_plugins.feature_families import baseline_grouped_architecture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.base.read_text(encoding="utf-8"))
    columns = config.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise SystemExit("base config requires non-empty feature_columns")
    config["feature_extractor_plugin"] = "grouped_features_extractor"
    config["feature_extractor_config"] = baseline_grouped_architecture(columns)
    config["agent_state_contract"] = "live_stationary_v2"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
