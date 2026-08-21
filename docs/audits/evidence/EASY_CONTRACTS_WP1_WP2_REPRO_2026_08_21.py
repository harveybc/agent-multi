#!/usr/bin/env python3
"""Independent counterexamples for Satoshi commit d9888aef."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def run_report(row: dict) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "source.json"
        result = root / "result.json"
        source.write_text(json.dumps({"history": [row]}))
        proc = subprocess.run(
            [sys.executable, str(REPO / "tools/rank_disagreement_study.py"),
             "--report", str(source), "--out-csv", str(root / "out.csv"),
             "--out-report", str(result)],
            cwd=REPO, text=True, capture_output=True, check=False)
        return proc.returncode, (json.loads(result.read_text())
                                 if result.exists() else {})


def main() -> int:
    base = {
        "epoch": 1,
        "train_tail_trades": 5,
        "val_trades": 5,
        "train_tail_return": 0.01,
        "val_return": 0.01,
        "train_tail_drawdown": 0.02,
        "val_drawdown": 0.02,
    }
    missing = dict(base)
    del missing["val_trades"]
    string_count = {**base, "val_trades": "3"}
    missing_economics = dict(base)
    del missing_economics["val_return"]
    outcomes = {}
    for name, row in (("missing_trade_count", missing),
                      ("string_trade_count", string_count),
                      ("missing_validation_return", missing_economics)):
        rc, report = run_report(row)
        outcomes[name] = {
            "returncode": rc,
            "accepted": rc == 0,
            "epochs": report.get("epochs"),
        }

    source = (REPO / "pipeline_plugins/_easy_contracts.py").read_text()
    references = subprocess.run(
        ["rg", "-l", "easy_checkpoint_monitor|easy_doin_candidate_fitness",
         ".", "--glob", "!docs/**", "--glob", "!tests/**"],
        cwd=REPO, text=True, capture_output=True, check=False).stdout.splitlines()
    executable_consumers = [path for path in references if path not in {
        "./pipeline_plugins/_easy_contracts.py",
        "./tools/rank_disagreement_study.py",
        "./tools/TOOL_DECLARATIONS.json",
    }]
    outcomes["executing_consumers"] = executable_consumers
    outcomes["coercions_present"] = {
        "int_trade_count": "closed_trades=int(vv_tr or 0)" in (
            REPO / "tools/rank_disagreement_study.py").read_text(),
        "defaulted_validation_return": (
            'row.get("val_return", row.get("validation_return",\n'
            '                                               0.0)) or 0.0'
        ) in (REPO / "tools/rank_disagreement_study.py").read_text(),
    }
    reproduced = (
        all(outcomes[name]["accepted"] for name in
            ("missing_trade_count", "string_trade_count",
             "missing_validation_return"))
        and not executable_consumers
        and outcomes["coercions_present"]["int_trade_count"]
    )
    packet = {"schema": "agent_multi.easy_contracts_audit_repro.v1",
              "target": "d9888aef", "reproduced": reproduced,
              "outcomes": outcomes,
              "module_loaded": "easy_checkpoint_monitor" in source}
    print(json.dumps(packet, indent=2, sort_keys=True))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
