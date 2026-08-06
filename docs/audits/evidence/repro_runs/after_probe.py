#!/usr/bin/env python3
"""Run each Musashi counterexample individually AFTER the corrections.

A counterexample is CORRECTED when it either (a) reports
`reproduced=false`, or (b) now raises — because the reproducer's path
was written to observe a fail-open behaviour that is now fail-closed.
Both outcomes are recorded verbatim; neither is inferred.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
SPEC = importlib.util.spec_from_file_location(
    "musashi_repro",
    REPO / "docs/audits/evidence/"
    "SATOSHI_III_128_134_CORRECTION_REPRO_2026_08_06.py")
repro = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repro)

CASES = {
    "inexact_rejoin": "reproduce_inexact_rejoin",
    "repair_validation_fail_open": "reproduce_repair_fail_open",
    "incomplete_exact_reuse": "reproduce_incomplete_exact_reuse",
    "terminal_reference_gap": "reproduce_terminal_reference_gap",
    "duplicate_seed_empty_identity_promotion":
        "reproduce_duplicate_seed_and_empty_identity",
    "incomplete_authority_join": "reproduce_incomplete_authority_join",
    "warmup_in_interval_score": "reproduce_warmup_scoring",
    "rt_identity_and_split_collision":
        "reproduce_rt_identity_collision",
}


def main() -> int:
    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for label, attr in CASES.items():
            func = getattr(repro, attr, None)
            if func is None:
                results[label] = {"probe": "function_not_found"}
                continue
            try:
                import inspect
                params = inspect.signature(func).parameters
                out = (func(root / label) if params else func())
                results[label] = {
                    "probe": "ran",
                    "still_reproduced": bool(
                        out.get("reproduced")) if isinstance(out, dict)
                    else None,
                    "detail": out,
                }
            except Exception as exc:
                results[label] = {
                    "probe": "raised_fail_closed",
                    "still_reproduced": False,
                    "exception": f"{type(exc).__name__}: {exc}"[:300],
                    "trace_tail": traceback.format_exc()
                    .strip().splitlines()[-1][:200],
                }
    payload = {
        "schema": "agent_multi.after_correction_probe.v1",
        "cases": results,
        "all_corrected": all(
            v.get("still_reproduced") is False for v in results.values()
            if v.get("probe") != "function_not_found"),
    }
    print(json.dumps(payload, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
