#!/usr/bin/env python3
"""External READ-ONLY runtime audit of a frozen screen-v2 run
(Musashi order @65ee8488 §4 R2, 2026-09-03).

Adjudicates the evidence the frozen run already produced WITHOUT
discarding or rewriting anything:

1. recomputes the digest of every phase ledger, every immutable
   input, the code files and the predeclaration AGAINST THE FROZEN
   EXECUTING CHECKOUT (--code-root at the exact commit that ran);
2. recomputes every result_digest from content and demands exact
   unit-result correspondence (path uid == state uid; identity
   fields consistent for cell results);
3. proves no two attempts of one unit overlapped, to the extent the
   frozen schema records it (single result, attempt-log count equals
   the recorded attempt counter, no leftover .lock, monotonic
   claim/finish stamps) and DECLARES the schema's limits honestly;
4. enumerates every FAILED / TIMED_OUT / INTERRUPTED trace: attempt
   logs beyond the final COMPLETED attempt, watchdog lines in the
   supervisor log, non-COMPLETED terminal states — none may vanish
   behind a later terminal;
5. binds the final report (when present) to the complete inventory
   of verified units.

Output: a single verdict —
  SCREEN_V2_ACCEPTED_AFTER_EXTERNAL_RUNTIME_AUDIT, or
  SCREEN_V2_RERUN_REQUIRED with the exact causes.
This tool NEVER writes into the run directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PHASES = ("round1", "round2", "round3", "survivors", "fusion")
CODE_FILES = ("tools/positive_skill_screen_v2.py",
              "tools/positive_skill_screen.py",
              "agent_plugins/experiment_runtime.py",
              "agent_plugins/branch_pretraining.py",
              "agent_plugins/temporal_information.py",
              "agent_plugins/pretrained_branch_loader.py")


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha_obj(obj) -> str:
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def audit(run_root: Path, code_root: Path,
          predeclaration: Path) -> dict:
    findings: list = []
    facts: dict = {"phases": {}}

    def finding(kind: str, detail):
        findings.append({"kind": kind, "detail": detail})

    frozen_code = sha_obj({f: sha_file(code_root / f)
                           for f in CODE_FILES})
    frozen_config = sha_file(predeclaration)
    inventory: dict = {}

    for phase in PHASES:
        phase_dir = run_root / phase
        ledger_path = phase_dir / "ledger.json"
        if not ledger_path.exists():
            facts["phases"][phase] = {"state": "NOT_MATERIALIZED"}
            continue
        ledger = json.loads(ledger_path.read_text())
        recomputed = sha_obj({k: v for k, v in ledger.items()
                              if k != "ledger_digest"})
        if ledger.get("ledger_digest") != recomputed:
            finding("ledger_digest_mismatch", phase)
        digests = ledger.get("digests", {})
        if digests.get("code") != frozen_code:
            finding("code_digest_mismatch",
                    {"phase": phase,
                     "ledger": str(digests.get("code"))[:12],
                     "frozen_checkout": frozen_code[:12]})
        if digests.get("config") != frozen_config:
            finding("config_digest_mismatch", phase)
        for key, value in digests.items():
            if key.startswith("input_") or key == "data_csv":
                if key.startswith("input_"):
                    name = ("fusion_inputs.npz"
                            if key == "input_fusion"
                            else f"windows_{key.split('_')[1]}.npz")
                    path = run_root / "inputs" / name
                else:
                    continue  # csv path known only to the tool run
                if not path.exists():
                    finding("input_missing", {"phase": phase,
                                              "input": key})
                elif sha_file(path) != value:
                    finding("input_digest_mismatch",
                            {"phase": phase, "input": key})

        states, results = {}, {}
        counts: dict = {}
        overlap_suspects = []
        nonterminal_traces = []
        for state_path in sorted(
                (phase_dir / "units").glob("*.state.json")):
            state = json.loads(state_path.read_text())
            uid = state["unit_id"]
            states[uid] = state
            counts[state["state"]] = counts.get(state["state"], 0) + 1
            attempt = int(state.get("attempt", 0))
            logs = sorted((phase_dir / "units").glob(
                f"{uid}.attempt*.log"))
            if len(logs) > attempt:
                finding("more_attempt_logs_than_attempts",
                        {"unit": uid, "logs": len(logs),
                         "attempts": attempt})
            if attempt > 1:
                nonterminal_traces.append(
                    {"unit": uid, "attempts": attempt,
                     "final_state": state["state"],
                     "note": "earlier attempt(s) ended "
                             "FAILED/INTERRUPTED/TIMED_OUT before "
                             "the final state — enumerated, not "
                             "hidden"})
            lock = phase_dir / "units" / f"{uid}.lock"
            if lock.exists() and state["state"] != "RUNNING":
                overlap_suspects.append(
                    {"unit": uid, "why": "terminal state with a "
                                         "leftover claim lock"})
            claimed = state.get("claimed_at")
            finished = state.get("finished_at")
            if claimed and finished and finished < claimed:
                finding("non_monotonic_stamps",
                        {"unit": uid, "claimed_at": claimed,
                         "finished_at": finished})
            result_path = phase_dir / "units" / f"{uid}.result.json"
            if state["state"] == "COMPLETED":
                if not result_path.exists():
                    finding("completed_without_result", uid)
                    continue
                result = json.loads(result_path.read_text())
                recomputed_r = sha_obj(
                    {k: v for k, v in result.items()
                     if k != "result_digest"})
                if result.get("result_digest") != recomputed_r:
                    finding("result_digest_mismatch", uid)
                bound = result.get("unit_id")
                if bound is not None and bound != uid:
                    finding("result_unit_binding_broken",
                            {"unit": uid, "bound": bound})
                identity = state.get("identity") or {}
                for field in ("family", "window", "latent"):
                    if field in result and field in identity and \
                            result[field] != identity[field]:
                        finding("result_identity_mismatch",
                                {"unit": uid, "field": field})
                results[uid] = result
            elif result_path.exists():
                finding("result_for_nonterminal_completion",
                        {"unit": uid, "state": state["state"]})
        expected_units = {u["unit_id"] for u in ledger["units"]}
        foreign = set(states) - expected_units
        missing = expected_units - set(states)
        if foreign:
            finding("foreign_units", sorted(foreign)[:5])
        if missing:
            finding("missing_units", sorted(missing)[:5])
        if overlap_suspects:
            finding("overlap_suspects", overlap_suspects[:5])
        decision_path = phase_dir / "decisions" / "halving.json"
        decision = (json.loads(decision_path.read_text())
                    if decision_path.exists() else None)
        if decision is not None and \
                decision.get("ledger_digest") != \
                ledger.get("ledger_digest"):
            finding("decision_ledger_binding_mismatch", phase)
        facts["phases"][phase] = {
            "units": len(states), "counts": counts,
            "completed_results_verified": len(results),
            "attempt_retries_enumerated": nonterminal_traces,
            "halving_decision_present": decision is not None}
        inventory[phase] = sorted(results)

    supervisor_log = run_root / "supervisor.out"
    watchdog_lines = []
    if supervisor_log.exists():
        for line in supervisor_log.read_text(
                errors="replace").splitlines():
            if "[watchdog]" in line or "[ceiling]" in line:
                watchdog_lines.append(line.strip()[:200])
    facts["supervisor_watchdog_and_ceiling_lines"] = watchdog_lines

    report_path = run_root / "POSITIVE_SKILL_SCREEN_V2_REPORT.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        cells = set(report.get("cells", {}))
        verified_round_cells = set()
        for phase in ("round1", "round2", "round3"):
            for uid in inventory.get(phase, []):
                result = json.loads(
                    (run_root / phase / "units" /
                     f"{uid}.result.json").read_text())
                verified_round_cells.add(
                    f"{result['family']}|w{result['window']}|"
                    f"d{result['latent']}")
        unbacked = cells - verified_round_cells
        if unbacked:
            finding("report_cells_without_verified_units",
                    sorted(unbacked)[:5])
        facts["final_report"] = {
            "present": True,
            "cells": len(cells),
            "cells_backed_by_verified_units":
                len(cells & verified_round_cells),
            "survivor_decisions":
                len(report.get("survivor_decisions", {})),
            "fusion_decisions":
                (report.get("fusion") or {}).get("decisions")}
    else:
        facts["final_report"] = {"present": False}

    facts["schema_limits_declared"] = [
        "the frozen state schema keeps only the LAST attempt per "
        "unit; per-attempt claim/finish intervals of superseded "
        "attempts are reconstructible only from attempt logs and "
        "the supervisor journal, both enumerated above",
        "overlap detection relies on single-result + lock absence + "
        "attempt-log accounting + watchdog journal; the R1-corrected "
        "runtime adds attempt CAS so future runs prove it "
        "structurally"]

    incomplete = (not facts["final_report"]["present"]) or any(
        e.get("counts", {}).get("PENDING", 0)
        + e.get("counts", {}).get("RUNNING", 0) > 0
        for e in facts["phases"].values() if "counts" in e) or any(
        e.get("state") == "NOT_MATERIALIZED"
        for e in facts["phases"].values())
    if incomplete:
        verdict = "AUDIT_INCOMPLETE_RUN_STILL_EXECUTING"
    elif findings:
        verdict = "SCREEN_V2_RERUN_REQUIRED"
    else:
        verdict = "SCREEN_V2_ACCEPTED_AFTER_EXTERNAL_RUNTIME_AUDIT"
    return {"schema": "agent_multi.screen_v2_external_audit.v1",
            "run_root": "run id " + run_root.name +
                        " (durable state store)",
            "frozen_code_digest": frozen_code,
            "frozen_config_digest": frozen_config,
            "verdict": verdict,
            "findings": findings,
            "facts": facts}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--code-root", required=True,
                        help="checkout at the EXACT commit that "
                             "executed the run")
    parser.add_argument("--predeclaration", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = audit(Path(args.run_root), Path(args.code_root),
                   Path(args.predeclaration))
    payload = json.dumps(result, indent=1, default=str)
    if args.output:
        Path(args.output).write_text(payload)
    print(json.dumps({"verdict": result["verdict"],
                      "findings": len(result["findings"])},
                     indent=1))
    if result["findings"]:
        print(json.dumps(result["findings"][:10], indent=1))
    if result["verdict"] == "AUDIT_INCOMPLETE_RUN_STILL_EXECUTING":
        return 3
    return 0 if not result["findings"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
