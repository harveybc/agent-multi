#!/usr/bin/env python3
"""Author amendment 16 (order B4 C49-C56): the v8 SUPERVISED
recovery generation — append-only after the byte-pinned
amendment 15. A PUBLISHED amendment can never be regenerated."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import b4_authority as b4a  # noqa: E402


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-publication-regenerate",
                    action="store_true")
    args = ap.parse_args()
    out = b4a.AMENDMENT_16_PATH
    if out.exists():
        tracked = subprocess.run(
            ["git", "-C", str(REPO), "ls-files",
             "--error-unmatch", str(out.relative_to(REPO))],
            capture_output=True).returncode == 0
        if tracked:
            raise SystemExit(
                "REFUSED: amendment 16 is PUBLISHED history — "
                "append a later amendment, never regenerate")
        if not args.pre_publication_regenerate:
            raise SystemExit(
                "REFUSED: amendment 16 exists (unpublished); "
                "pass --pre-publication-regenerate to re-author "
                "before its first commit")
        out.unlink()
    if sha_file(b4a.AMENDMENT_15_PATH) != b4a.AMENDMENT_15_SHA:
        raise SystemExit(
            "REFUSED: amendment 15 bytes differ from the pinned "
            "history — nothing is authored over a broken chain")
    quarantine_audit = (REPO / "docs/audits/"
                        "MUSASHI_B4_V7_RUNTIME_STALL_AND_"
                        "QUARANTINE_2026_09_09.md")
    pins = {rel: sha_file(REPO / rel) for rel in (
        "tools/b4_authority.py", "tools/b4_run_cell.py",
        "tools/b4_campaign_executor.py",
        "tools/b4_campaign_ledger.py",
        "tools/b4_campaign_orchestrator.py",
        "tools/b4_cell_supervisor.py",
        "tools/b4_cell_child.py",
        "tools/b4_adjudicator.py",
        "tools/materialize_b4_causal_sac.py",
        "pipeline_plugins/rl_pipeline_with_validation.py",
        "tests/test_b4_materializer_authority.py")}
    a16 = {
        "schema": ("agent_multi.b4_superseding_design_amendment."
                   "v14_supervised_recovery"),
        "amends_amendment_15_sha256": b4a.AMENDMENT_15_SHA,
        "v7_quarantine_audit_sha256": sha_file(quarantine_audit),
        "order": ("C49-C56 (runtime-stall recovery order "
                  "2026-09-09): the third v7 cell stalled ~29 h "
                  "outside callback control (epoch 111, "
                  "2,231,000 steps, CUDA busy, zero durable "
                  "writes) and was operator-stopped and "
                  "quarantined. This amendment records the "
                  "append-only v8 SUPERVISED recovery."),
        "change_disclosure": (
            "runtime supervision, liveness and accounting "
            "identity ONLY: (C51) every cell attempt runs in ONE "
            "supervised child process under a capability bound "
            "to generation/cell/attempt/materialization/"
            "authorization/parent identity/wall/budget; the "
            "parent deadline is monotonic and independent of "
            "callbacks, the GIL, model code, CUDA progress and "
            "child telemetry; (C52) ordered bounded escalation "
            "persisted step-by-step: durable stop -> graceful "
            "60 s -> SIGTERM -> 30 s -> SIGKILL -> mandatory "
            "reap with empty-process-group proof -> CUDA "
            "inventory -> no next cell; supervisor incident "
            "records only, never fabricated scientific "
            "terminals; late/partial terminals stay "
            "QUARANTINED_RUNTIME_STALL/QUARANTINED_EXTERNAL_"
            "STOP; (C53) parent-observed durable-progress "
            "liveness (status.json beats once per epoch — "
            "measured median 171 s / p90 184 s on the completed "
            "cells; checkpoint files are sparse, up to 12,528 s "
            "apart, and are NOT the signal) with "
            "LIVENESS_MAX_SILENCE_S = 1800 s (~10x the "
            "per-epoch p90; the audited stall was silent ~29 h);"
            " a busy CPU/GPU is not progress; GPU accounting "
            "closes at the externally observed reap time in an "
            "append-only supervisor record and ALL charges "
            "re-derive from durable intervals only. No "
            "population, config, data, genesis, comparator, "
            "budget, seed, observation-contract, model-recipe "
            "or decision-rule change."),
        "scientific_change":
            "NONE — external supervision/liveness/accounting "
            "only",
        "campaign_generation_v8": b4a.V8_GENERATION,
        "supersedes_generation": "b4_campaign_generation_v7_"
                                 "20260907",
        "supersedes_results_root_logical":
            b4a.V7_RESULTS_ROOT_LOGICAL,
        "v8_results_root_logical": b4a.V8_RESULTS_ROOT_LOGICAL,
        "v7_root_disposition": (
            "read-only historical evidence; both stop signals "
            "preserved; the quarantined third attempt "
            "(attempt_b459e72dc47a4d72) contributes NOTHING to "
            "v8 — no model, replay, RNG or checkpoint reuse; "
            "its cell reruns from its ORIGINAL zero-update "
            "genesis"),
        "population_identity": (
            "the same twelve cells, order, data, costs, "
            "comparators, seeds, observation contract, model "
            "recipe and decision rule as v7 — bound through the "
            "unchanged amendment-6 population identities and "
            "the v5 materialization digests"),
        "imported_completed_cells": {
            "o2022_seed101": {
                "adjudication": "COMPLETED_VERIFIED",
                "terminal_sha256":
                    "3a836df218702adfa8eb8d436b81064e3f517e59f"
                    "16ab9a750b58304a8c921d5",
                "seal_sha256":
                    "aea56c68cfbe12d1ed10e5c9dbb6751ec6ffddb4d"
                    "6c1d677aae69c9b65b6624e"},
            "o2022_seed202": {
                "adjudication": "COMPLETED_VERIFIED",
                "terminal_sha256":
                    "cd3e61e7e33772663e808745a1135a1d4cebf6ddd"
                    "7c4b67f54c134f0c3c5fd7c",
                "seal_sha256":
                    "ec763fd82b9958cf1e4789e2c9d178b8fc9f174c6"
                    "69e25172027f742c7118526"},
            "import_condition": (
                "imported ONLY because the productive "
                "adjudicator re-derived COMPLETED_VERIFIED and "
                "these digests equal the quarantine audit bytes "
                "at authoring time; the v8 final campaign "
                "verifier re-proves full bindings and exact "
                "bytes again before any scientific use — "
                "otherwise rerun decisions are external")},
        "gpu_accounting_identity": {
            "source": "durable intervals only; the rounded "
                      "44.69/51.30 dry-run summary is NEVER an "
                      "input",
            "v7_charges_seconds": {
                "o2022_seed101": 17773.5,
                "o2022_seed202": 17909.2,
                "o2022_seed303_quarantined_closed_at_operator_"
                "stop": 125074.2},
            "prior_generations_seconds": 39.1,
            "total_charged_seconds": 160795.9,
            "global_ceiling_seconds": 345600.0,
            "remaining_seconds": 184804.1,
            "quarantined_time_counts_against_ceiling": True},
        "final_code_pins": pins,
        "launch_boundary": (
            "the v8 launch is CLOSED: require_v6_launch_open() "
            "now delegates to the v8 gate, which demands BOTH "
            "the external Musashi v8 recovery acta AND the "
            "separate owner dispatch-scope record; the "
            "historical v7 acta opens nothing; only "
            "non-authorizing templates ship"),
        "chronology_truth": (
            "authored AFTER the C49-C56 PRE freeze and after "
            "the supervisor, child, orchestrator call-site and "
            "accounting code were finalized and hashed from "
            "disk; amendments 1-15 untouched"),
    }
    body = {k: a16[k] for k in sorted(a16)}
    a16["amendment_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    out.write_text(json.dumps(a16, indent=1))
    print(json.dumps({"amendment_16_sha256_field":
                      a16["amendment_sha256"],
                      "file_sha256": sha_file(out)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
