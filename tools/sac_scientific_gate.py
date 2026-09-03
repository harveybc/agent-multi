#!/usr/bin/env python3
"""Scientific gate BEFORE the paired SAC dispatch (Musashi
correction 2, 2026-09-03).

The fused representation must DEMONSTRATE that it conserves the
branch signal before eight GPU cells are spent on it: at least one
fusion variant must ADVANCE under the predeclared section-D gate
(positive out-of-sample monitor skill on >= 1 target AND no other
target degraded beyond tolerance vs the best non-fused branch, in
every seed) in the observable-runtime screen report.

``evaluate`` writes a typed gate artifact bound to the report digest:

* ``SAC_GATE_PASS``  -> the driver may dispatch (with everything
  else it already demands);
* ``SAC_GATE_FAIL_NEGATIVE_RESULT`` -> the eight cells are NOT
  launched; the negative result is returned to the auditor, or a
  different extractor is explicitly designed. Both outcomes are
  first-class science.

The dispatch driver REFUSES execution without a PASS artifact whose
report digest matches the report on disk."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

GATE_SCHEMA = "agent_multi.sac_scientific_gate.v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


GATE_FIELDS = ("schema", "gate", "advancing_fusion_variants",
               "fusion_decisions", "survivor_verdicts",
               "usable_branches", "screen_report",
               "screen_report_sha256", "external_audit",
               "external_audit_sha256", "external_audit_verdict",
               "consequence")
ACCEPTED_AUDIT_VERDICT = ("SCREEN_V2_NEGATIVE_RESULT_ACCEPTED_WITH_"
                          "LEGACY_BINDING_DISCLOSURE",
                          "SCREEN_V2_ACCEPTED_AFTER_EXTERNAL_"
                          "RUNTIME_AUDIT")


def evaluate(report_path: Path, audit_path: Path) -> dict:
    report = json.loads(Path(report_path).read_text())
    if report.get("schema") != "agent_multi.positive_skill_screen.v2":
        raise SystemExit(
            f"REFUSED: {report_path} is not a v2 observable-runtime "
            "screen report — the retired monolith's output confers "
            "no authority (Musashi correction 1)")
    fusion = report.get("fusion") or {}
    decisions = fusion.get("decisions") or {}
    if not decisions:
        raise SystemExit(
            "REFUSED: the report carries no fusion decisions — the "
            "gate cannot be evaluated on a partial screen")
    advancing = sorted(name for name, verdict in decisions.items()
                       if verdict == "ADVANCES")
    survivor = {k: v.get("verdict") for k, v in
                (report.get("survivor_decisions") or {}).items()}
    gate = ("SAC_GATE_PASS" if advancing
            else "SAC_GATE_FAIL_NEGATIVE_RESULT")
    # C3 (order @1649e7c0 §4): the gate BINDS the corrected external
    # audit and requires its accepted classification
    audit = json.loads(Path(audit_path).read_text())
    audit_verdict = audit.get("verdict")
    if audit_verdict not in ACCEPTED_AUDIT_VERDICT:
        raise SystemExit(
            f"REFUSED: the external audit verdict is "
            f"{audit_verdict!r} — a gate may only be derived over "
            "an ACCEPTED runtime audit")
    # the report the gate derives from must be THE audited report —
    # a doctored copy with a self-consistent re-derivation refuses
    audited_sha = audit.get("audited_report_sha256")
    if audited_sha is None:
        raise SystemExit(
            "REFUSED: the external audit does not name the audited "
            "report digest — regenerate the audit")
    if sha256_file(report_path) != audited_sha:
        raise SystemExit(
            "REFUSED: the report is not the one the external audit "
            "verified — a self-consistent derivation over a "
            "doctored report confers nothing")
    return {
        "schema": GATE_SCHEMA,
        "gate": gate,
        "advancing_fusion_variants": advancing,
        "fusion_decisions": decisions,
        "survivor_verdicts": survivor,
        "usable_branches": sorted(
            k for k, v in survivor.items()
            if v == "USABLE_PREDICTIVE_VALUE"),
        "screen_report": str(report_path),
        "screen_report_sha256": sha256_file(report_path),
        "external_audit": str(audit_path),
        "external_audit_sha256": sha256_file(audit_path),
        "external_audit_verdict": audit_verdict,
        "consequence": (
            "the eight paired-SAC cells MAY be dispatched (subject "
            "to every other driver refusal)" if advancing else
            "the eight paired-SAC cells are NOT dispatched; return "
            "the negative result or explicitly design another "
            "extractor (Musashi correction 2)"),
    }


def verify_gate_for_dispatch(gate_path: Path) -> dict:
    """C3 (order @1649e7c0 §4): the gate is DERIVED, never declared.
    Verification re-derives the entire artifact from the bound
    report and audit and requires the supplied artifact to equal the
    recomputation on EVERY field; missing and unknown fields refuse;
    an edited verdict, a forged advancing variant, a substituted
    report or audit, and a self-consistent re-digest all refuse
    because the derivation source is the report itself."""
    gate = json.loads(Path(gate_path).read_text())
    if gate.get("schema") != GATE_SCHEMA:
        raise SystemExit(
            f"REFUSED: {gate_path} is not a "
            f"{GATE_SCHEMA} artifact")
    supplied_fields = set(gate)
    expected_fields = set(GATE_FIELDS)
    if supplied_fields != expected_fields:
        raise SystemExit(
            "REFUSED: gate artifact fields do not match the exact "
            f"schema — missing {sorted(expected_fields
                                       - supplied_fields)}, "
            f"unknown {sorted(supplied_fields - expected_fields)}")
    report_path = Path(gate["screen_report"])
    if not report_path.exists():
        raise SystemExit(
            "REFUSED: the gate's screen report no longer exists")
    if sha256_file(report_path) != gate["screen_report_sha256"]:
        raise SystemExit(
            "REFUSED: the screen report changed after the gate was "
            "evaluated — re-run the gate")
    audit_path = Path(gate["external_audit"])
    if not audit_path.exists():
        raise SystemExit(
            "REFUSED: the gate's external audit no longer exists")
    if sha256_file(audit_path) != gate["external_audit_sha256"]:
        raise SystemExit(
            "REFUSED: the external audit changed after the gate "
            "was evaluated")
    recomputed = evaluate(report_path, audit_path)
    for field in GATE_FIELDS:
        if gate[field] != recomputed[field]:
            raise SystemExit(
                f"REFUSED: gate field {field!r} does not equal the "
                "recomputed derivation from the bound report/audit "
                "— a declared gate never overrides a derived one")
    if gate["gate"] != "SAC_GATE_PASS":
        raise SystemExit(
            "REFUSED: the scientific gate is "
            f"{gate['gate']} — the eight SAC cells are not launched "
            "(Musashi correction 2)")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("evaluate")
    ev.add_argument("--report", required=True)
    ev.add_argument("--audit", required=True,
                    help="the corrected external-audit artifact "
                         "(C3: the gate binds it and requires an "
                         "accepted classification)")
    ev.add_argument("--output", required=True)
    vf = sub.add_parser("verify")
    vf.add_argument("--gate", required=True)
    args = parser.parse_args()
    if args.cmd == "evaluate":
        gate = evaluate(Path(args.report), Path(args.audit))
        Path(args.output).write_text(json.dumps(gate, indent=1))
        print(json.dumps({k: gate[k] for k in
                          ("gate", "advancing_fusion_variants",
                           "consequence")}, indent=1))
        return 0
    gate = verify_gate_for_dispatch(Path(args.gate))
    print(json.dumps({"gate": gate["gate"],
                      "verified": True}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
