#!/usr/bin/env python3
"""T1 independent verifier (orders C12 + C16-C18 + C21-C24).

A fresh process that starts from the SEALED DESIGN and the raw
evidence (physically bound bank arrays + content-addressed NPZ
blobs + manifest-bound measurement records) — never the producer
summary. It refuses unless every executable byte it and the
producers run matches the design's complete code identity (C21),
re-derives EVERY decision-bearing metric of EVERY measured record,
recomputes the COMPLETE canonical adjudication and requires exact
equality with the publication — verdicts, reasons, distributions,
seed counts, flags, failures, metadata and counts (C22).

C23: this candidate-distributed tool can NEVER confer review
authority. Its strongest possible outcome is a self-consistent,
NON-AUTHORIZING review submission (exit 3). External review
authority is a separate Musashi record, created only after
independent reproduction and pinned by digest in the consuming T2
order — no input to this tool can change that."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import t1_adjudicator as adj  # noqa: E402


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--design-sha", required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    ap.add_argument("--measurements", type=Path, required=True)
    ap.add_argument("--measurement-manifest", type=Path,
                    required=True)
    ap.add_argument("--published", type=Path, required=True)
    ap.add_argument("--submission-out", type=Path, default=None)
    args = ap.parse_args()
    raw = args.design.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.design_sha:
        raise SystemExit("REFUSED: design digest mismatch")
    design = adj._strict_json(args.design)
    # C21: complete executable identity BEFORE any evidence read
    adj.verify_complete_code_identity(design)
    inventory = adj._strict_json(
        args.bank_dir / "BANK_INVENTORY.json")
    if design["bank_inventory_sha256"] != adj._sha_file(
            args.bank_dir / "BANK_INVENTORY.json"):
        raise SystemExit("REFUSED: inventory differs from the "
                         "sealed design")
    # C17: physical population re-derivation BEFORE any use
    adj.check_inventory(inventory)
    for uid in inventory["unit_ids"]:
        adj.verify_unit_bytes(inventory, args.bank_dir, uid)
    # C18: the measurement-population manifest binds the exact
    # measurement bytes being verified
    manifest = adj._strict_json(args.measurement_manifest)
    want_m = {"schema", "design_sha256", "bank_inventory_sha256",
              "measurements_sha256", "executed_code_identity",
              "records_total", "records_measured",
              "records_refused", "manifest_sha256"}
    if set(manifest) != want_m:
        raise SystemExit("REFUSED: measurement manifest keys are "
                         "not the exact schema")
    body = {k: manifest[k] for k in sorted(manifest)
            if k != "manifest_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest() != \
            manifest["manifest_sha256"]:
        raise SystemExit("REFUSED: manifest digest does not "
                         "re-derive")
    if manifest["design_sha256"] != args.design_sha or \
            manifest["bank_inventory_sha256"] != adj._sha_file(
                args.bank_dir / "BANK_INVENTORY.json"):
        raise SystemExit("REFUSED: manifest binds a foreign "
                         "design/inventory")
    # C21: the manifest must name the SAME executed identity the
    # design seals — a run under other bytes is not this run
    if manifest["executed_code_identity"] != \
            design["code_identity"]:
        raise SystemExit(
            "REFUSED: the manifest's executed code identity "
            "differs from the sealed design")
    meas_sha = adj._sha_file(args.measurements)
    if manifest["measurements_sha256"] != meas_sha:
        raise SystemExit(
            "REFUSED: measurement bytes differ from the "
            "population manifest — a rewritten population is not "
            "the measured one")
    m = adj._strict_json(args.measurements)
    if m.get("design_sha256") != args.design_sha:
        raise SystemExit("REFUSED: measurements name a foreign "
                         "design")
    if m.get("code_identity", {}).get("executed") != \
            design["code_identity"]:
        raise SystemExit(
            "REFUSED: the measurement payload's executed code "
            "identity differs from the sealed design")
    measured = [r for r in m["records"]
                if r["status"] == "MEASURED"]
    if len(m["records"]) != manifest["records_total"] or \
            len(measured) != manifest["records_measured"]:
        raise SystemExit("REFUSED: record counts differ from the "
                         "manifest")
    # C16: re-derive EVERY decision-bearing metric of EVERY
    # measured record — never a sample
    for r in measured:
        adj.rederive_all_facts(r, args.bank_dir, args.npz_dir)
    out = adj.adjudicate(design, inventory, m,
                         bank_dir=args.bank_dir,
                         npz_dir=args.npz_dir, rederive_sample=0)
    published = adj._strict_json(args.published)
    pub_sha = adj._sha_file(args.published)
    # C22: the COMPLETE canonical adjudication must equal the
    # re-derived object — labels and counts alone grant nothing
    adj.require_publication_equality(out, published)
    submission = {
        "schema": "agent_multi.t1_review_submission.v2",
        "design_sha256": args.design_sha,
        "bank_inventory_sha256":
            manifest["bank_inventory_sha256"],
        "measurement_manifest_sha256":
            manifest["manifest_sha256"],
        "measurements_sha256": meas_sha,
        "publication_sha256": pub_sha,
        "executed_code_identity": adj.executed_code_identity(),
        "records_rederived_from_arrays": len(measured),
        "note": "candidate submission — NON-AUTHORIZING. External "
                "review authority is a separate Musashi record; "
                "no candidate tool can create or consume it."}
    if args.submission_out:
        args.submission_out.write_text(
            json.dumps(submission, indent=1))
    # C23: the strongest possible outcome of candidate code.
    print(json.dumps({
        "independent_verification":
            "SELF_CONSISTENT_ONLY_NOT_AUTHORIZING",
        "records_rederived_from_arrays": len(measured),
        "complete_publication_equality": True,
        "verdict_counts": out["verdict_counts"],
        "measurements_sha256": meas_sha,
        "publication_sha256": pub_sha}, indent=1))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
