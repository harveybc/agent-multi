#!/usr/bin/env python3
"""T1 independent verifier (orders C12 + C16-C18).

A fresh process that starts from the SEALED DESIGN and the raw
evidence (physically bound bank arrays + content-addressed NPZ
blobs + manifest-bound measurement records) — never the producer
summary. It re-derives EVERY decision-bearing metric of EVERY
measured record (reconstruction, SNR, extremes, tails, all
downstream assays, residual value — never a sample), re-runs the
full adjudication, and emits a candidate REVIEW SUBMISSION.

C18: self-consistency NEVER authorizes — without an external
reviewer record naming the exact measurement and publication
digests, the outcome is SELF_CONSISTENT_ONLY_NOT_AUTHORIZING
(exit 3)."""
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
    ap.add_argument("--reviewed-record", type=Path, default=None)
    ap.add_argument("--submission-out", type=Path, default=None)
    args = ap.parse_args()
    raw = args.design.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.design_sha:
        raise SystemExit("REFUSED: design digest mismatch")
    design = adj._strict_json(args.design)
    inventory = adj._strict_json(
        args.bank_dir / "BANK_INVENTORY.json")
    if design["bank_inventory_sha256"] != adj._sha_file(
            args.bank_dir / "BANK_INVENTORY.json"):
        raise SystemExit("REFUSED: inventory differs from the "
                         "sealed design")
    # C17: the inventory must bind the physical population, and
    # every unit's bytes must re-derive BEFORE any use.
    adj.check_inventory(inventory)
    for uid in inventory["unit_ids"]:
        adj.verify_unit_bytes(inventory, args.bank_dir, uid)
    # C18: the measurement-population manifest must bind the exact
    # measurement bytes being verified.
    manifest = adj._strict_json(args.measurement_manifest)
    want_m = {"schema", "design_sha256", "bank_inventory_sha256",
              "measurements_sha256", "records_total",
              "records_measured", "records_refused",
              "manifest_sha256"}
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
    meas_sha = adj._sha_file(args.measurements)
    if manifest["measurements_sha256"] != meas_sha:
        raise SystemExit(
            "REFUSED: measurement bytes differ from the "
            "population manifest — a rewritten population is not "
            "the measured one")
    m = adj._strict_json(args.measurements)
    measured = [r for r in m["records"]
                if r["status"] == "MEASURED"]
    if len(m["records"]) != manifest["records_total"] or \
            len(measured) != manifest["records_measured"]:
        raise SystemExit("REFUSED: record counts differ from the "
                         "manifest")
    # C16: re-derive EVERY decision-bearing metric of EVERY
    # measured record — never a sample.
    for r in measured:
        adj.rederive_all_facts(r, args.bank_dir, args.npz_dir)
    out = adj.adjudicate(design, inventory, m,
                         bank_dir=args.bank_dir,
                         npz_dir=args.npz_dir, rederive_sample=0)
    published = adj._strict_json(args.published)
    pub_sha = adj._sha_file(args.published)
    if out["verdict_counts"] != published["verdict_counts"]:
        raise SystemExit(
            f"REFUSED: independent verdict counts "
            f"{out['verdict_counts']} differ from published "
            f"{published['verdict_counts']}")
    mismatches = [k for k in out["verdicts"]
                  if out["verdicts"][k]["verdict"]
                  != published["verdicts"][k]["verdict"]]
    if mismatches:
        raise SystemExit(
            f"REFUSED: {len(mismatches)} verdicts differ "
            f"(e.g. {mismatches[:3]})")
    submission = {
        "schema": "agent_multi.t1_review_submission.v1",
        "design_sha256": args.design_sha,
        "bank_inventory_sha256":
            manifest["bank_inventory_sha256"],
        "measurement_manifest_sha256":
            manifest["manifest_sha256"],
        "measurements_sha256": meas_sha,
        "publication_sha256": pub_sha,
        "records_rederived_from_arrays": len(measured),
        "note": "candidate submission — NON-AUTHORIZING until an "
                "external reviewer record names these digests"}
    if args.submission_out:
        args.submission_out.write_text(
            json.dumps(submission, indent=1))
    # C18: self-consistency NEVER authorizes. Only an external
    # reviewer record naming these exact digests promotes the
    # result to a reviewed identity.
    if args.reviewed_record is None:
        print(json.dumps({
            "independent_verification":
                "SELF_CONSISTENT_ONLY_NOT_AUTHORIZING",
            "records_rederived_from_arrays": len(measured),
            "verdict_counts": out["verdict_counts"],
            "measurements_sha256": meas_sha,
            "publication_sha256": pub_sha}, indent=1))
        return 3
    rr = adj._strict_json(args.reviewed_record)
    want_r = {"schema", "reviewer", "measurements_sha256",
              "publication_sha256"}
    if not want_r.issubset(set(rr)) or \
            rr.get("schema") != "agent_multi.t1_reviewed_record.v1":
        raise SystemExit("REFUSED: reviewed record is not the "
                         "exact schema")
    if rr["measurements_sha256"] != meas_sha or \
            rr["publication_sha256"] != pub_sha:
        raise SystemExit(
            "REFUSED: the reviewed record names DIFFERENT "
            "measurement/publication digests — a coherent rewrite "
            "is not the reviewed identity")
    print(json.dumps({
        "independent_verification":
            "REPRODUCED_UNDER_REVIEWED_IDENTITY",
        "reviewer": rr["reviewer"],
        "records_rederived_from_arrays": len(measured),
        "verdict_counts": out["verdict_counts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
