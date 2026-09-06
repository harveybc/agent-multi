#!/usr/bin/env python3
"""T1 independent verifier (order C12).

A fresh process that starts from the SEALED DESIGN and the raw
evidence (bank arrays + content-addressed NPZ blobs + measurement
records) — never the producer summary — and must reproduce the
published verdict counts exactly. It re-derives the score-role SNR
gate for EVERY measured record (not a sample) and re-runs the full
adjudication."""
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
    ap.add_argument("--published", type=Path, required=True)
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
    m = adj._strict_json(args.measurements)
    measured = [r for r in m["records"]
                if r["status"] == "MEASURED"]
    for r in measured:
        adj.rederive_gates(r, args.bank_dir, args.npz_dir)
    out = adj.adjudicate(design, inventory, m,
                         bank_dir=args.bank_dir,
                         npz_dir=args.npz_dir, rederive_sample=0)
    published = adj._strict_json(args.published)
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
    print(json.dumps({
        "independent_verification": "REPRODUCED",
        "records_rederived_from_arrays": len(measured),
        "verdict_counts": out["verdict_counts"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
