#!/usr/bin/env python
"""Cut a cell's per-row trace to a declared view-row range (the strict row-parity view
shared with M07/M05) and write PAIRED_ROWS.csv + its sha256 beside the cell.

    python tools/rl_temporal_paired_rows.py --cell-dir <out dir> --rows 13699 15888
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-dir", required=True)
    ap.add_argument("--rows", nargs=2, type=int, required=True, help="inclusive view-row range [a, b]")
    args = ap.parse_args(argv)
    d = Path(args.cell_dir)
    a, b = args.rows
    rows = []
    with (d / "trace.csv").open() as fh:
        for r in csv.DictReader(fh):
            v = int(r["view_row"])
            if a <= v <= b and r["is_context_prefix"] in ("False", "0", ""):
                rows.append(r)
    out = d / f"PAIRED_ROWS_{a}_{b}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["view_row"])
        w.writeheader()
        w.writerows(rows)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    first, last = (rows[0], rows[-1]) if rows else (None, None)
    receipt = {"schema": "rl_temporal.paired_rows.v1", "cut": [a, b], "rows_written": len(rows), "file": out.name, "sha256": sha,
               "equity_first": first and float(first["equity"]), "equity_last": last and float(last["equity"]),
               "net_return_on_cut": (float(last["equity"]) / float(first["equity"]) - 1.0) if rows else None,
               "trades_on_cut": (int(last["trades"]) - int(first["trades"])) if rows else None,
               "note": "secondary strict row-parity view; the primary result is the full episode in RESULT.json"}
    (d / f"PAIRED_ROWS_{a}_{b}.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
