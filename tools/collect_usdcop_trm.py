#!/usr/bin/env python3
"""USDCOP TRM collector (WP-DATA, Data-First order @7886de39).

Collects Colombia's official Tasa Representativa del Mercado from the
open-data Socrata dataset (datos.gov.co, dataset 32sa-8pi3, no
credentials) into a local sqlite store with full provenance.

AUTHORITY: REPORTING_ONLY. Owner order @7886de39 — the COP CDT 10%
nominal-annual hurdle is a reporting layer; TRM values NEVER enter any
fitness, objective, gate or promotion decision. Every run stamps that
authority into its provenance manifest.

Sanitization: manifests record logical store identities only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATASET_ID = "32sa-8pi3"
SOURCE_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"
AUTHORITY = ("REPORTING_ONLY — COP hurdle is a reporting layer; TRM "
             "never enters fitness/gates/promotion (owner order "
             "@7886de39)")
DEFAULT_STORE = (Path.home() / ".local/share/agent-multi/"
                 "market_reference/usdcop_trm.sqlite")


class TrmCollectorError(RuntimeError):
    """Typed refusal: malformed source rows or invalid arguments."""


def fetch_rows(start: str | None, end: str | None, latest: bool,
               timeout: float = 30.0) -> list[dict]:
    """Fetch TRM rows from the official Socrata endpoint. Injectable in
    tests via the ``fetcher`` parameter of :func:`collect`."""
    params: dict[str, str] = {"$order": "vigenciadesde DESC" if latest
                              else "vigenciadesde ASC",
                              "$limit": "1" if latest else "50000"}
    clauses = []
    if start:
        clauses.append(f"vigenciadesde >= '{start}T00:00:00.000'")
    if end:
        clauses.append(f"vigenciadesde <= '{end}T23:59:59.999'")
    if clauses:
        params["$where"] = " AND ".join(clauses)
    url = f"{SOURCE_URL}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode())


def normalize_row(row: dict) -> tuple:
    try:
        valor = float(row["valor"])
        desde = str(row["vigenciadesde"])[:10]
        hasta = str(row.get("vigenciahasta") or row["vigenciadesde"])[:10]
        unidad = str(row.get("unidad") or "COP")
    except (KeyError, TypeError, ValueError) as exc:
        raise TrmCollectorError(f"malformed TRM row {row!r}") from exc
    if valor <= 0:
        raise TrmCollectorError(f"non-positive TRM value {valor}")
    return desde, hasta, valor, unidad


def ensure_store(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE IF NOT EXISTS trm_observations (
            vigencia_desde TEXT PRIMARY KEY,
            vigencia_hasta TEXT NOT NULL,
            valor_cop_per_usd REAL NOT NULL,
            unidad TEXT NOT NULL,
            source_dataset TEXT NOT NULL,
            retrieved_at TEXT NOT NULL,
            authority TEXT NOT NULL
        )""")
    connection.commit()
    return connection


def collect(store_path: Path, start: str | None, end: str | None,
            latest: bool, fetcher=fetch_rows) -> dict:
    rows = fetcher(start, end, latest)
    if not rows:
        raise TrmCollectorError(
            "source returned no rows for the requested range")
    retrieved_at = datetime.now(timezone.utc).isoformat()
    connection = ensure_store(store_path)
    upserted = 0
    span = [None, None]
    try:
        for raw in rows:
            desde, hasta, valor, unidad = normalize_row(raw)
            connection.execute(
                """INSERT INTO trm_observations VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(vigencia_desde) DO UPDATE SET
                     vigencia_hasta=excluded.vigencia_hasta,
                     valor_cop_per_usd=excluded.valor_cop_per_usd,
                     unidad=excluded.unidad,
                     source_dataset=excluded.source_dataset,
                     retrieved_at=excluded.retrieved_at,
                     authority=excluded.authority""",
                (desde, hasta, valor, unidad, DATASET_ID, retrieved_at,
                 AUTHORITY))
            upserted += 1
            span[0] = desde if span[0] is None else min(span[0], desde)
            span[1] = desde if span[1] is None else max(span[1], desde)
        connection.commit()
        total = connection.execute(
            "SELECT COUNT(*) FROM trm_observations").fetchone()[0]
    finally:
        connection.close()
    manifest = {
        "schema": "agent_multi.usdcop_trm_provenance.v1",
        "authority": AUTHORITY,
        "source_dataset": DATASET_ID,
        "source_host": "www.datos.gov.co (Socrata open data, official "
                       "TRM publication)",
        "store": "store:agent-multi-local/market_reference/"
                 "usdcop_trm.sqlite#trm_observations",
        "retrieved_at": retrieved_at,
        "rows_upserted": upserted,
        "range_collected": {"first": span[0], "last": span[1]},
        "rows_in_store": int(total),
    }
    manifest_path = store_path.with_name(
        store_path.stem + "_provenance.json")
    manifest_path.write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--start", help="YYYY-MM-DD (vigencia desde)")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--latest", action="store_true",
                        help="fetch only the newest published TRM")
    args = parser.parse_args()
    if not args.latest and not args.start:
        raise SystemExit("REFUSED: pass --start (bounded backfill) or "
                         "--latest")
    manifest = collect(args.store, args.start, args.end, args.latest)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
