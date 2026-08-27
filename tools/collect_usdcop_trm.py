#!/usr/bin/env python3
"""USDCOP TRM collector (WP-DATA, Data-First order @7886de39;
DATA-SOTA-352 corrected).

Collects Colombia's official Tasa Representativa del Mercado from the
open-data Socrata dataset (datos.gov.co, dataset 32sa-8pi3, no
credentials) into a local sqlite store with full provenance.

AUTHORITY: REPORTING_ONLY. Owner order @7886de39 — the COP CDT 10%
nominal-annual hurdle is a reporting layer; TRM values NEVER enter any
fitness, objective, gate or promotion decision. Every run stamps that
authority into its provenance manifest.

DATA-SOTA-352 temporal contract:
* validity dates parse STRICTLY (impossible dates refuse), intervals
  must be ordered (vigencia_desde <= vigencia_hasta), values finite
  positive COP-per-USD;
* the provenance manifest is written atomically (fsync + rename);
* consumption goes through ONE as-of API — ``trm_as_of(store, ts)``
  returns exactly the observation whose validity interval contains the
  reporting timestamp, a typed Unavailable when none does, and a typed
  Ambiguous when intervals overlap. A future-effective publication is
  stored but NEVER returned early.

Sanitization: manifests record logical store identities only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
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


class TrmUnavailable(TrmCollectorError):
    """Typed as-of result: no stored TRM interval contains the
    reporting timestamp (future-effective rows are never used early)."""


class TrmAmbiguous(TrmCollectorError):
    """Typed as-of result: more than one stored interval contains the
    reporting timestamp (overlapping validity)."""


def parse_validity_date(value, label: str) -> date:
    """DATA-SOTA-352: strict calendar parsing — 'garbage-da' and
    2026-02-30 refuse instead of being sliced into the store."""
    text = str(value or "")
    day = text[:10]
    try:
        parsed = date.fromisoformat(day)
    except ValueError as exc:
        raise TrmCollectorError(
            f"{label}={text!r} is not a valid ISO date") from exc
    rest = text[10:]
    if rest and not rest.startswith(("T", " ")):
        raise TrmCollectorError(
            f"{label}={text!r} has a malformed time suffix")
    return parsed


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
    except (KeyError, TypeError, ValueError) as exc:
        raise TrmCollectorError(f"malformed TRM row {row!r}") from exc
    if not math.isfinite(valor) or valor <= 0:
        raise TrmCollectorError(
            f"TRM value must be finite positive COP-per-USD, got "
            f"{valor!r}")
    if "vigenciadesde" not in row:
        raise TrmCollectorError(f"malformed TRM row {row!r}")
    desde = parse_validity_date(row["vigenciadesde"], "vigenciadesde")
    hasta = parse_validity_date(
        row.get("vigenciahasta") or row["vigenciadesde"],
        "vigenciahasta")
    if desde > hasta:
        raise TrmCollectorError(
            f"inverted validity interval: vigencia_desde {desde} > "
            f"vigencia_hasta {hasta} (DATA-SOTA-352)")
    unidad = str(row.get("unidad") or "COP")
    if unidad.upper() != "COP":
        raise TrmCollectorError(
            f"unexpected TRM unit {unidad!r} (expected COP)")
    return desde.isoformat(), hasta.isoformat(), valor, unidad


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


def trm_as_of(store_path: Path, timestamp) -> dict:
    """DATA-SOTA-352: THE single consumption API. Returns the one TRM
    observation whose validity interval contains ``timestamp``'s date;
    raises TrmUnavailable when none applies (a future-effective
    publication is never returned early) and TrmAmbiguous when stored
    intervals overlap."""
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp)
        except ValueError as exc:
            raise TrmCollectorError(
                f"invalid as-of timestamp {timestamp!r}") from exc
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    as_of_day = timestamp.date().isoformat()
    connection = sqlite3.connect(store_path)
    try:
        rows = connection.execute(
            "SELECT vigencia_desde, vigencia_hasta, valor_cop_per_usd,"
            " unidad, retrieved_at FROM trm_observations WHERE"
            " vigencia_desde <= ? AND vigencia_hasta >= ?",
            (as_of_day, as_of_day)).fetchall()
    finally:
        connection.close()
    if not rows:
        raise TrmUnavailable(
            f"no TRM validity interval contains {as_of_day} "
            f"(future-effective publications are never used early)")
    if len(rows) > 1:
        raise TrmAmbiguous(
            f"{len(rows)} overlapping TRM intervals contain "
            f"{as_of_day}: "
            f"{sorted((r[0], r[1]) for r in rows)}")
    row = rows[0]
    return {"as_of": as_of_day, "vigencia_desde": row[0],
            "vigencia_hasta": row[1], "valor_cop_per_usd": row[2],
            "unidad": row[3], "retrieved_at": row[4],
            "authority": AUTHORITY}


def _atomic_write_text(path: Path, text: str) -> None:
    """DATA-SOTA-355: file fsync + atomic rename + PARENT-DIRECTORY
    fsync — without the last step a power loss can drop the renamed
    directory entry after success was reported."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


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
        "as_of_rule": ("consume ONLY via trm_as_of(store, timestamp); "
                       "future-effective rows are stored but never "
                       "returned early (DATA-SOTA-352)"),
    }
    manifest_path = store_path.with_name(
        store_path.stem + "_provenance.json")
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=1))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--start", help="YYYY-MM-DD (vigencia desde)")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--latest", action="store_true",
                        help="fetch only the newest published TRM")
    parser.add_argument("--as-of", dest="as_of",
                        help="report the TRM applicable at this ISO "
                             "timestamp (reads the store, no fetch)")
    args = parser.parse_args()
    if args.as_of:
        print(json.dumps(trm_as_of(args.store, args.as_of), indent=1))
        return 0
    if not args.latest and not args.start:
        raise SystemExit("REFUSED: pass --start (bounded backfill), "
                         "--latest, or --as-of")
    manifest = collect(args.store, args.start, args.end, args.latest)
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
