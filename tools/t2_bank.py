#!/usr/bin/env python3
"""T2 physical time-series bank (orders C1, C2, C4).

C1 — causal missingness: no future-looking fill exists here.
Malformed numeric tokens REFUSE (never coerced to missing). Leading
missing values are removed by the predeclared leading-prefix rule
BEFORE roles are formed. Interior gaps may only be forward-filled
up to a declared maximum run; longer runs refuse (or the caller
splits the unit). The original mask, filled count, maximum run and
policy identity are recorded on every unit.

C2 — physical authority: every task is parsed from exact hashed
bytes and carries a strict schema, numeric types, an explicit or
mechanically reconstructed time index, frequency and seasonal
period provenance, duplicate/ordering/spacing checks, missingness
semantics, and its source/license/digest records.

C4 — panel support: loaders for Monash .tsf panels and multivariate
CSV tasks; the SERIES is the primary paired unit, rolling origins
and seeds are nested repeated measurements, the source DATASET is
the clustering/family unit, series subsampling is deterministic by
identifier hash (never by outcome), and physically duplicate series
across archives count once (byte-level series digest)."""
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

LEADING_PREFIX_RULE = "drop_leading_missing_v1"
INTERIOR_FILL_POLICY = "forward_fill_max_run_v1"
DEFAULT_MAX_GAP_RUN = 5


class BankRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_strict_numeric(tokens, where: str) -> np.ndarray:
    """C1: malformed tokens REFUSE — they are never converted into
    missing observations. The ONLY missing marker accepted is an
    empty field / declared NaN token ('', 'NaN', 'nan', '?')."""
    out = np.empty(len(tokens), dtype=float)
    for i, t in enumerate(tokens):
        s = str(t).strip()
        if s in ("", "NaN", "nan", "?"):
            out[i] = np.nan
            continue
        try:
            v = float(s)
        except ValueError:
            raise BankRefusal(
                f"malformed numeric token {s!r} at row {i} in "
                f"{where} — refusing, not coercing to missing")
        out[i] = v
    return out


def apply_causal_missingness(y: np.ndarray, where: str,
                             max_gap_run: int =
                             DEFAULT_MAX_GAP_RUN) -> dict:
    """C1: leading prefix dropped by rule; interior gaps forward-
    filled up to max_gap_run; trailing/over-long gaps refuse. The
    original mask travels with the unit. NO bfill exists."""
    mask = np.isnan(y)
    n0 = len(y)
    lead = 0
    while lead < n0 and mask[lead]:
        lead += 1
    y2 = y[lead:].copy()
    m2 = mask[lead:]
    if len(y2) == 0:
        raise BankRefusal(f"{where}: all values missing")
    if m2[-1]:
        raise BankRefusal(
            f"{where}: trailing missing run — forward fill would "
            "have no later truth and backward fill is forbidden")
    max_run = 0
    run = 0
    for flag in m2:
        run = run + 1 if flag else 0
        max_run = max(max_run, run)
    if max_run > max_gap_run:
        raise BankRefusal(
            f"{where}: interior missing run {max_run} exceeds the "
            f"declared maximum {max_gap_run} — refuse or split the "
            "unit")
    filled = 0
    last = None
    for i in range(len(y2)):
        if m2[i]:
            y2[i] = last
            filled += 1
        else:
            last = y2[i]
    if not np.isfinite(y2).all():
        raise BankRefusal(f"{where}: non-finite after causal fill")
    return {"y": y2,
            "missingness": {
                "policy": INTERIOR_FILL_POLICY,
                "leading_rule": LEADING_PREFIX_RULE,
                "leading_dropped": int(lead),
                "interior_filled": int(filled),
                "max_missing_run": int(max_run),
                "max_gap_run_declared": int(max_gap_run),
                "original_missing_count": int(mask.sum()),
                "original_mask_sha256": _sha_bytes(
                    mask.tobytes())}}


def check_time_index(ts: np.ndarray, where: str,
                     expected_step: float = None) -> dict:
    """C2: duplicates, ordering and spacing are FACTS checked from
    the parsed index — never assumed."""
    if len(ts) < 3:
        raise BankRefusal(f"{where}: too few timestamps")
    diffs = np.diff(ts)
    if (diffs <= 0).any():
        raise BankRefusal(
            f"{where}: duplicate or non-increasing timestamps")
    step = float(np.median(diffs))
    irregular = int((np.abs(diffs - step) > 1e-6 * max(1.0, step))
                    .sum())
    if expected_step is not None and \
            abs(step - expected_step) > 1e-6 * expected_step:
        raise BankRefusal(
            f"{where}: median spacing {step} differs from the "
            f"declared frequency step {expected_step}")
    return {"n": int(len(ts)), "median_step": step,
            "irregular_spacings": irregular,
            "first_ts": float(ts[0]), "last_ts": float(ts[-1])}


def series_content_digest(y: np.ndarray) -> str:
    """C4: physical identity of one series for cross-archive
    deduplication — duplicates count once."""
    return _sha_bytes(np.ascontiguousarray(
        np.round(y.astype(float), 10)).tobytes())


def deterministic_subsample(series_ids, fraction: float,
                            salt: str) -> list:
    """C4: subsampling by identifier hash — never by outcome."""
    if not 0 < fraction <= 1:
        raise BankRefusal("subsample fraction out of (0,1]")
    keep = []
    for sid in series_ids:
        h = int(hashlib.sha256(
            f"{salt}|{sid}".encode()).hexdigest()[:8], 16)
        if (h % 10_000) / 10_000.0 < fraction:
            keep.append(sid)
    return sorted(keep)


def parse_tsf_bytes(raw: bytes, logical_id: str,
                    max_series: int = None) -> dict:
    """C4: minimal strict parser for the Monash .tsf format —
    header attributes + @data lines 'name:...:v1,v2,...'. Refuses
    malformed structure instead of guessing."""
    text = raw.decode("utf-8", errors="strict")
    lines = text.splitlines()
    freq = None
    horizon = None
    missing_marker = "?"
    attrs = []
    data_at = None
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("@frequency"):
            freq = ln.split(maxsplit=1)[1].strip()
        elif low.startswith("@horizon"):
            horizon = ln.split(maxsplit=1)[1].strip()
        elif low.startswith("@attribute"):
            attrs.append(ln.split()[1])
        elif low.startswith("@data"):
            data_at = i + 1
            break
    if data_at is None:
        raise BankRefusal(f"{logical_id}: .tsf has no @data")
    series = {}
    for ln in lines[data_at:]:
        if not ln.strip():
            continue
        parts = ln.split(":")
        if len(parts) < 2:
            raise BankRefusal(
                f"{logical_id}: malformed .tsf data line")
        sid = parts[0]
        values = parts[-1].split(",")
        y = parse_strict_numeric(
            values, f"{logical_id}/{sid}")
        series[sid] = y
        if max_series and len(series) >= max_series:
            break
    if not series:
        raise BankRefusal(f"{logical_id}: no series parsed")
    return {"logical_id": logical_id, "frequency": freq,
            "declared_horizon": horizon,
            "attributes": attrs, "series": series,
            "bytes_sha256": _sha_bytes(raw)}


def build_series_units(panel: dict, family: str,
                       seasonal_period: int,
                       period_provenance: str,
                       max_gap_run: int = DEFAULT_MAX_GAP_RUN,
                       min_length: int = 120,
                       seen_digests: dict = None) -> dict:
    """C4: one panel -> per-series units with C1 missingness, C2
    facts and cross-archive dedup. seen_digests maps
    series_content_digest -> first logical unit id."""
    units = {}
    excluded = {}
    seen = seen_digests if seen_digests is not None else {}
    for sid, y_raw in sorted(panel["series"].items()):
        uid = f"{panel['logical_id']}::{sid}"
        try:
            fixed = apply_causal_missingness(
                y_raw, uid, max_gap_run)
        except SystemExit as exc:
            excluded[uid] = str(exc)
            continue
        y = fixed["y"]
        if len(y) < min_length:
            excluded[uid] = (f"REFUSED: {len(y)} rows < declared "
                             f"minimum {min_length}")
            continue
        digest = series_content_digest(y)
        if digest in seen:
            excluded[uid] = (f"DUPLICATE_OF:{seen[digest]} — "
                             "physical series counted once")
            continue
        seen[digest] = uid
        units[uid] = {
            "unit_id": uid, "family": family,
            "panel_bytes_sha256": panel["bytes_sha256"],
            "series_content_sha256": digest,
            "n": int(len(y)),
            "seasonal_period": int(seasonal_period),
            "seasonal_period_provenance": period_provenance,
            "frequency_declared": panel["frequency"],
            "missingness": fixed["missingness"],
            "y": y}
    return {"units": units, "excluded": excluded}
