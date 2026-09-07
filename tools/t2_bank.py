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


def series_numeric_digest(y: np.ndarray) -> str:
    """C14 (honest name): EXACT NUMERIC EQUIVALENCE identity of one
    parsed float64 series — no rounding, no byte-level claim about
    the source text. Used for cross-archive deduplication:
    numerically identical series count once."""
    return _sha_bytes(np.ascontiguousarray(
        y.astype(np.float64)).tobytes())


# retired misnomer kept OFF the API surface deliberately
series_content_digest = None


# C26/C28: the ONE productive origin-geometry rule — shared by the
# design generator, the design validator and the fresh verifier so
# no two implementations can drift. Mirrors the frozen harness
# contract (unit_origins): base=int(n*frac), equal score windows,
# the last absorbing the remainder; train is always [0, o_lo).
SCREEN_MIN_LENGTH = 120
SCREEN_MIN_SCORE_WINDOW = 12          # RIDGE_LAGS(8) + 4


def origin_windows_for(n: int, rolling_origins: int,
                       base_frac: float) -> dict:
    """Exact per-unit causal windows derived from series LENGTH
    and the origin contract alone — before any result exists.
    Refuses (typed) when the frozen geometry cannot window the
    series; a refusal here is a population fact, never a score."""
    if isinstance(n, bool) or type(n) is not int or n <= 0:
        raise BankRefusal(f"origin geometry: length {n!r} is not "
                          "a positive int")
    if n < SCREEN_MIN_LENGTH:
        raise BankRefusal(
            f"GEOMETRY_INADMISSIBLE: length {n} < harness minimum "
            f"{SCREEN_MIN_LENGTH}")
    base = int(n * base_frac)
    w = (n - base) // rolling_origins
    if w < SCREEN_MIN_SCORE_WINDOW:
        raise BankRefusal(
            f"GEOMETRY_INADMISSIBLE: score window {w} < minimum "
            f"{SCREEN_MIN_SCORE_WINDOW} for {rolling_origins} "
            "rolling origins")
    out = {}
    for k_ in range(rolling_origins):
        lo = base + k_ * w
        hi = base + (k_ + 1) * w if k_ < rolling_origins - 1 \
            else n
        out[f"origin{k_}"] = {"train": [0, lo],
                              "score": [lo, hi]}
    return out


def geometry_admissible(n: int, rolling_origins: int,
                        base_frac: float) -> bool:
    try:
        origin_windows_for(n, rolling_origins, base_frac)
        return True
    except SystemExit:
        return False


def unit_time_identity(unit: dict) -> str:
    """C28: canonical TEMPORAL identity of a built unit — length,
    declared frequency, provenance and the parsed index facts —
    re-derivable from physical bytes by the fresh verifier."""
    ti = unit["time_index"]
    body = {"n": int(unit["n"]),
            "frequency_declared": unit["frequency_declared"],
            "time_provenance": unit["time_provenance"],
            "first_ts": ti["first_ts"], "last_ts": ti["last_ts"],
            "median_step": ti["median_step"]}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()


def family_top_k(ids_by_dataset: dict, k: int,
                 salt: str) -> list:
    """C21: top-k over the WHOLE FAMILY — all datasets of the
    family pooled first, global ids required unique, exact
    k=min(k,n) by lowest hash; dataset/series permutation cannot
    change the选 selection and the family total never exceeds k."""
    pooled = []
    for ds_ids in ids_by_dataset.values():
        pooled.extend(ds_ids)
    if len(pooled) != len(set(pooled)):
        raise BankRefusal(
            "duplicate global series identifier across the "
            "family's datasets")
    return deterministic_top_k(pooled, k, salt)


def deterministic_top_k(series_ids, k: int, salt: str) -> list:
    """C11: EXACT top-k selection by lowest identifier hash —
    order-independent, outcome-independent, never exceeds k."""
    if k <= 0:
        raise BankRefusal("selection k must be positive")
    ranked = sorted(
        set(series_ids),
        key=lambda sid: hashlib.sha256(
            f"{salt}|{sid}".encode()).hexdigest())
    return sorted(ranked[:min(k, len(ranked))])


def parse_tsf_bytes(raw: bytes, logical_id: str,
                    max_series: int = None) -> dict:
    """C4: minimal strict parser for the Monash .tsf format —
    header attributes + @data lines 'name:...:v1,v2,...'. Refuses
    malformed structure instead of guessing."""
    try:
        text = raw.decode("utf-8", errors="strict")
        encoding_used = "utf-8"
    except UnicodeDecodeError:
        # Monash headers occasionally carry latin-1 punctuation;
        # latin-1 is byte-bijective (no silent loss) and the
        # numeric data lines are ASCII. The choice is RECORDED.
        text = raw.decode("latin-1")
        encoding_used = "latin-1"
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
        if sid in series:
            raise BankRefusal(
                f"{logical_id}: duplicate .tsf series identifier "
                f"{sid!r} — identities are never silently "
                "overwritten")
        values = parts[-1].split(",")
        y = parse_strict_numeric(
            values, f"{logical_id}/{sid}")
        series[sid] = y
    if not series:
        raise BankRefusal(f"{logical_id}: no series parsed")
    if max_series is not None:
        raise BankRefusal(
            "max_series truncation is retired (C11): the census "
            "walks the COMPLETE panel and selects exact top-k by "
            "identifier hash afterwards")
    return {"logical_id": logical_id, "frequency": freq,
            "declared_horizon": horizon,
            "attributes": attrs, "series": series,
            "encoding_used": encoding_used,
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
        digest = series_numeric_digest(y)
        if digest in seen:
            excluded[uid] = (f"DUPLICATE_OF:{seen[digest]} — "
                             "physical series counted once")
            continue
        seen[digest] = uid
        # C12: every unit carries its per-unit temporal contract —
        # here an ORDINAL index reconstructed from the declared
        # panel frequency (never claimed as real timestamps), whose
        # length must equal the signal after the missingness
        # policy.
        ts = np.arange(len(y), dtype=float)
        time_facts = check_time_index(ts, uid, expected_step=1.0)
        if time_facts["n"] != len(y):
            raise BankRefusal(
                f"{uid}: time index length differs from the "
                "post-missingness signal")
        units[uid] = {
            "unit_id": uid, "family": family,
            "panel_bytes_sha256": panel["bytes_sha256"],
            "series_numeric_sha256": digest,
            "n": int(len(y)),
            "seasonal_period": int(seasonal_period),
            "seasonal_period_provenance": period_provenance,
            "frequency_declared": panel["frequency"],
            "time_index": time_facts,
            "time_provenance":
                "ordinal_reconstructed_from_declared_frequency",
            "missingness": fixed["missingness"],
            "y": y}
    return {"units": units, "excluded": excluded}
