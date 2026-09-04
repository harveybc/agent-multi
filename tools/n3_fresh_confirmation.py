#!/usr/bin/env python3
"""N3_FRESH_DATA_CONFIRMATION runner, v2 (orders agent-multi@a13671ab
and the publication-integrity correction @17f6e574; contract sealed in
N3_FRESH_CONFIRMATION_CONTRACT_2026_09_04.json + supersessions).

v2 corrections (order @17f6e574):
  C1  one typed timestamp→epoch-ms helper (unit-safe, range-guarded);
      continuity requires EXACTLY 2190 overlap + 1458 extension rows;
      empty overlap can never demonstrate continuity.
  C2  restricted staging custody: root 0700 / files 0600, created and
      VERIFIED from descriptor-derived facts; permissive or symlinked
      roots refuse — never chmodded into trust.
  C3  complete per-observation evidence: exact labels and class
      probabilities per anchor; every aggregate (decomposition with
      additive identity, Brier components, recall, calibration
      deciles) derived from them; rounding only at publication.
  C4  the offline verifier requires an EXTERNAL bundle sha256 before
      parsing, verifies the sealed-contract bytes, enforces exact
      schemas, derives completeness/licenses/supports from evidence,
      and recomputes every aggregate, contrast field and the decision.
      INTERNAL_CONSISTENCY_ONLY mode exists but cannot publish
      N3_BUNDLE_VERIFIED.
  C5  strict wire grammar: exactly 12 kline fields, duplicate-key and
      non-finite JSON rejection, integer non-boolean
      timestamps/counts, canonical decimal strings; the historical
      ffill's effect is MEASURED per role and any changed 2026 cell
      refuses.

Subcommands: acquire / reattest / regenerate / execute / verify."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import stat as stat_mod
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import sha_file, sha_obj  # noqa: E402
from agent_plugins.paired_inference import holm_adjust  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "target_horizon_census_n2",
    REPO / "tools" / "target_horizon_census_n2.py")
tcn2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tcn2)

FINDATA = Path.home() / "Documents/GitHub/financial-data"
PREDICTOR = Path.home() / "Documents/GitHub/predictor"
CONTRACT = ("docs/audits/evidence/"
            "N3_FRESH_CONFIRMATION_CONTRACT_2026_09_04.json")
LAKE_PARQUET = (FINDATA
                / "market_data/crypto/spot_top50/ethusdt/4h.parquet")
LAKE_SHA = ("7a6b79833355d7c22a3db30e6494ced078d628338d76e671af320"
            "a06b35fc9e5")
FROZEN_CSV = (PREDICTOR / "examples/data/project3/"
              "ethusdt_4h_tech_stat_full_model_ready.csv")
FROZEN_SHA = ("1b447c66e68495e826c53e2ab2b08ecd3922c8fdc7357476"
              "28f8d0435ebe440f")
BAR_MS = 14_400_000
H_MAX = 12
STRIDE = 4
WINDOW = 64
WARMUP_FLOOR = 321
BOOT_SEED = 707
BOOT_B = 2000
BLOCK_LEN = 6
MARGIN_SCALE = 0.01
MARGIN_REPR = 0.005
SUPPORT_MIN = 15
EXPECTED_OVERLAP = 2190
EXPECTED_EXTENSION = 1458
EPOCH_MS_MIN = 1_400_000_000_000   # 2014-05
EPOCH_MS_MAX = 1_800_000_000_000   # 2027-01
TARGETS = {"bar_h6": 6, "bar_h12": 12}
ARMS = ("arm1", "arm2", "arm3", "arm4", "arm5")
CONTRAST_FAMILY = (("arm2", "arm1"), ("arm3", "arm2"),
                   ("arm4", "arm1"), ("arm5", "arm2"))
REPRESENTATION = {("arm3", "arm2"), ("arm4", "arm1"),
                  ("arm5", "arm2")}
BLOCKS = (("B1_JanFeb", "2026-01-01 00:00", "2026-02-28 20:00", 354),
          ("B2_MarApr", "2026-03-01 00:00", "2026-04-30 20:00", 366),
          ("B3_MayJun", "2026-05-01 00:00", "2026-06-30 20:00", 366),
          ("B4_JulAug", "2026-07-01 00:00", "2026-08-31 20:00", 372))
ROLE_FIT = ("2017-09-28 04:00", "2024-12-31 20:00")
ROLE_CAL = ("2025-01-01 00:00", "2025-12-31 20:00")
CONF_START = "2026-01-01 00:00"
CONF_END = "2026-08-31 20:00"
BUNDLE_SCHEMA_V2 = "agent_multi.n3_fresh_bundle.v2"
DECIMAL_RE = re.compile(r"^[0-9]+\.[0-9]+$")
KLINE_FIELDS = 12


class FreshRefusal(ValueError):
    """Typed refusal for any boundary violation."""


# ------------------------------------------------------------------ #
# C1: the single typed timestamp -> epoch-milliseconds helper        #
# ------------------------------------------------------------------ #

def to_epoch_ms(values):
    """Unit-safe conversion of datetime-like values to int64 epoch
    milliseconds. The subtraction/Timedelta form normalizes ANY
    pandas datetime resolution (ms, us, ns) explicitly — an integer
    is never divided before its unit is known. Rejects nulls,
    booleans, non-datetimes and out-of-range results."""
    import pandas as pd
    if isinstance(values, (bool,)) or (
            hasattr(values, "dtype")
            and values.dtype == bool):
        raise FreshRefusal("boolean input to epoch conversion")
    s = pd.to_datetime(values, utc=True, errors="coerce")
    isna = s.isna()
    if bool(getattr(isna, "any", lambda: isna)()):
        raise FreshRefusal("null/unparseable timestamp in epoch "
                           "conversion")
    ms = (s - pd.Timestamp(0, tz="UTC")) \
        // pd.Timedelta(milliseconds=1)
    import numpy as np
    arr = np.asarray(ms, dtype="int64")
    if ((arr < EPOCH_MS_MIN) | (arr > EPOCH_MS_MAX)).any():
        raise FreshRefusal(
            "epoch-ms outside the project range "
            f"[{EPOCH_MS_MIN}, {EPOCH_MS_MAX}]")
    return pd.Series(arr, index=getattr(s, "index", None))


def strict_epoch_int(x, label: str) -> int:
    """C5: integer, non-boolean, in-range wire timestamp/count."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise FreshRefusal(f"{label}: not a non-boolean integer: "
                           f"{x!r}")
    return x


def strict_decimal(x, label: str) -> float:
    """C5: canonical Binance decimal string; no silent coercion."""
    if not isinstance(x, str) or not DECIMAL_RE.match(x):
        raise FreshRefusal(
            f"{label}: not a canonical decimal string: {x!r}")
    v = float(x)
    if not math.isfinite(v):
        raise FreshRefusal(f"{label}: non-finite decimal")
    return v


def _reject_const(name):
    raise FreshRefusal(f"non-finite JSON constant {name} refused")


def _no_dup_pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise FreshRefusal(f"duplicate JSON key {k!r} refused")
        d[k] = v
    return d


def strict_json(raw: bytes):
    """C5: JSON with duplicate-key and NaN/Inf rejection."""
    return json.loads(raw, parse_constant=_reject_const,
                      object_pairs_hook=_no_dup_pairs)


# ------------------------------------------------------------------ #
# C2 custody: restricted staging (0700 root, 0600 files)             #
# ------------------------------------------------------------------ #

def secure_root(path: Path, *, create: bool) -> int:
    """Open the staging root O_NOFOLLOW and verify from the
    DESCRIPTOR: regular directory, owned by the current uid, mode
    exactly 0700. Refuses permissive/foreign/symlinked roots — never
    chmods them into trust. Returns the directory fd."""
    if create:
        if path.exists() or path.is_symlink():
            raise FreshRefusal(
                f"staging root already exists — refusing to reuse "
                f"or repair it: {path.name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        os.mkdir(path, mode=0o700)
        os.chmod(path, 0o700)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY
                     | os.O_NOFOLLOW)
    except OSError as exc:
        raise FreshRefusal(
            f"staging root not an openable real directory: "
            f"{exc}") from exc
    st = os.fstat(fd)
    if not stat_mod.S_ISDIR(st.st_mode):
        os.close(fd)
        raise FreshRefusal("staging root is not a directory")
    if st.st_uid != os.geteuid():
        os.close(fd)
        raise FreshRefusal("staging root owned by another uid")
    if (st.st_mode & 0o777) != 0o700:
        os.close(fd)
        raise FreshRefusal(
            f"staging root mode {oct(st.st_mode & 0o777)} != 0700 "
            "— permissive custody refused, not repaired")
    return fd


def secure_write(dir_fd: int, name: str, payload: bytes) -> None:
    """Create a file at exact mode 0600 under the verified root,
    O_EXCL|O_NOFOLLOW, and verify the mode from the descriptor."""
    if "/" in name:
        raise FreshRefusal("secure_write takes a bare file name")
    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | os.O_NOFOLLOW, 0o600, dir_fd=dir_fd)
    try:
        os.fchmod(fd, 0o600)
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode) or \
                (st.st_mode & 0o777) != 0o600:
            raise FreshRefusal(f"{name}: bad mode after create")
        os.write(fd, payload)
    finally:
        os.close(fd)


def secure_read(root: Path, name: str) -> bytes:
    """Read a file under a verified root, refusing symlinks,
    non-regular files, foreign owners and permissive modes."""
    dir_fd = secure_root(root, create=False)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW,
                     dir_fd=dir_fd)
    except OSError as exc:
        os.close(dir_fd)
        raise FreshRefusal(f"{name}: {exc}") from exc
    try:
        st = os.fstat(fd)
        if not stat_mod.S_ISREG(st.st_mode):
            raise FreshRefusal(f"{name}: not a regular file")
        if st.st_uid != os.geteuid():
            raise FreshRefusal(f"{name}: foreign owner")
        if (st.st_mode & 0o777) != 0o600:
            raise FreshRefusal(
                f"{name}: mode {oct(st.st_mode & 0o777)} != 0600")
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)
        os.close(dir_fd)


# ------------------------------------------------------------------ #
# geometry and decision (sealed)                                     #
# ------------------------------------------------------------------ #

def _utc(ts: str):
    return datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(
        tzinfo=timezone.utc)


def role_ledger() -> dict:
    return {"schema": "agent_multi.n3_role_ledger.v1",
            "roles": {"history_fit": list(ROLE_FIT),
                      "calibration": list(ROLE_CAL),
                      "confirmation": [CONF_START, CONF_END]},
            "blocks": {name: [start, end, bars]
                       for name, start, end, bars in BLOCKS},
            "purge_bars": H_MAX, "stride": STRIDE,
            "window": WINDOW}


def scoring_anchor_offsets(block_bars: int) -> list:
    return [i for i in range(0, block_bars, STRIDE)
            if i + H_MAX < block_bars]


def canonical_anchor_datetimes(block_name: str) -> list:
    """The exact anchor timestamp list implied by the sealed blocks,
    stride and purge — derived from the CONTRACT, not the bundle."""
    from datetime import timedelta
    for name, start, end, bars in BLOCKS:
        if name == block_name:
            t0 = _utc(start).replace(tzinfo=None)
            return [(t0 + timedelta(hours=4 * i)).strftime(
                "%Y-%m-%d %H:%M:%S")
                for i in scoring_anchor_offsets(bars)]
    raise FreshRefusal(f"unknown block {block_name}")


def decide(contrast_stats: dict, blocks_complete: bool,
           licenses_ok: bool) -> str:
    if not blocks_complete:
        return "FRESH_CONFIRMATION_INSUFFICIENT"
    if not licenses_ok:
        return "FRESH_CONFIRMATION_INCONCLUSIVE"

    def ok(key, margin):
        s = contrast_stats[key]
        return (s["all_blocks_positive"]
                and s["pooled_skill"] >= margin
                and s["holm_p"] < 0.05)
    if any(ok((t, a, b), MARGIN_REPR)
           for t in TARGETS for (a, b) in REPRESENTATION):
        return "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
    if all(ok((t, "arm2", "arm1"), MARGIN_SCALE) for t in TARGETS):
        return ("TARGET_SCALE_EFFECT_CONFIRMED_NO_REPRESENTATION"
                "_SIGNAL")
    return "TARGET_SCALE_EFFECT_NOT_CONFIRMED"


# ------------------------------------------------------------------ #
# C3: per-observation metrics — everything derives from             #
# (labels, probabilities); rounding only at publication             #
# ------------------------------------------------------------------ #

def unit_metrics(labels, probs) -> dict:
    """All predeclared metrics for one (unit, arm) from exact labels
    and class probabilities. Refuses invalid simplexes, length
    disagreement, non-finite values; proves the additive identity
    multiclass = hit/censored + hit_indicator * direction|hit."""
    import numpy as np
    y = np.asarray(labels)
    p = np.asarray(probs, dtype="float64")
    if p.ndim != 2 or p.shape[1] != 3 or len(y) != len(p):
        raise FreshRefusal("labels/probability shape disagreement")
    if not np.isfinite(p).all():
        raise FreshRefusal("non-finite probability")
    if (p < 0).any() or (p > 1).any() or \
            (np.abs(p.sum(axis=1) - 1.0) > 1e-9).any():
        raise FreshRefusal("invalid probability simplex")
    if not np.isin(y, (0, 1, 2)).all():
        raise FreshRefusal("label outside {0,1,2}")
    pc = np.clip(p[np.arange(len(y)), y], 1e-12, None)
    lm = -np.log(pc)
    p_hit = np.clip(p[:, 0] + p[:, 1], 1e-12, 1 - 1e-12)
    is_hit = y < 2
    l_hit = np.where(is_hit, -np.log(p_hit), -np.log(1 - p_hit))
    l_dir = np.zeros(len(y))
    hi = np.where(is_hit)[0]
    if len(hi):
        l_dir[hi] = -np.log(np.clip(
            p[hi, y[hi]] / p_hit[hi], 1e-12, None))
    if not np.allclose(lm, l_hit + l_dir, atol=1e-9):
        raise FreshRefusal("additive identity violated")
    onehot = np.eye(3)[y]
    pred = p.argmax(axis=1)
    recall = {}
    unavailable = []
    for c in (0, 1, 2):
        n_c = int((y == c).sum())
        if n_c == 0:
            recall[str(c)] = None
            unavailable.append(c)
        else:
            recall[str(c)] = round(float(
                ((pred == c) & (y == c)).sum() / n_c), 6)
    edges = np.quantile(p_hit, np.linspace(0, 1, 11))
    deciles = []
    for d in range(10):
        lo, hi_e = edges[d], edges[d + 1]
        sel = (p_hit >= lo) & (p_hit <= hi_e if d == 9
                               else p_hit < hi_e)
        if sel.sum():
            deciles.append({
                "bin": d, "n": int(sel.sum()),
                "mean_predicted_hit": round(
                    float(p_hit[sel].mean()), 6),
                "observed_hit_rate": round(
                    float(is_hit[sel].mean()), 6)})
    return {
        "multiclass_logloss_mean": round(float(lm.mean()), 6),
        "hit_vs_censored_mean": round(float(l_hit.mean()), 6),
        "direction_given_hit_mean": (
            round(float(l_dir[is_hit].mean()), 6)
            if is_hit.any() else None),
        "additive_identity_max_abs_gap": round(float(
            np.abs(lm - (l_hit + l_dir)).max()), 12),
        "brier": round(float(
            ((p - onehot) ** 2).sum(axis=1).mean()), 6),
        "brier_components": {
            str(c): round(float(
                ((p[:, c] - (y == c)) ** 2).mean()), 6)
            for c in (0, 1, 2)},
        "recall_argmax": recall,
        "recall_unavailable_classes": unavailable,
        "calibration_deciles_hit": deciles,
    }


def derive_losses(labels, probs):
    """Per-observation multiclass log loss from exact evidence."""
    import numpy as np
    y = np.asarray(labels)
    p = np.asarray(probs, dtype="float64")
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))


# ------------------------------------------------------------------ #
# shared contrast rederivation (single bootstrap implementation)     #
# ------------------------------------------------------------------ #

def _rederive(units):
    """Contrasts, bootstrap p, Holm and decision inputs derived
    purely from unit (labels, probabilities) evidence."""
    import numpy as np
    losses = {}
    for u in units:
        for arm, rec in u.get("arms", {}).items():
            losses[(u["unit"], arm)] = derive_losses(
                u["labels"], rec["probs"])
    pvals, stats = {}, {}
    complete = True
    for tkey in TARGETS:
        tunits = {u["block"]: u for u in units
                  if u["unit"].startswith(tkey)}
        for (a, b) in CONTRAST_FAMILY:
            per_block = {}
            diffs = []
            for name, _, _, _ in BLOCKS:
                u = tunits.get(name)
                if u is None or a not in u.get("arms", {}) \
                        or b not in u.get("arms", {}):
                    complete = False
                    continue
                la = losses[(u["unit"], a)]
                lb = losses[(u["unit"], b)]
                per_block[name] = round(
                    1.0 - float(la.sum() / lb.sum()), 6)
                diffs.append(lb - la)
            if len(per_block) < 4:
                complete = False
                continue
            pooled = round(1.0 - float(
                sum(losses[(tunits[nm]["unit"], a)].sum()
                    for nm, _, _, _ in BLOCKS)
                / sum(losses[(tunits[nm]["unit"], b)].sum()
                      for nm, _, _, _ in BLOCKS)), 6)
            rng = np.random.default_rng(BOOT_SEED)
            n_low = 0
            for _ in range(BOOT_B):
                parts = []
                for d in diffs:
                    m = len(d)
                    n_blocks = math.ceil(m / BLOCK_LEN)
                    starts = rng.integers(0, m, size=n_blocks)
                    idx = (starts[:, None]
                           + np.arange(BLOCK_LEN)[None, :]
                           ).reshape(-1) % m
                    parts.append(d[idx[:m]])
                if float(np.concatenate(parts).mean()) <= 0.0:
                    n_low += 1
            p = (1 + n_low) / (BOOT_B + 1)
            ckey = f"{tkey}:{a}-vs-{b}"
            pvals[ckey] = min(1.0, p)
            stats[(tkey, a, b)] = {
                "pooled_skill": pooled,
                "per_block_skill": per_block,
                "all_blocks_positive": all(
                    v > 0 for v in per_block.values()),
                "bootstrap_p": ("<= 1/2001"
                                if p <= 1 / (BOOT_B + 1) + 1e-12
                                else round(p, 6))}
    holm = holm_adjust(pvals) if pvals else {}
    contrast_stats, contrasts_out = {}, {}
    for (tkey, a, b), s in stats.items():
        ckey = f"{tkey}:{a}-vs-{b}"
        s["holm_p"] = round(holm[ckey], 6)
        contrast_stats[(tkey, a, b)] = {
            "pooled_skill": s["pooled_skill"],
            "all_blocks_positive": s["all_blocks_positive"],
            "holm_p": s["holm_p"]}
        contrasts_out[ckey] = s
    return contrasts_out, contrast_stats, complete


# ------------------------------------------------------------------ #
# wire-row validation (C5 grammar) + continuity (C1)                 #
# ------------------------------------------------------------------ #

def validate_wire_rows(rows, acquired_at_ms: int):
    """Exact Binance kline grammar and market validity."""
    if not rows:
        raise FreshRefusal("empty acquisition")
    for r in rows:
        if not isinstance(r, list) or len(r) != KLINE_FIELDS:
            raise FreshRefusal(
                f"schema drift: kline row with "
                f"{len(r) if isinstance(r, list) else '?'} fields "
                f"!= {KLINE_FIELDS}")
        strict_epoch_int(r[0], "open_time")
        strict_epoch_int(r[6], "close_time")
        strict_epoch_int(r[8], "trade_count")
        if r[8] < 0:
            raise FreshRefusal("negative trade_count")
        o = strict_decimal(r[1], "open")
        h = strict_decimal(r[2], "high")
        low = strict_decimal(r[3], "low")
        c = strict_decimal(r[4], "close")
        vol = strict_decimal(r[5], "volume")
        strict_decimal(r[7], "quote_volume")
        strict_decimal(r[9], "taker_buy_base_volume")
        strict_decimal(r[10], "taker_buy_quote_volume")
        if min(o, h, low, c) <= 0 or vol < 0:
            raise FreshRefusal("non-positive OHLC or negative "
                               "volume")
        if h < max(o, c) or low > min(o, c):
            raise FreshRefusal("invalid OHLC geometry")
        if r[0] % BAR_MS != 0:
            raise FreshRefusal("open_time not on the 4h grid")
        if r[6] != r[0] + BAR_MS - 1:
            raise FreshRefusal("close_time != open_time + 4h - 1ms")
        if r[6] >= acquired_at_ms:
            raise FreshRefusal("partially open terminal bar")
    opens = [r[0] for r in rows]
    if len(set(opens)) != len(opens):
        raise FreshRefusal("duplicate open_time")
    for prev, cur in zip(opens, opens[1:]):
        if cur - prev != BAR_MS:
            raise FreshRefusal(
                f"grid gap/overlap between {prev} and {cur}")


MARKET_FIELDS = [(1, "open"), (2, "high"), (3, "low"),
                 (4, "close"), (5, "volume"),
                 (9, "taker_buy_base_volume")]
DERIVED_FIELDS = [(7, "quote_volume"),
                  (10, "taker_buy_quote_volume")]


def _ulp_distance(a: float, b: float) -> int:
    import struct
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    return abs(ia - ib)


def _verify_overlap(overlap, lake_2025, lake_ms_2025):
    """Continuity per the disclosed field-class amendment
    (@17f6e574 supersessions): MARKET fields, timestamps and
    trade_count must be float64-BITWISE exact (adversary 5: a
    revision hidden by float32 coercion still refuses); the two
    DERIVED-AGGREGATE fields tolerate at most 1 ulp of Binance
    re-serialization jitter, with counts returned for the receipt.
    Anything beyond refuses."""
    ulp_report = {col: {"cells_1ulp": 0, "max_ulp": 0}
                  for _, col in DERIVED_FIELDS}
    for i, r in enumerate(overlap):
        if r[0] != int(lake_ms_2025.iloc[i]):
            raise FreshRefusal(
                "SOURCE_CONTINUITY_NOT_DEMONSTRATED: timestamp "
                f"order mismatch at overlap row {i}")
        for j, col in MARKET_FIELDS:
            if float(r[j]) != float(lake_2025[col].iloc[i]):
                raise FreshRefusal(
                    "SOURCE_CONTINUITY_NOT_DEMONSTRATED: MARKET "
                    f"field {col} revised at open_time {r[0]}: "
                    f"api {r[j]} vs frozen "
                    f"{lake_2025[col].iloc[i]}")
        for j, col in DERIVED_FIELDS:
            fa, fl = float(r[j]), float(lake_2025[col].iloc[i])
            if fa != fl:
                d = _ulp_distance(fa, fl)
                if d > 1:
                    raise FreshRefusal(
                        "SOURCE_CONTINUITY_NOT_DEMONSTRATED: "
                        f"derived field {col} deviates {d} ulp at "
                        f"open_time {r[0]} — beyond the disclosed "
                        "1-ulp serialization tolerance")
                ulp_report[col]["cells_1ulp"] += 1
                ulp_report[col]["max_ulp"] = max(
                    ulp_report[col]["max_ulp"], d)
        if int(r[8]) != int(lake_2025["trade_count"].iloc[i]):
            raise FreshRefusal(
                "SOURCE_CONTINUITY_NOT_DEMONSTRATED: trade_count "
                f"revised at {r[0]}")
    return ulp_report


def continuity(rows):
    """C1: exact 2190/1458 accounting; empty overlap NEVER
    demonstrates continuity."""
    import pandas as pd
    if sha_file(LAKE_PARQUET) != LAKE_SHA:
        raise FreshRefusal("frozen lake parquet digest changed")
    lake = pd.read_parquet(LAKE_PARQUET)
    lake_ms = to_epoch_ms(lake["open_time"])
    start_ms = int(_utc("2025-01-01 00:00").timestamp() * 1000)
    sel = lake_ms >= start_ms
    lake_2025 = lake[sel].reset_index(drop=True)
    lake_ms_2025 = lake_ms[sel].reset_index(drop=True)
    lake_last = int(lake_ms.iloc[-1])
    overlap = [r for r in rows if r[0] <= lake_last]
    extension = [r for r in rows if r[0] > lake_last]
    if len(overlap) != EXPECTED_OVERLAP:
        raise FreshRefusal(
            "SOURCE_CONTINUITY_NOT_DEMONSTRATED: overlap rows "
            f"{len(overlap)} != {EXPECTED_OVERLAP} — an empty or "
            "short overlap can never demonstrate continuity")
    if len(extension) != EXPECTED_EXTENSION:
        raise FreshRefusal(
            f"extension rows {len(extension)} != "
            f"{EXPECTED_EXTENSION}")
    if len(lake_2025) != EXPECTED_OVERLAP:
        raise FreshRefusal(
            f"frozen 2025 rows {len(lake_2025)} != "
            f"{EXPECTED_OVERLAP}")
    ulp_report = _verify_overlap(overlap, lake_2025,
                                 lake_ms_2025)
    ext_first = extension[0][0]
    conf_start_ms = int(_utc(CONF_START).timestamp() * 1000)
    if ext_first != conf_start_ms:
        raise FreshRefusal(
            f"first extension bar {ext_first} != confirmation "
            f"start {conf_start_ms}")
    return overlap, extension, lake_last, ulp_report


# ------------------------------------------------------------------ #
# D2: acquire (network) and C1/C2: reattest from frozen pages        #
# ------------------------------------------------------------------ #

def acquire(staging: Path) -> dict:
    import requests
    dir_fd = secure_root(staging, create=True)
    os.close(dir_fd)
    start_ms = int(_utc("2025-01-01 00:00").timestamp() * 1000)
    end_open_ms = int(_utc(CONF_END).timestamp() * 1000)
    acquired_at = datetime.now(timezone.utc)
    acquired_at_ms = int(acquired_at.timestamp() * 1000)
    if end_open_ms + BAR_MS >= acquired_at_ms:
        raise FreshRefusal("terminal confirmation bar not yet "
                           "closed at acquisition time")
    receipts, rows = [], []
    cursor, page = start_ms, 0
    while cursor <= end_open_ms:
        params = {"symbol": "ETHUSDT", "interval": "4h",
                  "startTime": cursor,
                  "endTime": end_open_ms + BAR_MS - 1,
                  "limit": 1000}
        try:
            resp = requests.get(
                "https://api.binance.com/api/v3/klines",
                params=params, timeout=30)
        except Exception as exc:
            raise FreshRefusal(
                f"PUBLIC_DATA_ACQUISITION_BLOCKED: {exc}") from exc
        if resp.status_code == 429:
            time.sleep(10)
            continue
        if resp.status_code != 200:
            raise FreshRefusal(
                f"PUBLIC_DATA_ACQUISITION_BLOCKED: HTTP "
                f"{resp.status_code}")
        raw = resp.content
        payload = strict_json(raw)
        if not payload:
            break
        dir_fd = secure_root(staging, create=False)
        try:
            secure_write(dir_fd, f"page_{page:03d}.json", raw)
        finally:
            os.close(dir_fd)
        receipts.append({
            "page": page,
            "request": {"symbol": "ETHUSDT", "interval": "4h",
                        "startTime": params["startTime"],
                        "endTime": params["endTime"],
                        "limit": 1000},
            "status": resp.status_code,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "acquired_at_utc": datetime.now(
                timezone.utc).isoformat(),
            "first_open_time": payload[0][0],
            "last_open_time": payload[-1][0],
            "n_rows": len(payload)})
        rows.extend(payload)
        cursor = payload[-1][0] + BAR_MS
        page += 1
        time.sleep(0.4)
    return _attest(staging, rows, receipts, acquired_at_ms,
                   version=1, supersedes=None)


def _attest(staging: Path, rows, page_receipts, acquired_at_ms,
            *, version: int, supersedes) -> dict:
    import pandas as pd
    validate_wire_rows(rows, acquired_at_ms)
    overlap, extension, lake_last, ulp_report = continuity(rows)
    receipt = {
        "schema": f"agent_multi.n3_acquisition_receipt.v{version}",
        "contract": CONTRACT,
        "attested_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_identity": "Binance Spot public /api/v3/klines",
        "pages": page_receipts,
        "rows_total": len(rows),
        "rows_overlap_verified_exact": len(overlap),
        "rows_2026": len(extension),
        "derived_field_serialization_jitter": {
            "rule": "disclosed 1-ulp tolerance for the two "
                    "derived-aggregate fields only; market fields "
                    "bitwise exact",
            **ulp_report},
        "last_closed_open_time": rows[-1][0],
        "verdict": "SOURCE_CONTINUITY_DEMONSTRATED"}
    if supersedes:
        receipt["supersedes"] = supersedes
    table = pd.DataFrame(
        rows, columns=["open_time", "open", "high", "low", "close",
                       "volume", "close_time", "quote_volume",
                       "trade_count", "taker_buy_base_volume",
                       "taker_buy_quote_volume", "ignore"])
    import io
    buf = io.BytesIO()
    table.to_parquet(buf)
    dir_fd = secure_root(staging, create=False)
    try:
        secure_write(dir_fd, "acquired.parquet", buf.getvalue())
        receipt["acquired_parquet_sha256"] = hashlib.sha256(
            buf.getvalue()).hexdigest()
        secure_write(dir_fd, "acquisition_receipt.json",
                     (json.dumps(receipt, indent=1) + "\n")
                     .encode())
    finally:
        os.close(dir_fd)
    return receipt


def reattest(v1_staging: Path, v2_staging: Path,
             v1_receipt_path: Path) -> dict:
    """C1+C2: rebuild custody and continuity from the ALREADY
    acquired raw page bytes. No network request. Copies only bytes
    whose digest matches the frozen v1 page receipts, into a fresh
    0700 root with 0600 files."""
    v1_receipt_bytes = v1_receipt_path.read_bytes()
    v1 = strict_json(v1_receipt_bytes)
    dir_fd = secure_root(v2_staging, create=True)
    os.close(dir_fd)
    rows = []
    page_receipts = []
    for page_rec in v1["pages"]:
        name = f"page_{page_rec['page']:03d}.json"
        raw = (v1_staging / name).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != page_rec["sha256"]:
            raise FreshRefusal(
                f"{name}: bytes changed after their receipt digest "
                f"({digest[:12]} != {page_rec['sha256'][:12]})")
        dir_fd = secure_root(v2_staging, create=False)
        try:
            secure_write(dir_fd, name, raw)
        finally:
            os.close(dir_fd)
        payload = strict_json(raw)
        rows.extend(payload)
        page_receipts.append(page_rec)
    acquired_at_ms = int(datetime.now(timezone.utc)
                         .timestamp() * 1000)
    receipt = _attest(
        v2_staging, rows, page_receipts, acquired_at_ms,
        version=2,
        supersedes={
            "v1_receipt_sha256": hashlib.sha256(
                v1_receipt_bytes).hexdigest(),
            "v1_status": "PRESERVED UNCHANGED, SUPERSEDED",
            "correction_map": {
                "rows_overlap_verified_exact": {
                    "v1": v1["rows_overlap_verified_exact"],
                    "v2": EXPECTED_OVERLAP,
                    "cause": "timestamp-unit bug (datetime64[ms] "
                             "int64 divided by 1e6 again) made both "
                             "overlap sets empty; 0 == 0 passed "
                             "vacuously"},
                "rows_2026": {"v1": v1["rows_2026"],
                              "v2": EXPECTED_EXTENSION,
                              "cause": "same unit bug classified "
                                       "every acquired row as "
                                       "extension"},
                "custody": {"v1": "root 0775, files 0664",
                            "v2": "root 0700, files 0600, "
                                  "descriptor-verified"}}})
    return receipt


# ------------------------------------------------------------------ #
# D3: regenerate (v2 semantics)                                      #
# ------------------------------------------------------------------ #

def _load_stage22():
    spec = importlib.util.spec_from_file_location(
        "stage22", FINDATA / "_scripts/workers/"
        "stage22_trading_features_worker.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def regenerate(staging: Path) -> dict:
    import numpy as np
    import pandas as pd
    stage22 = _load_stage22()
    lake = pd.read_parquet(LAKE_PARQUET)
    acq = pd.read_parquet(
        __import__("io").BytesIO(
            secure_read(staging, "acquired.parquet")))
    acq_ms = acq["open_time"].apply(
        lambda x: strict_epoch_int(int(x), "open_time"))
    lake_ms = to_epoch_ms(lake["open_time"])
    lake_last_ms = int(lake_ms.iloc[-1])
    ext = acq[acq_ms > lake_last_ms].copy()
    if len(ext) != EXPECTED_EXTENSION:
        raise FreshRefusal(
            f"extension selection {len(ext)} != "
            f"{EXPECTED_EXTENSION} rows after the true final "
            "frozen timestamp")
    first_ext_ms = int(acq_ms[acq_ms > lake_last_ms].iloc[0])
    if first_ext_ms != int(_utc(CONF_START).timestamp() * 1000):
        raise FreshRefusal(
            "first extension row is not 2026-01-01T00:00:00Z")
    ext["open_time"] = pd.to_datetime(
        pd.to_numeric(ext["open_time"]), unit="ms", utc=True)
    ext["close_time"] = pd.to_datetime(
        pd.to_numeric(ext["close_time"]), unit="ms", utc=True)
    for col in ("open", "high", "low", "close", "volume",
                "quote_volume", "taker_buy_base_volume",
                "taker_buy_quote_volume"):
        ext[col] = pd.to_numeric(ext[col])
    ext["trade_count"] = pd.to_numeric(ext["trade_count"])
    full_raw = pd.concat(
        [lake, ext[lake.columns]], ignore_index=True)
    base = full_raw.rename(columns={"open_time": "timestamp"})
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    if base["timestamp"].duplicated().any():
        raise FreshRefusal(
            "duplicate timestamp between frozen lake and "
            "extension — an overlap row would be replaced; refused")
    base = base.sort_values("timestamp").reset_index(drop=True)
    for col in ("open", "high", "low", "close", "volume"):
        base[col] = pd.to_numeric(base[col], errors="coerce")
    t26 = base["timestamp"]
    conf = base[(t26 >= _utc(CONF_START))
                & (t26 <= _utc(CONF_END))]
    if len(conf) != EXPECTED_EXTENSION:
        raise FreshRefusal(
            f"confirmation grid incomplete: {len(conf)} != "
            f"{EXPECTED_EXTENSION}")
    tech = stage22.compute_technical(base)
    stat = stage22.compute_statistical(base)
    stat = stat.rename(columns={"log_return_1":
                                "statistical__log_return_1"})
    merged = pd.DataFrame({
        "DATE_TIME": base["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S"),
        "typical_price": ((base["high"] + base["low"]
                           + base["close"]) / 3),
        "OPEN": base["open"], "HIGH": base["high"],
        "LOW": base["low"], "CLOSE": base["close"],
        "VOLUME": base["volume"]})
    frozen_cols = json.loads(
        (PREDICTOR / "examples/data/project3/"
         "ethusdt_4h_tech_stat_export_metadata.json").read_text()
    )["columns"]
    feature_cols = [c for c in frozen_cols
                    if c not in merged.columns]
    for col in feature_cols:
        if col in tech.columns:
            merged[col] = tech[col].to_numpy()
        elif col in stat.columns:
            merged[col] = stat[col].to_numpy()
        else:
            raise FreshRefusal(f"feature {col} produced by neither "
                               "stage-2.2 table")
    merged = merged[frozen_cols]
    numeric = [c for c in frozen_cols if c != "DATE_TIME"]
    # C5: measure the historical-compatibility ffill BY ROLE and
    # refuse if it changes any 2026 input cell
    before = merged[numeric].replace([np.inf, -np.inf], np.nan)
    after = before.ffill()
    changed = before.isna() & after.notna()
    is_2026 = (pd.to_datetime(merged["DATE_TIME"])
               >= _utc(CONF_START).replace(tzinfo=None))
    changed_2026 = int(changed[is_2026.to_numpy()].sum().sum())
    changed_hist = int(changed[~is_2026.to_numpy()].sum().sum())
    if changed_2026 != 0:
        raise FreshRefusal(
            f"ffill changed {changed_2026} confirmation-era cells "
            "— imputation into 2026 refused")
    merged[numeric] = after
    model_ready = merged.dropna(
        subset=[c for c in feature_cols]).reset_index(drop=True)
    if sha_file(FROZEN_CSV) != FROZEN_SHA:
        raise FreshRefusal("frozen model-ready CSV digest changed")
    frozen = pd.read_csv(FROZEN_CSV)
    n_over = len(frozen)
    regen_over = model_ready.iloc[:n_over]
    if list(regen_over["DATE_TIME"]) != list(frozen["DATE_TIME"]):
        raise FreshRefusal("SOURCE_OR_PIPELINE_DRIFT: DATE_TIME "
                           "sequence mismatch on overlap")
    if list(model_ready.columns) != list(frozen.columns):
        raise FreshRefusal("SOURCE_OR_PIPELINE_DRIFT: column order "
                           "mismatch")
    exact_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME",
                  "vol_regime_high", "vol_regime_low"]
    parity, drift = {}, []
    for col in [c for c in frozen_cols if c != "DATE_TIME"]:
        a = regen_over[col].to_numpy(dtype="float64")
        b = frozen[col].to_numpy(dtype="float64")
        if col in exact_cols:
            ok = bool(np.array_equal(a, b))
            parity[col] = {"class": "exact", "equal": ok}
            if not ok:
                drift.append(col)
            continue
        a32 = a.astype("float32")
        b32 = b.astype("float32")
        bit = a32 == b32
        both_nan = np.isnan(a) & np.isnan(b)
        mism = ~(bit | both_nan)
        absdev = np.abs(a - b)
        with np.errstate(divide="ignore", invalid="ignore"):
            reldev = absdev / np.maximum(np.abs(b), 1e-30)
        inside = (absdev <= 1e-6) | (reldev <= 1e-5)
        bad = mism & ~inside & ~both_nan
        parity[col] = {
            "class": "float32_envelope",
            "bit_exact_frac": round(
                float((bit | both_nan).mean()), 6),
            "max_abs_dev": float(np.nanmax(absdev))
            if len(absdev) else 0.0,
            "max_rel_dev": float(np.nanmax(reldev))
            if len(reldev) else 0.0,
            "cells_outside_envelope": int(bad.sum())}
        if bad.sum():
            drift.append(col)
    verdict = ("SOURCE_OR_PIPELINE_DRIFT" if drift
               else "OVERLAP_PARITY_DEMONSTRATED")
    report = {"schema": "agent_multi.n3_parity_report.v2",
              "contract": CONTRACT,
              "overlap_rows": n_over,
              "regenerated_rows": len(model_ready),
              "rows_2026": len(model_ready) - n_over,
              "ffill_cells_changed": {
                  "historical_warmup_compat": changed_hist,
                  "confirmation_2026": changed_2026},
              "verdict": verdict,
              "drifted_features": drift,
              "per_feature": parity}
    import io
    buf = io.BytesIO()
    model_ready.to_parquet(buf)
    dir_fd = secure_root(staging, create=False)
    try:
        secure_write(dir_fd, "model_ready_extended.parquet",
                     buf.getvalue())
        report["extended_sha256"] = hashlib.sha256(
            buf.getvalue()).hexdigest()
        secure_write(dir_fd, "parity_report.json",
                     (json.dumps(report, indent=1) + "\n")
                     .encode())
    finally:
        os.close(dir_fd)
    if drift:
        raise FreshRefusal(f"SOURCE_OR_PIPELINE_DRIFT: {drift[:5]}")
    return report


# ------------------------------------------------------------------ #
# D4: frozen five-arm execution (v2 bundle)                          #
# ------------------------------------------------------------------ #

def _anchor_indices(df, start, end):
    import pandas as pd
    ts = pd.to_datetime(df["DATE_TIME"])
    lo = ts.searchsorted(pd.Timestamp(_utc(start).replace(
        tzinfo=None)))
    hi = ts.searchsorted(pd.Timestamp(_utc(end).replace(
        tzinfo=None)), side="right")
    return list(range(lo, hi))


def execute(staging: Path, out_bundle: Path,
            v1_bundle: Path | None = None) -> dict:
    import numpy as np
    import pandas as pd
    from agent_plugins.branch_pretraining import barrier_hit_labels
    started = time.time()
    df = pd.read_parquet(
        __import__("io").BytesIO(
            secure_read(staging, "model_ready_extended.parquet")))
    n = len(df)
    closes = df["CLOSE"].to_numpy()
    returns = np.diff(np.log(closes))
    max_a = n - H_MAX - 1
    steps = list(range(WINDOW + 4 * 3 + 1, max_a + 2))
    labels_all = barrier_hit_labels(
        df["OPEN"].to_numpy(), df["HIGH"].to_numpy(),
        df["LOW"].to_numpy(), closes, steps, [6, 12], 64, 2.0, 2.0,
        1e-8)
    label_row = {s - 1: i for i, s in enumerate(steps)}
    sq = np.concatenate([[0.0], np.cumsum(returns ** 2)])
    scale = np.full((n, 4), np.nan)
    for k in range(4):
        a = np.arange(WINDOW + 4 * 3 + 1, n)
        a2 = a - 4 * k
        valid = a2 >= 65
        av, a2v = a[valid], a2[valid]
        scale[av, k] = np.sqrt(
            (sq[a2v] - sq[a2v - 64]) / 64.0) + 1e-8
    from agent_plugins.branch_pretraining import (
        collect_preprocessed_windows)
    from agent_plugins.pretrained_branch_loader import verify_source
    pretrain_dir = (Path.home() / ".local/share/agent-multi/"
                    "restricted_evidence/"
                    "candidate_full5_pcgrad_o2022_20260828")
    split_contract = json.loads(
        (REPO / "examples/config/phase_3_eth_sac_dynamics/splits/"
         "eth_nested_split_contract_o2022_paired_v1.json")
        .read_text())
    source = verify_source(pretrain_dir, REPO,
                           Path(split_contract["source_csv"]))
    contract = source["contract"]
    env_source = contract["observation_pipeline"]["source_config"]
    env_config = json.loads(
        (Path(env_source) if Path(env_source).is_absolute()
         else REPO / env_source).read_text())
    contract_w = {**contract, "window_size": WINDOW}
    env_w = {**env_config, "window_size": WINDOW}

    def summary_for(rows):
        steps_w = [a + 1 for a in rows]
        win = collect_preprocessed_windows(df, contract_w, env_w,
                                           steps_w)
        return np.concatenate(
            [win[:, -1, :], win.mean(axis=1), win.std(axis=1)],
            axis=1).astype("float64")

    def stride_rows(start, end):
        rows = _anchor_indices(df, start, end)
        boundary = rows[-1] + 1
        return [r for r in rows[::STRIDE]
                if r + H_MAX < boundary and r in label_row
                and r >= WARMUP_FLOOR]
    fit_rows = stride_rows(*ROLE_FIT)
    cal_rows = stride_rows(*ROLE_CAL)
    block_rows = {}
    blocks_complete = True
    for name, start, end, bars in BLOCKS:
        rows = _anchor_indices(df, start, end)
        if len(rows) != bars:
            blocks_complete = False
        offs = scoring_anchor_offsets(len(rows))
        block_rows[name] = [rows[i] for i in offs
                            if rows[i] in label_row]
    ledger = role_ledger()
    ledger["anchor_counts"] = {
        "fit": len(fit_rows), "cal": len(cal_rows),
        **{k: len(v) for k, v in block_rows.items()}}
    score_rows = [r for v in block_rows.values() for r in v]
    all_rows = fit_rows + cal_rows + score_rows
    summ = summary_for(all_rows)
    summaries = {r: summ[i] for i, r in enumerate(all_rows)}

    def xmat(rows, arm):
        if arm == "arm2":
            x = np.array([[scale[r, 0]] for r in rows])
        elif arm == "arm3":
            x = np.array([scale[r] for r in rows])
        elif arm == "arm4":
            x = np.array([summaries[r] for r in rows])
        else:
            x = np.array(
                [np.concatenate([scale[r], summaries[r]])
                 for r in rows])
        if not np.isfinite(x).all():
            raise FreshRefusal(
                f"non-finite feature cell in {arm} matrix — "
                "refused, never imputed")
        return x

    units = []
    licenses_ok = True
    for tkey, h in TARGETS.items():
        hcol = 0 if h == 6 else 1
        y = {r: int(labels_all[label_row[r], hcol])
             for r in all_rows}
        yf = np.array([y[r] for r in fit_rows])
        yc = np.array([y[r] for r in cal_rows])
        counts = np.bincount(np.concatenate([yf, yc]),
                             minlength=3)
        prior = np.clip(counts / counts.sum(), 1e-12, None)
        prior = prior / prior.sum()
        arm_probs, arm_recs = {}, {}
        degenerate = None
        for arm in ARMS:
            if arm == "arm1":
                arm_probs[arm] = np.tile(prior,
                                         (len(score_rows), 1))
                arm_recs[arm] = {"prior_from": "fit+calibration"}
                continue
            probs, rec = tcn2._logistic(
                xmat(fit_rows, arm), yf, xmat(cal_rows, arm), yc,
                xmat(score_rows, arm))
            if probs is None:
                degenerate = arm
                break
            arm_probs[arm] = probs
            arm_recs[arm] = rec
        row_pos = {r: i for i, r in enumerate(score_rows)}
        for name, _, _, _ in BLOCKS:
            rows_s = block_rows[name]
            ys = [y[r] for r in rows_s]
            payload = {"unit": f"{tkey}:{name}",
                       "horizon": h, "block": name,
                       "n_score": len(rows_s),
                       "anchor_datetimes": [
                           str(df["DATE_TIME"].iloc[r])
                           for r in rows_s],
                       "fit_cal_label_histogram": [
                           int(c) for c in counts],
                       "labels": [int(v) for v in ys],
                       "class_support_score": {
                           str(c): int(sum(1 for v in ys
                                           if v == c))
                           for c in (0, 1, 2)},
                       "arms": {}}
            if min(payload["class_support_score"]["0"],
                   payload["class_support_score"]["1"]) \
                    < SUPPORT_MIN:
                licenses_ok = False
                payload["license_failure"] = "class_support"
            if degenerate is not None:
                licenses_ok = False
                payload["license_failure"] = \
                    f"degenerate:{degenerate}"
                units.append(payload)
                continue
            idx = [row_pos[r] for r in rows_s]
            for arm in ARMS:
                probs = arm_probs[arm][idx]
                payload["arms"][arm] = {
                    "record": arm_recs[arm],
                    "probs": [[float(v) for v in row]
                              for row in probs],
                    "metrics": unit_metrics(ys, probs)}
            units.append(payload)
    contrasts_out, contrast_stats, complete = _rederive(units)
    if not complete:
        licenses_ok = False
    verdict = decide(contrast_stats, blocks_complete, licenses_ok)
    v1_map = None
    if v1_bundle is not None and v1_bundle.exists():
        v1 = json.loads(v1_bundle.read_text())
        # C9 (order @a1e7b739): every key and value of all eight
        # contrast objects, never a two-field proxy
        same_contrasts = v1["contrasts"] == contrasts_out
        v1_map = {
            "v1_bundle_sha256": hashlib.sha256(
                v1_bundle.read_bytes()).hexdigest(),
            "v1_status": "PRESERVED UNCHANGED, SUPERSEDED",
            "decisions_equal": v1["verdict"] == verdict,
            "complete_contrast_objects_equal": bool(same_contrasts),
            "added_in_v2": ["labels", "per-anchor class "
                            "probabilities", "direction_given_hit",
                            "additive identity", "Brier components",
                            "recall", "calibration deciles"],
            "removed_in_v2": ["per_obs_logloss (now derived from "
                              "labels+probs)",
                              "blocks_complete/licenses_ok as "
                              "bundle authority (verifier derives "
                              "them)"]}
    bundle = {
        "schema": BUNDLE_SCHEMA_V2,
        "contract": CONTRACT,
        "contract_sha256": sha_file(REPO / CONTRACT),
        "role_ledger": ledger,
        "digests": {
            "acquired_parquet": hashlib.sha256(
                secure_read(staging, "acquired.parquet"))
            .hexdigest(),
            "model_ready_extended": hashlib.sha256(
                secure_read(staging,
                            "model_ready_extended.parquet"))
            .hexdigest(),
            "frozen_csv": FROZEN_SHA,
            "lake_parquet": LAKE_SHA,
            "code": _code_digest()},
        "units": units,
        "contrasts": contrasts_out,
        "verdict": verdict,
        "elapsed_s": round(time.time() - started, 1),
        "decision_constants": {
            "margin_scale": MARGIN_SCALE,
            "margin_repr": MARGIN_REPR,
            "support_min": SUPPORT_MIN,
            "boot_b": BOOT_B, "boot_seed": BOOT_SEED,
            "block_len": BLOCK_LEN}}
    if v1_map:
        bundle["v1_correction_map"] = v1_map
    for u in units:
        u["payload_sha256"] = sha_obj(
            {k: v for k, v in u.items() if k != "payload_sha256"})
    out_bundle.write_text(json.dumps(bundle, indent=1,
                                     default=float) + "\n")
    return bundle


def _code_digest() -> str:
    return sha_obj({
        "n3_fresh_confirmation.py": sha_file(
            REPO / "tools/n3_fresh_confirmation.py"),
        "target_horizon_census_n2.py": sha_file(
            REPO / "tools/target_horizon_census_n2.py"),
        "paired_inference.py": sha_file(
            REPO / "agent_plugins/paired_inference.py"),
        "branch_pretraining.py": sha_file(
            REPO / "agent_plugins/branch_pretraining.py")})


# ------------------------------------------------------------------ #
# C6-C8: the verifier — reviewer authority, exact nested schemas,    #
# typed evidence, everything derived (order @a1e7b739)               #
# ------------------------------------------------------------------ #

# Reviewer-owned metadata registry (order @13fdf18c C11): the
# candidate verifier NEVER consumes it for authority and candidate
# tools NEVER write it — independent acceptance lives in separately
# committed auditor review records. validate_reviewer_registry()
# below offers hygiene validation of the metadata file only.
REVIEWER_REGISTRY = ("docs/audits/evidence/"
                     "N3_REVIEWED_PUBLICATION_DIGESTS.json")
RECEIPT_V2 = ("docs/audits/evidence/"
              "N3_ACQUISITION_RECEIPT_V2_2026_09_04.json")
PARITY_V2 = ("docs/audits/evidence/"
             "N3_PARITY_REPORT_V2_2026_09_04.json")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
TOP_KEYS_REQUIRED = {"schema", "contract", "contract_sha256",
                     "role_ledger", "digests", "units", "contrasts",
                     "verdict", "elapsed_s", "decision_constants"}
TOP_KEYS_OPTIONAL = {"v1_correction_map", "v2_correction_map"}
UNIT_KEYS_EXACT = {"unit", "horizon", "block", "n_score",
                   "anchor_datetimes", "fit_cal_label_histogram",
                   "labels", "class_support_score", "arms",
                   "payload_sha256"}
ARM_KEYS_EXACT = {"record", "probs", "metrics"}
METRIC_KEYS_EXACT = {"multiclass_logloss_mean",
                     "hit_vs_censored_mean",
                     "direction_given_hit_mean",
                     "additive_identity_max_abs_gap", "brier",
                     "brier_components", "recall_argmax",
                     "recall_unavailable_classes",
                     "calibration_deciles_hit"}
DECILE_KEYS_EXACT = {"bin", "n", "mean_predicted_hit",
                     "observed_hit_rate"}
CONTRAST_KEYS_EXACT = {"pooled_skill", "per_block_skill",
                       "all_blocks_positive", "bootstrap_p",
                       "holm_p"}
DIGEST_KEYS_EXACT = {"acquired_parquet", "model_ready_extended",
                     "frozen_csv", "lake_parquet", "code"}
ROLE_LEDGER_KEYS_EXACT = {"schema", "roles", "blocks",
                          "purge_bars", "stride", "window",
                          "anchor_counts"}
# C12 (order @13fdf18c): no gate label exists in this tool — no
# executing consumer exists, so no verdict here may imply one.


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return (_is_int(x) or isinstance(x, float)) \
        and math.isfinite(x)


def _need(cond: bool, msg: str) -> None:
    if not cond:
        raise FreshRefusal(msg)


def _exact_keys(obj: dict, exact: set, label: str,
                optional: set = frozenset()) -> None:
    _need(isinstance(obj, dict), f"{label}: not an object")
    unknown = set(obj) - exact - optional
    missing = exact - set(obj)
    _need(not unknown and not missing,
          f"{label}: schema violation — unknown="
          f"{sorted(unknown)} missing={sorted(missing)}")


def _validate_role_ledger(ledger: dict) -> None:
    """C6.2: the role ledger must DERIVE from the sealed contract
    and frozen geometry, not merely exist. fit/cal anchor counts are
    INFORMATIONAL (they depend on the frozen data grid's pre-2020
    gaps); block anchor counts are strictly derived."""
    _exact_keys(ledger, ROLE_LEDGER_KEYS_EXACT, "role_ledger")
    canon = role_ledger()
    _need(ledger["schema"] == canon["schema"],
          "role_ledger.schema differs")
    _need(ledger["roles"] == canon["roles"],
          "role_ledger.roles differ from the sealed roles")
    _need(ledger["blocks"] == canon["blocks"],
          "role boundary moved: blocks differ from the sealed "
          "contract")
    sealed = strict_json((REPO / CONTRACT).read_bytes())
    for name, spec in sealed["role_ledger"]["blocks_utc"].items():
        got = ledger["blocks"].get(name)
        norm = lambda b: [str(b[0]).replace("T", " "),
                          str(b[1]).replace("T", " "), b[2]]
        _need(got is not None and norm(got) == norm(spec),
              f"role boundary moved: block {name} differs from "
              "the sealed contract")
    _need(_is_int(ledger["purge_bars"])
          and ledger["purge_bars"] == H_MAX,
          "role_ledger.purge_bars != sealed max horizon")
    _need(_is_int(ledger["stride"]) and ledger["stride"] == STRIDE,
          "role_ledger.stride != sealed stride")
    _need(_is_int(ledger["window"]) and ledger["window"] == WINDOW,
          "role_ledger.window != sealed window")
    counts = ledger["anchor_counts"]
    expected_keys = {"fit", "cal"} | {b[0] for b in BLOCKS}
    _exact_keys(counts, expected_keys,
                "role_ledger.anchor_counts")
    for k, v in counts.items():
        _need(_is_int(v) and v >= 0,
              f"anchor_counts[{k}] not a non-negative integer")
    for name, _, _, bars in BLOCKS:
        _need(counts[name] == len(scoring_anchor_offsets(bars)),
              f"anchor_counts[{name}] != canonical derived count")
    # fit/cal counts: informational (typed above, not rederived)


def _validate_digests(digests: dict, code_digest_expected: str
                      ) -> None:
    """C6.4: exactly five canonical sha256 fields, each BOUND —
    acquisition to the committed strict v2 receipt, model-ready to
    the committed parity record, data to the frozen constants, code
    to the expected identity (current, or the allowlist entry's
    recorded historical identity)."""
    _exact_keys(digests, DIGEST_KEYS_EXACT, "digests")
    for k, v in digests.items():
        _need(isinstance(v, str) and HEX64.match(v),
              f"digests[{k}] is not a canonical lowercase sha256")
    _need(digests["frozen_csv"] == FROZEN_SHA,
          "digests.frozen_csv differs from the frozen constant")
    _need(digests["lake_parquet"] == LAKE_SHA,
          "digests.lake_parquet differs from the frozen constant")
    receipt = strict_json((REPO / RECEIPT_V2).read_bytes())
    _need(digests["acquired_parquet"]
          == receipt["acquired_parquet_sha256"],
          "digests.acquired_parquet does not equal the committed "
          "strict v2 receipt")
    parity = strict_json((REPO / PARITY_V2).read_bytes())
    _need(digests["model_ready_extended"]
          == parity["extended_sha256"],
          "digests.model_ready_extended does not equal the "
          "committed parity record")
    _need(digests["code"] == code_digest_expected,
          "code identity differs from the expected identity for "
          "this candidate")


def _validate_unit_typed(u: dict) -> None:
    """C7: typed per-observation evidence, validated BEFORE any
    numpy conversion can coerce authority-bearing values."""
    if "license_failure" in u:
        raise FreshRefusal(
            f"unit {u.get('unit')}: license failure "
            f"{u['license_failure']} beside the decision")
    _exact_keys(u, UNIT_KEYS_EXACT, f"unit {u.get('unit')}")
    name = u["unit"]
    _need(isinstance(name, str) and ":" in name,
          "unit name malformed")
    target, block = name.split(":", 1)
    _need(target in TARGETS, f"unit {name}: unknown target")
    _need(block in {b[0] for b in BLOCKS},
          f"unit {name}: unknown block")
    _need(u["block"] == block,
          f"unit {name}: block field != name")
    _need(_is_int(u["horizon"])
          and u["horizon"] == TARGETS[target],
          f"unit {name}: horizon does not equal the target's "
          "sealed horizon")
    canon = canonical_anchor_datetimes(block)
    _need(_is_int(u["n_score"]) and u["n_score"] == len(canon),
          f"unit {name}: n_score != canonical anchor count")
    _need(u["anchor_datetimes"] == canon,
          f"unit {name}: anchors differ from the canonical "
          "sealed-geometry anchors")
    hist = u["fit_cal_label_histogram"]
    _need(isinstance(hist, list) and len(hist) == 3
          and all(_is_int(x) and x >= 0 for x in hist)
          and sum(hist) > 0,
          f"unit {name}: histogram not three non-negative "
          "integers")
    labels = u["labels"]
    _need(isinstance(labels, list)
          and len(labels) == u["n_score"]
          and all(_is_int(x) and x in (0, 1, 2) for x in labels),
          f"unit {name}: labels must be JSON integers in {{0,1,2}} "
          "(booleans refuse)")
    support = u["class_support_score"]
    _exact_keys(support, {"0", "1", "2"},
                f"unit {name}.class_support_score")
    for c in ("0", "1", "2"):
        _need(_is_int(support[c]) and support[c] >= 0,
              f"unit {name}: support[{c}] not a non-negative "
              "integer")
        _need(support[c] == sum(1 for v in labels
                                if v == int(c)),
              f"unit {name}: class support does not derive from "
              "the labels")
    _need(min(support["0"], support["1"]) >= SUPPORT_MIN,
          f"unit {name}: derived class support below "
          f"{SUPPORT_MIN} — unlicensed evidence beside a decision")
    _exact_keys(u["arms"], set(ARMS), f"unit {name}.arms")
    for arm, rec in u["arms"].items():
        _exact_keys(rec, ARM_KEYS_EXACT, f"{name}.{arm}")
        record = rec["record"]
        if arm == "arm1":
            _exact_keys(record, {"prior_from"},
                        f"{name}.arm1.record")
            _need(record["prior_from"] == "fit+calibration",
                  f"{name}.arm1: prior source differs from the "
                  "declared fit+calibration")
        else:
            _exact_keys(record, {"C", "cal_loss", "coef_norm"},
                        f"{name}.{arm}.record")
            _need(_is_num(record["C"])
                  and record["C"] in tcn2.LOGISTIC_CS,
                  f"{name}.{arm}: C outside the sealed search set")
            _need(_is_num(record["cal_loss"])
                  and _is_num(record["coef_norm"]),
                  f"{name}.{arm}: non-finite fit record")
        probs = rec["probs"]
        _need(isinstance(probs, list)
              and len(probs) == u["n_score"],
              f"{name}.{arm}: probability rows != n_score")
        for row in probs:
            _need(isinstance(row, list) and len(row) == 3,
                  f"{name}.{arm}: malformed probability row")
            for v in row:
                _need(_is_num(v) and 0.0 <= v <= 1.0,
                      f"{name}.{arm}: probability must be a JSON "
                      "number in [0,1] (booleans and strings "
                      "refuse)")
            _need(abs(sum(row) - 1.0) <= 1e-9,
                  f"{name}.{arm}: probability row does not sum "
                  "to one")
        _validate_metrics_typed(rec["metrics"],
                                f"{name}.{arm}.metrics")


def _validate_metrics_typed(m: dict, label: str) -> None:
    _exact_keys(m, METRIC_KEYS_EXACT, label)
    for k in ("multiclass_logloss_mean", "hit_vs_censored_mean",
              "additive_identity_max_abs_gap", "brier"):
        _need(_is_num(m[k]), f"{label}.{k} not finite")
    _need(m["direction_given_hit_mean"] is None
          or _is_num(m["direction_given_hit_mean"]),
          f"{label}.direction_given_hit_mean invalid")
    _exact_keys(m["brier_components"], {"0", "1", "2"},
                f"{label}.brier_components")
    for v in m["brier_components"].values():
        _need(_is_num(v), f"{label}: brier component not finite")
    _exact_keys(m["recall_argmax"], {"0", "1", "2"},
                f"{label}.recall_argmax")
    for v in m["recall_argmax"].values():
        _need(v is None or _is_num(v),
              f"{label}: recall value invalid")
    _need(isinstance(m["recall_unavailable_classes"], list)
          and all(_is_int(x) and x in (0, 1, 2)
                  for x in m["recall_unavailable_classes"]),
          f"{label}: recall_unavailable_classes invalid")
    _need(isinstance(m["calibration_deciles_hit"], list),
          f"{label}: calibration deciles not a list")
    for d in m["calibration_deciles_hit"]:
        _exact_keys(d, DECILE_KEYS_EXACT, f"{label}.decile")
        _need(_is_int(d["bin"]) and 0 <= d["bin"] <= 9
              and _is_int(d["n"]) and d["n"] > 0
              and _is_num(d["mean_predicted_hit"])
              and _is_num(d["observed_hit_rate"]),
              f"{label}: malformed calibration decile")


def _validate_contrasts_typed(contrasts: dict) -> None:
    expected = {f"{t}:{a}-vs-{b}" for t in TARGETS
                for (a, b) in CONTRAST_FAMILY}
    _exact_keys(contrasts, expected, "contrasts")
    for ckey, s in contrasts.items():
        _exact_keys(s, CONTRAST_KEYS_EXACT, f"contrasts[{ckey}]")
        _need(_is_num(s["pooled_skill"]),
              f"{ckey}: pooled_skill not finite")
        _exact_keys(s["per_block_skill"],
                    {b[0] for b in BLOCKS},
                    f"{ckey}.per_block_skill")
        for v in s["per_block_skill"].values():
            _need(_is_num(v), f"{ckey}: block skill not finite")
        _need(isinstance(s["all_blocks_positive"], bool),
              f"{ckey}: all_blocks_positive not boolean")
        _need(s["bootstrap_p"] == "<= 1/2001"
              or _is_num(s["bootstrap_p"]),
              f"{ckey}: bootstrap_p invalid")
        _need(_is_num(s["holm_p"]), f"{ckey}: holm_p invalid")


REGISTRY_ENTRY_KEYS = {
    "reviewed": {"artifact", "status", "reviewed_by",
                 "code_digest", "decision"},
    "pending_review": {"artifact", "status", "submitted_by",
                       "code_digest", "decision"}}


def validate_reviewer_registry(path: Path) -> dict:
    """C12-C13 §4 hygiene validation of the reviewer-owned
    METADATA file. This function grants nothing: it checks shape.
    The registry is not consumed by verify() and is never written
    by candidate tools."""
    reg = strict_json(path.read_bytes())
    _exact_keys(reg, {"schema", "doc", "entries"},
                "reviewer_registry")
    _need(reg["schema"]
          == "agent_multi.n3_reviewed_publication_digests.v1",
          "reviewer_registry: unknown schema")
    for sha, entry in reg["entries"].items():
        _need(bool(HEX64.match(sha)),
              f"registry key {sha[:12]}: not canonical sha256")
        status = entry.get("status")
        _need(status in REGISTRY_ENTRY_KEYS,
              f"registry {sha[:12]}: status {status!r} outside "
              "the enum")
        _exact_keys(entry, REGISTRY_ENTRY_KEYS[status],
                    f"registry entry {sha[:12]}")
        _need(bool(HEX64.match(entry["code_digest"])),
              f"registry {sha[:12]}: code_digest not sha256")
        for k in ("artifact", "decision"):
            _need(isinstance(entry[k], str) and entry[k],
                  f"registry {sha[:12]}: {k} not a string")
    return {"entries": len(reg["entries"]),
            "note": "metadata hygiene only — no authority"}


def verify(bundle_path: Path, supplied_sha256: str | None = None,
           internal_only: bool = False) -> dict:
    """C8 authority separation: a caller-supplied digest proves the
    candidate matches a value the CALLER chose — byte match plus
    semantic consistency, never publication authority. The
    gate-bearing N3_PUBLICATION_VERIFIED label requires the digest
    to appear with status 'reviewed' in the committed reviewer
    allowlist, which no candidate can generate for itself."""
    raw = bundle_path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if not internal_only:
        if not supplied_sha256:
            raise FreshRefusal(
                "an expected bundle sha256 is required — a digest "
                "carried inside the mutable bundle is a checksum, "
                "not authority")
        if actual_sha != supplied_sha256:
            raise FreshRefusal(
                f"bundle bytes do not match the supplied digest "
                f"({actual_sha[:12]} != {supplied_sha256[:12]})")
    bundle = strict_json(raw)
    _need(bundle.get("schema") == BUNDLE_SCHEMA_V2,
          "unknown bundle schema")
    _exact_keys(bundle, TOP_KEYS_REQUIRED, "bundle",
                optional=TOP_KEYS_OPTIONAL)
    # C6.1: contract path AND bytes
    _need(bundle["contract"] == CONTRACT,
          "contract path differs from the canonical committed "
          "contract")
    _need(bundle["contract_sha256"] == sha_file(REPO / CONTRACT),
          "contract_sha256 does not match the sealed contract "
          "bytes")
    # C6.3: decision constants equal the rederivation constants
    _exact_keys(bundle["decision_constants"],
                {"margin_scale", "margin_repr", "support_min",
                 "boot_b", "boot_seed", "block_len"},
                "decision_constants")
    expected_constants = {"margin_scale": MARGIN_SCALE,
                          "margin_repr": MARGIN_REPR,
                          "support_min": SUPPORT_MIN,
                          "boot_b": BOOT_B,
                          "boot_seed": BOOT_SEED,
                          "block_len": BLOCK_LEN}
    _need(bundle["decision_constants"] == expected_constants,
          "decision_constants differ from the sealed rederivation "
          "constants")
    # C6.2: role ledger derives from the sealed contract
    _validate_role_ledger(bundle["role_ledger"])
    # C6.4/C11: a candidate is validated against the CURRENT code
    # identity only; historical publications are verified by their
    # historical code, retrieved from git by the reviewer
    _validate_digests(bundle["digests"], _code_digest())
    # units: exact set, typed evidence, self-digest
    expected_units = {f"{t}:{b[0]}" for t in TARGETS
                      for b in BLOCKS}
    seen = [u.get("unit") for u in bundle["units"]]
    _need(len(seen) == len(set(seen)), "duplicate units")
    _need(set(seen) == expected_units,
          f"missing/extra units: "
          f"{sorted(set(seen) ^ expected_units)[:4]}")
    for u in bundle["units"]:
        _validate_unit_typed(u)
        claimed = u.get("payload_sha256")
        _need(sha_obj({k: v for k, v in u.items()
                       if k != "payload_sha256"}) == claimed,
              f"unit {u['unit']}: payload altered (digest)")
        hist = u["fit_cal_label_histogram"]
        import numpy as np
        prior = np.clip(np.array(hist, dtype="float64")
                        / sum(hist), 1e-12, None)
        prior = prior / prior.sum()
        p1 = np.asarray(u["arms"]["arm1"]["probs"],
                        dtype="float64")
        _need(bool(np.allclose(p1, prior[None, :], atol=1e-12)),
              f"unit {u['unit']}: arm1 probabilities are not the "
              "fit+cal prior — different label histories")
        for arm, rec in u["arms"].items():
            derived = unit_metrics(u["labels"], rec["probs"])
            _need(derived == rec["metrics"],
                  f"unit {u['unit']} {arm}: published metrics do "
                  "not derive from evidence")
    # typed contrast schema, then full rederivation equality
    _validate_contrasts_typed(bundle["contrasts"])
    contrasts_out, contrast_stats, complete = _rederive(
        bundle["units"])
    _need(complete, "missing or failed unit beside the decision")
    verdict = decide(contrast_stats, True, True)
    _need(verdict == bundle["verdict"],
          f"report edited: rederived verdict {verdict} != "
          f"bundled {bundle['verdict']}")
    _need(contrasts_out == bundle["contrasts"],
          "report edited: complete contrast objects do not "
          "rederive from unit evidence")
    # C13: correction maps are informational and UNVERIFIED —
    # typed as objects and NAMED as unverified in the output; they
    # are excluded from the semantic-verification claim
    informational = sorted(TOP_KEYS_OPTIONAL & set(bundle))
    for key in informational:
        _need(isinstance(bundle[key], dict),
              f"{key}: informational map must be an object")
    # C11/C12 outcome vocabulary: content and consistency ONLY.
    # This tool can NEVER emit a label implying independent review
    # or bear a gate — independent acceptance is a separately
    # committed auditor review record, outside this process.
    label = ("N3_INTERNAL_CONSISTENCY_ONLY" if internal_only
             else "N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST")
    return {"verdict": label,
            "rederived_decision": verdict,
            "units_verified": len(bundle["units"]),
            "bundle_sha256": actual_sha,
            "authority": ("none — this verifier establishes byte "
                          "identity and semantic consistency only; "
                          "independent review authority cannot be "
                          "minted from candidate-controlled "
                          "inputs"),
            "informational_unverified_fields": informational}


# ------------------------------------------------------------------ #
# C9/C10: v3 publication envelope without scientific re-execution    #
# ------------------------------------------------------------------ #

def _full_contrast_equality(a: dict, b: dict) -> bool:
    """C9: every key and value of all eight contrast objects."""
    return a == b


PUBLICATION_ONLY_PATHS = ("digests.code", "v1_correction_map",
                          "v2_correction_map")
SCIENTIFIC_FIELDS_INCLUDED = (
    "units[*].labels", "units[*].anchor_datetimes",
    "units[*].arms[*].probs")


def publication_diff(a: dict, b: dict,
                     allowed_paths=PUBLICATION_ONLY_PATHS) -> list:
    """C15: complete structural comparison of two bundle objects
    AFTER removing an explicit allowlist of publication-only paths.
    Returns every unexpected added/removed/changed top-or-nested
    path; the caller refuses when the list is non-empty."""
    def strip(obj):
        out = json.loads(json.dumps(obj, default=float))
        for path in allowed_paths:
            node = out
            parts = path.split(".")
            for part in parts[:-1]:
                node = node.get(part, {})
            node.pop(parts[-1], None)
        return out

    def walk(x, y, prefix, diffs):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                path = f"{prefix}.{k}" if prefix else k
                if k not in x:
                    diffs.append(f"added:{path}")
                elif k not in y:
                    diffs.append(f"removed:{path}")
                else:
                    walk(x[k], y[k], path, diffs)
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                diffs.append(f"changed:{prefix}(length)")
                return
            for i, (xi, yi) in enumerate(zip(x, y)):
                walk(xi, yi, f"{prefix}[{i}]", diffs)
        elif x != y:
            diffs.append(f"changed:{prefix}")

    diffs = []
    walk(strip(a), strip(b), "", diffs)
    return diffs


def reissue(prev_path: Path, v1_path: Path, v2_path: Path,
            out_path: Path, receipt_path: Path) -> dict:
    """C10/C15: derive a successor publication envelope from the
    EXACT prior evidence — only because the verifier code identity
    changed. Emits a candidate-submission RECEIPT; NEVER writes the
    reviewer-owned registry. Names its equalities truthfully:
    scientific_fields_equal covers the recorded field subset, and a
    complete structural comparison beyond the declared
    publication-only paths must be empty."""
    prev_raw = prev_path.read_bytes()
    prev = strict_json(prev_raw)
    v1 = strict_json(v1_path.read_bytes())
    v2_raw = v2_path.read_bytes()
    v2 = strict_json(v2_raw)
    v4 = json.loads(json.dumps(prev, default=float))

    def science_fields_digest(bundle):
        return sha_obj([[u["labels"], u["anchor_datetimes"],
                         {a: r["probs"]
                          for a, r in u["arms"].items()}]
                        for u in sorted(bundle["units"],
                                        key=lambda x: x["unit"])])
    sci_prev = science_fields_digest(prev)
    contrasts_out, contrast_stats, complete = _rederive(
        prev["units"])
    if not complete:
        raise FreshRefusal("prior evidence incomplete")
    decision = decide(contrast_stats, True, True)
    if decision != prev["verdict"]:
        raise FreshRefusal("prior decision does not rederive")
    if not _full_contrast_equality(contrasts_out,
                                   prev["contrasts"]):
        raise FreshRefusal("prior contrasts do not fully rederive")
    v4["digests"]["code"] = _code_digest()
    v4["v1_correction_map"] = {
        "v1_bundle_sha256": hashlib.sha256(
            v1_path.read_bytes()).hexdigest(),
        "v1_status": "PRESERVED UNCHANGED, SUPERSEDED",
        "decisions_equal": v1["verdict"] == decision,
        "complete_contrast_objects_equal":
            _full_contrast_equality(v1["contrasts"],
                                    prev["contrasts"]),
        "comparison_scope": "every key and value of all eight "
                            "contrast objects, plus the decision "
                            "(order @a1e7b739 C9)"}
    v4["v2_correction_map"] = {
        "v2_bundle_sha256": hashlib.sha256(v2_raw).hexdigest(),
        "v2_status": "PRESERVED UNCHANGED, SUPERSEDED (scientific "
                     "anchor of the reviewed publication)",
        "reason": "verifier code-identity change only (orders "
                  "@a1e7b739 C10, @13fdf18c C11-C15); no "
                  "scientific re-execution",
        "scientific_fields_equal":
            science_fields_digest(v4) == science_fields_digest(v2),
        "scientific_fields_included": list(
            SCIENTIFIC_FIELDS_INCLUDED),
        "publication_only_paths": list(PUBLICATION_ONLY_PATHS),
        "full_object_diff_beyond_publication_paths": [],
        "decisions_equal": v2["verdict"] == decision,
        "complete_contrast_objects_equal":
            _full_contrast_equality(v2["contrasts"],
                                    prev["contrasts"])}
    if not v4["v2_correction_map"]["scientific_fields_equal"]:
        raise FreshRefusal("scientific fields drifted during "
                           "reissue")
    diffs = publication_diff(v2, v4)
    if diffs:
        raise FreshRefusal(
            f"unexpected non-publication differences vs the "
            f"scientific anchor: {diffs[:5]}")
    if science_fields_digest(v4) != sci_prev:
        raise FreshRefusal("science drifted vs the prior envelope")
    out_path.write_text(json.dumps(v4, indent=1, default=float)
                        + "\n")
    v4_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    # candidate-submission receipt — SEPARATE from the
    # reviewer-owned registry, which candidate tools never touch
    receipt = {
        "schema": "agent_multi.n3_candidate_submission_receipt.v1",
        "candidate_artifact": out_path.name,
        "candidate_sha256": v4_sha,
        "derived_from": {
            "prior_envelope": prev_path.name,
            "prior_sha256": hashlib.sha256(prev_raw).hexdigest(),
            "scientific_anchor": v2_path.name,
            "scientific_anchor_sha256": hashlib.sha256(
                v2_raw).hexdigest()},
        "code_digest": v4["digests"]["code"],
        "decision": decision,
        "statement": "candidate submission awaiting the "
                     "independent reviewer; this receipt grants "
                     "nothing and is not the reviewer-owned "
                     "registry"}
    receipt_path.write_text(json.dumps(receipt, indent=1) + "\n")
    return {"v4_sha256": v4_sha, "decision": decision,
            "scientific_fields_equal": True,
            "full_object_diff_beyond_publication_paths": [],
            "receipt": receipt_path.name}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--staging", required=True)
    r = sub.add_parser("reattest")
    r.add_argument("--v1-staging", required=True)
    r.add_argument("--v2-staging", required=True)
    r.add_argument("--v1-receipt", required=True)
    g = sub.add_parser("regenerate")
    g.add_argument("--staging", required=True)
    e = sub.add_parser("execute")
    e.add_argument("--staging", required=True)
    e.add_argument("--out-bundle", required=True)
    e.add_argument("--v1-bundle", default=None)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", required=True)
    v.add_argument("--expected-sha256", default=None)
    v.add_argument("--internal-only", action="store_true")
    i = sub.add_parser("reissue")
    i.add_argument("--prev-bundle", required=True)
    i.add_argument("--v1-bundle", required=True)
    i.add_argument("--v2-bundle", required=True)
    i.add_argument("--out-bundle", required=True)
    i.add_argument("--receipt", required=True)
    args = parser.parse_args()
    try:
        if args.cmd == "acquire":
            out = acquire(Path(args.staging))
            print(json.dumps({"verdict": out["verdict"],
                              "rows": out["rows_total"],
                              "overlap": out[
                                  "rows_overlap_verified_exact"],
                              "rows_2026": out["rows_2026"]}))
        elif args.cmd == "reattest":
            out = reattest(Path(args.v1_staging),
                           Path(args.v2_staging),
                           Path(args.v1_receipt))
            print(json.dumps({"verdict": out["verdict"],
                              "rows": out["rows_total"],
                              "overlap": out[
                                  "rows_overlap_verified_exact"],
                              "rows_2026": out["rows_2026"]}))
        elif args.cmd == "regenerate":
            out = regenerate(Path(args.staging))
            print(json.dumps({"verdict": out["verdict"],
                              "rows_2026": out["rows_2026"],
                              "ffill": out["ffill_cells_changed"]}))
        elif args.cmd == "execute":
            out = execute(Path(args.staging),
                          Path(args.out_bundle),
                          Path(args.v1_bundle)
                          if args.v1_bundle else None)
            print(json.dumps({"verdict": out["verdict"],
                              "elapsed_s": out["elapsed_s"]}))
        elif args.cmd == "reissue":
            print(json.dumps(reissue(Path(args.prev_bundle),
                                     Path(args.v1_bundle),
                                     Path(args.v2_bundle),
                                     Path(args.out_bundle),
                                     Path(args.receipt)),
                             indent=1))
        else:
            print(json.dumps(verify(
                Path(args.bundle), args.expected_sha256,
                internal_only=args.internal_only), indent=1))
        return 0
    except FreshRefusal as refusal:
        print(json.dumps({"refusal": str(refusal)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
