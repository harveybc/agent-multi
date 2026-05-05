"""Stage B return-trace emission scaffold for Project 3.

This module is the *single* place where agent-multi serializes per-step
return traces and the matching metadata sidecar consumed by
financial-data's Stage B statistical evaluator (DSR / PBO / Reality
Check / SPA). It contains no DSR/PBO math — it only emits raw inputs.

Hard rules enforced here:

* Stage C is data on or after ``HELDOUT_BOUNDARY = "2025-01-01"``. A
  trace that contains any timestamp at or beyond that boundary is
  refused unless the caller config explicitly sets
  ``final_stage_c_evaluation: true`` AND
  ``stage_c_acknowledged: true``. Final Stage C mode itself is *not*
  implemented; the flag exists only so emergency dry-runs can be
  flagged without silently producing Stage C traces.
* The trace CSV always carries an explicit ``split`` column so a
  downstream loader can never confuse train/validation/test rows.
* The sidecar JSON records hashes of the source config and data file
  so a Stage B reviewer can confirm the trace matches the run.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# ---------------------------------------------------------------------------
SCHEMA_VERSION = "stage_b_return_trace_v1"
HELDOUT_BOUNDARY = "2025-01-01"

ALLOWED_SPLITS = (
    "train",
    "validation",
    "test",
    "train_epoch",
    "validation_epoch",
    "evaluation",
)

# Public schema. Keep this stable: financial-data binds to these names.
TRACE_FIELDNAMES: Sequence[str] = (
    "step",
    "timestamp",
    "asset",
    "timeframe",
    "split",
    "episode_id",
    "run_id",
    "seed",
    "bar_index",
    "price",
    "action_raw",
    "position",
    "reward",
    "gross_return",
    "net_return",
    "equity",
    "pnl",
    "commission_paid",
    "slippage_paid",
    "trade_cost",
    "trades",
)


# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_config(cfg: Mapping[str, Any]) -> str:
    blob = json.dumps(_jsonable(cfg), sort_keys=True, default=str).encode("utf-8")
    return _sha256_bytes(blob)


def _hash_feature_list(features: Any) -> Optional[str]:
    if features is None:
        return None
    try:
        blob = json.dumps(_jsonable(features), sort_keys=True, default=str).encode("utf-8")
    except Exception:
        return None
    return _sha256_bytes(blob)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ts_at_or_after_heldout(ts: str) -> bool:
    """True if ``ts`` parses to a date >= HELDOUT_BOUNDARY.

    This helper is intentionally only called after write_return_trace()
    has already validated that every timestamp is parseable. Timestamp
    parse failures are fatal because treating an unknown timestamp as
    pre-heldout would make the Stage C guard optimistic.
    """
    parsed = _parse_iso(ts)
    if parsed is None:
        raise TraceTimestampError(f"unparseable return_trace timestamp: {ts!r}")
    return parsed.date() >= _dt.date(2025, 1, 1)


def _parse_iso(ts: Any) -> Optional[_dt.datetime]:
    if ts in (None, "", b""):
        return None
    s = str(ts)
    # Trim trailing Z and microseconds shenanigans for fromisoformat compat.
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace("T", " ")
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d",
    ):
        try:
            return _dt.datetime.strptime(s.strip()[: len(fmt) + 6], fmt)
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(s.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
class StageCAccessError(RuntimeError):
    """Raised when a trace would include Stage C rows without authorization."""


class TraceTimestampError(ValueError):
    """Raised when trace timestamps are missing, invalid, or unordered."""


def _is_stage_c_authorized(config: Mapping[str, Any]) -> bool:
    return bool(
        config.get("final_stage_c_evaluation")
        and config.get("stage_c_acknowledged")
    )


# ---------------------------------------------------------------------------
def build_trace_row(
    *,
    env: Any,
    step: int,
    action: Any,
    reward: float,
    info: Mapping[str, Any],
    prev_equity: Optional[float],
    asset: str,
    timeframe: str,
    split: str,
    seed: int,
    run_id: str,
    episode_id: str,
) -> Dict[str, Any]:
    """Produce one row of the Stage B trace.

    ``net_return`` is the equity-based step return (post-cost). When a
    ``commission_paid``/``trade_cost`` is reported by the env, an
    approximate ``gross_return`` (pre-cost) is computed; otherwise it
    falls back to ``net_return``. Either field may be ``None`` when the
    env cannot report ``equity``.
    """
    bar_index = int(info.get("bar_index") or 0)
    equity = _safe_float(info.get("equity"))
    commission_paid = _safe_float(info.get("commission_paid"))
    slippage_paid = _safe_float(info.get("slippage_paid"))
    trade_cost = _safe_float(info.get("trade_cost"))

    net_return: Optional[float] = None
    gross_return: Optional[float] = None
    if prev_equity not in (None, 0.0) and equity is not None:
        net_return = (equity - float(prev_equity)) / float(prev_equity)
        gross_return = net_return
        cost_terms = 0.0
        for c in (commission_paid, slippage_paid, trade_cost):
            if c is not None:
                cost_terms += float(c)
        if cost_terms and float(prev_equity) != 0.0:
            gross_return = net_return + cost_terms / float(prev_equity)

    return {
        "step": int(step),
        "timestamp": _timestamp_for_bar(env, bar_index),
        "asset": asset,
        "timeframe": timeframe,
        "split": split,
        "episode_id": episode_id,
        "run_id": run_id,
        "seed": int(seed),
        "bar_index": bar_index,
        "price": _safe_float(info.get("price")),
        "action_raw": _safe_action_value(action),
        "position": _safe_float(info.get("position")),
        "reward": float(reward),
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
        "pnl": _safe_float(info.get("pnl")),
        "commission_paid": commission_paid,
        "slippage_paid": slippage_paid,
        "trade_cost": trade_cost,
        "trades": info.get("trades"),
    }


def _safe_action_value(action: Any) -> Any:
    try:
        import numpy as np

        arr = np.asarray(action).reshape(-1)
        if len(arr):
            return float(arr[0])
    except Exception:
        pass
    try:
        return float(action)
    except Exception:
        return str(action)


def _unwrap_env(env: Any):
    base = env
    while hasattr(base, "env") and not hasattr(base, "summary"):
        base = base.env
    return base


def _timestamp_for_bar(env: Any, bar_index: int) -> str:
    base = _unwrap_env(env)
    df = getattr(base, "dataframe", None)
    if df is None or len(df) == 0:
        return ""
    idx = max(0, min(int(bar_index) - 1, len(df) - 1))
    cfg = getattr(base, "config", None) or {}
    date_col = cfg.get("date_column", "DATE_TIME") if isinstance(cfg, Mapping) else "DATE_TIME"
    try:
        if date_col in df.columns:
            return str(df.iloc[idx][date_col])
        return str(df.index[idx])
    except Exception:
        return ""


# ---------------------------------------------------------------------------
def write_return_trace(
    trace_path: str,
    rows: Iterable[Dict[str, Any]],
    *,
    config: Mapping[str, Any],
    split: str,
    seed: int,
    asset: Optional[str] = None,
    timeframe: Optional[str] = None,
    run_id: Optional[str] = None,
    episode_id: Optional[str] = None,
    feature_list: Any = None,
    extra_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the trace CSV and its metadata sidecar.

    Returns the metadata dict that was written. Raises
    :class:`StageCAccessError` if the trace contains rows at or after
    the Stage C heldout boundary and the config has not explicitly
    authorized Stage C evaluation.
    """
    if split not in ALLOWED_SPLITS:
        raise ValueError(
            f"return_trace split={split!r} not in {ALLOWED_SPLITS!r}"
        )

    rows_list: List[Dict[str, Any]] = list(rows)
    parsed_timestamps: List[_dt.datetime] = []
    for idx, row in enumerate(rows_list):
        ts = row.get("timestamp", "")
        parsed = _parse_iso(ts)
        if parsed is None:
            raise TraceTimestampError(
                f"return_trace row {idx} has missing or unparseable timestamp: {ts!r}"
            )
        if parsed_timestamps and parsed <= parsed_timestamps[-1]:
            raise TraceTimestampError(
                "return_trace timestamps must be strictly increasing within "
                f"each split; row {idx} timestamp {ts!r} is not after the "
                f"previous timestamp {rows_list[idx - 1].get('timestamp', '')!r}"
            )
        parsed_timestamps.append(parsed)

    contains_heldout = any(
        _ts_at_or_after_heldout(r.get("timestamp", "")) for r in rows_list
    )
    stage_c_ok = _is_stage_c_authorized(config)
    if contains_heldout and not stage_c_ok:
        raise StageCAccessError(
            "return_trace would include Stage C rows "
            f"(timestamp >= {HELDOUT_BOUNDARY}). Set "
            "final_stage_c_evaluation=true AND stage_c_acknowledged=true "
            "in the config to authorize. Final Stage C mode is NOT "
            "implemented in this repo."
        )

    out = Path(trace_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(TRACE_FIELDNAMES))
        writer.writeheader()
        for r in rows_list:
            writer.writerow({k: r.get(k) for k in TRACE_FIELDNAMES})

    timestamps = [r.get("timestamp", "") for r in rows_list if r.get("timestamp")]
    first_ts = timestamps[0] if timestamps else None
    last_ts = timestamps[-1] if timestamps else None

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "trace_file": str(out.resolve()),
        "trace_file_sha256": _sha256_file(str(out)),
        "row_count": len(rows_list),
        "asset": asset or config.get("asset"),
        "timeframe": timeframe or config.get("timeframe") or config.get("timeframe_label"),
        "split": split,
        "seed": int(seed),
        "run_id": run_id,
        "episode_id": episode_id,
        "config_hash": _hash_config(config),
        "data_file": config.get("input_data_file"),
        "data_file_hash": _sha256_file(config.get("input_data_file")),
        "feature_list_hash": _hash_feature_list(feature_list),
        "split_boundaries": {
            "first_timestamp": first_ts,
            "last_timestamp": last_ts,
        },
        "heldout_boundary": HELDOUT_BOUNDARY,
        "contains_heldout_rows": bool(contains_heldout),
        "stage_c_authorized": bool(stage_c_ok),
        "fields": list(TRACE_FIELDNAMES),
    }
    if extra_metadata:
        metadata["extra"] = _jsonable(dict(extra_metadata))

    sidecar = out.with_suffix(out.suffix + ".meta.json")
    with sidecar.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)
    metadata["metadata_file"] = str(sidecar.resolve())
    return metadata


def derive_split_trace_path(trace_dir: str, split: str) -> str:
    """Return ``<trace_dir>/<split>_return_trace.csv``.

    Centralized so both pipelines and tests agree on the layout.
    """
    return str(Path(trace_dir) / f"{split}_return_trace.csv")


def make_run_id(config: Mapping[str, Any]) -> str:
    """Best-effort run identifier (does not write anything to disk)."""
    save_model = config.get("save_model")
    if save_model:
        return Path(str(save_model)).parent.name or Path(str(save_model)).stem
    asset = str(config.get("asset", "asset"))
    seed = config.get("train_seed", config.get("eval_seed", 0))
    return f"{asset}_seed{seed}"


# ---------------------------------------------------------------------------
# Run-level evidence index
#
# financial-data needs to discover every trace + metadata file produced by a
# given agent-multi run *without* scraping logs. The evidence index is the
# single durable artifact that points at all per-split traces, hashes, and
# Stage C status flags. The index is written next to the trace(s) when
# tracing is enabled and is also surfaced inline in the pipeline summary.
# ---------------------------------------------------------------------------

EVIDENCE_SCHEMA_VERSION = "project3_return_trace_evidence_v1"


class EvidenceConsistencyError(RuntimeError):
    """Raised when the evidence index would be misleading or unsafe.

    Examples:
        - a referenced trace file is missing on disk;
        - a metadata sidecar is missing on disk;
        - duplicate split labels (would silently overwrite a row);
        - per-trace items disagree on schema version, asset, seed, or
          config hash for the same run;
        - a metadata item asserts ``contains_heldout_rows`` without
          ``stage_c_authorized`` (defense in depth — should already
          have been caught by ``write_return_trace``).
    """


def _require_existing_file(label: str, path: Optional[str]) -> str:
    if not path:
        raise EvidenceConsistencyError(f"evidence: missing {label} path")
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise EvidenceConsistencyError(
            f"evidence: {label} not found on disk: {path}"
        )
    return str(p.resolve())


def _evidence_trace_entry(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Project a per-trace metadata sidecar into an evidence trace entry."""
    if not isinstance(meta, Mapping):
        raise EvidenceConsistencyError(
            f"evidence: metadata item must be a mapping, got {type(meta)!r}"
        )
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceConsistencyError(
            "evidence: trace metadata schema_version="
            f"{meta.get('schema_version')!r} does not match expected "
            f"{SCHEMA_VERSION!r}"
        )
    contains_heldout = bool(meta.get("contains_heldout_rows"))
    stage_c_ok = bool(meta.get("stage_c_authorized"))
    if contains_heldout and not stage_c_ok:
        raise EvidenceConsistencyError(
            "evidence: trace metadata reports heldout rows without Stage C "
            "authorization (defense in depth)"
        )
    trace_file = _require_existing_file(
        "trace_file", meta.get("trace_file"),
    )
    metadata_file = _require_existing_file(
        "metadata_file", meta.get("metadata_file"),
    )
    boundaries = meta.get("split_boundaries") or {}
    return {
        "split": meta.get("split"),
        "trace_file": trace_file,
        "trace_file_sha256": meta.get("trace_file_sha256"),
        "metadata_file": metadata_file,
        "row_count": int(meta.get("row_count") or 0),
        "first_timestamp": boundaries.get("first_timestamp"),
        "last_timestamp": boundaries.get("last_timestamp"),
        "contains_heldout_rows": contains_heldout,
        "stage_c_authorized": stage_c_ok,
        "episode_id": meta.get("episode_id"),
    }


def _consistent_value(items: Iterable[Mapping[str, Any]], key: str) -> Any:
    """Return the unique value for ``key`` across items, or fail closed."""
    values = []
    for it in items:
        if key in it and it[key] is not None:
            values.append(it[key])
    if not values:
        return None
    first = values[0]
    for v in values[1:]:
        if v != first:
            raise EvidenceConsistencyError(
                f"evidence: trace metadata items disagree on {key!r}: "
                f"{first!r} vs {v!r}"
            )
    return first


def build_return_trace_evidence(
    metadata_items: Sequence[Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
    run_id: Optional[str] = None,
    pipeline_plugin: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the run-level evidence index from per-trace metadata sidecars.

    Parameters
    ----------
    metadata_items
        The list of metadata dicts produced by
        :func:`write_return_trace` for this run. May be empty (callers
        should simply skip evidence emission in that case rather than
        passing an empty list).
    config
        The agent-multi config dict the run was launched with.
    run_id
        Override for the run identifier. When omitted, the value is
        cross-checked from the metadata items first, then derived from
        the config.
    pipeline_plugin
        Name of the pipeline plugin that produced the traces (e.g.
        ``"rl_pipeline"`` or ``"rl_pipeline_with_validation"``).

    Returns
    -------
    dict
        A serializable evidence index following the
        ``project3_return_trace_evidence_v1`` schema. The dict can be
        written verbatim to disk via :func:`write_return_trace_evidence`
        or surfaced inline in the pipeline summary.

    Raises
    ------
    EvidenceConsistencyError
        On any of the conditions documented in the class docstring.
    """
    if not metadata_items:
        raise EvidenceConsistencyError(
            "evidence: refusing to build an empty evidence index"
        )

    items = [dict(m) for m in metadata_items]

    # Duplicate split detection.
    splits = [m.get("split") for m in items]
    seen: set = set()
    for s in splits:
        if s in seen:
            raise EvidenceConsistencyError(
                f"evidence: duplicate split label {s!r}"
            )
        seen.add(s)

    # Cross-item consistency on identity-defining fields.
    asset = _consistent_value(items, "asset") or config.get("asset")
    timeframe = (
        _consistent_value(items, "timeframe")
        or config.get("timeframe")
        or config.get("timeframe_label")
    )
    seed = _consistent_value(items, "seed")
    config_hash = _consistent_value(items, "config_hash") or _hash_config(config)
    data_file = _consistent_value(items, "data_file") or config.get("input_data_file")
    data_file_hash = _consistent_value(items, "data_file_hash")
    feature_list_hash = _consistent_value(items, "feature_list_hash")
    items_run_id = _consistent_value(items, "run_id")

    resolved_run_id = run_id or items_run_id or make_run_id(config)

    trace_entries = [_evidence_trace_entry(m) for m in items]

    contains_heldout_any = any(t["contains_heldout_rows"] for t in trace_entries)
    stage_c_authorized_any = any(t["stage_c_authorized"] for t in trace_entries)

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "generated_at": _utcnow_iso(),
        "run_id": resolved_run_id,
        "pipeline_plugin": pipeline_plugin,
        "trace_schema_version": SCHEMA_VERSION,
        "asset": asset,
        "timeframe": timeframe,
        "seed": seed,
        "config_hash": config_hash,
        "data_file": data_file,
        "data_file_hash": data_file_hash,
        "feature_list_hash": feature_list_hash,
        "heldout_boundary": HELDOUT_BOUNDARY,
        "contains_heldout_rows": contains_heldout_any,
        "stage_c_authorized": stage_c_authorized_any,
        "traces": trace_entries,
    }


def write_return_trace_evidence(
    evidence: Mapping[str, Any],
    output_path: str,
) -> str:
    """Persist an evidence index dict to ``output_path``.

    Returns the absolute path of the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(_jsonable(dict(evidence)), fh, indent=2, sort_keys=True, default=str)
    return str(out.resolve())


def derive_evidence_path(
    *,
    trace_file: Optional[str] = None,
    trace_dir: Optional[str] = None,
) -> str:
    """Return the canonical ``evidence.json`` location.

    Layout:
      - if ``trace_dir`` is provided: ``<trace_dir>/evidence.json``;
      - else if ``trace_file`` is provided:
        ``<dir(trace_file)>/evidence.json``.
    """
    if trace_dir:
        return str(Path(trace_dir) / "evidence.json")
    if trace_file:
        return str(Path(trace_file).parent / "evidence.json")
    raise ValueError("derive_evidence_path requires trace_file or trace_dir")
