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

    Unparseable timestamps are treated as *not* heldout — the caller is
    expected to provide ISO/parseable strings; see ``_parse_iso``.
    """
    parsed = _parse_iso(ts)
    if parsed is None:
        return False
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
