"""WP1 — typed nested chronological split contract (doc 38 §3).

ONE implementation of split materialization for the nested L1/L2
program: fit_train / train_monitor / inner_validation /
outer_validation / sealed_test, with derived-and-verified row counts,
causal context prefixes that cannot leak into scores, a persisted split
manifest, and fail-closed mismatch handling. Pipelines consume this
module; they never parse nested dates themselves.

Invariants (order §5, doc 38 §3.2):
- prose row counts are assertions to VERIFY, never inputs to trust;
- train_monitor is a SUBSET of fit_train (the last complete fit year);
- fit_train -> inner -> outer -> sealed are contiguous, non-overlapping
  half-open ranges; any gap, overlap or reordering refuses;
- every evaluation split carries `context_bars` closed bars immediately
  before its score start; prefix bars initialize causal state ONLY —
  the ContextPrefixWrapper forces holds, tags `is_context_prefix`, and
  refuses if account state changes during the prefix;
- sealed_test cannot be materialized outside `release` mode;
- an artifact evaluated under a different split manifest fails closed.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

SCHEMA = "agent_multi.nested_split_contract.v1"
MANIFEST_SCHEMA = "agent_multi.nested_split_manifest.v1"
ROLES = ("fit_train", "train_monitor", "inner_validation",
         "outer_validation", "sealed_test")
EVAL_ROLES = ("train_monitor", "inner_validation", "outer_validation",
              "sealed_test")
MODES = ("l1", "l2", "train", "release")


class NestedSplitError(ValueError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_context_bars(*, window_size: int, scaling_window: int,
                         max_feature_lookback: int) -> int:
    """context_bars = max of the three causal lookbacks. All three are
    REQUIRED typed inputs; there is no silent default (order §5.2)."""
    for name, value in (("window_size", window_size),
                        ("scaling_window", scaling_window),
                        ("max_feature_lookback", max_feature_lookback)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise NestedSplitError(f"{name} must be an integer")
        if value < 0:
            raise NestedSplitError(f"{name} must not be negative")
    return max(window_size, scaling_window, max_feature_lookback)


def load_contract(path: Path) -> dict:
    contract = json.loads(Path(path).read_text())
    if contract.get("schema") != SCHEMA:
        raise NestedSplitError(
            f"unknown nested split contract schema: {contract.get('schema')}")
    for key in ("source_csv", "source_sha256", "date_column", "roles",
                "window_size", "scaling_window", "max_feature_lookback"):
        if key not in contract:
            raise NestedSplitError(f"contract missing {key}")
    missing = [role for role in ROLES if role not in contract["roles"]]
    if missing:
        raise NestedSplitError(f"contract missing roles: {missing}")
    contract["_contract_sha256"] = _sha(Path(path))
    return contract


def _role_bounds(contract: Mapping[str, Any]) -> Dict[str, tuple]:
    bounds = {}
    for role in ROLES:
        spec = contract["roles"][role]
        start = pd.Timestamp(spec["start"])
        end = pd.Timestamp(spec["end"])
        if not start < end:
            raise NestedSplitError(f"{role}: start must precede end")
        bounds[role] = (start, end)
    return bounds


def _verify_chronology(bounds: Dict[str, tuple]) -> None:
    fit_start, fit_end = bounds["fit_train"]
    monitor_start, monitor_end = bounds["train_monitor"]
    if not (fit_start <= monitor_start and monitor_end <= fit_end):
        raise NestedSplitError(
            "train_monitor must be a subset of fit_train")
    chain = ("fit_train", "inner_validation", "outer_validation",
             "sealed_test")
    for earlier, later in zip(chain, chain[1:]):
        if bounds[earlier][1] != bounds[later][0]:
            raise NestedSplitError(
                f"gap or overlap between {earlier} and {later}: "
                f"{bounds[earlier][1]} != {bounds[later][0]}")


def materialize_nested_splits(contract: Mapping[str, Any], out_dir: Path,
                              *, mode: str) -> dict:
    """Materialize per-role CSVs with causal context prefixes and emit
    the split manifest. Fail-closed everywhere."""
    if mode not in MODES:
        raise NestedSplitError(f"unknown mode {mode!r}")
    source = Path(contract["source_csv"])
    actual_sha = _sha(source)
    if actual_sha != contract["source_sha256"]:
        raise NestedSplitError(
            f"source sha {actual_sha[:16]} != contract"
            f" {contract['source_sha256'][:16]}")
    date_col = contract["date_column"]
    df = pd.read_csv(source)
    if date_col not in df.columns:
        raise NestedSplitError(f"date column {date_col!r} missing")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    if df[date_col].isna().any():
        raise NestedSplitError("unparseable dates in source")
    if not df[date_col].is_monotonic_increasing:
        raise NestedSplitError(
            "source rows are not chronologically ordered; refusing"
            " to silently reorder a decision dataset")
    if df[date_col].duplicated().any():
        raise NestedSplitError("duplicate timestamps in source")

    bounds = _role_bounds(contract)
    _verify_chronology(bounds)
    context_bars = compute_context_bars(
        window_size=int(contract["window_size"]),
        scaling_window=int(contract["scaling_window"]),
        max_feature_lookback=int(contract["max_feature_lookback"]))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_roles = {}
    for role in ROLES:
        if role == "sealed_test" and mode != "release":
            manifest_roles[role] = {
                "status": "SEALED",
                "reason": f"sealed_test is inaccessible in mode {mode!r}",
            }
            continue
        start, end = bounds[role]
        scored_mask = (df[date_col] >= start) & (df[date_col] < end)
        scored = df.loc[scored_mask]
        if scored.empty:
            raise NestedSplitError(f"{role}: no rows in range")
        expected = (contract.get("expected_rows") or {}).get(role)
        if expected is not None and len(scored) != int(expected):
            raise NestedSplitError(
                f"{role}: derived {len(scored)} rows != asserted"
                f" {expected}")
        if role in EVAL_ROLES:
            score_start_index = scored.index[0]
            prefix_start = max(0, score_start_index - context_bars)
            prefix = df.iloc[prefix_start:score_start_index]
            if len(prefix) < context_bars:
                raise NestedSplitError(
                    f"{role}: only {len(prefix)} context rows available,"
                    f" {context_bars} required — the first scored row"
                    " would lack complete causal context")
            frame = pd.concat([prefix, scored])
            context_rows = len(prefix)
        else:
            frame = scored
            context_rows = 0
        csv_path = out_dir / f"{role}.csv"
        frame.to_csv(csv_path, index=False)
        manifest_roles[role] = {
            "status": "MATERIALIZED",
            "csv": str(csv_path),
            "csv_sha256": _sha(csv_path),
            "context_start": (str(frame[date_col].iloc[0])
                              if context_rows else None),
            "score_start": str(scored[date_col].iloc[0]),
            "score_end": str(scored[date_col].iloc[-1]),
            "context_rows": context_rows,
            "scored_rows": len(scored),
        }

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "mode": mode,
        "source_csv": str(source),
        "source_sha256": actual_sha,
        "contract_sha256": contract.get("_contract_sha256"),
        "date_column": date_col,
        "context_bars": context_bars,
        "context_inputs": {
            "window_size": int(contract["window_size"]),
            "scaling_window": int(contract["scaling_window"]),
            "max_feature_lookback": int(contract["max_feature_lookback"]),
        },
        "roles": manifest_roles,
    }
    manifest_path = out_dir / "nested_split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def verify_split_manifest(manifest_path: Path) -> dict:
    """Reload and re-hash every materialized split; any drift refuses.
    Loading an artifact against a different manifest is a fail-closed
    comparison of `contract_sha256` + per-role csv hashes."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise NestedSplitError("unknown split manifest schema")
    for role, entry in manifest["roles"].items():
        if entry.get("status") != "MATERIALIZED":
            continue
        csv_path = Path(entry["csv"])
        if not csv_path.is_file():
            raise NestedSplitError(f"{role}: split csv missing")
        if _sha(csv_path) != entry["csv_sha256"]:
            raise NestedSplitError(f"{role}: split csv hash drift")
    return manifest


def assert_same_split_identity(manifest_a: Mapping[str, Any],
                               manifest_b: Mapping[str, Any]) -> None:
    for key in ("contract_sha256", "source_sha256", "context_bars"):
        if manifest_a.get(key) != manifest_b.get(key):
            raise NestedSplitError(
                f"split identity mismatch on {key}: an artifact from one"
                " split contract cannot be evaluated under another")


# Account facts that must not move while the causal prefix runs
# (order §5.2; finding 231 requirement 4). Equity and trades were the
# original two; positions, open orders and paid execution costs are the
# other directly observable proofs that no order was submitted and no
# account state was mutated by a prefix row.
PREFIX_GUARDED_KEYS = ("position", "position_units", "open_order_count",
                       "open_orders", "orders", "commission_paid",
                       "slippage_paid", "trade_cost")


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


class ContextPrefixWrapper:
    """Reusable env-adapter boundary for causal prefixes (order §5.2).

    For the first `context_rows` steps after reset the agent's action is
    REPLACED with hold (0.0), `info['is_context_prefix']=True` is
    tagged, and any account mutation during the prefix raises. Metrics,
    replay and traces must exclude rows via this flag, never via row
    position.

    The wrapper also counts what it separated — `context_prefix_steps`
    and `scored_steps` — so a caller can prove the score boundary was
    reached instead of assuming it, and refuses any order/position
    mutation during the prefix, not only equity and closed trades.
    """

    def __init__(self, env, context_rows: int):
        if isinstance(context_rows, bool) or not isinstance(
                context_rows, int) or context_rows < 0:
            raise NestedSplitError("context_rows must be a non-negative int")
        self.env = env
        self.context_rows = context_rows
        self.context_prefix_steps = 0
        self.scored_steps = 0
        self._steps = 0
        self._prefix_equity = None
        self._prefix_state: Dict[str, Any] = {}

    def reset(self, **kwargs):
        self._steps = 0
        self.context_prefix_steps = 0
        self.scored_steps = 0
        obs, info = self.env.reset(**kwargs)
        self._prefix_equity = info.get("equity")
        self._prefix_state = {key: info.get(key)
                              for key in PREFIX_GUARDED_KEYS}
        info["is_context_prefix"] = self.context_rows > 0
        return obs, info

    def step(self, action):
        in_prefix = self._steps < self.context_rows
        if in_prefix:
            action = 0.0                      # forced hold
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        info["is_context_prefix"] = in_prefix
        if in_prefix:
            self.context_prefix_steps += 1
            equity = info.get("equity")
            trades = info.get("trades")
            if (self._prefix_equity is not None and equity is not None
                    and not math.isclose(float(equity),
                                         float(self._prefix_equity),
                                         rel_tol=0.0, abs_tol=1e-9)):
                raise NestedSplitError(
                    "account equity changed during the context prefix:"
                    f" {self._prefix_equity} -> {equity}")
            if trades not in (None, 0):
                raise NestedSplitError(
                    f"trades occurred during the context prefix: {trades}")
            self._assert_no_account_mutation(info)
        else:
            self.scored_steps += 1
        return obs, reward, terminated, truncated, info

    def _assert_no_account_mutation(self, info) -> None:
        """No order, position or execution cost may move during the
        prefix: the prefix is input to the observation window only."""
        for key in PREFIX_GUARDED_KEYS:
            now = info.get(key)
            if now is None:
                continue
            before = self._prefix_state.get(key)
            now_value = _numeric(now)
            before_value = _numeric(before)
            if before_value is not None and now_value is not None:
                if math.isclose(now_value, before_value,
                                rel_tol=0.0, abs_tol=1e-9):
                    continue
            elif before is None:
                if now_value is None or abs(now_value) <= 1e-9:
                    continue
            elif now == before:
                continue
            raise NestedSplitError(
                "account state changed during the context prefix: "
                f"{key} {before!r} -> {now!r} — the prefix forces hold "
                "and may not open orders or positions")

    def __getattr__(self, name):
        return getattr(self.env, name)
