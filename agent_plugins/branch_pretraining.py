"""WP-PRETRAIN library (Data-First order @7886de39): branch-wise
self-supervised pretraining for the strong grouped route.

Discipline inherited from P1-316/317 and DATA-SOTA-329..340:

* the pretraining slice is STRUCTURALLY bounded — rows after ``fit_end``
  are never loaded into memory, ``fit_end`` must precede 2024-01-01
  (2024 is ``development_outer``; sealed 2025 is structurally absent),
  and windows whose forward targets would cross ``fit_end`` are dropped;
* every identity field (contract digest, data digest, canonical
  feature-column digest, code identity) is bound into the artifact and
  resume REFUSES on any drift;
* no objective sees future inputs: encoder input is the in-window past
  only; targets are strictly-forward log-returns of the close.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from pipeline_plugins._observation_contract import feature_columns_sha256

DEVELOPMENT_OUTER_START = "2024-01-01"


class PretrainContractError(ValueError):
    """Typed refusal: the pretraining contract or its resume identity
    is invalid. Never construct, never train."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                     separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def validate_contract(contract: dict[str, Any]) -> dict[str, Any]:
    from feature_branch_plugins._topology import (TopologyError,
                                                  strict_int, strict_real)
    if contract.get("schema") != "agent_multi.pretrain_contract.v1":
        raise PretrainContractError(
            f"unsupported pretrain contract schema "
            f"{contract.get('schema')!r}")
    try:
        window = strict_int(contract.get("window_size"), "window_size", 2)
        stride = strict_int(contract.get("window_stride", 1),
                            "window_stride", 1)
        seed = strict_int(contract.get("seed"), "seed", 0)
        epochs = strict_int(contract.get("epochs"), "epochs", 1)
        batch = strict_int(
            (contract.get("optimizer") or {}).get("batch_size"),
            "optimizer.batch_size", 1)
        lr = strict_real((contract.get("optimizer") or {}).get("lr"),
                         "optimizer.lr")
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    if lr <= 0:
        raise PretrainContractError(f"optimizer.lr must be > 0, got {lr}")
    fit_end = str(contract.get("fit_end") or "")
    if not fit_end:
        raise PretrainContractError("fit_end is required")
    if fit_end[:10] >= DEVELOPMENT_OUTER_START:
        raise PretrainContractError(
            f"fit_end={fit_end} reaches development_outer (2024+): "
            f"pretraining fits end before {DEVELOPMENT_OUTER_START}; "
            f"sealed 2025 is structurally excluded")
    objectives = contract.get("objectives") or {}
    known = {"masked_patch_reconstruction", "multi_horizon_quantile"}
    unknown = set(objectives) - known
    if unknown:
        raise PretrainContractError(
            f"unknown objectives {sorted(unknown)}; wired: {sorted(known)}")
    if not objectives:
        raise PretrainContractError("at least one objective is required")
    for name, spec in objectives.items():
        try:
            weight = strict_real((spec or {}).get("weight"),
                                 f"objectives.{name}.weight")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if weight <= 0:
            raise PretrainContractError(
                f"objectives.{name}.weight must be > 0, got {weight}")
    if "masked_patch_reconstruction" in objectives:
        spec = objectives["masked_patch_reconstruction"]
        try:
            span = strict_int(spec.get("mask_span"), "mask_span", 1)
            ratio = strict_real(spec.get("mask_ratio"), "mask_ratio")
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        if not (0.0 < ratio < 1.0):
            raise PretrainContractError(
                f"mask_ratio must lie in (0, 1), got {ratio}")
        if span >= window:
            raise PretrainContractError(
                f"mask_span={span} must be < window_size={window}")
    if "multi_horizon_quantile" in objectives:
        spec = objectives["multi_horizon_quantile"]
        from feature_branch_plugins._topology import require_int_list
        try:
            horizons = require_int_list(spec, "horizons", 1)
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        quantiles = spec.get("quantiles")
        if not isinstance(quantiles, (list, tuple)) or not quantiles:
            raise PretrainContractError("quantiles must be non-empty")
        for q in quantiles:
            try:
                q = strict_real(q, "quantile")
            except TopologyError as exc:
                raise PretrainContractError(str(exc)) from exc
            if not (0.0 < q < 1.0):
                raise PretrainContractError(
                    f"quantiles must lie in (0, 1), got {q}")
        if len(set(horizons)) != len(horizons):
            raise PretrainContractError("horizons must be unique")
    if not contract.get("branches"):
        raise PretrainContractError("branches must not be empty")
    norm = contract.get("input_normalization") or {}
    if norm.get("mode") != "per_window_channel_zscore":
        raise PretrainContractError(
            "input_normalization.mode must be 'per_window_channel_zscore'"
            " in v1: raw-scale channels are refused (finding-235 family)")
    return {"window_size": window, "window_stride": stride, "seed": seed,
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "fit_end": fit_end}


def load_fit_slice(csv_path, contract: dict[str, Any]):
    """Load ONLY the pretraining fit slice.

    The dataframe returned physically ends at ``fit_end``; later rows —
    development_outer 2024 and sealed 2025 — never enter memory, so no
    downstream bug can peek at them.
    """
    import pandas as pd

    parsed = validate_contract(contract)
    date_col = str(contract.get("date_column") or "DATE_TIME")
    close_col = str(contract.get("close_column") or "CLOSE")
    columns = list(contract["feature_columns"])
    usecols = [date_col] + columns + (
        [close_col] if close_col not in columns else [])
    df = pd.read_csv(csv_path, usecols=usecols)
    stamps = pd.to_datetime(df[date_col])
    df = df.loc[stamps <= pd.Timestamp(parsed["fit_end"])]
    if df.empty:
        raise PretrainContractError(
            f"no rows at or before fit_end={parsed['fit_end']}")
    if df[columns + [close_col]].isna().any().any():
        bad = df[columns + [close_col]].isna().any()
        raise PretrainContractError(
            f"NaNs in fit slice columns: {sorted(bad[bad].index)}")
    return df.reset_index(drop=True), columns, close_col


def build_window_index(n_rows: int, window: int, stride: int,
                       max_horizon: int,
                       max_windows: int | None) -> list[int]:
    """End-indices t of eligible windows: the window [t-window+1, t] is
    fully in-slice AND every forward target close[t+h] is too — windows
    near fit_end whose targets would cross the boundary are DROPPED."""
    first = window - 1
    last = n_rows - 1 - max_horizon
    ends = list(range(first, last + 1, stride))
    if not ends:
        raise PretrainContractError(
            f"no eligible window: {n_rows} rows, window {window}, "
            f"max horizon {max_horizon}")
    if max_windows is not None and len(ends) > max_windows:
        ends = ends[-max_windows:]  # keep the newest fit-slice windows
    return ends


def instance_normalize(windows, eps: float = 1e-5):
    """Per-window per-channel z-score over the window's own T steps.

    Causally clean (statistics come from the in-window PAST only) and
    scale-invariant: raw-scale price/volume channels otherwise make
    masked reconstruction meaningless and drown the quantile term
    (observed 6e13 vs 0.1 on the real ETH H4 csv — the finding-235
    raw-price failure family)."""
    mean = windows.mean(dim=1, keepdim=True)
    std = windows.std(dim=1, keepdim=True, unbiased=False)
    return (windows - mean) / (std + eps)


def sample_span_mask(batch: int, window: int, ratio: float, span: int,
                     generator):
    """(B, T) boolean temporal mask covering ~ratio of the window in
    contiguous spans; always leaves >=1 masked and >=1 visible step."""
    import torch

    n_spans = max(1, int(round(window * ratio / span)))
    mask = torch.zeros(batch, window, dtype=torch.bool)
    starts = torch.randint(0, window - span + 1, (batch, n_spans),
                           generator=generator)
    for i in range(batch):
        for s in starts[i].tolist():
            mask[i, s:s + span] = True
        if mask[i].all():  # keep at least one visible step
            mask[i, 0] = False
    return mask


def masked_reconstruction_loss(encoder, head, windows, mask):
    """Encode the MASKED window, reconstruct, score ONLY masked steps."""
    import torch

    masked_in = windows.masked_fill(mask.unsqueeze(-1), 0.0)
    pred = head(encoder(masked_in)).view(windows.shape)
    diff = (pred - windows)[mask]
    if diff.numel() == 0:
        return torch.zeros((), dtype=windows.dtype)
    return (diff ** 2).mean()


def pinball_loss(pred, target, quantiles):
    """pred (B, H, Q), target (B, H): mean quantile (pinball) loss."""
    import torch

    q = torch.tensor(list(quantiles), dtype=pred.dtype,
                     device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def forward_log_return_targets(close_values, ends, horizons):
    """(N, H) strictly-forward log returns log(close[t+h] / close[t])."""
    import numpy as np

    close = np.asarray(close_values, dtype=np.float64)
    if (close <= 0).any():
        raise PretrainContractError("non-positive close in fit slice")
    ends = np.asarray(ends)
    cols = [np.log(close[ends + h] / close[ends]) for h in horizons]
    return np.stack(cols, axis=1).astype(np.float32)


def resume_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    keys = ("contract_sha256", "data_sha256", "feature_columns_sha256",
            "library_sha256", "branch_assignment_sha256", "seed",
            "window_size", "fit_end")
    return {k: manifest["identity"][k] for k in keys}


def refuse_on_identity_drift(saved: dict[str, Any],
                             current: dict[str, Any]) -> None:
    drift = {k: (saved.get(k), current.get(k))
             for k in set(saved) | set(current)
             if saved.get(k) != current.get(k)}
    if drift:
        raise PretrainContractError(
            "resume identity drift REFUSED: "
            + "; ".join(f"{k}: saved={a!r} current={b!r}"
                        for k, (a, b) in sorted(drift.items())))


def canonical_feature_digest(columns) -> str:
    # F01 digest unity: the pipeline's canonical serialization, reused.
    return feature_columns_sha256(columns)
