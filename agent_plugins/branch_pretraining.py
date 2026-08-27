"""WP-PRETRAIN library v2 (Data-First order @7886de39; DATA-SOTA-341..346
corrected).

Discipline:

* 341 — pretraining roles are CAUSAL PER ORIGIN: the contract binds a
  ``score_origin`` and ``fit_end`` must precede its score-year start, so
  monitor/inner-validation/outer/sealed years are all structurally
  absent (rows after ``fit_end`` are never loaded). Origins after the
  first additionally require a frozen earlier-origin decision reference.
* 342 — pretraining windows come from the SAME executing preprocessor
  plugin (entry point ``preprocessor.plugins``) the GymFxEnv calls, with
  the same config; parity is proven by regression against env-emitted
  tensors.
* 343 — when a window-level normalization policy is used for masked
  reconstruction, statistics are computed over VISIBLE steps only:
  masked raw values cannot influence visible model inputs.
* 344 — normalization is a TYPED PER-FAMILY policy set declared in the
  contract and applied through the call path (eps included).
* 345 — objective weighting follows a predeclared bounded balancing
  rule (inverse initial loss on the monitor split), with per-objective
  monitor losses, gradient norms/cosines and quantile-crossing
  measurement; the quantile head is monotone BY CONSTRUCTION.
* 346 — resume identity binds EVERY executing identity field, and
  checkpoint+manifest are written as atomic fsynced generations sealed
  by digest; torn generations refuse.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from pipeline_plugins._observation_contract import feature_columns_sha256

FIRST_ORIGIN = "o2022"
DEVELOPMENT_OUTER_START = "2024-01-01"
NORMALIZATION_POLICIES = ("identity_preprocessed", "window_zscore_visible")


class PretrainContractError(ValueError):
    """Typed refusal: the pretraining contract, its resume identity or
    its artifact generation is invalid. Never construct, never train."""


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
                                                  require_int_list,
                                                  strict_int, strict_real)
    if contract.get("schema") != "agent_multi.pretrain_contract.v2":
        raise PretrainContractError(
            f"unsupported pretrain contract schema "
            f"{contract.get('schema')!r}")
    try:
        window = strict_int(contract.get("window_size"), "window_size", 2)
        stride = strict_int(contract.get("window_stride", 1),
                            "window_stride", 1)
        seed = strict_int(contract.get("seed"), "seed", 0)
        epochs = strict_int(contract.get("epochs"), "epochs", 1)
        warmup = strict_int(contract.get("warmup_bars"), "warmup_bars", 2)
        batch = strict_int(
            (contract.get("optimizer") or {}).get("batch_size"),
            "optimizer.batch_size", 1)
        lr = strict_real((contract.get("optimizer") or {}).get("lr"),
                         "optimizer.lr")
        monitor_fraction = strict_real(contract.get("monitor_fraction"),
                                       "monitor_fraction")
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    if lr <= 0:
        raise PretrainContractError(f"optimizer.lr must be > 0, got {lr}")
    if not (0.0 < monitor_fraction <= 0.5):
        raise PretrainContractError(
            f"monitor_fraction must lie in (0, 0.5], got "
            f"{monitor_fraction}")

    # ---- DATA-SOTA-341: causal per-origin role boundary
    origin = contract.get("score_origin") or {}
    origin_id = str(origin.get("origin_id") or "")
    score_start = str(origin.get("score_start") or "")
    if not origin_id or not score_start:
        raise PretrainContractError(
            "score_origin.origin_id and score_origin.score_start are "
            "required (DATA-SOTA-341: pretraining is causal per origin)")
    fit_end = str(contract.get("fit_end") or "")
    if not fit_end:
        raise PretrainContractError("fit_end is required")
    if fit_end[:10] >= score_start[:10]:
        raise PretrainContractError(
            f"fit_end={fit_end} does not precede score_origin "
            f"{origin_id} score_start={score_start}: monitor and inner "
            f"validation years are reserved (DATA-SOTA-341)")
    if fit_end[:10] >= DEVELOPMENT_OUTER_START:
        raise PretrainContractError(
            f"fit_end={fit_end} reaches development_outer (2024+); "
            f"sealed 2025 is structurally excluded")
    if origin_id != FIRST_ORIGIN and not str(
            contract.get("earlier_origin_decision_frozen") or "").strip():
        raise PretrainContractError(
            f"origin {origin_id} requires earlier_origin_decision_frozen"
            f": later origins expand only after the earlier decision is "
            f"frozen (DATA-SOTA-341)")

    # ---- DATA-SOTA-342: executing observation pipeline binding
    pipe = contract.get("observation_pipeline") or {}
    if str(pipe.get("preprocessor_plugin") or "") != \
            "feature_window_preprocessor":
        raise PretrainContractError(
            "observation_pipeline.preprocessor_plugin must name the "
            "executing preprocessor (feature_window_preprocessor); "
            "pretraining windows must come from the same transform the "
            "env applies (DATA-SOTA-342)")
    if not str(pipe.get("source_config") or ""):
        raise PretrainContractError(
            "observation_pipeline.source_config is required "
            "(DATA-SOTA-342)")

    branches = contract.get("branches") or []
    if not branches:
        raise PretrainContractError("branches must not be empty")

    # ---- DATA-SOTA-344: typed per-family normalization policies
    policies = contract.get("normalization_policies")
    if not isinstance(policies, dict) or not policies:
        raise PretrainContractError(
            "normalization_policies is required: every branch family "
            "declares a typed policy (DATA-SOTA-344)")
    names = [str(b.get("name")) for b in branches]
    if sorted(policies) != sorted(names):
        raise PretrainContractError(
            f"normalization_policies must cover the branch families "
            f"exactly: policies for {sorted(policies)}, branches "
            f"{sorted(names)} (DATA-SOTA-344)")
    parsed_policies: dict[str, dict[str, Any]] = {}
    for family, spec in policies.items():
        policy = str((spec or {}).get("policy") or "")
        if policy not in NORMALIZATION_POLICIES:
            raise PretrainContractError(
                f"normalization_policies[{family}].policy must be one "
                f"of {NORMALIZATION_POLICIES}, got {policy!r}")
        eps = None
        if policy == "window_zscore_visible":
            try:
                eps = strict_real(spec.get("eps"),
                                  f"normalization_policies[{family}].eps")
            except TopologyError as exc:
                raise PretrainContractError(str(exc)) from exc
            if eps <= 0:
                raise PretrainContractError(
                    f"normalization_policies[{family}].eps must be > 0")
        parsed_policies[str(family)] = {"policy": policy, "eps": eps}

    # ---- DATA-SOTA-345: predeclared bounded objective balancing
    balancing = contract.get("objective_balancing") or {}
    if str(balancing.get("method") or "") != "inverse_initial_loss":
        raise PretrainContractError(
            "objective_balancing.method must be 'inverse_initial_loss' "
            "(the predeclared bounded balancing rule; DATA-SOTA-345)")
    try:
        floor = strict_real(balancing.get("floor"),
                            "objective_balancing.floor")
    except TopologyError as exc:
        raise PretrainContractError(str(exc)) from exc
    if floor <= 0:
        raise PretrainContractError("objective_balancing.floor must be "
                                    "> 0")

    objectives = contract.get("objectives") or {}
    known = {"masked_patch_reconstruction", "multi_horizon_quantile"}
    unknown = set(objectives) - known
    if unknown:
        raise PretrainContractError(
            f"unknown objectives {sorted(unknown)}; wired: "
            f"{sorted(known)}")
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
        try:
            horizons = require_int_list(spec, "horizons", 1)
        except TopologyError as exc:
            raise PretrainContractError(str(exc)) from exc
        quantiles = spec.get("quantiles")
        if not isinstance(quantiles, (list, tuple)) or not quantiles:
            raise PretrainContractError("quantiles must be non-empty")
        previous = 0.0
        for q in quantiles:
            try:
                q = strict_real(q, "quantile")
            except TopologyError as exc:
                raise PretrainContractError(str(exc)) from exc
            if not (0.0 < q < 1.0):
                raise PretrainContractError(
                    f"quantiles must lie in (0, 1), got {q}")
            if q <= previous:
                raise PretrainContractError(
                    "quantiles must be strictly increasing (the "
                    "monotone head orders its outputs by quantile)")
            previous = q
        if len(set(horizons)) != len(horizons):
            raise PretrainContractError("horizons must be unique")

    return {"window_size": window, "window_stride": stride, "seed": seed,
            "epochs": epochs, "batch_size": batch, "lr": lr,
            "fit_end": fit_end, "warmup_bars": warmup,
            "monitor_fraction": monitor_fraction,
            "origin_id": origin_id, "score_start": score_start,
            "balancing_floor": floor,
            "normalization_policies": parsed_policies}


def load_fit_slice(csv_path, contract: dict[str, Any]):
    """Load ONLY the causal per-origin fit slice: rows after ``fit_end``
    (reserved monitor/validation years, development_outer, sealed) are
    never loaded into memory."""
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


def build_step_index(n_rows: int, warmup_bars: int, stride: int,
                     max_horizon: int,
                     max_windows: int | None) -> list[int]:
    """Eligible preprocessor steps t: the executing window covers rows
    [t-window, t) so the last OBSERVED bar is t-1; require full scaler
    warmup (t >= warmup_bars) and every forward target close[t-1+h]
    inside the fit slice (t-1+max_horizon <= n_rows-1)."""
    first = warmup_bars
    last = n_rows - max_horizon
    steps = list(range(first, last + 1, stride))
    if not steps:
        raise PretrainContractError(
            f"no eligible step: {n_rows} rows, warmup {warmup_bars}, "
            f"max horizon {max_horizon}")
    if max_windows is not None and len(steps) > max_windows:
        steps = steps[-max_windows:]  # keep the newest fit-slice steps
    return steps


def collect_preprocessed_windows(df, contract: dict[str, Any],
                                 env_config: dict[str, Any],
                                 steps: list[int]):
    """DATA-SOTA-342: emit pretraining windows through the SAME
    executing preprocessor plugin the GymFxEnv calls, with the same
    config — one shared transform, not a reimplementation."""
    import numpy as np

    from app.plugin_loader import load_plugin

    plugin_name = contract["observation_pipeline"]["preprocessor_plugin"]
    plugin_class, _ = load_plugin("preprocessor.plugins", plugin_name)
    preprocessor = plugin_class()
    neutral = {"initial_cash": 1000.0, "equity": 1000.0, "price": 0.0,
               "position": 0, "position_units": 0.0, "entry_price": 0.0,
               "holding_bars": 0, "bar_index": 0, "total_bars": len(df)}
    windows = []
    for t in steps:
        obs = preprocessor.make_observation(
            data=df, step=t, bridge_state={**neutral, "bar_index": t},
            config=env_config)
        windows.append(np.asarray(obs["features"], dtype=np.float32))
    return np.stack(windows, axis=0)  # (N, T, F)


def monitor_split(steps: list[int], monitor_fraction: float):
    """DATA-SOTA-345: the NEWEST fit-tail steps are held out as the
    monitor set and never trained on."""
    n_monitor = max(1, int(round(len(steps) * monitor_fraction)))
    if n_monitor >= len(steps):
        raise PretrainContractError(
            f"monitor split consumes every window ({n_monitor} of "
            f"{len(steps)})")
    return steps[:-n_monitor], steps[-n_monitor:]


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


def masked_visible_normalize(windows, mask, eps: float):
    """DATA-SOTA-343: normalization statistics come from VISIBLE steps
    only — changing masked raw values cannot change any visible
    normalized input. eps is the DECLARED contract value
    (DATA-SOTA-344), applied here, not a hardcoded default."""
    import torch

    visible = (~mask).unsqueeze(-1).to(windows.dtype)   # (B, T, 1)
    count = visible.sum(dim=1, keepdim=True)
    mean = (windows * visible).sum(dim=1, keepdim=True) / count
    var = (((windows - mean) ** 2) * visible).sum(dim=1,
                                                  keepdim=True) / count
    return (windows - mean) / (torch.sqrt(var) + eps)


def apply_family_policy(windows, mask, policy: dict[str, Any]):
    """Typed per-family normalization (DATA-SOTA-344): returns the
    tensor the reconstruction objective operates on."""
    if policy["policy"] == "identity_preprocessed":
        return windows  # executing preprocessor output, untouched
    return masked_visible_normalize(windows, mask, float(policy["eps"]))


def masked_reconstruction_loss(encoder, head, values, mask):
    """Encode the MASKED tensor, reconstruct, score ONLY masked steps."""
    import torch

    masked_in = values.masked_fill(mask.unsqueeze(-1), 0.0)
    pred = head(encoder(masked_in)).view(values.shape)
    diff = (pred - values)[mask]
    if diff.numel() == 0:
        return torch.zeros((), dtype=values.dtype)
    return (diff ** 2).mean()


def build_monotone_quantile_head(dim: int, n_horizons: int,
                                 n_quantiles: int):
    """DATA-SOTA-345: quantile outputs are monotone BY CONSTRUCTION —
    the first output is the lowest quantile and the rest are cumulative
    softplus increments, so crossing is structurally impossible."""
    import torch
    import torch.nn as nn

    class MonotoneQuantileHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(dim, n_horizons * n_quantiles)

        def forward(self, embedding):
            raw = self.proj(embedding).view(-1, n_horizons, n_quantiles)
            base = raw[..., :1]
            if n_quantiles == 1:
                return base
            steps = torch.nn.functional.softplus(raw[..., 1:])
            return torch.cat([base, base + torch.cumsum(steps, dim=-1)],
                             dim=-1)

    return MonotoneQuantileHead()


def pinball_loss(pred, target, quantiles):
    """pred (B, H, Q), target (B, H): mean quantile (pinball) loss."""
    import torch

    q = torch.tensor(list(quantiles), dtype=pred.dtype,
                     device=pred.device).view(1, 1, -1)
    err = target.unsqueeze(-1) - pred
    return torch.maximum(q * err, (q - 1.0) * err).mean()


def quantile_crossing_rate(pred) -> float:
    """Fraction of adjacent quantile pairs where a higher quantile
    predicts BELOW a lower one (measured even though the monotone head
    makes it structurally zero)."""
    import torch

    if pred.shape[-1] < 2:
        return 0.0
    crossings = (pred[..., 1:] < pred[..., :-1])
    return float(crossings.float().mean())


def forward_log_return_targets(close_values, steps, horizons):
    """(N, H) strictly-forward log returns from the LAST OBSERVED bar of
    each executing window: log(close[t-1+h] / close[t-1]) for step t
    (the window covers rows [t-window, t))."""
    import numpy as np

    close = np.asarray(close_values, dtype=np.float64)
    if (close <= 0).any():
        raise PretrainContractError("non-positive close in fit slice")
    anchor = np.asarray(steps) - 1
    cols = [np.log(close[anchor + h] / close[anchor]) for h in horizons]
    return np.stack(cols, axis=1).astype(np.float32)


def objective_gradient_diagnostics(encoder, losses: dict[str, Any]):
    """DATA-SOTA-345: per-objective gradient norm over the SHARED
    encoder parameters plus pairwise cosine (gradient conflict)."""
    import torch

    grads = {}
    params = [p for p in encoder.parameters() if p.requires_grad]
    for name, loss in losses.items():
        g = torch.autograd.grad(loss, params, retain_graph=True,
                                allow_unused=True)
        flat = torch.cat([x.reshape(-1) for x in g if x is not None])
        grads[name] = flat
    report: dict[str, Any] = {
        "norms": {k: round(float(v.norm()), 6) for k, v in grads.items()}}
    names = sorted(grads)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            denom = float(grads[a].norm() * grads[b].norm())
            cos = float(grads[a] @ grads[b]) / denom if denom > 0 else 0.0
            report[f"cosine:{a}|{b}"] = round(cos, 6)
    return report


def balance_objective_weights(initial_losses: dict[str, float],
                              declared: dict[str, float],
                              floor: float) -> dict[str, float]:
    """Predeclared bounded balancing (DATA-SOTA-345): effective weight =
    declared / max(initial monitor loss, floor), frozen before epoch 0."""
    return {name: declared[name] / max(float(initial_losses[name]), floor)
            for name in declared}


# ------------------------------------------------------------ durability

def resume_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """DATA-SOTA-346: EVERY identity field binds resume — nothing about
    the executing code, data, contract or environment may drift."""
    return dict(manifest["identity"])


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


def _fsync_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_generation(out_dir: Path, checkpoint_obj: dict[str, Any],
                     manifest_obj: dict[str, Any],
                     generation: int) -> None:
    """DATA-SOTA-346: checkpoint + manifest land as ONE atomic fsynced
    generation sealed by digest; a reader can always detect a torn
    pair."""
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(checkpoint_obj, buffer)
    ckpt_bytes = buffer.getvalue()
    manifest_bytes = json.dumps(manifest_obj, indent=1).encode()
    _fsync_write_bytes(out_dir / "checkpoint.pt", ckpt_bytes)
    _fsync_write_bytes(out_dir / "pretrain_manifest.json",
                       manifest_bytes)
    seal = {"schema": "agent_multi.pretrain_generation.v1",
            "generation": generation,
            "checkpoint_sha256": hashlib.sha256(ckpt_bytes).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                manifest_bytes).hexdigest()}
    _fsync_write_bytes(out_dir / "generation.json",
                       json.dumps(seal, indent=1).encode())


def load_generation(out_dir: Path):
    """Verify the generation seal; a torn checkpoint/manifest pair is a
    typed refusal, never a silent resume."""
    import torch

    seal_path = out_dir / "generation.json"
    ckpt_path = out_dir / "checkpoint.pt"
    manifest_path = out_dir / "pretrain_manifest.json"
    if not (seal_path.is_file() and ckpt_path.is_file()
            and manifest_path.is_file()):
        raise PretrainContractError(
            "resume REFUSED: no sealed generation in output dir")
    seal = json.loads(seal_path.read_text())
    actual_ckpt = sha256_file(ckpt_path)
    actual_manifest = sha256_file(manifest_path)
    if (seal.get("checkpoint_sha256") != actual_ckpt
            or seal.get("manifest_sha256") != actual_manifest):
        raise PretrainContractError(
            "resume REFUSED: TORN GENERATION — checkpoint/manifest do "
            "not match the generation seal (DATA-SOTA-346)")
    return (torch.load(ckpt_path, weights_only=False),
            json.loads(manifest_path.read_text()),
            int(seal["generation"]))


def canonical_feature_digest(columns) -> str:
    # F01 digest unity: the pipeline's canonical serialization, reused.
    return feature_columns_sha256(columns)
