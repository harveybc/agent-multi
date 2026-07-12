#!/usr/bin/env python3
"""Train-only event-token transformer encoder for the Project 3 weekly pool.

This module implements the heavier sibling of ``event_token_attention_v1``:

    context_embedding_profile.family = event_token_transformer_v1

It is callable from ``project3_weekly_materialize.py`` with the exact same
contract as the attention bridge: given the loaded weekly-pool CSV rows, the
resolved source columns and the *train-only* row indices, it mutates ``rows``
in place with fixed per-row embedding/diagnostic columns and returns the
manifest fragments (``training_summary`` / ``model_config`` / generated column
lists).

Design goals (see PROJECT3_EVENT_TOKEN_TRANSFORMER_AGENT_SPEC_2026_06_17.md):

* Fit *only* on the subjob train window. Token value normalization, the
  supervised auxiliary readout, and every learned statistic are computed from
  train rows exclusively. Validation/test rows are transformed with the frozen
  train-fit encoder but never contribute to any fit.
* Small and robust: a couple of multi-head self-attention encoder blocks over
  one token per source column, followed by attention pooling to a single fixed
  vector per row. CPU-safe and fast enough for unit tests.
* Deterministic by seed. The encoder weights are drawn from a seeded
  ``numpy.random.Generator`` (a frozen random-projection transformer), and the
  only trained component is a closed-form ridge readout, so two runs with the
  same seed and inputs produce byte-identical embeddings.
* Fail clearly when dependencies are unavailable.

The encoder weights are random projections rather than back-propagated
parameters; the supervised signal lives in the closed-form ridge readout fit on
the train-only next-bar return. This is intentionally the "start small" v1 from
the spec, not the final architecture.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Sequence

try:  # numpy is required only for this family; keep the import lazy/clear.
    import numpy as _np
except Exception as _exc:  # pragma: no cover - exercised only without numpy
    _np = None
    _NUMPY_IMPORT_ERROR: Exception | None = _exc
else:
    _NUMPY_IMPORT_ERROR = None


FAMILY = "event_token_transformer_v1"
FRAMEWORK = "numpy_random_transformer_v1"


def _require_numpy() -> Any:
    if _np is None:
        raise RuntimeError(
            "context_embedding_profile.family=event_token_transformer_v1 requires "
            "numpy, which is not importable in this environment. Install numpy or "
            "use family=event_token_attention_v1."
        ) from _NUMPY_IMPORT_ERROR
    return _np


def _source_group(column: str, prefixes: Sequence[str]) -> str:
    for prefix in prefixes:
        if prefix and column.startswith(prefix):
            return prefix
    return "__other__"


def _layernorm(x: Any, eps: float = 1.0e-5) -> Any:
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / _np.sqrt(var + eps)


def _softmax(x: Any, axis: int = -1) -> Any:
    shifted = x - x.max(axis=axis, keepdims=True)
    exps = _np.exp(shifted)
    return exps / exps.sum(axis=axis, keepdims=True)


def _multi_head_self_attention(
    x: Any,
    *,
    w_q: Any,
    w_k: Any,
    w_v: Any,
    w_o: Any,
    num_heads: int,
) -> Any:
    n, s, h = x.shape
    head_dim = h // num_heads
    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    def split(t: Any) -> Any:
        return t.reshape(n, s, num_heads, head_dim).transpose(0, 2, 1, 3)

    qs = split(q)
    ks = split(k)
    vs = split(v)
    scores = qs @ ks.transpose(0, 1, 3, 2) / math.sqrt(head_dim)
    attn = _softmax(scores, axis=-1)
    ctx = attn @ vs
    ctx = ctx.transpose(0, 2, 1, 3).reshape(n, s, h)
    return ctx @ w_o


def encode_event_token_transformer(
    *,
    rows: list[dict[str, Any]],
    source_columns: list[str],
    train_indices: list[int],
    price_column: str,
    profile: dict[str, Any],
    output_prefix: str,
    safe_float: Callable[[Any], float],
) -> dict[str, Any]:
    """Fit (train-only) and apply the transformer encoder to every row.

    ``rows`` is mutated in place with the generated embedding/diagnostic
    columns. Returns the manifest fragment for the caller to merge.
    """
    np = _require_numpy()
    if not source_columns:
        raise ValueError("event_token_transformer_v1 requires at least one source column")
    if not train_indices:
        raise ValueError("event_token_transformer_v1 requires at least one train row")

    embedding_dim = int(profile.get("embedding_dim", 16))
    if embedding_dim <= 0 or embedding_dim > 256:
        raise ValueError("context embedding_dim must be between 1 and 256")
    hidden_size = int(profile.get("hidden_size", 16))
    num_heads = int(profile.get("num_heads", 2))
    if hidden_size <= 0 or num_heads <= 0 or hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be a positive multiple of num_heads")
    num_blocks = int(profile.get("num_blocks", 2))
    if num_blocks < 1 or num_blocks > 4:
        raise ValueError("num_blocks must be between 1 and 4")
    ff_dim = int(profile.get("ff_dim", 2 * hidden_size))
    value_clip = float(profile.get("value_clip", 8.0))
    ridge_lambda = float(profile.get("ridge_lambda", 1.0))
    batch_size = int(profile.get("batch_size", 512))
    if batch_size <= 0 or batch_size > 8192:
        raise ValueError("context transformer batch_size must be between 1 and 8192")
    seed = int(profile.get("seed", 0))
    prefixes = [str(p) for p in (profile.get("source_prefixes") or ["event_"])]

    n_rows = len(rows)
    n_tokens = len(source_columns)

    # --- Train-only token value normalization -------------------------------
    raw = np.zeros((n_rows, n_tokens), dtype=np.float32)
    manifest_raw = np.zeros((n_rows, n_tokens), dtype=np.float64)
    present = np.zeros((n_rows, n_tokens), dtype=bool)
    for j, col in enumerate(source_columns):
        for i, row in enumerate(rows):
            cell = row.get(col)
            value = safe_float(cell)
            raw[i, j] = value
            manifest_raw[i, j] = value
            present[i, j] = cell not in (None, "")
    train_idx = np.asarray(sorted(train_indices), dtype=np.int64)
    train_raw = raw[train_idx]
    manifest_train_raw = manifest_raw[train_idx]
    manifest_means = manifest_train_raw.mean(axis=0, dtype=np.float64)
    manifest_stds = (
        manifest_train_raw.std(axis=0, ddof=1)
        if len(train_idx) > 1
        else np.ones(n_tokens, dtype=np.float64)
    )
    manifest_stds = np.where(manifest_stds > 1.0e-9, manifest_stds, 1.0)
    means = train_raw.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = (
        train_raw.astype(np.float64).std(axis=0, ddof=1).astype(np.float32)
        if len(train_idx) > 1
        else np.ones(n_tokens, dtype=np.float32)
    )
    stds = np.where(stds > 1.0e-9, stds, 1.0)
    z = np.clip((raw - means) / stds, -value_clip, value_clip).astype(np.float32)

    # --- Deterministic frozen encoder weights -------------------------------
    rng = np.random.default_rng(seed)

    def randn(*shape: int) -> Any:
        return (rng.standard_normal(shape) / math.sqrt(hidden_size)).astype(np.float32)

    groups = [_source_group(col, prefixes) for col in source_columns]
    unique_groups = sorted(set(groups))
    type_emb = {g: (rng.standard_normal(hidden_size) * 0.1).astype(np.float32) for g in unique_groups}
    # Per-token input projection: value vector + (type + identity) bias.
    value_proj = np.stack([randn(hidden_size).ravel() for _ in source_columns])  # (S, H)
    col_bias = np.stack(
        [
            type_emb[groups[j]]
            + (rng.standard_normal(hidden_size) * 0.05).astype(np.float32)
            for j in range(n_tokens)
        ]
    ).astype(np.float32)  # (S, H)

    blocks = []
    for _ in range(num_blocks):
        blocks.append(
            {
                "w_q": randn(hidden_size, hidden_size),
                "w_k": randn(hidden_size, hidden_size),
                "w_v": randn(hidden_size, hidden_size),
                "w_o": randn(hidden_size, hidden_size),
                "w1": randn(hidden_size, ff_dim),
                "w2": randn(ff_dim, hidden_size),
            }
        )
    q_pool = (rng.standard_normal(hidden_size) / math.sqrt(hidden_size)).astype(np.float32)
    w_out = randn(hidden_size, embedding_dim)
    b_out = (rng.standard_normal(embedding_dim) * 0.01).astype(np.float32)

    # --- Forward pass (identical math for train/val/test rows) --------------
    embeddings = np.zeros((n_rows, embedding_dim), dtype=np.float32)
    attn_mass = np.zeros(n_rows, dtype=np.float32)
    token_count = present.sum(axis=1).astype(np.int32)  # (N,)

    for start in range(0, n_rows, batch_size):
        stop = min(n_rows, start + batch_size)
        x = z[start:stop, :, None] * value_proj[None, :, :] + col_bias[None, :, :]
        x = x.astype(np.float32, copy=False)
        for blk in blocks:
            attn = _multi_head_self_attention(
                x,
                w_q=blk["w_q"],
                w_k=blk["w_k"],
                w_v=blk["w_v"],
                w_o=blk["w_o"],
                num_heads=num_heads,
            )
            x = _layernorm(x + attn).astype(np.float32, copy=False)
            hidden = np.maximum(0.0, x @ blk["w1"])
            ff = hidden @ blk["w2"]
            x = _layernorm(x + ff).astype(np.float32, copy=False)

        # attention pooling to a single fixed vector per row
        pool_scores = (x @ q_pool) / math.sqrt(hidden_size)  # (B, S)
        pool_weights = _softmax(pool_scores, axis=1).astype(np.float32, copy=False)  # (B, S)
        pooled = np.einsum("ns,nsh->nh", pool_weights, x)  # (B, H)
        token_norms = np.linalg.norm(x, axis=2)  # (B, S)
        attn_mass[start:stop] = np.einsum("ns,ns->n", pool_weights, token_norms)
        embeddings[start:stop] = np.tanh(pooled @ w_out + b_out)

    # --- Train-only supervised auxiliary readout (closed-form ridge) --------
    train_set = set(int(i) for i in train_idx)
    aux_indices: list[int] = []
    targets_list: list[float] = []
    for i in train_idx.tolist():
        if i + 1 in train_set:
            now_price = max(1.0e-12, safe_float(rows[i].get(price_column)))
            next_price = max(1.0e-12, safe_float(rows[i + 1].get(price_column)))
            aux_indices.append(i)
            targets_list.append(math.log(next_price / now_price))
    if aux_indices:
        target_idx = np.asarray(aux_indices, dtype=np.int64)
        targets = np.asarray(targets_list, dtype=np.float64)
    else:
        target_idx = train_idx[:0]
        targets = np.zeros(0, dtype=np.float64)
    x_train = embeddings[target_idx]
    design = np.concatenate([x_train, np.ones((len(target_idx), 1))], axis=1)
    gram = design.T @ design
    reg = ridge_lambda * np.eye(design.shape[1])
    reg[-1, -1] = 0.0  # do not regularize the intercept
    if len(targets) > 0:
        try:
            beta = np.linalg.solve(gram + reg, design.T @ targets)
        except np.linalg.LinAlgError:  # pragma: no cover - singular fallback
            beta, *_ = np.linalg.lstsq(design, targets, rcond=None)
        preds = design @ beta
        resid = targets - preds
        ss_res = float(resid @ resid)
        centered = targets - targets.mean()
        ss_tot = float(centered @ centered)
        train_mse = ss_res / max(1, len(targets))
        train_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 1.0e-18 else 0.0
        if ss_tot > 1.0e-18 and ss_res >= 0.0:
            pred_centered = preds - preds.mean()
            denom = math.sqrt(float(pred_centered @ pred_centered) * ss_tot)
            train_pearson = abs(float(pred_centered @ centered) / denom) if denom > 1.0e-18 else 0.0
        else:
            train_pearson = 0.0
    else:
        beta = np.zeros(embedding_dim + 1, dtype=np.float64)
        train_mse = 0.0
        train_r2 = 0.0
        train_pearson = 0.0

    # --- Write generated columns -------------------------------------------
    embedding_columns = [f"{output_prefix}_{i:02d}" for i in range(embedding_dim)]
    diagnostic_columns = [f"{output_prefix}_attn_mass", f"{output_prefix}_token_count"]
    for i, row in enumerate(rows):
        for d, col in enumerate(embedding_columns):
            row[col] = f"{float(embeddings[i, d]):.10g}"
        row[diagnostic_columns[0]] = f"{float(attn_mass[i]):.10g}"
        row[diagnostic_columns[1]] = str(int(token_count[i]))

    normalization = {
        col: {"mean": float(manifest_means[j]), "std": float(manifest_stds[j])}
        for j, col in enumerate(source_columns)
    }
    model_config = {
        "framework": FRAMEWORK,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_blocks": num_blocks,
        "ff_dim": ff_dim,
        "embedding_dim": embedding_dim,
        "value_clip": value_clip,
        "ridge_lambda": ridge_lambda,
        "seed": seed,
        "batch_size": batch_size,
        "forward_dtype": "float32",
        "source_groups": {col: groups[j] for j, col in enumerate(source_columns)},
        "normalization": normalization,
        "aux_target": "next_bar_log_return",
    }
    training_summary = {
        "fit_scope": "train_only",
        "n_train_rows": int(len(train_idx)),
        "n_aux_target_rows": int(len(targets)),
        "n_total_rows": int(n_rows),
        "n_source_tokens": int(n_tokens),
        "aux_target": "next_bar_log_return",
        "ridge_lambda": ridge_lambda,
        "batch_size": batch_size,
        "forward_dtype": "float32",
        "train_mse": train_mse,
        "train_r2": train_r2,
        "train_pearson_abs": train_pearson,
        "readout_l2_norm": float(np.linalg.norm(beta)),
        "embedding_dim": embedding_dim,
    }
    return {
        "embedding_columns": embedding_columns,
        "diagnostic_columns": diagnostic_columns,
        "training_summary": training_summary,
        "model_config": model_config,
    }
