"""Common five-objective probe surface (routing order 2026-08-27 R2).

Every trained route encoder is evaluated on the SAME five downstream
probes — including objectives it did not train — so route cardinality
can never change the evaluation surface (DATA-SOTA-369). The encoder
is FROZEN structurally: probe adapters fit on CACHED embeddings, so no
gradient can reach the encoder. Adapters are fresh, identically
initialized (fixed seed), with fixed capacity, optimizer, steps and
stopping (fixed step count — no early stop), and they never transfer.

A missing or degenerate probe REFUSES the route rather than shrinking
the evaluation surface.
"""
from __future__ import annotations

from typing import Any

from agent_plugins.branch_pretraining import (
    PretrainContractError, build_monotone_quantile_head,
    build_projection_head, frozen_class_weights, pinball_loss,
    sample_span_mask)

PROBE_TASKS = ("reconstruction", "quantile", "contrastive",
               "volatility", "barrier")


class ProbeRefusal(PretrainContractError):
    """Typed refusal: a probe is missing or degenerate — the route is
    refused instead of facing a smaller exam."""


def _fit_adapter(adapter, embeddings_fit, target_fn_fit, score_fn,
                 protocol):
    """Fixed-protocol adapter fit on cached embeddings; returns
    (probe_score, convergence)."""
    import torch

    optimizer = torch.optim.Adam(adapter.parameters(),
                                 lr=float(protocol["lr"]))
    n = embeddings_fit.shape[0]
    batch = int(protocol["batch_size"])
    generator = torch.Generator().manual_seed(int(protocol["seed"]))
    first = last = None
    for step in range(int(protocol["steps"])):
        idx = torch.randint(0, n, (min(batch, n),),
                            generator=generator)
        loss = target_fn_fit(adapter, idx)
        if not torch.isfinite(loss):
            raise ProbeRefusal("non-finite adapter fit loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        first = value if first is None else first
        last = value
    with torch.no_grad():
        score = float(score_fn(adapter))
    return score, {"fit_first": round(first, 6),
                   "fit_last": round(last, 6),
                   "converged": last < first}


def common_probe_surface(*, embeddings_fit, embeddings_score,
                         masked_embeddings_fit, masked_embeddings_score,
                         windows_fit, windows_score, mask_fit,
                         mask_score, quantile_targets_fit,
                         quantile_targets_score, quantile_quantiles,
                         volatility_targets_fit,
                         volatility_targets_score, barrier_labels_fit,
                         barrier_labels_score, positions_fit,
                         positions_score, contrastive_exclusion: int,
                         contrastive_temperature: float,
                         protocol: dict[str, Any]) -> dict[str, Any]:
    """Fit fresh adapters for ALL FIVE probes on probe-fit; score every
    one on probe-score. All encoder outputs arrive pre-computed
    (frozen encoder, cached)."""
    import torch
    import torch.nn.functional as F

    dim = int(embeddings_fit.shape[1])
    window_numel = int(windows_fit.shape[1] * windows_fit.shape[2])
    report: dict[str, Any] = {"encoder_output_std": round(
        float(embeddings_score.std()), 6)}
    if report["encoder_output_std"] < 1e-4:
        raise ProbeRefusal("degenerate encoder output variance")
    results = {}

    torch.manual_seed(int(protocol["seed"]))  # identical adapter init

    # 1. masked reconstruction: embedding of the MASKED window -> window
    adapter = torch.nn.Linear(dim, window_numel)

    def rec_fit(a, idx):
        pred = a(masked_embeddings_fit[idx]).view(
            -1, windows_fit.shape[1], windows_fit.shape[2])
        diff = (pred - windows_fit[idx])[mask_fit[idx]]
        return (diff ** 2).mean()

    def rec_score(a):
        pred = a(masked_embeddings_score).view(windows_score.shape)
        diff = (pred - windows_score)[mask_score]
        return (diff ** 2).mean()
    results["reconstruction"] = _fit_adapter(adapter, embeddings_fit,
                                             rec_fit, rec_score,
                                             protocol)

    # 2. multi-horizon quantiles (monotone head)
    n_h = quantile_targets_fit.shape[1]
    adapter = build_monotone_quantile_head(dim, n_h,
                                           len(quantile_quantiles))

    def quant_fit(a, idx):
        return pinball_loss(a(embeddings_fit[idx]),
                            quantile_targets_fit[idx],
                            quantile_quantiles)

    def quant_score(a):
        return pinball_loss(a(embeddings_score),
                            quantile_targets_score,
                            quantile_quantiles)
    results["quantile"] = _fit_adapter(adapter, embeddings_fit,
                                       quant_fit, quant_score, protocol)

    # 3. contrastive/retrieval: projection InfoNCE between anchor and
    # smoothed-view embeddings (both pre-computed? — the view embedding
    # equals the anchor here would be degenerate; instead use the
    # MASKED embedding as the corrupted positive view: retrieval of the
    # intact window from its corrupted view)
    adapter = build_projection_head(dim, int(protocol[
        "projection_dim"]))

    def info_nce(a, anchor_e, view_e, positions):
        z_a = F.normalize(a(anchor_e), dim=-1)
        z_v = F.normalize(a(view_e), dim=-1)
        logits = z_a @ z_v.T / contrastive_temperature
        distance = (positions[:, None] - positions[None, :]).abs()
        negative_mask = distance > contrastive_exclusion
        positive = torch.diagonal(logits)
        masked = logits.masked_fill(~negative_mask, float("-inf"))
        denominator = torch.logsumexp(
            torch.cat([positive.unsqueeze(1), masked], dim=1), dim=1)
        return (denominator - positive).mean()

    def con_fit(a, idx):
        return info_nce(a, embeddings_fit[idx],
                        masked_embeddings_fit[idx], positions_fit[idx])

    def con_score(a):
        return info_nce(a, embeddings_score, masked_embeddings_score,
                        positions_score)
    results["contrastive"] = _fit_adapter(adapter, embeddings_fit,
                                          con_fit, con_score, protocol)

    # 4. realized volatility
    adapter = torch.nn.Linear(dim, volatility_targets_fit.shape[1])

    def vol_fit(a, idx):
        return torch.nn.functional.mse_loss(
            a(embeddings_fit[idx]), volatility_targets_fit[idx])

    def vol_score(a):
        return torch.nn.functional.mse_loss(
            a(embeddings_score), volatility_targets_score)
    results["volatility"] = _fit_adapter(adapter, embeddings_fit,
                                         vol_fit, vol_score, protocol)

    # 5. OHLC barrier hit: weighted CE, class weights FROZEN from
    # probe-fit labels (declared)
    n_bh = barrier_labels_fit.shape[1]
    weights = frozen_class_weights(barrier_labels_fit.numpy())
    support = {}
    for col in range(barrier_labels_score.shape[1]):
        import numpy as np
        counts = {int(k): int(v) for k, v in zip(*np.unique(
            barrier_labels_score[:, col].numpy(),
            return_counts=True))}
        support[f"h{col}"] = counts
        if len(counts) < 2:
            raise ProbeRefusal(
                f"degenerate barrier probe support at horizon index "
                f"{col}: {counts}")
    adapter = torch.nn.Linear(dim, n_bh * 3)

    def bar_fit(a, idx):
        from agent_plugins.branch_pretraining import barrier_loss
        return barrier_loss(a(embeddings_fit[idx]),
                            barrier_labels_fit[idx], weights)

    def bar_score(a):
        from agent_plugins.branch_pretraining import barrier_loss
        return barrier_loss(a(embeddings_score),
                            barrier_labels_score, weights)
    results["barrier"] = _fit_adapter(adapter, embeddings_fit, bar_fit,
                                      bar_score, protocol)

    report["probes"] = {task: {"probe_score": round(score, 8),
                               **convergence}
                        for task, (score, convergence)
                        in results.items()}
    report["barrier_probe_support"] = support
    for task, facts in report["probes"].items():
        if not facts["converged"]:
            raise ProbeRefusal(
                f"probe adapter for {task} did not converge on "
                f"probe-fit — route refused, surface never shrinks")
    return report
