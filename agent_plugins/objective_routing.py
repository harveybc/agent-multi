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


# ---------------- P1/P2 validated protocol (final probe order 2026-08-28)

def split_adapter_train_val(fit_indices, purge_steps: int,
                            train_fraction: float = 0.7):
    """P1: split the probe-fit block CAUSALLY into adapter-train
    (oldest 70%) and adapter-val (newest remainder) with a purge
    between them; probe-score stays untouched until final scoring."""
    import numpy as np

    fit_indices = np.asarray(fit_indices)
    n = len(fit_indices)
    n_train = int(n * train_fraction)
    if n_train < 1 or n - n_train - purge_steps < 1:
        raise ProbeRefusal("adapter train/val split leaves an empty "
                           "block")
    return fit_indices[:n_train], fit_indices[n_train + purge_steps:]


def fit_adapter_validated(build_adapter, fit_loss_fn, val_loss_fn,
                          score_fn, protocol: dict):
    """P1 (DATA-SOTA-371): per fixed seed — early-stopped, best-state
    restored, finite-curve, minimum-improvement adapter fit; median +
    dispersion across seeds; material seed instability REFUSES."""
    import copy

    import torch

    seeds = list(protocol["adapter_seeds"])
    max_steps = int(protocol["max_steps"])
    min_steps = int(protocol["min_steps"])
    cadence = int(protocol["validation_cadence_steps"])
    patience = int(protocol["patience_steps"])
    scores = []
    curves = []
    for seed in seeds:
        torch.manual_seed(seed)
        adapter = build_adapter()
        optimizer = torch.optim.Adam(adapter.parameters(),
                                     lr=float(protocol["lr"]))
        generator = torch.Generator().manual_seed(seed + 1)
        with torch.no_grad():
            initial_val = float(val_loss_fn(adapter))
        best_val = initial_val
        best_state = copy.deepcopy(adapter.state_dict())
        best_step = 0
        history = {"initial_val": round(initial_val, 8), "val": []}
        step = 0
        while step < max_steps:
            loss = fit_loss_fn(adapter, generator)
            if not torch.isfinite(loss):
                raise ProbeRefusal(
                    f"non-finite adapter train loss (seed {seed})")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % cadence == 0:
                with torch.no_grad():
                    val = float(val_loss_fn(adapter))
                if val != val or abs(val) == float("inf"):
                    raise ProbeRefusal(
                        f"non-finite adapter val loss (seed {seed})")
                history["val"].append(round(val, 8))
                if val < best_val:
                    best_val = val
                    best_state = copy.deepcopy(adapter.state_dict())
                    best_step = step
                if (step >= min_steps
                        and step - best_step >= patience):
                    break
        adapter.load_state_dict(best_state)
        if best_val > initial_val * (1.0 - float(
                protocol["minimum_improvement_fraction"])):
            raise ProbeRefusal(
                f"ADAPTER_FAILED_TO_FIT (seed {seed}): best val "
                f"{best_val:.6f} did not improve >= "
                f"{protocol['minimum_improvement_fraction']:.0%} over "
                f"initial {initial_val:.6f}")
        with torch.no_grad():
            scores.append(float(score_fn(adapter)))
        history.update({"best_val": round(best_val, 8),
                        "best_step": best_step,
                        "stopped_at": step})
        curves.append(history)
    ordered = sorted(scores)
    median = ordered[len(ordered) // 2]
    dispersion = ordered[-1] - ordered[0]
    if abs(median) > 0 and dispersion / abs(median) > 0.5:
        raise ProbeRefusal(
            f"MATERIAL_SEED_INSTABILITY: dispersion {dispersion:.6f} "
            f"> 0.5 x |median| {abs(median):.6f} — the best seed is "
            f"never selected")
    return {"probe_score_median": round(median, 8),
            "probe_scores_by_seed": [round(s, 8) for s in scores],
            "dispersion": round(dispersion, 8),
            "curves": curves}


def normalized_skill(loss_random: float, loss_route: float,
                     loss_solo: float):
    """P2 (DATA-SOTA-372): skill with random=0 and solo=1; ill-ordered
    or near-zero denominators are DIAGNOSTIC_INVALID for ranking (raw
    losses always preserved by the caller)."""
    denominator = loss_random - loss_solo
    if loss_solo >= loss_random:
        return None, "DIAGNOSTIC_INVALID: ill-ordered (solo >= random)"
    if denominator < 0.05 * abs(loss_random):
        return None, "DIAGNOSTIC_INVALID: near-zero denominator"
    return round((loss_random - loss_route) / denominator, 4), None


def common_probe_surface_v2(*, embeddings_fit, embeddings_score,
                            masked_embeddings_fit,
                            masked_embeddings_score, windows_fit,
                            windows_score, mask_fit, mask_score,
                            quantile_targets_fit, quantile_targets_score,
                            quantile_quantiles, volatility_targets_fit,
                            volatility_targets_score, barrier_labels_fit,
                            barrier_labels_score, positions_fit,
                            positions_score, contrastive_exclusion: int,
                            contrastive_temperature: float,
                            adapter_train_pos, adapter_val_pos,
                            protocol: dict[str, Any],
                            floor_mode: bool = False,
                            only_tasks=None) -> dict[str, Any]:
    """P1-validated surface: every probe adapter fits on the causal
    adapter-train segment, early-stops on adapter-val, restores its
    best state, and only then scores on the untouched probe-score
    block; three fixed seeds, median + dispersion, instability
    refusal. Positions index INTO the fit arrays."""
    import torch
    import torch.nn.functional as F

    dim = int(embeddings_fit.shape[1])
    window_numel = int(windows_fit.shape[1] * windows_fit.shape[2])
    batch = int(protocol["batch_size"])
    train_pos = torch.as_tensor(adapter_train_pos)
    val_pos = torch.as_tensor(adapter_val_pos)
    report: dict[str, Any] = {"encoder_output_std": round(
        float(embeddings_score.std()), 6)}
    if report["encoder_output_std"] < 1e-4:
        raise ProbeRefusal("degenerate encoder output variance")

    def sampler(generator):
        pick = torch.randint(0, len(train_pos),
                             (min(batch, len(train_pos)),),
                             generator=generator)
        return train_pos[pick]

    def rec_loss(a, idx):
        pred = a(masked_embeddings_fit[idx]).view(
            -1, windows_fit.shape[1], windows_fit.shape[2])
        diff = (pred - windows_fit[idx])[mask_fit[idx]]
        return (diff ** 2).mean()

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

    weights = frozen_class_weights(
        barrier_labels_fit[train_pos].numpy())
    import numpy as np
    support = {}
    for col in range(barrier_labels_score.shape[1]):
        counts = {int(k): int(v) for k, v in zip(*np.unique(
            barrier_labels_score[:, col].numpy(), return_counts=True))}
        support[f"h{col}"] = counts
        if len(counts) < 2:
            raise ProbeRefusal(
                f"degenerate barrier probe support at horizon index "
                f"{col}: {counts}")
    from agent_plugins.branch_pretraining import barrier_loss
    tasks = {
        "reconstruction": {
            "build": lambda: torch.nn.Linear(dim, window_numel),
            "fit": lambda a, g: rec_loss(a, sampler(g)),
            "val": lambda a: rec_loss(a, val_pos),
            "score": lambda a: ((a(masked_embeddings_score).view(
                windows_score.shape) - windows_score)[mask_score]
                ** 2).mean()},
        "quantile": {
            "build": lambda: build_monotone_quantile_head(
                dim, quantile_targets_fit.shape[1],
                len(quantile_quantiles)),
            "fit": lambda a, g: (lambda idx: pinball_loss(
                a(embeddings_fit[idx]), quantile_targets_fit[idx],
                quantile_quantiles))(sampler(g)),
            "val": lambda a: pinball_loss(
                a(embeddings_fit[val_pos]),
                quantile_targets_fit[val_pos], quantile_quantiles),
            "score": lambda a: pinball_loss(
                a(embeddings_score), quantile_targets_score,
                quantile_quantiles)},
        "contrastive": {
            "build": lambda: build_projection_head(
                dim, int(protocol["projection_dim"])),
            "fit": lambda a, g: (lambda idx: info_nce(
                a, embeddings_fit[idx], masked_embeddings_fit[idx],
                positions_fit[idx]))(sampler(g)),
            "val": lambda a: info_nce(
                a, embeddings_fit[val_pos],
                masked_embeddings_fit[val_pos],
                positions_fit[val_pos]),
            "score": lambda a: info_nce(
                a, embeddings_score, masked_embeddings_score,
                positions_score)},
        "volatility": {
            "build": lambda: torch.nn.Linear(
                dim, volatility_targets_fit.shape[1]),
            "fit": lambda a, g: (lambda idx: F.mse_loss(
                a(embeddings_fit[idx]),
                volatility_targets_fit[idx]))(sampler(g)),
            "val": lambda a: F.mse_loss(
                a(embeddings_fit[val_pos]),
                volatility_targets_fit[val_pos]),
            "score": lambda a: F.mse_loss(
                a(embeddings_score), volatility_targets_score)},
        "barrier": {
            "build": lambda: torch.nn.Linear(
                dim, barrier_labels_fit.shape[1] * 3),
            "fit": lambda a, g: (lambda idx: barrier_loss(
                a(embeddings_fit[idx]), barrier_labels_fit[idx],
                weights))(sampler(g)),
            "val": lambda a: barrier_loss(
                a(embeddings_fit[val_pos]),
                barrier_labels_fit[val_pos], weights),
            "score": lambda a: barrier_loss(
                a(embeddings_score), barrier_labels_score, weights)},
    }
    report["probes"] = {}
    for task, spec in tasks.items():
        if only_tasks is not None and task not in only_tasks:
            continue
        try:
            report["probes"][task] = fit_adapter_validated(
                spec["build"], spec["fit"], spec["val"], spec["score"],
                protocol)
        except ProbeRefusal as refusal:
            if not floor_mode:
                raise
            # protocol ADDENDUM 2026-08-28: an unfittable/unstable
            # probe on the RANDOM floor is the floor's own signal
            if "MATERIAL_SEED_INSTABILITY" in str(refusal):
                report["probes"][task] = {
                    "probe_score_median": None,
                    "floor_diagnostic_invalid": str(refusal)}
            else:
                import torch as _torch

                _torch.manual_seed(int(protocol["adapter_seeds"][0]))
                fallback = spec["build"]()
                with _torch.no_grad():
                    score = float(spec["score"](fallback))
                report["probes"][task] = {
                    "probe_score_median": round(score, 8),
                    "floor_fit_marginal": True,
                    "refusal_recorded": str(refusal)[:100]}
    report["barrier_probe_support"] = support
    return report


PREDICTIVE_TASKS = ("quantile", "volatility", "barrier")


def select_routes(families: dict) -> dict:
    """C1 (DATA-SOTA-374/376) — the SELECTION AUTHORITY as one pure,
    regression-testable function over already-measured facts:

    * a ROUTE_REFUSED arm is NOT EVALUABLE and can never enter
      ``selected`` — including through any fallback;
    * eligibility requires ALL THREE predictive skills present, finite
      and >= -0.05; an arm missing any predictive skill is
      INCOMPLETE_EVIDENCE, never eligible and never fallback material;
    * the conservative full5 fallback applies ONLY when full5 itself is
      fully evaluable (all three predictive skills valid) but worse
      than random — 'valid but worse than random' is distinguished
      from 'not evaluable'.
    """
    import statistics

    verdicts = {}
    selected = {}
    for family, payload in families.items():
        arms = payload["arms"]
        evaluable = {}
        incomplete = {}
        for arm, facts in arms.items():
            if "ROUTE_REFUSED" in facts:
                continue
            skills = facts.get("skills") or {}
            predictive = {t: skills.get(t) for t in PREDICTIVE_TASKS}
            if any(v is None or v != v for v in predictive.values()):
                incomplete[arm] = [t for t, v in predictive.items()
                                   if v is None or v != v]
                continue
            evaluable[arm] = {
                "predictive": predictive,
                "eligible": all(v >= -0.05
                                for v in predictive.values()),
                "median_predictive": round(statistics.median(
                    predictive.values()), 4),
                "median_all": round(statistics.median(
                    [v for v in skills.values()
                     if v is not None]), 4)}
        eligible = {a: f for a, f in evaluable.items() if f["eligible"]}
        if eligible:
            ranked = sorted(eligible.items(),
                            key=lambda kv: (-kv[1]["median_predictive"],
                                            -kv[1]["median_all"]))
            best_arm, best = ranked[0]
            ties = [a for a, f in ranked[1:]
                    if abs(f["median_predictive"]
                           - best["median_predictive"]) <= 0.02
                    and abs(f["median_all"] - best["median_all"])
                    <= 0.02]
            if ties:
                verdicts[family] = (f"INCONCLUSIVE: {best_arm} ties "
                                    f"{ties}")
                selected[family] = None
            else:
                verdicts[family] = (
                    f"SELECTED: {best_arm} (median predictive skill "
                    f"{best['median_predictive']})")
                selected[family] = {"arm": best_arm,
                                    "label": "SELECTED"}
        elif evaluable:
            if "full5_control" in evaluable:
                verdicts[family] = (
                    "WORSE_THAN_RANDOM on a predictive probe in every "
                    "evaluable arm -> full5_control as CONSERVATIVE "
                    "DIAGNOSTIC candidate (full5 itself fully "
                    "evaluable), not proven optimal")
                selected[family] = {"arm": "full5_control",
                                    "label": "CONSERVATIVE_DIAGNOSTIC"}
            else:
                verdicts[family] = (
                    "NOT_EVALUABLE_FOR_SELECTION: no eligible arm and "
                    "full5_control is not fully evaluable — no "
                    "candidate can come from this probe")
                selected[family] = None
        elif incomplete:
            verdicts[family] = (
                f"INCOMPLETE_EVIDENCE: no arm has all three predictive "
                f"probes valid (missing e.g. {incomplete})")
            selected[family] = None
        else:
            verdicts[family] = ("NOT_EVALUABLE_FOR_SELECTION: every "
                                "arm ROUTE_REFUSED")
            selected[family] = None
    return {"verdicts": verdicts, "selected": selected}
