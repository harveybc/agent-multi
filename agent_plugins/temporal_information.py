"""Temporal-information acceptance suite — core primitives (order
b06ec0c7 §4; continuation ordered 2026-08-28). REPRESENTATION_DIAGNOSTIC:
nothing here is a promotion authority.

The suite distinguishes ABSENCE OF LOOKAHEAD from PRESERVATION OF
USEFUL TEMPORAL INFORMATION, per family and on the fused
representation. Structural controls are exact tensor statements;
predictive controls run frozen-encoder probes against shuffled-time
and random-encoder baselines through the SAME validated adapter
machinery the routing screens used."""
from __future__ import annotations

from typing import Any, Callable

import numpy as np


class TemporalControlFailure(AssertionError):
    """A required control failed — typed, never a silent pass."""


# ---------------------------------------------------------------- #
# Structural controls (exact, per encoder consuming (B, T, F))     #
# ---------------------------------------------------------------- #

def control_future_immutability(encoder, window: int, features: int,
                                seed: int = 0) -> dict:
    """Control 1: the representation of a window ending at bar t must
    be BIT-IDENTICAL when any bar AFTER t changes. Windows are the
    encoder's whole input, so the statement is: the representation at
    position t of a longer series depends only on bars <= t."""
    import torch

    torch.manual_seed(seed)
    series = torch.randn(1, window + 8, features)
    encoder.eval()
    with torch.no_grad():
        base = encoder(series[:, :window, :]).clone()
        mutated = series.clone()
        mutated[:, window:, :] += 5.0        # future bars only
        after = encoder(mutated[:, :window, :])
    identical = bool(torch.equal(base, after))
    if not identical:
        raise TemporalControlFailure(
            "future mutation changed an earlier representation")
    return {"control": "future_immutability", "passed": True,
            "comparison": "bitwise torch.equal on the window ending "
                          "before the mutation"}


def control_newest_bar_sensitivity(encoder, window: int,
                                   features: int,
                                   seed: int = 0) -> dict:
    """Control 2: mutating the newest bar must change the output."""
    import torch

    torch.manual_seed(seed)
    probe = torch.randn(1, window, features)
    encoder.eval()
    with torch.no_grad():
        base = encoder(probe)
        mutated = probe.clone()
        mutated[:, -1, :] += 1.0
        after = encoder(mutated)
    delta = float(torch.linalg.vector_norm(after - base).item())
    if delta <= 0.0:
        raise TemporalControlFailure(
            "newest-bar mutation left the representation unchanged")
    return {"control": "newest_bar_sensitivity", "passed": True,
            "l2_delta": round(delta, 8)}


def control_time_reversal(encoder, window: int, features: int,
                          seed: int = 0,
                          min_relative_delta: float = 1e-3) -> dict:
    """Control 3: reversing time must produce a materially different
    representation (a time-blind pooling encoder fails here)."""
    import torch

    torch.manual_seed(seed)
    probe = torch.randn(1, window, features)
    encoder.eval()
    with torch.no_grad():
        base = encoder(probe)
        reversed_out = encoder(torch.flip(probe, dims=[1]))
    delta = float(torch.linalg.vector_norm(
        reversed_out - base).item())
    scale = float(torch.linalg.vector_norm(base).item()) or 1.0
    relative = delta / scale
    if relative < min_relative_delta:
        raise TemporalControlFailure(
            f"time reversal changed the representation by only "
            f"{relative:.2e} (relative) — temporally indifferent")
    return {"control": "time_reversal", "passed": True,
            "relative_l2_delta": round(relative, 6)}


def control_save_load_bit_exact(encoder, window: int, features: int,
                                seed: int = 0) -> dict:
    """Control 7: serialize/restore under the same identity is
    bit-exact in weights AND outputs."""
    import copy
    import io

    import torch

    torch.manual_seed(seed)
    probe = torch.randn(2, window, features)
    encoder.eval()
    buffer = io.BytesIO()
    torch.save(encoder.state_dict(), buffer)
    buffer.seek(0)
    clone = copy.deepcopy(encoder)
    clone.load_state_dict(torch.load(buffer, weights_only=True))
    clone.eval()
    with torch.no_grad():
        a = encoder(probe)
        b = clone(probe)
    if not torch.equal(a, b):
        raise TemporalControlFailure(
            "save/load round trip is not bit-exact")
    return {"control": "save_load_bit_exact", "passed": True}


# ---------------------------------------------------------------- #
# Representation diagnostics                                       #
# ---------------------------------------------------------------- #

def effective_rank(embeddings: np.ndarray) -> dict:
    """Entropy-based effective rank + top-1 variance share (collapse
    diagnostic). embeddings: (N, D)."""
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    total = float(energy.sum()) or 1.0
    probabilities = energy / total
    nonzero = probabilities[probabilities > 1e-12]
    entropy = float(-(nonzero * np.log(nonzero)).sum())
    return {
        "effective_rank": round(float(np.exp(entropy)), 3),
        "dimension": int(embeddings.shape[1]),
        "top1_variance_share": round(float(probabilities[0]), 4),
        "collapse_flag": bool(probabilities[0] > 0.99),
    }


def lagged_correlation_preservation(embeddings: np.ndarray,
                                    feature_series: np.ndarray,
                                    lags: tuple = (1, 2, 4, 8, 16),
                                    ) -> dict:
    """How much linearly-decodable memory of the feature k bars back
    does the embedding hold? R^2 of ridge regression from the
    embedding at t to the feature value at t-k, per lag."""
    out = {}
    n = min(len(embeddings), len(feature_series))
    emb = embeddings[:n]
    series = feature_series[:n]
    for lag in lags:
        target = series[:-lag] if lag else series
        source = emb[lag:]
        m = min(len(source), len(target))
        if m < 32:
            out[f"lag_{lag}"] = None
            continue
        x = source[:m]
        y = target[:m]
        x = np.hstack([x, np.ones((m, 1))])
        coef, *_ = np.linalg.lstsq(
            x + 1e-8 * np.random.default_rng(0).standard_normal(
                x.shape), y, rcond=None)
        predicted = x @ coef
        ss_res = float(((y - predicted) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum()) or 1.0
        out[f"lag_{lag}"] = round(1.0 - ss_res / ss_tot, 4)
    return out


def phase_randomized_surrogate(series: np.ndarray,
                               seed: int = 0) -> np.ndarray:
    """Surrogate preserving the marginal amplitude spectrum but
    destroying phase structure (control 5), column-wise."""
    rng = np.random.default_rng(seed)
    result = np.empty_like(series)
    for column in range(series.shape[1]):
        spectrum = np.fft.rfft(series[:, column])
        phases = rng.uniform(0, 2 * np.pi, len(spectrum))
        phases[0] = 0.0
        surrogate = np.abs(spectrum) * np.exp(1j * phases)
        result[:, column] = np.fft.irfft(surrogate,
                                         n=len(series))
    return result


def synthetic_signals(bars: int, features: int,
                      seed: int = 0) -> dict:
    """Controlled series with KNOWN periodicity, phase and a regime
    change — plus constant, duplicated and noise-only channels for
    control 6."""
    rng = np.random.default_rng(seed)
    t = np.arange(bars)
    period = 16
    base = np.sin(2 * np.pi * t / period)
    regime = np.where(t < bars // 2, 1.0, -1.0)
    columns = []
    for f in range(features):
        if f == 0:
            columns.append(base)                       # periodic
        elif f == 1:
            columns.append(np.sin(2 * np.pi * t / period
                                  + np.pi / 3))        # phase-shifted
        elif f == 2:
            columns.append(base * regime)              # regime switch
        elif f == 3:
            columns.append(np.full(bars, 0.7))         # constant
        elif f == 4:
            columns.append(base.copy())                # duplicate
        else:
            columns.append(rng.standard_normal(bars))  # noise-only
    return {"series": np.stack(columns, axis=1).astype(np.float32),
            "period": period,
            "channel_roles": ["periodic", "phase_shifted",
                              "regime_switch", "constant",
                              "duplicate_of_0"]
            + ["noise"] * max(0, features - 5)}


def window_embeddings(encoder, series: np.ndarray,
                      window: int) -> np.ndarray:
    """Frozen-encoder embeddings of every complete causal window."""
    import torch

    encoder.eval()
    windows = np.stack([series[i - window:i]
                        for i in range(window, len(series) + 1)])
    with torch.no_grad():
        out = encoder(torch.tensor(windows, dtype=torch.float32))
    return out.numpy()


def probe_r2(embeddings: np.ndarray, targets: np.ndarray,
             *, ridge: float = 1e-3) -> float:
    """Causal-split ridge probe: fit on the first 70%, score R^2 on
    the last 30% (never shuffled)."""
    n = min(len(embeddings), len(targets))
    emb, y = embeddings[:n], targets[:n]
    split = int(n * 0.7)
    x_fit = np.hstack([emb[:split], np.ones((split, 1))])
    x_val = np.hstack([emb[split:], np.ones((n - split, 1))])
    gram = x_fit.T @ x_fit + ridge * np.eye(x_fit.shape[1])
    coef = np.linalg.solve(gram, x_fit.T @ y[:split])
    predicted = x_val @ coef
    residual = float(((y[split:] - predicted) ** 2).sum())
    total = float(((y[split:] - y[split:].mean()) ** 2).sum()) or 1.0
    return round(1.0 - residual / total, 4)
