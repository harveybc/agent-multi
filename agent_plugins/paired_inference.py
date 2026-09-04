"""Reusable paired-inference statistics (order @8fce8da0 §4 R3).

Repairs, before any successor consumes them:

1. ``holm_adjust``: proper step-down Holm with the REQUIRED
   cumulative maximum in sorted order (monotone non-decreasing
   adjusted p-values), capped at 1.
2. ``paired_t``: zero-variance paired differences produce FINITE,
   predeclared outcomes — positive constant -> p_one_sided = 0.0 and
   a degenerate CI at the constant; zero constant -> p = 1.0, CI
   [0, 0]; negative constant -> p = 1.0, CI at the constant. The
   ``inf -> linspace -> NaN`` path is structurally impossible.
3. NaN and infinite scores, differences, statistics and adjusted
   p-values are REJECTED with a typed error, never propagated.
"""
from __future__ import annotations

import math
from typing import Sequence


class PairedInferenceError(ValueError):
    """Typed refusal: non-finite input or invalid statistic."""


def _require_finite(name: str, values: Sequence[float]) -> list:
    out = []
    for i, v in enumerate(values):
        f = float(v)
        if not math.isfinite(f):
            raise PairedInferenceError(
                f"{name}[{i}] is not finite: {v!r} — refused")
        out.append(f)
    return out


def t_survival(t: float, df: int) -> float:
    """One-sided P(T_df > t) by numerical integration of the exact
    t density. Requires finite t (zero-variance cases are handled
    BEFORE any t statistic exists)."""
    if not math.isfinite(t):
        raise PairedInferenceError(
            f"t statistic is not finite: {t!r} — the zero-variance "
            "branch must be taken before computing t")
    if df < 1:
        raise PairedInferenceError(f"df must be >= 1, got {df}")
    import numpy as np
    grid = np.linspace(abs(t), abs(t) + 80, 320000)
    coeff = (math.gamma((df + 1) / 2.0)
             / (math.sqrt(df * math.pi) * math.gamma(df / 2.0)))
    pdf = coeff * (1 + grid ** 2 / df) ** (-(df + 1) / 2.0)
    integrate = getattr(np, "trapezoid", None) or np.trapz
    p = float(integrate(pdf, grid))
    p = min(max(p, 0.0), 1.0)
    return p if t > 0 else 1.0 - p


def paired_t(diffs: Sequence[float], *,
             t_crit: float | None = None) -> dict:
    """Paired one-sample t analysis with predeclared finite
    outcomes for zero-variance differences."""
    d = _require_finite("diffs", diffs)
    n = len(d)
    if n < 2:
        raise PairedInferenceError(
            f"paired_t needs >= 2 differences, got {n}")
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    sd = math.sqrt(var)
    df = n - 1
    if t_crit is None:
        # two-sided 95% critical values for small df (exact table)
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
                 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
                 9: 2.262, 10: 2.228}
        t_crit = table.get(df, 1.96)
    if sd == 0.0:
        # predeclared finite outcomes — no t statistic exists
        if mean > 0:
            return {"n": n, "mean": mean, "sd": 0.0,
                    "ci95": [mean, mean], "t_stat": None,
                    "p_one_sided": 0.0,
                    "zero_variance": "positive_constant"}
        if mean < 0:
            return {"n": n, "mean": mean, "sd": 0.0,
                    "ci95": [mean, mean], "t_stat": None,
                    "p_one_sided": 1.0,
                    "zero_variance": "negative_constant"}
        return {"n": n, "mean": 0.0, "sd": 0.0,
                "ci95": [0.0, 0.0], "t_stat": None,
                "p_one_sided": 1.0,
                "zero_variance": "zero_constant"}
    se = sd / math.sqrt(n)
    t_stat = mean / se
    p = t_survival(t_stat, df)
    result = {"n": n, "mean": mean, "sd": sd,
              "ci95": [mean - t_crit * se, mean + t_crit * se],
              "t_stat": t_stat, "p_one_sided": p,
              "zero_variance": None}
    for key in ("mean", "sd", "t_stat", "p_one_sided"):
        if result[key] is not None and \
                not math.isfinite(result[key]):
            raise PairedInferenceError(
                f"non-finite statistic {key}={result[key]!r}")
    return result


def holm_adjust(pvals: dict) -> dict:
    """Step-down Holm with the REQUIRED cumulative maximum: sort
    ascending, multiply p_(i) by (m - i), then enforce monotone
    non-decreasing adjusted values via a running maximum, capped at
    1. Rejects non-finite inputs."""
    if not pvals:
        return {}
    _require_finite("pvals", list(pvals.values()))
    for k, v in pvals.items():
        if not (0.0 <= float(v) <= 1.0):
            raise PairedInferenceError(
                f"pvals[{k!r}]={v!r} outside [0, 1]")
    m = len(pvals)
    ordered = sorted(pvals.items(), key=lambda kv: kv[1])
    adjusted = {}
    running_max = 0.0
    for rank, (key, p) in enumerate(ordered):
        raw = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, raw)
        adjusted[key] = running_max
    return adjusted
