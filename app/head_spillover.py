"""Measured head-delta decomposition; no inferred information-capacity bound."""
import numpy as np


def decompose_final_layer(hidden0, hidden1, weight0, weight1, bias0, bias1):
    h0, h1, w0, w1, b0, b1 = [np.asarray(x, dtype=np.float64)
                               for x in (hidden0, hidden1, weight0, weight1, bias0, bias1)]
    if (h0.ndim != 2 or h1.shape != h0.shape or w0.ndim != 2
            or w1.shape != w0.shape or w0.shape[1] != h0.shape[1]
            or b0.shape != (w0.shape[0],) or b1.shape != b0.shape
            or not all(np.isfinite(x).all() for x in (h0, h1, w0, w1, b0, b1))):
        raise ValueError('SPILLOVER_SHAPE_OR_FINITE')
    dh, dw = h1-h0, w1-w0
    return dict(weight=h0@dw.T, hidden=dh@w0.T,
                interaction=dh@dw.T, bias=np.broadcast_to(b1-b0, (len(h0), len(b0))))


def spectrum(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('SPILLOVER_SPECTRUM_INPUT')
    s = np.linalg.svd(x-x.mean(0), full_matrices=False, compute_uv=False)
    energy = s*s
    total = float(energy.sum())
    p = energy/total if total else np.zeros_like(energy)
    positive = p[p > 0]
    return dict(rows=len(x), width=x.shape[1], singular_values=s.tolist(),
                effective_rank=float(np.exp(-(positive*np.log(positive)).sum())) if total else 0.,
                dimensions_99pct=int(np.searchsorted(np.cumsum(p), .99)+1) if total else 0,
                constant_dimensions=int((np.ptp(x, axis=0) == 0).sum()))


def nearest_rows(values, anchor, count=20):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not 0 <= anchor < len(x) or count < 1 or not np.isfinite(x).all():
        raise ValueError('SPILLOVER_NEIGHBOURS_INPUT')
    distances = np.linalg.norm(x-x[anchor], axis=1)
    ids = np.arange(len(x))
    order = np.lexsort((ids, distances))
    chosen = order[order != anchor][:count]
    return chosen, distances[chosen]
