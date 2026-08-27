"""M1 treatment combiner: PCGrad-style conflict projection over the
ENCODER gradients only (heads receive solely their own objective
gradients — separation enforced by the runner's two-optimizer split).

Deterministic: objectives are processed in SORTED-NAME order (declared);
for each objective k, its weighted gradient is projected against every
OTHER objective j in that order: when g_k . g_j < 0,
g_k <- g_k - (g_k.g_j / (|g_j|^2 + epsilon)) g_j. A gradient with norm
below epsilon is skipped (declared zero-gradient behavior). Pre/post
pairwise dot products and norms are persisted per call.
"""
from __future__ import annotations


class Plugin:
    plugin_params = {"epsilon": 1e-12,
                     "order": "sorted_objective_names"}

    @staticmethod
    def combine(weighted_gradients, params):
        import torch

        epsilon = float(params["epsilon"])
        names = sorted(weighted_gradients)  # declared deterministic
        pre_dots = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pre_dots[f"{a}|{b}"] = float(
                    weighted_gradients[a] @ weighted_gradients[b])
        projected = {name: weighted_gradients[name].clone()
                     for name in names}
        projections = 0
        for k in names:
            for j in names:
                if j == k:
                    continue
                g_j = weighted_gradients[j]
                denominator = float(g_j @ g_j)
                if denominator < epsilon:
                    continue  # declared zero-gradient behavior: skip
                dot = float(projected[k] @ g_j)
                if dot < 0:
                    projected[k] = projected[k] - (
                        dot / (denominator + epsilon)) * g_j
                    projections += 1
        combined = torch.zeros_like(projected[names[0]])
        for name in names:
            combined = combined + projected[name]
        post_dots = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                post_dots[f"{a}|{b}"] = float(
                    projected[a] @ projected[b])
        report = {"combiner": "pcgrad",
                  "projections": projections,
                  "pre_dot_mean": round(sum(pre_dots.values())
                                        / len(pre_dots), 8)
                  if pre_dots else 0.0,
                  "post_dot_mean": round(sum(post_dots.values())
                                         / len(post_dots), 8)
                  if post_dots else 0.0,
                  "pre_negative_pairs": sum(1 for v in pre_dots.values()
                                            if v < 0),
                  "post_negative_pairs": sum(
                      1 for v in post_dots.values() if v < 0)}
        return combined, report
