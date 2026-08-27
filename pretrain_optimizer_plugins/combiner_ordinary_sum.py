"""Control combiner: the weighted per-objective encoder gradients are
summed unchanged."""
from __future__ import annotations


class Plugin:
    plugin_params = {}

    @staticmethod
    def combine(weighted_gradients, params):
        import torch

        names = sorted(weighted_gradients)
        combined = torch.zeros_like(weighted_gradients[names[0]])
        for name in names:
            combined = combined + weighted_gradients[name]
        return combined, {"combiner": "ordinary_sum",
                          "projections": 0}
