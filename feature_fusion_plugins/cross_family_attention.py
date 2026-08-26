"""Cross-family attention fusion (Data-First order §3): each branch
embedding becomes a FAMILY TOKEN; multi-head self-attention lets
families condition on each other; gated residual + pooled projection.
Contract: build(branch_dims, params) -> (module, features_dim);
module consumes a list of (B, D_i) tensors."""
from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "d_model": 64,
        "n_heads": 4,
        "dropout": 0.0,
        "output_dim": 128,
    }

    @staticmethod
    def build(branch_dims: list[int], config: dict[str, Any]):
        import torch
        import torch.nn as nn

        d_model = int(config["d_model"])
        n_heads = int(config["n_heads"])
        out_dim = int(config["output_dim"])
        if d_model % n_heads:
            raise ValueError("cross_family_attention d_model % n_heads")
        if not branch_dims:
            raise ValueError("cross_family_attention needs branches")

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.ModuleList(
                    [nn.Linear(d, d_model) for d in branch_dims])
                self.family_pos = nn.Parameter(
                    torch.zeros(1, len(branch_dims), d_model))
                self.attn = nn.MultiheadAttention(
                    d_model, n_heads, dropout=float(config["dropout"]),
                    batch_first=True)
                self.gate = nn.Linear(d_model, d_model)
                self.norm = nn.LayerNorm(d_model)
                self.out = nn.Linear(len(branch_dims) * d_model,
                                     out_dim)

            def forward(self, encoded):
                tokens = torch.stack(
                    [p(e) for p, e in zip(self.proj, encoded)],
                    dim=1) + self.family_pos          # (B, N, D)
                attended, _ = self.attn(tokens, tokens, tokens,
                                        need_weights=False)
                gated = torch.sigmoid(self.gate(attended)) * attended
                fused = self.norm(tokens + gated)
                return self.out(fused.flatten(1))

        return Fusion(), out_dim
