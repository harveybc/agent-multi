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
        "family_ids": [],
    }

    @staticmethod
    def build(branch_dims: list[int], config: dict[str, Any]):
        import torch
        import torch.nn as nn

        from feature_branch_plugins._topology import (
            require_dropout, require_heads_divide, require_positive_int)
        d_model, n_heads = require_heads_divide(config, "d_model",
                                                "n_heads")
        out_dim = require_positive_int(config, "output_dim")
        require_dropout(config)
        if not branch_dims or any(int(d) < 1 for d in branch_dims):
            raise ValueError("cross_family_attention needs positive "
                             "branch dims")
        family_ids = list(config.get("family_ids") or [])
        if family_ids and len(family_ids) != len(branch_dims):
            raise ValueError("family_ids must match branch count")
        expected_dims = [int(d) for d in branch_dims]

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
                # DATA-SOTA-332: exact count/rank/width — never a
                # silent zip truncation
                if len(encoded) != len(expected_dims):
                    raise ValueError(
                        f"cross_family_attention expected "
                        f"{len(expected_dims)} branches "
                        f"({family_ids or 'unnamed'}), got "
                        f"{len(encoded)}")
                for i, (e, d) in enumerate(zip(encoded, expected_dims)):
                    if e.dim() != 2 or e.shape[-1] != d:
                        name = family_ids[i] if family_ids else i
                        raise ValueError(
                            f"branch {name!r} must be (B, {d}), got "
                            f"{tuple(e.shape)}")
                tokens = torch.stack(
                    [p(e) for p, e in zip(self.proj, encoded)],
                    dim=1) + self.family_pos          # (B, N, D)
                attended, _ = self.attn(tokens, tokens, tokens,
                                        need_weights=False)
                gated = torch.sigmoid(self.gate(attended)) * attended
                fused = self.norm(tokens + gated)
                return self.out(fused.flatten(1))

        fusion = Fusion()
        fusion.family_ids = family_ids
        fusion.expected_dims = expected_dims
        return fusion, out_dim
