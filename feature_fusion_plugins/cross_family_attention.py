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
        from feature_branch_plugins._topology import strict_int
        if not branch_dims:
            raise ValueError("cross_family_attention needs branches")
        expected_dims = [strict_int(d, f"branch_dims[{i}]", 1)
                         for i, d in enumerate(branch_dims)]
        # DATA-SOTA-336: family identity is REQUIRED, unique, nonempty
        family_ids = list(config.get("family_ids") or [])
        if len(family_ids) != len(expected_dims):
            raise ValueError(
                "cross_family_attention requires one family_id per "
                f"branch ({len(expected_dims)}), got {len(family_ids)}")
        if any(not isinstance(f, str) or not f.strip()
               for f in family_ids):
            raise ValueError("family_ids must be nonempty strings")
        if len(set(family_ids)) != len(family_ids):
            raise ValueError(
                f"duplicate family_ids refused: {family_ids}")

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
                # DATA-SOTA-332/336: NAMED runtime records — identity,
                # count, rank and width all refuse; a same-width swap
                # refuses BY NAME, not by numeric accident.
                if len(encoded) != len(expected_dims):
                    raise ValueError(
                        f"cross_family_attention expected "
                        f"{len(expected_dims)} named branches "
                        f"{family_ids}, got {len(encoded)}")
                tensors = []
                for i, record in enumerate(encoded):
                    if (not isinstance(record, (tuple, list))
                            or len(record) != 2):
                        raise ValueError(
                            "cross_family_attention consumes NAMED "
                            "records (family_id, tensor); positional "
                            f"input at index {i} refused")
                    name, e = record
                    if name != family_ids[i]:
                        raise ValueError(
                            f"family identity mismatch at position "
                            f"{i}: expected {family_ids[i]!r}, got "
                            f"{name!r}")
                    d = expected_dims[i]
                    if e.dim() != 2 or e.shape[-1] != d:
                        raise ValueError(
                            f"branch {name!r} must be (B, {d}), got "
                            f"{tuple(e.shape)}")
                    tensors.append(e)
                encoded = tensors
                tokens = torch.stack(
                    [p(e) for p, e in zip(self.proj, encoded)],
                    dim=1) + self.family_pos          # (B, N, D)
                attended, _ = self.attn(tokens, tokens, tokens,
                                        need_weights=False)
                gated = torch.sigmoid(self.gate(attended)) * attended
                fused = self.norm(tokens + gated)
                return self.out(fused.flatten(1))

        import hashlib
        fusion = Fusion()
        fusion.family_ids = family_ids
        fusion.expected_dims = expected_dims
        fusion.consumes_named = True
        fusion.family_digest = hashlib.sha256(
            "\n".join(family_ids).encode()).hexdigest()
        return fusion, out_dim
