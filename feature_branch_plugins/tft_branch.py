"""TFT-style branch (Data-First order §3): per-timestep VARIABLE
SELECTION (GRN-gated softmax over features) + GRU temporal core +
causal interpretable temporal attention. Compact but faithful to the
TFT mechanism family. Contract: (B, T, F) -> (B, dim)."""
from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "hidden": 64,
        "n_heads": 4,
        "dropout": 0.0,
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch
        import torch.nn as nn

        hidden = int(config["hidden"])
        n_heads = int(config["n_heads"])
        drop = float(config["dropout"])
        if hidden % n_heads:
            raise ValueError("tft_branch hidden must divide n_heads")

        class GRN(nn.Module):
            def __init__(self, inp, out):
                super().__init__()
                self.fc1 = nn.Linear(inp, out)
                self.fc2 = nn.Linear(out, out)
                self.gate = nn.Linear(out, out)
                self.skip = (nn.Identity() if inp == out
                             else nn.Linear(inp, out))
                self.norm = nn.LayerNorm(out)
                self.drop = nn.Dropout(drop)

            def forward(self, x):
                h = torch.nn.functional.elu(self.fc1(x))
                h = self.drop(self.fc2(h))
                g = torch.sigmoid(self.gate(h))
                return self.norm(self.skip(x) + g * h)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.var_grn = GRN(input_channels, input_channels)
                self.feat_embed = nn.Linear(1, hidden)
                self.post_select = GRN(hidden, hidden)
                self.gru = nn.GRU(hidden, hidden, batch_first=True)
                self.attn = nn.MultiheadAttention(
                    hidden, n_heads, dropout=drop, batch_first=True)
                self.out_grn = GRN(hidden, hidden)
                mask = torch.triu(
                    torch.full((window_size, window_size),
                               float("-inf")), diagonal=1)
                self.register_buffer("causal_mask", mask)

            def forward(self, values):  # (B, T, F)
                weights = torch.softmax(
                    self.var_grn(values), dim=-1)      # (B,T,F)
                embedded = self.feat_embed(
                    values.unsqueeze(-1))              # (B,T,F,H)
                selected = (weights.unsqueeze(-1) * embedded).sum(2)
                selected = self.post_select(selected)  # (B,T,H)
                core, _ = self.gru(selected)
                attended, _ = self.attn(core, core, core,
                                        attn_mask=self.causal_mask,
                                        need_weights=False)
                return self.out_grn(core + attended)[:, -1, :]

        return Encoder(), hidden
