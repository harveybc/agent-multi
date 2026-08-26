"""TimesNet-style branch (Data-First order §3): rFFT top-k period
selection over the (fully historical) window, period-folded 2D
convolution, amplitude-weighted aggregation. All inputs are <= t-1 by
the window contract, so intra-window spectral analysis is causal.
Contract: (B, T, F) -> (B, dim)."""
from __future__ import annotations

from typing import Any


class Plugin:
    plugin_params = {
        "top_k": 2,
        "d_model": 64,
        "kernel": 3,
        "dropout": 0.0,
    }

    @staticmethod
    def build(input_channels: int, window_size: int, config: dict[str, Any]):
        import torch
        import torch.nn as nn

        from feature_branch_plugins._topology import (
            require_dropout, require_odd_kernel, require_positive_int,
            require_spectral_viability, require_window)
        top_k = require_positive_int(config, "top_k")
        d_model = require_positive_int(config, "d_model")
        kernel = require_odd_kernel(config)
        require_dropout(config)
        require_window(window_size, 4, "timesnet_branch")
        require_spectral_viability(window_size, top_k)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Linear(input_channels, d_model)
                self.conv = nn.Sequential(
                    nn.Conv2d(d_model, d_model, kernel,
                              padding=kernel // 2),
                    nn.GELU(),
                    nn.Dropout(float(config["dropout"])),
                    nn.Conv2d(d_model, d_model, kernel,
                              padding=kernel // 2))
                self.norm = nn.LayerNorm(d_model)

            def forward(self, values):  # (B, T, F)
                x = self.embed(values)                 # (B, T, D)
                b, t, d = x.shape
                spec = torch.fft.rfft(x, dim=1).abs().mean(-1)  # (B,Fq)
                spec[:, 0] = 0.0
                k = min(top_k, spec.shape[1] - 1)
                amps, freqs = torch.topk(spec, k, dim=1)  # (B,k)
                result = torch.zeros(b, d, device=x.device,
                                     dtype=x.dtype)
                weight = torch.softmax(amps, dim=1)
                for i in range(k):
                    periods = torch.clamp(
                        t // torch.clamp(freqs[:, i], min=1), 1, t)
                    for sample in range(b):
                        p = int(periods[sample])
                        cycles = t // p
                        if cycles < 1:
                            cycles, p = 1, t
                        seg = x[sample, t - cycles * p:, :]
                        folded = seg.reshape(cycles, p, d)
                        img = folded.permute(2, 0, 1).unsqueeze(0)
                        conv = self.conv(img)
                        result[sample] += weight[sample, i] * \
                            conv[0, :, -1, -1]
                return self.norm(result)

        return Encoder(), d_model
