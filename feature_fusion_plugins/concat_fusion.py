from __future__ import annotations


class Plugin:
    plugin_params = {"output_dim": 128, "hidden_dim": 256, "dropout": 0.0}

    @staticmethod
    def build(branch_dims: list[int], config):
        import torch
        import torch.nn as nn

        input_dim = sum(branch_dims)
        output_dim = int(config["output_dim"])

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, int(config["hidden_dim"])),
                    nn.GELU(),
                    nn.Dropout(float(config["dropout"])),
                    nn.Linear(int(config["hidden_dim"]), output_dim),
                    nn.LayerNorm(output_dim),
                )

            def forward(self, values):
                return self.network(torch.cat(values, dim=-1))

        return Fusion(), output_dim

