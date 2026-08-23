from __future__ import annotations


class Plugin:
    plugin_params = {"common_dim": 64, "output_dim": 128, "dropout": 0.0}

    @staticmethod
    def build(branch_dims: list[int], config):
        import torch
        import torch.nn as nn

        common = int(config["common_dim"])
        output = int(config["output_dim"])

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.projections = nn.ModuleList(nn.Linear(dim, common) for dim in branch_dims)
                self.gates = nn.Linear(common * len(branch_dims), len(branch_dims))
                self.output = nn.Sequential(nn.Dropout(float(config["dropout"])), nn.Linear(common, output), nn.LayerNorm(output))

            def forward(self, values):
                projected = [layer(value) for layer, value in zip(self.projections, values)]
                weights = torch.softmax(self.gates(torch.cat(projected, dim=-1)), dim=-1)
                mixed = sum(value * weights[:, index : index + 1] for index, value in enumerate(projected))
                return self.output(mixed)

        return Fusion(), output

