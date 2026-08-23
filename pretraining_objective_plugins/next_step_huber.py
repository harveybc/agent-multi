from __future__ import annotations


class Plugin:
    plugin_params = {"delta": 1.0}

    @staticmethod
    def output_dim(input_channels: int) -> int:
        return input_channels

    @staticmethod
    def target(batch, config):
        return batch["next_features"]

    @staticmethod
    def loss(prediction, target, config):
        import torch.nn.functional as functional

        return functional.huber_loss(prediction, target, delta=float(config["delta"]))
