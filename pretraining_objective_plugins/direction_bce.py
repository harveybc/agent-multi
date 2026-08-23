from __future__ import annotations


class Plugin:
    plugin_params = {"threshold": 0.0}

    @staticmethod
    def output_dim(input_channels: int) -> int:
        return input_channels

    @staticmethod
    def target(batch, config):
        return (batch["next_features"] > float(config["threshold"])).float()

    @staticmethod
    def loss(prediction, target, config):
        import torch.nn.functional as functional

        return functional.binary_cross_entropy_with_logits(prediction, target)
