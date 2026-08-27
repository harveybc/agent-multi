"""M1 treatment balancing: per-objective encoder-gradient scales
derived ONCE from the CALIBRATION partition before epoch 0, frozen,
provenance persisted. The monitor structurally cannot influence them —
the inputs are calibration gradient norms computed before training."""
from __future__ import annotations


class Plugin:
    plugin_params = {"floor": 1e-8}

    @staticmethod
    def compute(*, declared_weights, initial_calibration_losses,
                calibration_gradient_norms, params):
        floor = float(params["floor"])
        weights = {name: declared_weights[name]
                   / max(float(calibration_gradient_norms[name]), floor)
                   for name in declared_weights}
        provenance = {"method": "frozen_gradient_norm",
                      "calibration_encoder_gradient_norms": {
                          k: round(float(v), 8) for k, v in
                          calibration_gradient_norms.items()},
                      "floor": floor,
                      "formula": "declared / max(calibration encoder "
                                 "grad norm, floor)",
                      "source": "calibration partition only, computed "
                                "BEFORE epoch 0, frozen"}
        return weights, provenance
