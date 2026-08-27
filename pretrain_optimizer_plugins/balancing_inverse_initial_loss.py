"""Control balancing (existing lineage): effective weight = declared /
max(initial CALIBRATION loss, floor). The monitor never participates."""
from __future__ import annotations


class Plugin:
    plugin_params = {"floor": 1e-6}

    @staticmethod
    def compute(*, declared_weights, initial_calibration_losses,
                calibration_gradient_norms, params):
        floor = float(params["floor"])
        weights = {name: declared_weights[name]
                   / max(float(initial_calibration_losses[name]), floor)
                   for name in declared_weights}
        provenance = {"method": "inverse_initial_loss",
                      "initial_calibration_losses": {
                          k: round(float(v), 8) for k, v in
                          initial_calibration_losses.items()},
                      "floor": floor,
                      "formula": "declared / max(initial_loss, floor)",
                      "source": "calibration partition only"}
        return weights, provenance
