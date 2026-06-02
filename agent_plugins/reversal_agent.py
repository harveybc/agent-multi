"""Mean-reversion alias around the deterministic momentum baseline."""
from __future__ import annotations

from typing import Any, Dict

from .momentum_agent import Plugin as _MomentumPlugin


class Plugin(_MomentumPlugin):
    plugin_params: Dict[str, Any] = {
        **_MomentumPlugin.plugin_params,
        "momentum_reversal": True,
    }

    def save(self, model, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("reversal_agent\n")
