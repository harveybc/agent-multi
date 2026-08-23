"""Versioned semantic grouping for the current ETH technical/statistical set."""
from __future__ import annotations

from collections.abc import Iterable


FAMILY_PATTERNS = {
    "returns_momentum": (
        "return_", "log_return_", "roc_", "mom_", "statistical__log_return_",
    ),
    "trend_level": (
        "sma_", "ema_", "close_sma_ratio_", "macd", "trend_", "ema_cross_", "zscore_close_",
    ),
    "oscillators": ("rsi_", "stoch_", "williams_", "cci_", "bb_pct_b", "mfi_"),
    "volatility_distribution": (
        "bb_upper", "bb_middle", "bb_lower", "bb_width", "atr_", "natr_", "hist_vol_",
        "roll_", "realized_var_", "autocorr_", "sqret_", "vol_regime_", "hurst_",
    ),
    "volume_flow": ("obv", "volume_", "vwap_"),
}


def semantic_feature_families(columns: Iterable[str]) -> dict[str, list[str]]:
    """Assign every column once; ambiguity and unknowns are contract failures."""
    grouped = {name: [] for name in FAMILY_PATTERNS}
    for column in columns:
        matches = [
            family
            for family, prefixes in FAMILY_PATTERNS.items()
            if any(column.startswith(prefix) for prefix in prefixes)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"feature {column!r} maps to {len(matches)} semantic families: {matches}"
            )
        grouped[matches[0]].append(column)
    return {name: values for name, values in grouped.items() if values}


DEFAULT_BRANCH_PLUGIN = {
    "returns_momentum": "tcn_branch",
    "trend_level": "transformer_branch",
    "oscillators": "gru_branch",
    "volatility_distribution": "tcn_branch",
    "volume_flow": "gru_branch",
}


def baseline_grouped_architecture(columns: Iterable[str]) -> dict:
    """Create the declared baseline; this is a search starting point, not a winner."""
    columns = list(columns)
    groups = semantic_feature_families(columns)
    branches = []
    for name, features in groups.items():
        plugin = DEFAULT_BRANCH_PLUGIN[name]
        params = {
            "tcn_branch": {"channels": [64, 64]},
            "transformer_branch": {
                "model_dim": 64, "num_heads": 4, "num_layers": 2,
            },
            "gru_branch": {"hidden_size": 64, "num_layers": 1},
        }[plugin]
        branches.append({
            "name": name,
            "features": features,
            "plugin": plugin,
            "params": params,
        })
    return {
        "schema": "agent_multi.grouped_features.v1",
        "feature_columns": columns,
        "branches": branches,
        "state_keys": [
            "position", "equity_norm", "unrealized_pnl_norm", "holding_duration_norm",
        ],
        "state_branch": {
            "plugin": "mlp_branch",
            "params": {"hidden_dims": [32], "output_dim": 32},
        },
        "fusion": {
            "plugin": "gated_fusion",
            "params": {"common_dim": 64, "output_dim": 128},
        },
        "share_features_extractor": False,
    }

