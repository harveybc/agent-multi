"""Fail-closed validation for feature-aware policy observations."""
from __future__ import annotations

from typing import Any, Dict


def validate_observation_contract(config: Dict[str, Any]) -> None:
    """Reject enriched-data runs that silently fall back to raw prices.

    The guard is opt-in so legacy experiments remain reproducible. Phase 1
    asset-policy configs enable it because their declared data profiles are
    only meaningful when the engineered columns reach the policy.
    """
    if not bool(config.get("require_feature_aware_preprocessor", False)):
        return

    plugin = str(config.get("preprocessor_plugin") or "").strip()
    if plugin != "feature_window_preprocessor":
        raise ValueError(
            "feature-aware observation contract requires "
            "preprocessor_plugin='feature_window_preprocessor'; "
            f"got {plugin!r}"
        )

    columns = config.get("feature_columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError(
            "feature-aware observation contract requires a non-empty, "
            "materialized feature_columns list"
        )
    if len(columns) != len(set(map(str, columns))):
        raise ValueError("feature_columns contains duplicates")

    scaling = str(config.get("feature_scaling") or "").strip().lower()
    if scaling not in {"rolling_zscore", "expanding_zscore"}:
        raise ValueError(
            "feature-aware observation contract requires causal z-score "
            f"scaling; got {scaling!r}"
        )
    if bool(config.get("include_price_window", True)):
        raise ValueError(
            "feature-aware observation contract forbids the unscaled raw "
            "price window; set include_price_window=false"
        )
