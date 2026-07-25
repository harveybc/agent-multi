from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from project3_evidence_metrics import canonical_trading_metrics  # noqa: E402


def test_canonical_metrics_have_explicit_weekly_and_annual_horizons() -> None:
    timestamps = pd.date_range("2023-01-01", periods=365, freq="D", tz="UTC")
    equity = np.linspace(1.0, 1.12, len(timestamps))

    result = canonical_trading_metrics(
        timestamps=timestamps,
        equity=equity,
        risk_penalty_lambda=0.5,
    )

    by_name = {row["metric_name"]: row for row in result["metrics"]}
    assert by_name["mean_weekly_return"]["horizon"] == "week"
    assert by_name["mean_weekly_return"]["unit"] == "fraction"
    assert by_name["annualized_return"]["horizon"] == "year"
    assert by_name["mean_weekly_rap"]["horizon"] == "week"
    assert by_name["annual_rap"]["horizon"] == "year"
    assert result["canonical"]["annualized_return"] == pytest.approx(0.12, abs=0.002)


def test_canonical_metrics_reject_nonpositive_initial_equity() -> None:
    with pytest.raises(ValueError, match="initial equity"):
        canonical_trading_metrics(
            timestamps=pd.date_range("2023-01-01", periods=2, freq="D", tz="UTC"),
            equity=[0.0, 1.0],
        )
