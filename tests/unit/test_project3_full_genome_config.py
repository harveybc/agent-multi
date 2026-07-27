from __future__ import annotations

from tools.project3_full_genome_config import _feature_group, _mixed_genome


def test_feature_group_classification_is_explicit():
    assert _feature_group("log_return_20") == "returns_momentum"
    assert _feature_group("ema_100") == "trend"
    assert _feature_group("rsi_14") == "oscillators"
    assert _feature_group("hist_vol_20") == "volatility"
    assert _feature_group("wavelet_energy_L4") == "wavelet"
    assert _feature_group("ht_phase") == "hilbert"
    assert _feature_group("learned_lstm_12") == "learned"
    assert _feature_group("sota_pair_btcusdt_ethusdt_corr") == "cross_asset"
    assert _feature_group("sota_funding_rate") == "derivatives"
    assert _feature_group("event_cpi_week_flag") == "economic_events"
    assert _feature_group("external__macro_fred__DFF") == "macro"


def test_context_hours_decode_to_timeframe_specific_native_windows():
    genes, patches = _mixed_genome(
        timeframe_hours=4,
        feature_groups={"trend": ["ema_20"]},
    )
    context = next(item for item in genes if item["name"] == "context_hours")

    assert context["choices"] == [24, 72, 168, 336, 720]
    assert patches["168"]["window_size"] == 42
    assert any(item["name"] == "feature_group__trend" for item in genes)
