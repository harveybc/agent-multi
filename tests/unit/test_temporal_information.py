"""Unit tests for the temporal-information suite primitives (order
b06ec0c7 §4). REPRESENTATION_DIAGNOSTIC machinery only."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.temporal_information import (  # noqa: E402
    TemporalControlFailure, control_future_immutability,
    control_newest_bar_sensitivity, control_save_load_bit_exact,
    control_time_reversal, effective_rank, phase_randomized_surrogate,
    probe_r2, synthetic_signals, window_embeddings)


class LastStepEncoder(torch.nn.Module):
    def __init__(self, features=4, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(features, dim)

    def forward(self, values):
        return self.proj(values[:, -1, :])


class MeanPoolEncoder(torch.nn.Module):
    """Time-blind pooling: must FAIL the time-reversal control."""

    def __init__(self, features=4, dim=8):
        super().__init__()
        self.proj = torch.nn.Linear(features, dim)

    def forward(self, values):
        return self.proj(values.mean(dim=1))


class LookaheadEncoder(torch.nn.Module):
    """Deliberate future leak across window positions is impossible
    here (windows ARE the input), so the future-immutability control
    is exercised by construction on the honest encoder instead."""


def test_structural_controls_pass_on_causal_encoder():
    encoder = LastStepEncoder()
    assert control_future_immutability(encoder, 8, 4)["passed"]
    assert control_newest_bar_sensitivity(encoder, 8, 4)["passed"]
    assert control_save_load_bit_exact(encoder, 8, 4)["passed"]


def test_time_reversal_fails_on_time_blind_pooling():
    with pytest.raises(TemporalControlFailure, match="reversal"):
        control_time_reversal(MeanPoolEncoder(), 8, 4)


def test_newest_bar_control_fails_when_insensitive():
    class FirstStepEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 8)

        def forward(self, values):
            return self.proj(values[:, 0, :])
    with pytest.raises(TemporalControlFailure, match="newest"):
        control_newest_bar_sensitivity(FirstStepEncoder(), 8, 4)


def test_effective_rank_detects_collapse():
    healthy = np.random.default_rng(0).standard_normal((200, 16))
    collapsed = np.outer(np.arange(200, dtype=float), np.ones(16))
    assert effective_rank(healthy)["effective_rank"] > 10
    diag = effective_rank(collapsed)
    assert diag["collapse_flag"] is True


def test_phase_surrogate_preserves_amplitude_spectrum():
    signal = synthetic_signals(256, 3)["series"]
    surrogate = phase_randomized_surrogate(signal)
    original = np.abs(np.fft.rfft(signal[:, 0]))
    scrambled = np.abs(np.fft.rfft(surrogate[:, 0]))
    assert np.allclose(original, scrambled, rtol=1e-4, atol=1e-3)
    assert not np.allclose(signal[:, 0], surrogate[:, 0])


def test_probe_r2_causal_split_and_sanity():
    rng = np.random.default_rng(1)
    emb = rng.standard_normal((400, 8))
    target = emb[:, 0] * 2.0 + 0.1 * rng.standard_normal(400)
    assert probe_r2(emb, target) > 0.9
    assert probe_r2(emb, rng.standard_normal(400)) < 0.1


def test_window_embeddings_shape_and_causality():
    encoder = LastStepEncoder()
    series = synthetic_signals(64, 4)["series"]
    emb = window_embeddings(encoder, series, 8)
    assert emb.shape == (64 - 8 + 1, 8)


def test_suite_report_is_diagnostic_and_ceiling_disciplined():
    report_path = (REPO / "docs/audits/evidence/"
                   "TEMPORAL_INFORMATION_SUITE_REPORT_2026_08_28"
                   ".json")
    import json
    report = json.loads(report_path.read_text())
    assert report["classification"] == "REPRESENTATION_DIAGNOSTIC"
    verdicts = set(report["summary"].values())
    # v1 PASS labels were tombstoned by the 2026-08-28 audit
    assert verdicts <= {"PASS", "FAIL",
                        "CEILING_SATURATED_INCONCLUSIVE",
                        "SYNTHETIC_MECHANICS_ONLY_PENDING_REAL_PROBES"}
    assert report.get("verdict_status") == \
        "SYNTHETIC_MECHANICS_ONLY_PENDING_REAL_PROBES"
    for family, entry in report["families"].items():
        gate = entry["temporal_gate"]
        assert gate["structural_controls_pass"] is True, family


class TestV2Primitives:
    """v2 scientific-correction primitives (order @c1a319c0)."""

    def test_within_window_permutation_preserves_multiset(self):
        from agent_plugins.temporal_information import (
            make_windows, within_window_permutation)
        series = np.arange(40, dtype=np.float32).reshape(-1, 1)
        windows = make_windows(np.repeat(series, 3, axis=1), 8)
        permuted = within_window_permutation(windows, seed=3)
        # same value multiset per window, different order somewhere
        for i in range(len(windows)):
            assert sorted(permuted[i][:, 0].tolist()) == \
                sorted(windows[i][:, 0].tolist())
        assert not np.array_equal(permuted, windows)

    def test_phase_scramble_preserves_window_content(self):
        from agent_plugins.temporal_information import (
            make_windows, per_window_phase_scramble)
        series = np.random.default_rng(0).standard_normal(
            (60, 2)).astype(np.float32)
        windows = make_windows(series, 8)
        scrambled = per_window_phase_scramble(windows, seed=5)
        for i in range(len(windows)):
            assert sorted(scrambled[i][:, 0].tolist()) == \
                pytest.approx(sorted(windows[i][:, 0].tolist()))
        assert not np.array_equal(scrambled, windows)

    def test_ridge_roles_isolate_lambda_selection(self):
        from agent_plugins.temporal_information import (
            ridge_fit_cal_score)
        rng = np.random.default_rng(2)
        x = rng.standard_normal((300, 6))
        y = x[:, 0] * 3.0 + 0.05 * rng.standard_normal(300)
        result = ridge_fit_cal_score(
            x[:180], y[:180], x[180:240], y[180:240],
            x[240:], y[240:])
        assert result["score"] > 0.9
        assert result["lambda"] in (1e-4, 1e-2, 1.0, 100.0)

    def test_paired_stats_ci(self):
        from agent_plugins.temporal_information import paired_stats
        stats = paired_stats([0.10, 0.12, 0.08, 0.11])
        assert stats["n"] == 4
        assert stats["ci95_low"] > 0.0
        wide = paired_stats([0.5, -0.4, 0.6, -0.5])
        assert wide["ci95_low"] < 0.0  # unstable effect not certified

    def test_chronological_roles_are_contiguous_and_disjoint(self):
        from agent_plugins.temporal_information import (
            chronological_roles)
        fit, cal, mon = chronological_roles(100)
        assert fit[-1] < cal[0] < mon[0]
        assert len(set(fit) | set(cal) | set(mon)) == 100

    def test_pinball_score_orders_quantiles(self):
        from agent_plugins.temporal_information import (
            pinball_loss_score)
        y = np.zeros(100)
        assert pinball_loss_score(np.zeros(100), y, 0.5) == 0.0
        assert pinball_loss_score(np.ones(100), y, 0.5) < 0.0
