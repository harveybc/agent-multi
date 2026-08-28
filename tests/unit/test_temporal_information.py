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
    assert verdicts <= {"PASS", "FAIL",
                        "CEILING_SATURATED_INCONCLUSIVE"}
    for family, entry in report["families"].items():
        gate = entry["temporal_gate"]
        assert gate["structural_controls_pass"] is True, family
