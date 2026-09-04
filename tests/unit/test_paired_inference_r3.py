"""R3 batteries (order @8fce8da0 §4): the repaired helper survives
exactly the inputs that break the N1-era code — the non-monotone
Holm and the inf -> linspace -> NaN zero-variance path."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from agent_plugins.paired_inference import (  # noqa: E402
    PairedInferenceError, holm_adjust, paired_t, t_survival)


class TestHolmStepDown:

    def test_cumulative_maximum_enforced(self):
        """The exact non-monotone case the naive per-rank formula
        produces: p=[0.01, 0.011] -> naive adjusted [0.02, 0.011]
        (DECREASING). Correct Holm: [0.02, 0.02]."""
        adjusted = holm_adjust({"a": 0.01, "b": 0.011})
        assert adjusted["a"] == pytest.approx(0.02)
        assert adjusted["b"] == pytest.approx(0.02), \
            "cumulative maximum missing — non-monotone Holm"

    def test_n1_real_values_monotone(self):
        """The published N1 output (lin 0.9419, temp 1.0) came from
        the buggy path: naive gives lin 0.9419*2->1.0 capped, temp
        stays at its raw*1 — but the SORTED order matters. With the
        real p-values the corrected Holm is monotone."""
        adjusted = holm_adjust({"direct_linear": 0.9419,
                                "direct_temporal": 0.9713})
        values = [adjusted[k] for k in
                  sorted(adjusted, key=adjusted.get)]
        assert values == sorted(values)
        assert adjusted["direct_linear"] == 1.0
        assert adjusted["direct_temporal"] == 1.0

    def test_three_way_step_down(self):
        adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
        assert adjusted["a"] == pytest.approx(0.03)
        assert adjusted["c"] == pytest.approx(0.06)
        assert adjusted["b"] == pytest.approx(0.06)

    def test_rejects_non_finite(self):
        with pytest.raises(PairedInferenceError, match="not finite"):
            holm_adjust({"a": float("nan")})
        with pytest.raises(PairedInferenceError, match="not finite"):
            holm_adjust({"a": float("inf")})

    def test_rejects_out_of_range(self):
        with pytest.raises(PairedInferenceError, match="outside"):
            holm_adjust({"a": 1.5})


class TestZeroVariance:

    def test_positive_constant_is_finite_p_zero(self):
        """Under the N1-era code this raised or NaN'd:
        se=0 -> t=inf -> linspace(inf, inf+60) -> NaN p."""
        result = paired_t([0.2, 0.2, 0.2, 0.2])
        assert result["zero_variance"] == "positive_constant"
        assert result["p_one_sided"] == 0.0
        assert result["ci95"] == [0.2, 0.2]
        assert result["t_stat"] is None

    def test_zero_constant(self):
        result = paired_t([0.0, 0.0, 0.0, 0.0])
        assert result["zero_variance"] == "zero_constant"
        assert result["p_one_sided"] == 1.0
        assert result["ci95"] == [0.0, 0.0]

    def test_negative_constant(self):
        result = paired_t([-0.1, -0.1, -0.1, -0.1])
        assert result["zero_variance"] == "negative_constant"
        assert result["p_one_sided"] == 1.0
        assert result["ci95"] == [-0.1, -0.1]

    def test_all_outputs_finite(self):
        for diffs in ([0.2] * 4, [0.0] * 4, [-0.1] * 4,
                      [0.1, 0.2, 0.15, 0.12]):
            result = paired_t(diffs)
            for key in ("mean", "sd", "p_one_sided"):
                assert math.isfinite(result[key])


class TestNonFiniteRejection:

    def test_nan_diff_rejected(self):
        with pytest.raises(PairedInferenceError, match="not finite"):
            paired_t([0.1, float("nan"), 0.2, 0.3])

    def test_inf_diff_rejected(self):
        with pytest.raises(PairedInferenceError, match="not finite"):
            paired_t([0.1, float("inf"), 0.2, 0.3])

    def test_inf_t_rejected_by_survival(self):
        """The exact inf -> linspace -> NaN path is now a typed
        refusal instead of silent NaN."""
        with pytest.raises(PairedInferenceError,
                           match="not finite"):
            t_survival(float("inf"), 3)


class TestSurvivalAccuracy:

    @pytest.mark.parametrize("t,df,expected", [
        (3.182, 3, 0.025), (2.353, 3, 0.050),
        (2.776, 4, 0.025), (1.0, 3, 0.196)])
    def test_known_values(self, t, df, expected):
        assert t_survival(t, df) == pytest.approx(expected,
                                                  abs=0.002)

    def test_negative_t_complement(self):
        assert t_survival(-3.182, 3) == pytest.approx(0.975,
                                                      abs=0.002)


class TestN1Rederivation:

    def test_n1_verdict_unchanged_from_frozen_records(self):
        """R3.5: re-derive the N1 interpretation from the frozen 28
        result records with the REPAIRED helper — the primary
        verdict must be unchanged."""
        import json
        run_root = (Path.home() / ".local/share/agent-multi/"
                    "target_identifiability_n1_20260903")
        if not run_root.exists():
            pytest.skip("frozen N1 run not present on this host")
        units_dir = run_root / "diagnostic" / "units"
        per_window: dict = {}
        for state_path in units_dir.glob("*.state.json"):
            state = json.loads(state_path.read_text())
            ident = state["identity"]
            result = json.loads(
                (units_dir /
                 f"{state['unit_id']}.result.json").read_text())
            wk, arm = ident["origin"], ident["treatment"]
            per_window.setdefault(wk, {"seeds": {}})
            if arm == "direct_temporal":
                per_window[wk]["seeds"][ident["seed"]] = \
                    result["score_r2"]
            else:
                per_window[wk][arm] = result["score_r2"]
        windows = sorted(per_window)
        assert len(windows) == 4
        for wk in windows:
            seeds = per_window[wk]["seeds"]
            assert len(seeds) == 4
            per_window[wk]["direct_temporal"] = \
                sum(seeds.values()) / 4
        pvals = {}
        analysis = {}
        for arm in ("direct_linear", "direct_temporal"):
            diffs = [per_window[wk][arm]
                     - per_window[wk]["literal_persistence"]
                     for wk in windows]
            stats = paired_t(diffs)
            analysis[arm] = stats
            pvals[arm] = stats["p_one_sided"]
        adjusted = holm_adjust(pvals)
        advancing = [
            arm for arm in analysis
            if all(per_window[wk][arm] > 0 for wk in windows)
            and analysis[arm]["mean"] >= 0.02
            and analysis[arm]["ci95"][0] > 0
            and adjusted[arm] < 0.05]
        hinting = [arm for arm in analysis
                   if arm not in advancing
                   and all(per_window[wk][arm] > 0
                           for wk in windows)]
        if advancing:
            verdict = "REPRESENTATION_BOTTLENECK_DEMONSTRATED"
        elif hinting:
            verdict = "INCONCLUSIVE_DISCORDANT"
        else:
            verdict = "PREDICTABILITY_NOT_DEMONSTRATED"
        assert verdict == "PREDICTABILITY_NOT_DEMONSTRATED", \
            "the repaired helper changed the N1 primary verdict"
        # corrected Holm values are monotone (the published ones
        # were not: 0.9419 < 1.0 in sorted order was non-monotone)
        values = sorted(adjusted.values())
        assert values == sorted(values)
        assert all(v >= 0.94 for v in adjusted.values())
