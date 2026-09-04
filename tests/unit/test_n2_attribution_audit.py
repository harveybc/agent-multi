"""C3 unit tests (order @4c1f1532): the log-loss decomposition is
exactly additive, the sealed verdict rule is pure and total, and the
Monte Carlo p floor is reported as an inequality."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n2a", REPO / "tools" / "n2_attribution_audit.py")
n2a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n2a)


class TestDecomposition:

    def test_exactly_additive(self):
        rng = np.random.default_rng(3)
        probs = rng.dirichlet([1, 1, 1], size=50)
        y = rng.integers(0, 3, size=50)
        l_multi = n2a._logloss3(probs, y)
        l_hit, l_dir, is_hit = n2a.decompose(probs, y)
        assert np.allclose(l_multi, l_hit + l_dir, atol=1e-9)

    def test_censored_rows_have_zero_direction_loss(self):
        probs = np.array([[0.2, 0.3, 0.5]])
        y = np.array([2])
        l_hit, l_dir, is_hit = n2a.decompose(probs, y)
        assert l_dir[0] == 0.0 and not is_hit[0]
        assert l_hit[0] == pytest.approx(-np.log(0.5))


class TestVerdictRule:

    def _stats(self, **over):
        base = {}
        for t in n2a.TARGETS:
            for (a, b) in n2a.CONTRASTS:
                base[(t, a, b)] = {"pooled_skill": -0.01,
                                   "all_windows_positive": False,
                                   "holm_p": 1.0}
        for k, v in over.items():
            t, a, b = k.split("|")
            base[(t, a, b)] = v
        return base

    def _good(self, skill):
        return {"pooled_skill": skill,
                "all_windows_positive": True, "holm_p": 0.001}

    def test_scale_explains(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.022),
            "bar_h12|arm2|arm1": self._good(0.021)})
        assert n2a.decide(stats) == \
            "BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE"

    def test_incremental_structure_wins_over_scale(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.022),
            "bar_h12|arm2|arm1": self._good(0.021),
            "bar_h6|arm5|arm2": self._good(0.006)})
        assert n2a.decide(stats) == \
            "INCREMENTAL_DEVELOPMENT_STRUCTURE_OBSERVED"

    def test_weak_scale_is_inconclusive(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.022)})
        assert n2a.decide(stats) == "ATTRIBUTION_INCONCLUSIVE"

    def test_incremental_below_margin_does_not_trigger(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.022),
            "bar_h12|arm2|arm1": self._good(0.021),
            "bar_h6|arm3|arm2": self._good(0.004)})
        assert n2a.decide(stats) == \
            "BARRIER_SIGNAL_EXPLAINED_BY_TARGET_DEFINITION_SCALE"


class TestPFloor:

    def test_floor_reported_as_inequality(self):
        assert n2a._fmt_p(1.0 / 2001) == "<= 1/2001"
        assert n2a._fmt_p(0.01) == 0.01
