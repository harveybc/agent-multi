"""N3 pre-acquisition refusal battery (order @a13671ab §§4, 7):
geometry matches the sealed contract, the decision table is pure and
total, and the offline verifier refuses the ordered adversaries on a
synthetic bundle BEFORE any network request exists."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n3f", REPO / "tools" / "n3_fresh_confirmation.py")
n3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3f)

from agent_plugins.experiment_runtime import sha_obj  # noqa: E402

SEALED = json.loads(
    (REPO / "docs/audits/evidence/"
     "N3_FRESH_CONFIRMATION_CONTRACT_2026_09_04.json").read_text())


class TestGeometry:

    def test_expected_scoring_anchors_match_contract(self):
        expected = SEALED["role_ledger"][
            "expected_scoring_anchors"]
        for name, start, end, bars in n3f.BLOCKS:
            assert len(n3f.scoring_anchor_offsets(bars)) == \
                expected[name]
        assert sum(len(n3f.scoring_anchor_offsets(b[3]))
                   for b in n3f.BLOCKS) == expected["total"]

    def test_role_ledger_blocks_equal_sealed_contract(self):
        ledger = n3f.role_ledger()
        def norm(b):
            return [str(b[0]).replace("T", " "),
                    str(b[1]).replace("T", " "), b[2]]
        for k, v in SEALED["role_ledger"]["blocks_utc"].items():
            assert norm(ledger["blocks"][k]) == norm(v)

    def test_purge_keeps_labels_inside_block(self):
        for _, _, _, bars in n3f.BLOCKS:
            offs = n3f.scoring_anchor_offsets(bars)
            assert all(i + n3f.H_MAX < bars for i in offs)

    def test_confirmation_grid_total(self):
        assert sum(b[3] for b in n3f.BLOCKS) == 1458


class TestDecisionTable:

    def _stats(self, **over):
        base = {}
        for t in n3f.TARGETS:
            for (a, b) in n3f.CONTRAST_FAMILY:
                base[(t, a, b)] = {"pooled_skill": -0.01,
                                   "all_blocks_positive": False,
                                   "holm_p": 1.0}
        for key, v in over.items():
            t, a, b = key.split("|")
            base[(t, a, b)] = v
        return base

    def _good(self, skill):
        return {"pooled_skill": skill,
                "all_blocks_positive": True, "holm_p": 0.001}

    def test_insufficient_dominates(self):
        assert n3f.decide({}, False, True) == \
            "FRESH_CONFIRMATION_INSUFFICIENT"

    def test_license_failure_is_inconclusive(self):
        assert n3f.decide({}, True, False) == \
            "FRESH_CONFIRMATION_INCONCLUSIVE"

    def test_scale_confirmed_both_horizons(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.02),
            "bar_h12|arm2|arm1": self._good(0.02)})
        assert n3f.decide(stats, True, True) == \
            "TARGET_SCALE_EFFECT_CONFIRMED_NO_REPRESENTATION_SIGNAL"

    def test_scale_on_one_horizon_only_not_confirmed(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.02)})
        assert n3f.decide(stats, True, True) == \
            "TARGET_SCALE_EFFECT_NOT_CONFIRMED"

    def test_representation_precedence(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.02),
            "bar_h12|arm2|arm1": self._good(0.02),
            "bar_h12|arm5|arm2": self._good(0.006)})
        assert n3f.decide(stats, True, True) == \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"

    def test_representation_below_margin_does_not_trigger(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.02),
            "bar_h12|arm2|arm1": self._good(0.02),
            "bar_h12|arm3|arm2": self._good(0.004)})
        assert n3f.decide(stats, True, True) == \
            "TARGET_SCALE_EFFECT_CONFIRMED_NO_REPRESENTATION_SIGNAL"


def _synthetic_bundle():
    """A fully consistent synthetic bundle: scale beats prior by
    ~2%, representation arms neutral; verdict derives from the
    module's own rederivation."""
    rng = np.random.default_rng(11)
    units = []
    hist = [900, 1000, 1500]
    prior = np.array(hist) / sum(hist)
    prior_loss = -np.log(prior)
    for tkey in n3f.TARGETS:
        for name, start, end, bars in n3f.BLOCKS:
            n_s = len(n3f.scoring_anchor_offsets(bars))
            ys = rng.integers(0, 3, size=n_s)
            t0 = datetime.strptime(start, "%Y-%m-%d %H:%M")
            anchors = [(t0 + timedelta(hours=4 * i)).strftime(
                "%Y-%m-%d %H:%M:%S")
                for i in n3f.scoring_anchor_offsets(bars)][:n_s]
            l1 = np.round(prior_loss[ys], 8)
            base = {"arm1": l1,
                    "arm2": np.round(l1 * 0.98
                                     + 0.001 * rng.standard_normal(
                                         n_s), 8),
                    "arm3": np.round(l1 * 0.98
                                     + 0.002 * rng.standard_normal(
                                         n_s), 8)}
            base["arm4"] = np.round(
                l1 * 1.01 + 0.002 * rng.standard_normal(n_s), 8)
            base["arm5"] = np.round(
                base["arm2"] * 1.005
                + 0.002 * rng.standard_normal(n_s), 8)
            payload = {
                "unit": f"{tkey}:{name}", "horizon":
                    n3f.TARGETS[tkey], "block": name,
                "n_score": n_s,
                "anchor_datetimes": anchors,
                "fit_cal_label_histogram": hist,
                "class_support_score": {
                    str(c): int((ys == c).sum())
                    for c in (0, 1, 2)},
                "arms": {a: {"record": {},
                             "multiclass_logloss_mean": round(
                                 float(v.mean()), 6),
                             "hit_vs_censored_mean": 0.0,
                             "per_obs_logloss": [float(x)
                                                 for x in v]}
                         for a, v in base.items()}}
            payload["payload_sha256"] = sha_obj(payload)
            units.append(payload)
    contrasts, stats, complete = n3f._rederive(units)
    assert complete
    verdict = n3f.decide(stats, True, True)
    return {"schema": "agent_multi.n3_fresh_bundle.v1",
            "contract": n3f.CONTRACT,
            "role_ledger": n3f.role_ledger(),
            "blocks_complete": True, "licenses_ok": True,
            "units": units, "contrasts": contrasts,
            "verdict": verdict}


@pytest.fixture(scope="module")
def bundle():
    return _synthetic_bundle()


def _write(tmp_path, b):
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(b, default=float))
    return p


class TestVerifier:

    def test_consistent_bundle_verifies(self, bundle, tmp_path):
        out = n3f.verify(_write(tmp_path, bundle))
        assert out["verdict"] == "N3_BUNDLE_VERIFIED"
        assert out["units_verified"] == 8

    def test_adv1_pre2026_anchor(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["units"][0]["anchor_datetimes"][0] = \
            "2025-12-30 04:00:00"
        b["units"][0]["payload_sha256"] = sha_obj(
            {k: v for k, v in b["units"][0].items()
             if k != "payload_sha256"})
        with pytest.raises(n3f.FreshRefusal, match="pre-2026"):
            n3f.verify(_write(tmp_path, b))

    def test_adv2_moved_boundary(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["role_ledger"]["blocks"]["B4_JulAug"] = [
            "2026-07-01 00:00", "2026-09-15 20:00", 372]
        with pytest.raises(n3f.FreshRefusal, match="boundary"):
            n3f.verify(_write(tmp_path, b))

    def test_adv3_future_anchor(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        u = [x for x in b["units"]
             if x["block"] == "B4_JulAug"][0]
        u["anchor_datetimes"][-1] = "2026-09-02 00:00:00"
        u["payload_sha256"] = sha_obj(
            {k: v for k, v in u.items()
             if k != "payload_sha256"})
        with pytest.raises(n3f.FreshRefusal, match="beyond"):
            n3f.verify(_write(tmp_path, b))

    def test_adv4_altered_payload(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["units"][2]["arms"]["arm2"]["per_obs_logloss"][0] += 0.5
        with pytest.raises(n3f.FreshRefusal, match="altered"):
            n3f.verify(_write(tmp_path, b))

    def test_adv8_label_history_mismatch(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][1]
        u["fit_cal_label_histogram"] = [100, 2000, 1300]
        u["payload_sha256"] = sha_obj(
            {k: v for k, v in u.items()
             if k != "payload_sha256"})
        with pytest.raises(n3f.FreshRefusal,
                           match="label histories"):
            n3f.verify(_write(tmp_path, b))

    def test_adv9_missing_unit(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["units"] = b["units"][:-1]
        with pytest.raises(n3f.FreshRefusal,
                           match="missing/extra"):
            n3f.verify(_write(tmp_path, b))

    def test_adv9b_license_failure_beside_decision(
            self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][3]
        u["license_failure"] = "class_support"
        u["payload_sha256"] = sha_obj(
            {k: v for k, v in u.items()
             if k != "payload_sha256"})
        with pytest.raises(n3f.FreshRefusal, match="license"):
            n3f.verify(_write(tmp_path, b))

    def test_adv10_edited_verdict(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["verdict"] = \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
        with pytest.raises(n3f.FreshRefusal,
                           match="report edited"):
            n3f.verify(_write(tmp_path, b))

    def test_adv10b_edited_contrast(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        key = next(iter(b["contrasts"]))
        b["contrasts"][key]["pooled_skill"] = 0.5
        with pytest.raises(n3f.FreshRefusal,
                           match="report edited"):
            n3f.verify(_write(tmp_path, b))

    def test_duplicate_unit(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["units"].append(b["units"][0])
        with pytest.raises(n3f.FreshRefusal, match="duplicate"):
            n3f.verify(_write(tmp_path, b))
