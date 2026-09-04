"""N4 acceptance battery v2 (orders @13fdf18c + @9fd016b0 §11):
contract-derived licensing, the complete 14-slot family, truthful
terminal-return and volatility-history semantics, design-to-execution
binding, and the pure adjudicator."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4a)

CENSUS_V2 = (REPO / "docs/audits/evidence/"
             "N4_TARGET_CENSUS_V2_2026_09_04.json")
RESULT_V2 = (REPO / "docs/audits/evidence/"
             "N4_SCREEN_RESULT_V2_2026_09_04.json")
DESIGN_V2 = (REPO / "docs/audits/evidence/"
             "N4_TARGET_AUDIT_DESIGN_V2_2026_09_04.json")


@pytest.fixture(scope="module")
def result_v2():
    return json.loads(RESULT_V2.read_text())


@pytest.fixture(scope="module")
def records(result_v2):
    return json.loads(json.dumps(
        result_v2["per_window_records"]))


class TestLicensing:

    def test_1_ternaries_unlicensed_for_class2(self, result_v2):
        for ck in ("tm_h6", "tm_h12", "tm_h24"):
            e = result_v2["per_candidate"][ck]
            assert e["outcome"] == "UNLICENSED"
            assert any("class 2" in r
                       for r in e["license_reasons"])

    def test_2_binary_requires_both_classes(self, result_v2):
        for ck in ("lm_h6", "lm_h12"):
            e = result_v2["per_candidate"][ck]
            assert e["outcome"] == "UNLICENSED"
            assert any("class 1" in r
                       for r in e["license_reasons"])

    def test_3_valid_ternary_fixture_requires_all_three(
            self, records):
        recs = copy.deepcopy(records)
        rng = np.random.default_rng(3)
        for wk, rec in recs["tm_h6"].items():
            rec["class_support_score"] = {"0": 80, "1": 80,
                                          "2": 53}
        out = n4a.adjudicate(recs)
        assert out["per_candidate"]["tm_h6"]["outcome"] != \
            "UNLICENSED"
        # and dropping class 0 support re-unlicenses it
        recs["tm_h6"]["w2"]["class_support_score"]["0"] = 5
        out2 = n4a.adjudicate(recs)
        assert out2["per_candidate"]["tm_h6"]["outcome"] == \
            "UNLICENSED"
        assert any("class 0" in r for r in
                   out2["per_candidate"]["tm_h6"][
                       "license_reasons"])


class TestFamilyOf14:

    def test_4_family_is_exactly_14_ordered(self, result_v2):
        assert result_v2["family_cardinality_proven"] == 14
        slots = [f["slot"] for f in result_v2["family"]]
        assert slots == list(n4a.FAMILY_SLOTS)
        assert len(slots) == 14

    def test_5_no_unlicensed_slot_can_pass(self, result_v2,
                                           records):
        for f in result_v2["family"]:
            if f["status"] == "UNLICENSED_PLACEHOLDER":
                assert f["raw_p"] == 1.0
                assert f["passes"] is False
        # even absurdly good losses cannot make an unlicensed
        # candidate pass: supports gate first
        recs = copy.deepcopy(records)
        for wk, rec in recs["tm_h6"].items():
            rec["losses"]["volatility_history"] = [
                v * 0.5 for v in rec["losses"]["prior"]]
        out = n4a.adjudicate(recs)
        assert "tm_h6:volatility_history" not in out["passers"]


class TestTruthfulSemantics:

    def test_6_terminal_differs_from_intrahorizon_touch(self):
        closes = np.array([100.0] + [100.0] * 5 + [100.05])
        plane = {"anchors": np.array([0]), "closes": closes,
                 "highs": np.array([100., 105., 100, 100, 100,
                                    100, 100.05]),
                 "lows": closes * 0.999}
        t = n4a.build_targets(plane)
        # +5% intrahorizon touch, terminal +0.05% -> class 2
        assert int(t["tm_h6"][0]) == 2

    def test_7_arm_name_is_volatility_history(self, result_v2):
        for ck, per_w in result_v2["per_window_records"].items():
            for rec in per_w.values():
                if rec.get("losses"):
                    assert set(rec["losses"]) == {
                        "prior", "volatility_history",
                        "causal_linear"}
        src = (REPO / "tools/n4_target_audit.py").read_text()
        assert "tradeable_move" not in src
        assert src.count('"target_history"') == 1  # legacy alias
        assert "LEGACY_ARM_ALIAS" in src

    def test_7b_alias_declared_legacy_in_artifact(self, result_v2):
        assert result_v2["supersession"][
            "legacy_arm_alias_applied"] == {
            "target_history": "volatility_history"}


class TestDesignBinding:

    def test_8_each_binding_mutation_refused(self, tmp_path,
                                             monkeypatch):
        design = json.loads(DESIGN_V2.read_text())
        cases = [
            ("margin", 0.5),
            ("round_trip_cost", 0.5),
            ("boot_seed", 999),
            ("family_slots_ordered",
             design["executable_binding"][
                 "family_slots_ordered"][:13]),
            ("fitted_arms", ["target_history", "causal_linear"]),
            ("verdict_labels", ["YES", "NO"]),
        ]
        for field, bad in cases:
            mutated = json.loads(json.dumps(design))
            mutated["executable_binding"][field] = bad
            p = tmp_path / f"design_{field}.json"
            p.write_text(json.dumps(mutated))
            monkeypatch.setattr(n4a, "DESIGN_V2", str(p))
            sha = hashlib.sha256(p.read_bytes()).hexdigest()
            with pytest.raises(n4a.N4Refusal,
                               match="mismatch|schema"):
                n4a.validate_design(sha)

    def test_8b_wrong_digest_refused(self):
        with pytest.raises(n4a.N4Refusal, match="digest"):
            n4a.validate_design("ab" * 32)

    def test_8c_committed_design_validates(self):
        sha = hashlib.sha256(DESIGN_V2.read_bytes()).hexdigest()
        d = n4a.validate_design(sha)
        assert d["executable_binding"] == n4a.executable_binding()


class TestPureAdjudicator:

    def test_9_incomplete_duplicate_nonfinite_forged_refused(
            self, records):
        recs = copy.deepcopy(records)
        del recs["mfemae_h6"]
        with pytest.raises(n4a.N4Refusal, match="candidate set"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        del recs["tm_h6"]["w3"]
        with pytest.raises(n4a.N4Refusal, match="window set"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["mfemae_h6"]["w1"]["losses"]["prior"][0] = \
            float("nan")
        with pytest.raises(n4a.N4Refusal, match="finite"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["mfemae_h6"]["w1"]["losses"][
            "volatility_history"] = recs["mfemae_h6"]["w1"][
            "losses"]["volatility_history"][:-1]
        with pytest.raises(n4a.N4Refusal, match="unequal"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["mfemae_h6"]["w1"]["licensed"] = True
        with pytest.raises(n4a.N4Refusal, match="producer"):
            n4a.adjudicate(recs)

    def test_10_adjudicator_reproduces_committed_artifact(
            self, result_v2, records):
        out = n4a.adjudicate(records)
        assert out["verdict"] == result_v2["verdict"] == \
            "TARGET_FORMULATION_NOT_IDENTIFIED"
        assert out["passers"] == result_v2["passers"] == []
        assert out["family"] == result_v2["family"]
        assert {ck: e["outcome"]
                for ck, e in out["per_candidate"].items()} == \
            {ck: e["outcome"] for ck, e in
             result_v2["per_candidate"].items()}


class TestScopedConclusion:

    def test_11_exact_scope_no_universal_claim(self, result_v2):
        note = result_v2["scope_note"]
        assert "ONLY" in note
        assert "ETH H4 tech_stat" in note
        assert "no claim about all possible" in note
        assert "new data or a separately motivated scientific "\
            "design" in note
        joined = json.dumps(result_v2)
        assert "all possible forecasting targets are exhausted" \
            not in joined

    def test_supersession_relation_explicit(self, result_v2):
        sup = result_v2["supersession"]
        v1_sha = hashlib.sha256(
            (REPO / "docs/audits/evidence/"
             "N4_SCREEN_RESULT_2026_09_04.json")
            .read_bytes()).hexdigest()
        assert sup["v1_result_sha256"] == v1_sha
        assert sup["loss_vectors_identical_to_v1"] is True


class TestCensusV2:

    def test_census_v2_truthful_and_complete(self):
        c = json.loads(CENSUS_V2.read_text())
        assert len(c["targets"]) == 16
        joined = json.dumps(c)
        assert "tradeable_move" not in joined
        assert "ANY trade" not in joined
        tm = c["targets"]["tm_h6"]
        assert "TERMINAL" in tm["definition"]
