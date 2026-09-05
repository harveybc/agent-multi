"""N4 acceptance battery v3 (orders @13fdf18c + @9fd016b0 +
@af1ca667): strict design boundary (C17), evidence-complete
adjudication (C18), source-bound rebind (C19), typed refusals (C20),
truthful chronology (C21) and the corrected ledger (C22) — with the
eight mandatory permanent regressions of §6."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest.mock as um
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n4a", REPO / "tools" / "n4_target_audit.py")
n4a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n4a)

EV = REPO / "docs/audits/evidence"
RESULT_V1 = EV / "N4_SCREEN_RESULT_2026_09_04.json"
RESULT_V3 = EV / "N4_SCREEN_RESULT_V3_2026_09_04.json"
DESIGN_V3 = EV / "N4_TARGET_AUDIT_DESIGN_V3_2026_09_04.json"
LEDGER_V2 = EV / "N5_TRANSITION_LEDGER_V2_2026_09_04.json"
V1_SHA = ("d696886c4e0d8f59378e29505eaea509ffeef80b6fa8b1c5c951758"
          "7333d7400")
V1_DESIGN_SHA = ("ae05f1878305cc3aee9003849d4f147f2685a159ed3afbdc"
                 "3870ec7e8c58f4ef")


@pytest.fixture(scope="module")
def result_v3():
    return json.loads(RESULT_V3.read_text())


@pytest.fixture(scope="module")
def records(result_v3):
    return json.loads(json.dumps(
        result_v3["per_window_records"]))


# ------------------------------------------------------------------ #
# C17: strict design boundary                                        #
# ------------------------------------------------------------------ #

class TestC17StrictDesign:

    def test_committed_design_validates(self):
        sha = hashlib.sha256(DESIGN_V3.read_bytes()).hexdigest()
        d = n4a.validate_design(sha)
        assert d["executable_binding"] == n4a.executable_binding()

    def _try(self, tmp_path, text):
        p = tmp_path / "d.json"
        p.write_text(text)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        with um.patch.object(n4a, "DESIGN_V3", str(p)):
            n4a.validate_design(sha)

    def test_unknown_top_field_refused(self, tmp_path):
        d = json.loads(DESIGN_V3.read_text())
        d["unknown_top_level_field"] = 1
        with pytest.raises(n4a.N4Refusal, match="top-level"):
            self._try(tmp_path, json.dumps(d))

    def test_duplicate_key_refused(self, tmp_path):
        text = DESIGN_V3.read_text().replace(
            '"experiment":', '"experiment": "x", "experiment":',
            1)
        with pytest.raises(n4a.N4Refusal, match="duplicate"):
            self._try(tmp_path, text)

    def test_float_support_min_refused(self, tmp_path):
        d = json.loads(DESIGN_V3.read_text())
        d["executable_binding"]["support_min"] = 30.0
        with pytest.raises(n4a.N4Refusal, match="integer"):
            self._try(tmp_path, json.dumps(d))

    def test_boolean_never_an_integer(self, tmp_path):
        d = json.loads(DESIGN_V3.read_text())
        d["executable_binding"]["support_min"] = True
        with pytest.raises(n4a.N4Refusal, match="boolean|integer"):
            self._try(tmp_path, json.dumps(d))

    def test_regression8_altered_design_refused_before_records(
            self, tmp_path):
        d = json.loads(DESIGN_V3.read_text())
        d["executable_binding"]["margin"] = 0.5
        with pytest.raises(n4a.N4Refusal, match="binding"):
            self._try(tmp_path, json.dumps(d))


# ------------------------------------------------------------------ #
# C18/C20: evidence-complete adjudication — mandatory regressions    #
# ------------------------------------------------------------------ #

class TestMandatoryRegressions:

    def test_r1_fabricated_supports_cannot_license_tm_h6(
            self, records):
        """Support is DERIVED from labels; the old bypass (declare
        counts, sweeten losses) has no field to attack — a
        producer-supplied support field now refuses outright."""
        recs = copy.deepcopy(records)
        for wk, rec in recs["tm_h6"].items():
            rec["class_support_score"] = {"0": 80, "1": 80,
                                          "2": 60}
        with pytest.raises(n4a.N4Refusal, match="producer"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        for wk, rec in recs["tm_h6"].items():
            rec["losses"]["volatility_history"] = [
                round(v * 0.5, 8) for v in rec["losses"]["prior"]]
        out = n4a.adjudicate(recs)
        assert out["per_candidate"]["tm_h6"]["outcome"] == \
            "UNLICENSED"
        assert "tm_h6:volatility_history" not in out["passers"]

    def test_r2_missing_window_or_n_score_refuses(self, records):
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"].pop("window")
        with pytest.raises(n4a.N4Refusal, match="schema"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"].pop("n_score")
        with pytest.raises(n4a.N4Refusal, match="schema"):
            n4a.adjudicate(recs)

    def test_r3_support_totals_bound_to_cardinality(self, records):
        """Labels ARE the support source; truncating them breaks
        the cardinality equation and refuses."""
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"]["labels"] = \
            recs["tm_h6"]["w1"]["labels"][:-4]
        with pytest.raises(n4a.N4Refusal, match="cardinality"):
            n4a.adjudicate(recs)

    def test_r4_numeric_string_loss_refuses_even_unlicensed(
            self, records):
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"]["losses"]["prior"][0] = str(
            recs["tm_h6"]["w1"]["losses"]["prior"][0])
        with pytest.raises(n4a.N4Refusal, match="finite JSON"):
            n4a.adjudicate(recs)

    def test_r5_boolean_target_value_refuses(self, records):
        recs = copy.deepcopy(records)
        recs["mfemae_h6"]["w1"]["target_values"][0] = True
        with pytest.raises(n4a.N4Refusal, match="finite JSON"):
            n4a.adjudicate(recs)

    def test_r6_altered_source_digest_refuses_before_parsing(
            self, tmp_path):
        forged = tmp_path / "v1.json"
        forged.write_text(RESULT_V1.read_text() + " ")
        with pytest.raises(n4a.N4Refusal, match="reviewed"):
            n4a.rebind(forged, V1_SHA, V1_DESIGN_SHA,
                       hashlib.sha256(
                           DESIGN_V3.read_bytes()).hexdigest(),
                       Path("/nonexistent"),
                       tmp_path / "out.json")

    def test_r7_forged_positive_v1_cannot_mint_a_successor(
            self, tmp_path):
        """The C19 attack frozen: sweetened continuous losses in a
        forged v1 — rejected at the source-identity gate, never
        adjudicated."""
        v1 = json.loads(RESULT_V1.read_text())
        for ck in ("mfemae_h6", "mfemae_h12"):
            for wk, rec in v1["per_window_records"][ck].items():
                rec["losses"]["target_history"] = [
                    round(v * 0.01, 8)
                    for v in rec["losses"]["prior"]]
        forged = tmp_path / "forged_v1.json"
        forged.write_text(json.dumps(v1, default=float))
        with pytest.raises(n4a.N4Refusal, match="reviewed"):
            n4a.rebind(forged, V1_SHA, V1_DESIGN_SHA,
                       hashlib.sha256(
                           DESIGN_V3.read_bytes()).hexdigest(),
                       Path("/nonexistent"),
                       tmp_path / "out.json")
        # and supplying the forged file's own digest is not the
        # reviewed identity: the caller cannot substitute it
        fsha = hashlib.sha256(forged.read_bytes()).hexdigest()
        assert fsha != V1_SHA

    def test_more_typed_refusals(self, records):
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"]["window"] = "w2"
        with pytest.raises(n4a.N4Refusal, match="declared window"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"]["labels"][0] = 7
        with pytest.raises(n4a.N4Refusal, match="contract"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["mfemae_h6"]["w1"]["losses"]["prior"] = [0.0] * \
            recs["mfemae_h6"]["w1"]["n_score"]
        with pytest.raises(n4a.N4Refusal, match="denominator"):
            n4a.adjudicate(recs)
        recs = copy.deepcopy(records)
        recs["tm_h6"]["w1"]["n_score"] = float(
            recs["tm_h6"]["w1"]["n_score"])
        with pytest.raises(n4a.N4Refusal, match="positive integer"):
            n4a.adjudicate(recs)


# ------------------------------------------------------------------ #
# C18 derivation + verdict stability                                 #
# ------------------------------------------------------------------ #

class TestDerivedAdjudication:

    def test_supports_and_variance_are_derived(self, result_v3):
        facts = result_v3["derived_licensing_facts"]
        assert facts["tm_h6:w1"]["support"] == {"0": 113,
                                               "1": 97, "2": 4}
        assert facts["mfemae_h6:w1"]["response_var"] > 0

    def test_verdict_and_outcomes(self, result_v3):
        assert result_v3["verdict"] == \
            "TARGET_FORMULATION_NOT_IDENTIFIED"
        assert result_v3["passers"] == []
        assert result_v3["family_cardinality_proven"] == 14
        outcomes = {ck: e["outcome"] for ck, e in
                    result_v3["per_candidate"].items()}
        assert outcomes == {
            "tm_h6": "UNLICENSED", "tm_h12": "UNLICENSED",
            "tm_h24": "UNLICENSED", "mfemae_h6": "FAILS",
            "mfemae_h12": "FAILS", "lm_h6": "UNLICENSED",
            "lm_h12": "UNLICENSED"}

    def test_alignment_proofs_complete(self, result_v3):
        proofs = result_v3["alignment_proofs"]
        assert len(proofs) == 28
        assert set(proofs.values()) == {"prior_vector_exact"}

    def test_adjudicator_reproduces_artifact(self, result_v3,
                                             records):
        out = n4a.adjudicate(records)
        assert out["verdict"] == result_v3["verdict"]
        assert out["family"] == result_v3["family"]


# ------------------------------------------------------------------ #
# C21/C22: chronology and ledger                                     #
# ------------------------------------------------------------------ #

class TestChronologyAndLedger:

    def test_v3_never_called_predeclared(self, result_v3):
        chron = result_v3["chronology"]
        assert "AUDITOR_PRESCRIBED_CORRECTIVE_ADJUDICATION_"\
            "NO_NEW_HYPOTHESIS" in chron["v3"]
        assert "NOT a" in chron["v2"] or "not a" in chron["v2"]
        design = json.loads(DESIGN_V3.read_text())
        assert "sealed_before_any_new_score" not in design
        assert "untruthful" in design["chronology"]["v2"]

    def test_ledger_v2_roadmap_and_state(self):
        led = json.loads(LEDGER_V2.read_text())
        assert led["roadmap_next_node"]["node"].startswith(
            "Screen B / B4")
        assert "81fa5a2b" in led["roadmap_next_node"][
            "surviving_branch"]
        assert "DEFERRED" in led["feature_selection"]["status"]
        wf = led["weekly_flat_economic_work"]
        assert "ACCEPTED" in wf["c38_status"]
        assert "FLAT 0 positions / 0 orders" in wf[
            "mt5_last_known_state"]
        assert "6140" in wf["mt5_last_known_state"]
        assert wf["collector_status"].startswith(
            "COORDINATED_WINDOW_REQUIRED")
        assert led["owner_ratifications_consumed"][
            "observation_contract_v2"].startswith(
            "OWNER_RATIFIED")


class TestB4RecoveryArtifacts:
    """B4-R0/R1 (order @af1ca667 §§9-10): the compatibility matrix
    and superseding-design proposal are internally consistent and
    honor the ratified observation identity."""

    def test_matrix_names_exactly_one_semantic_change(self):
        m = json.loads((EV / "B4_COMPATIBILITY_MATRIX_2026_09_04"
                        ".json").read_text())
        assert len(m["semantic_changes_unavoidable"]) == 1
        assert "gymfx_env_semantics" in \
            m["semantic_changes_unavoidable"][0]
        assert m["matrix"]["supervised_pretraining_closure"][
            "verdict"] == "COMPATIBLE"

    def test_ratified_observation_terms_match_live_file(self):
        c = json.loads(
            (REPO / "examples/config/phase_3_eth_sac_dynamics/"
             "systems/ethusdt_4h_l1_system_v2.json").read_text())
        assert c["status"] == "OWNER_RATIFIED"
        cols = c["observation"]["feature_columns"]
        cols_sha = hashlib.sha256(json.dumps(
            cols, separators=(",", ":")).encode()).hexdigest()
        assert len(cols) == 83
        assert cols_sha.startswith("c4697681")
        assert c["observation"]["include_price_window"] is False
        assert c["ratification"]["record_sha256"].startswith(
            "399483a1")
        assert c["ratification"]["proposed_file_sha256"]\
            .startswith("0ecc3d00")

    def test_design_proposal_defers_execution(self):
        d = json.loads(
            (EV / "B4_SUPERSEDING_DESIGN_PROPOSAL_2026_09_04"
             ".json").read_text())
        assert d["status"] == "CANDIDATE_AWAITING_MUSASHI_REVIEW"
        assert any("mechanics cell" in x for x in
                   d["explicitly_not_authorized_here"])
        assert any("GPU" in x for x in
                   d["explicitly_not_authorized_here"])
        assert "invariant_either_way" in \
            d["the_single_semantic_decision_for_review"]
