"""C4 adversarial tests (order @4c1f1532): the offline bundle
verifier refuses missing, extra, duplicate, altered, forged and
non-terminal units, and refuses semantic drift against the
committed N2 trace — using the REAL committed bundle as substrate."""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location(
    "n2b", REPO / "tools" / "n2_result_bundle.py")
n2b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n2b)

BUNDLE = (REPO / "docs/audits/evidence/"
          "TARGET_HORIZON_CENSUS_N2_BUNDLE_2026_09_03.json")
TRACE = (REPO / "docs/audits/evidence/"
         "TARGET_HORIZON_CENSUS_N2_VERDICT_TRACE_2026_09_03.json")


@pytest.fixture()
def home_tmp():
    root = (Path.home() / ".cache" / "n2b_tests" / uuid.uuid4().hex)
    root.mkdir(parents=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _tampered(home, mutate):
    bundle = json.loads(BUNDLE.read_text())
    mutate(bundle)
    p = home / "bundle.json"
    p.write_text(json.dumps(bundle))
    return p


class TestVerifierAcceptsTruth:

    def test_real_bundle_verifies_semantically_equal(self):
        out = n2b.verify(BUNDLE, TRACE)
        assert out["verdict"] == "BUNDLE_VERIFIED_SEMANTICALLY_EQUAL"
        assert out["units_verified"] == 60
        assert out["reaggregated_verdict"] == \
            "TARGET_CANDIDATE_FOUND"


class TestVerifierRefusals:

    def test_missing_unit(self, home_tmp):
        p = _tampered(home_tmp,
                      lambda b: b["units"].pop(7))
        with pytest.raises(n2b.BundleRefusal, match="missing"):
            n2b.verify(p, TRACE)

    def test_duplicate_unit(self, home_tmp):
        p = _tampered(home_tmp,
                      lambda b: b["units"].append(b["units"][0]))
        with pytest.raises(n2b.BundleRefusal, match="duplicate"):
            n2b.verify(p, TRACE)

    def test_extra_unit_not_in_ledger(self, home_tmp):
        def mutate(b):
            fake = dict(b["units"][0])
            fake["unit_id"] = "f" * 20
            b["units"].append(fake)
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal, match="extra"):
            n2b.verify(p, TRACE)

    def test_altered_result_bytes(self, home_tmp):
        def mutate(b):
            b["units"][3]["result_text"] = \
                b["units"][3]["result_text"].replace(
                    '"n_score": 216', '"n_score": 215', 1)
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal, match="altered"):
            n2b.verify(p, TRACE)

    def test_forged_identity(self, home_tmp):
        def mutate(b):
            b["units"][5]["identity"] = dict(
                b["units"][5]["identity"], origin="w9")
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal,
                           match="identity differs"):
            n2b.verify(p, TRACE)

    def test_ledger_identity_forged_fails_unit_id_recompute(
            self, home_tmp):
        def mutate(b):
            uid = b["units"][5]["unit_id"]
            for u in b["ledger"]["units"]:
                if u["unit_id"] == uid:
                    u["identity"] = dict(u["identity"], origin="w9")
            b["units"][5]["identity"] = dict(
                b["units"][5]["identity"], origin="w9")
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal,
                           match="does not hash"):
            n2b.verify(p, TRACE)

    def test_non_completed_state(self, home_tmp):
        def mutate(b):
            b["units"][2]["state"] = "FAILED"
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal,
                           match="not COMPLETED"):
            n2b.verify(p, TRACE)

    def test_self_digest_forgery(self, home_tmp):
        """Re-serialize a tampered payload with a CORRECT file
        sha but stale runtime self-digest — caught by the
        canonical self-digest recompute."""
        import hashlib
        def mutate(b):
            payload = json.loads(b["units"][4]["result_text"])
            payload["selected_model"] = "forged_model"
            raw = json.dumps(payload)
            b["units"][4]["result_text"] = raw
            b["units"][4]["result_sha256"] = hashlib.sha256(
                raw.encode()).hexdigest()
        p = _tampered(home_tmp, mutate)
        with pytest.raises(n2b.BundleRefusal,
                           match="self-digest"):
            n2b.verify(p, TRACE)

    def test_semantic_drift_vs_trace(self, home_tmp):
        trace = json.loads(TRACE.read_text())
        trace["candidates"]["bar_h6"][
            "pooled_skill_vs_strongest"] = 0.5
        tp = home_tmp / "trace.json"
        tp.write_text(json.dumps(trace))
        with pytest.raises(n2b.BundleRefusal,
                           match="semantic drift"):
            n2b.verify(BUNDLE, tp)
