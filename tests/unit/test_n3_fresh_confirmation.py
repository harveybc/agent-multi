"""N3 battery v3 (orders @a13671ab + @17f6e574 + @a1e7b739):
typed epoch helper, restricted custody, strict wire grammar, typed
per-observation evidence, exact nested schemas (P1-P8 frozen as
semantic regressions on the REAL published v2 bundle), and the
authority ladder — a caller-supplied digest can never mint the
gate-bearing publication label."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat as stat_mod
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
V2_PATH = (REPO / "docs/audits/evidence/"
           "N3_FRESH_CONFIRMATION_BUNDLE_V2_2026_09_04.json")
V3_PATH = (REPO / "docs/audits/evidence/"
           "N3_FRESH_CONFIRMATION_BUNDLE_V3_2026_09_04.json")
V2_REVIEWED_SHA = ("f2c4ae1dc9628b1d9ab733a1ed4f28b1de3f32c31a713"
                   "9efff89f3945e592c82")


# ------------------------------------------------------------------ #
# C1 helper / C2 custody / C5 grammar (unchanged disciplines)        #
# ------------------------------------------------------------------ #

class TestEpochHelper:

    def test_ms_and_ns_resolutions_agree(self):
        for res in ("datetime64[ms, UTC]", "datetime64[ns, UTC]"):
            s = pd.Series(pd.to_datetime([1767211200000],
                                         unit="ms",
                                         utc=True)).astype(res)
            assert int(n3f.to_epoch_ms(s).iloc[0]) == 1767211200000

    def test_frozen_parquet_final_bar_exact(self):
        lake = pd.read_parquet(n3f.LAKE_PARQUET)
        assert int(n3f.to_epoch_ms(
            lake["open_time"]).iloc[-1]) == 1767211200000

    def test_null_bool_range_refuse(self):
        with pytest.raises(n3f.FreshRefusal):
            n3f.to_epoch_ms(pd.Series([pd.NaT]))
        with pytest.raises(n3f.FreshRefusal):
            n3f.to_epoch_ms(pd.Series([True, False]))
        with pytest.raises(n3f.FreshRefusal):
            n3f.to_epoch_ms(pd.Series(pd.to_datetime(
                [0], unit="ms", utc=True)))


class TestCustody:

    @pytest.fixture()
    def base(self, tmp_path):
        old = os.umask(0o000)
        yield tmp_path
        os.umask(old)

    def test_create_0700_files_0600_under_umask_000(self, base):
        root = base / "staging"
        fd = n3f.secure_root(root, create=True)
        try:
            assert (os.fstat(fd).st_mode & 0o777) == 0o700
            n3f.secure_write(fd, "a.bin", b"x")
        finally:
            os.close(fd)
        assert (os.stat(root / "a.bin").st_mode & 0o777) == 0o600
        assert n3f.secure_read(root, "a.bin") == b"x"

    def test_permissive_root_refused_not_repaired(self, base):
        root = base / "permissive"
        root.mkdir(mode=0o775)
        os.chmod(root, 0o775)
        with pytest.raises(n3f.FreshRefusal):
            n3f.secure_root(root, create=False)
        assert (os.stat(root).st_mode & 0o777) == 0o775

    def test_symlinks_refused(self, base):
        real = base / "real"
        real.mkdir(mode=0o700)
        os.chmod(real, 0o700)
        (base / "link").symlink_to(real)
        with pytest.raises(n3f.FreshRefusal):
            n3f.secure_root(base / "link", create=False)


class TestStrictParsing:

    def test_json_nan_and_duplicates_refused(self):
        with pytest.raises(n3f.FreshRefusal):
            n3f.strict_json(b'{"a": NaN}')
        with pytest.raises(n3f.FreshRefusal):
            n3f.strict_json(b'{"a": 1, "a": 2}')

    def test_decimal_grammar(self):
        assert n3f.strict_decimal("2345.10000000", "x") == 2345.1
        for bad in ("1e5", "NaN", True, 5, "5", ".5"):
            with pytest.raises(n3f.FreshRefusal):
                n3f.strict_decimal(bad, "x")

    def test_twelve_fields_and_bool_count_refused(self):
        row = [1735689600000, "100.0", "110.0", "90.0", "105.0",
               "10.5", 1735703999999, "1050.0", 7, "5.25",
               "525.0", "0"]
        n3f.validate_wire_rows([row], 1735689600000 + 10 ** 9)
        with pytest.raises(n3f.FreshRefusal, match="12"):
            n3f.validate_wire_rows([row[:11]],
                                   1735689600000 + 10 ** 9)
        bad = list(row)
        bad[8] = True
        with pytest.raises(n3f.FreshRefusal, match="integer"):
            n3f.validate_wire_rows([bad], 1735689600000 + 10 ** 9)


class TestOverlapExactness:

    def test_sub_float32_revision_refuses(self):
        base = 2345.123456789
        true_close = base + 1
        revised = true_close * (1 + 3e-8)
        assert np.float32(revised) == np.float32(true_close)
        rows = [[1735689600000, str(base), str(base + 10),
                 str(base - 10), repr(revised), "100.5",
                 1735703999999, "23451.2", 777, "50.25",
                 "11725.6", "0"]]
        lake = pd.DataFrame({
            "open": [base], "high": [base + 10],
            "low": [base - 10], "close": [true_close],
            "volume": [100.5], "quote_volume": [23451.2],
            "trade_count": [777],
            "taker_buy_base_volume": [50.25],
            "taker_buy_quote_volume": [11725.6]})
        with pytest.raises(n3f.FreshRefusal, match="revised"):
            n3f._verify_overlap(rows, lake,
                                pd.Series([1735689600000]))

    def test_one_ulp_derived_field_tolerated_and_counted(self):
        base = 2345.123456789
        import struct
        qv = 23451.2
        qv_1ulp = struct.unpack("<d", struct.pack(
            "<q", struct.unpack("<q", struct.pack(
                "<d", qv))[0] + 1))[0]
        rows = [[1735689600000, str(base), str(base + 10),
                 str(base - 10), str(base + 1), "100.5",
                 1735703999999, repr(qv_1ulp), 777, "50.25",
                 "11725.6", "0"]]
        lake = pd.DataFrame({
            "open": [base], "high": [base + 10],
            "low": [base - 10], "close": [base + 1],
            "volume": [100.5], "quote_volume": [qv],
            "trade_count": [777],
            "taker_buy_base_volume": [50.25],
            "taker_buy_quote_volume": [11725.6]})
        rep = n3f._verify_overlap(rows, lake,
                                  pd.Series([1735689600000]))
        assert rep["quote_volume"]["cells_1ulp"] == 1
        assert rep["quote_volume"]["max_ulp"] == 1


# ------------------------------------------------------------------ #
# geometry and decision (sealed)                                     #
# ------------------------------------------------------------------ #

class TestGeometryAndDecision:

    def test_canonical_anchors_match_contract(self):
        expected = SEALED["role_ledger"][
            "expected_scoring_anchors"]
        for name, start, end, bars in n3f.BLOCKS:
            canon = n3f.canonical_anchor_datetimes(name)
            assert len(canon) == expected[name]

    def test_decision_table(self):
        base = {}
        for t in n3f.TARGETS:
            for (a, b) in n3f.CONTRAST_FAMILY:
                base[(t, a, b)] = {"pooled_skill": -0.01,
                                   "all_blocks_positive": False,
                                   "holm_p": 1.0}
        assert n3f.decide(base, True, True) == \
            "TARGET_SCALE_EFFECT_NOT_CONFIRMED"
        assert n3f.decide({}, False, True) == \
            "FRESH_CONFIRMATION_INSUFFICIENT"
        good = {"pooled_skill": 0.02,
                "all_blocks_positive": True, "holm_p": 0.001}
        rep = dict(base)
        rep[("bar_h6", "arm2", "arm1")] = good
        rep[("bar_h12", "arm2", "arm1")] = good
        rep[("bar_h12", "arm5", "arm2")] = {
            "pooled_skill": 0.006,
            "all_blocks_positive": True, "holm_p": 0.001}
        assert n3f.decide(rep, True, True) == \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"


# ------------------------------------------------------------------ #
# P1-P8: the eight Musashi semantic mutations, frozen as             #
# regressions on the REAL published v2 bundle (self-supplied sha     #
# crosses the byte layer; refusal must be SEMANTIC)                  #
# ------------------------------------------------------------------ #

def _redigest(u):
    u["payload_sha256"] = sha_obj(
        {k: v for k, v in u.items() if k != "payload_sha256"})


@pytest.fixture(scope="module")
def v2():
    return json.loads(V2_PATH.read_text())


def _probe(tmp_path, v2, mutate):
    b = copy.deepcopy(v2)
    mutate(b)
    b["digests"]["code"] = n3f._code_digest()  # coherent forger
    p = tmp_path / "probe.json"
    p.write_text(json.dumps(b, default=float))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    return p, sha


class TestSemanticRegressionsP1P8:

    def test_p1_decision_constants(self, v2, tmp_path):
        p, sha = _probe(tmp_path, v2, lambda b: b[
            "decision_constants"].__setitem__("margin_repr", 999))
        with pytest.raises(n3f.FreshRefusal,
                           match="decision_constants"):
            n3f.verify(p, sha)

    def test_p2_stride(self, v2, tmp_path):
        p, sha = _probe(tmp_path, v2, lambda b: b[
            "role_ledger"].__setitem__("stride", 999))
        with pytest.raises(n3f.FreshRefusal, match="stride"):
            n3f.verify(p, sha)

    def test_p3_horizon(self, v2, tmp_path):
        def m(b):
            u = [x for x in b["units"] if x["horizon"] == 6][0]
            u["horizon"] = 999
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="horizon"):
            n3f.verify(p, sha)

    def test_p4_contract_path(self, v2, tmp_path):
        p, sha = _probe(tmp_path, v2, lambda b: b.__setitem__(
            "contract", "docs/README_UNRELATED.md"))
        with pytest.raises(n3f.FreshRefusal,
                           match="contract path"):
            n3f.verify(p, sha)

    def test_p5_acquired_digest(self, v2, tmp_path):
        p, sha = _probe(tmp_path, v2, lambda b: b[
            "digests"].__setitem__("acquired_parquet", "0" * 64))
        with pytest.raises(n3f.FreshRefusal, match="receipt"):
            n3f.verify(p, sha)

    def test_p6_boolean_label_support_preserving(self, v2,
                                                 tmp_path):
        def m(b):
            u = b["units"][0]
            u["labels"][u["labels"].index(1)] = True
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal,
                           match="JSON integers"):
            n3f.verify(p, sha)

    def test_p7_string_probability(self, v2, tmp_path):
        def m(b):
            u = b["units"][1]
            u["arms"]["arm2"]["probs"][0][0] = str(
                u["arms"]["arm2"]["probs"][0][0])
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="JSON number"):
            n3f.verify(p, sha)

    def test_p8_undeclared_digest_key(self, v2, tmp_path):
        p, sha = _probe(tmp_path, v2, lambda b: b[
            "digests"].__setitem__("undeclared_extra", "f" * 64))
        with pytest.raises(n3f.FreshRefusal, match="schema"):
            n3f.verify(p, sha)


class TestTypedEvidenceCounterexamples:

    def test_fractional_count(self, v2, tmp_path):
        def m(b):
            u = b["units"][2]
            u["n_score"] = float(u["n_score"])
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal):
            n3f.verify(p, sha)

    def test_bool_in_histogram(self, v2, tmp_path):
        def m(b):
            u = b["units"][3]
            u["fit_cal_label_histogram"][0] = True
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="histogram"):
            n3f.verify(p, sha)

    def test_malformed_probability_row(self, v2, tmp_path):
        def m(b):
            u = b["units"][4]
            u["arms"]["arm3"]["probs"][5] = [0.5, 0.5]
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="malformed"):
            n3f.verify(p, sha)

    def test_out_of_simplex_row(self, v2, tmp_path):
        def m(b):
            u = b["units"][5]
            u["arms"]["arm4"]["probs"][0] = [0.6, 0.6, 0.6]
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="sum"):
            n3f.verify(p, sha)

    def test_c_outside_sealed_grid(self, v2, tmp_path):
        def m(b):
            u = b["units"][6]
            u["arms"]["arm2"]["record"]["C"] = 3.14
            _redigest(u)
        p, sha = _probe(tmp_path, v2, m)
        with pytest.raises(n3f.FreshRefusal, match="search set"):
            n3f.verify(p, sha)


# ------------------------------------------------------------------ #
# C8: the authority ladder on the REAL artifacts                     #
# ------------------------------------------------------------------ #

class TestAuthorityLadder:

    def test_reviewed_v2_is_gate_bearing(self):
        out = n3f.verify(V2_PATH, V2_REVIEWED_SHA)
        assert out["verdict"] == "N3_PUBLICATION_VERIFIED"
        assert out["gate_bearing"] is True
        assert out["rederived_decision"] == \
            "TARGET_SCALE_EFFECT_NOT_CONFIRMED"

    def test_pending_v3_is_not_gate_bearing(self):
        sha = hashlib.sha256(V3_PATH.read_bytes()).hexdigest()
        out = n3f.verify(V3_PATH, sha)
        assert out["verdict"] == \
            "N3_CANDIDATE_CONSISTENT_PENDING_REVIEW"
        assert out["gate_bearing"] is False

    def test_internal_mode_cannot_mint_authority(self):
        out = n3f.verify(V3_PATH, internal_only=True)
        assert out["verdict"] == "N3_INTERNAL_CONSISTENCY_ONLY"
        assert out["gate_bearing"] is False

    def test_no_external_digest_refused(self):
        with pytest.raises(n3f.FreshRefusal, match="required"):
            n3f.verify(V3_PATH)

    def test_wrong_digest_refused_before_parsing(self):
        with pytest.raises(n3f.FreshRefusal, match="match"):
            n3f.verify(V3_PATH, "ab" * 32)

    def test_coherent_fake_passer_is_untrusted_never_verified(
            self, v2, tmp_path):
        """Acceptance 4+5: a fully coherent forgery — probs,
        metrics, digests, contrasts, verdict and code identity all
        recomputed — earns at most the supplied-digest consistency
        label: UNTRUSTED, non-gate, and never a label implying
        independent approval."""
        b = copy.deepcopy(v2)
        for u in b["units"]:
            if u["horizon"] == 6:
                y = np.asarray(u["labels"])
                prior = np.asarray(
                    u["fit_cal_label_histogram"],
                    dtype=float)
                prior = prior / prior.sum()
                fake = 0.95 * np.asarray(
                    u["arms"]["arm2"]["probs"]) \
                    + 0.05 * np.eye(3)[y]
                fake = fake / fake.sum(axis=1, keepdims=True)
                u["arms"]["arm3"]["probs"] = [
                    [float(x) for x in row] for row in fake]
                u["arms"]["arm3"]["metrics"] = n3f.unit_metrics(
                    y, fake)
                _redigest(u)
        contrasts, stats, _ = n3f._rederive(b["units"])
        b["contrasts"] = contrasts
        b["verdict"] = n3f.decide(stats, True, True)
        assert b["verdict"] == \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
        b["digests"]["code"] = n3f._code_digest()
        p = tmp_path / "fake.json"
        p.write_text(json.dumps(b, default=float))
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        out = n3f.verify(p, sha)
        assert out["verdict"] == \
            "N3_BUNDLE_CONSISTENT_WITH_SUPPLIED_DIGEST"
        assert out["gate_bearing"] is False
        assert "UNTRUSTED" in out["authority"]


# ------------------------------------------------------------------ #
# C9/C10: the v3 envelope                                            #
# ------------------------------------------------------------------ #

class TestReissueV3:

    def test_v3_science_byte_equal_and_full_compare(self):
        v3 = json.loads(V3_PATH.read_text())
        m = v3["v2_correction_map"]
        assert m["science_byte_equal"] is True
        assert m["complete_contrast_objects_equal"] is True
        assert m["decisions_equal"] is True
        m1 = v3["v1_correction_map"]
        assert m1["complete_contrast_objects_equal"] is True
        assert "every key and value" in m1["comparison_scope"]

    def test_v2_and_v3_contrasts_identical(self, v2):
        v3 = json.loads(V3_PATH.read_text())
        assert v3["contrasts"] == v2["contrasts"]
        assert v3["verdict"] == v2["verdict"]
        assert [u["labels"] for u in v3["units"]] == \
            [u["labels"] for u in v2["units"]]


class TestUnitMetrics:

    def test_additive_identity(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 3, 40)
        p = rng.dirichlet([1, 1, 1], size=40)
        m = n3f.unit_metrics(y, p)
        assert m["additive_identity_max_abs_gap"] < 1e-9

    def test_invalid_inputs_refuse(self):
        with pytest.raises(n3f.FreshRefusal):
            n3f.unit_metrics([0, 1], [[0.2, 0.3, 0.5]])
        with pytest.raises(n3f.FreshRefusal):
            n3f.unit_metrics([0], [[np.nan, 0.5, 0.5]])
