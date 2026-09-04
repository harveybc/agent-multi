"""N3 v2 battery (orders @a13671ab + correction @17f6e574):
typed epoch helper, restricted custody under umask 000, strict wire
grammar, evidence-derived metrics, and a verifier with EXTERNAL
authority that refuses the four Musashi forgeries and every earlier
adversary."""
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


# ------------------------------------------------------------------ #
# C1: the typed epoch helper                                         #
# ------------------------------------------------------------------ #

class TestEpochHelper:

    def test_ms_resolution_series_is_unit_safe(self):
        s = pd.Series(pd.to_datetime([1767211200000], unit="ms",
                                     utc=True)).astype(
            "datetime64[ms, UTC]")
        assert int(n3f.to_epoch_ms(s).iloc[0]) == 1767211200000

    def test_ns_resolution_series_same_answer(self):
        s = pd.Series(pd.to_datetime([1767211200000], unit="ms",
                                     utc=True)).astype(
            "datetime64[ns, UTC]")
        assert int(n3f.to_epoch_ms(s).iloc[0]) == 1767211200000

    def test_frozen_parquet_final_bar_exact(self):
        lake = pd.read_parquet(n3f.LAKE_PARQUET)
        ms = n3f.to_epoch_ms(lake["open_time"])
        assert int(ms.iloc[-1]) == 1767211200000

    def test_null_refuses(self):
        with pytest.raises(n3f.FreshRefusal, match="null"):
            n3f.to_epoch_ms(pd.Series([pd.NaT]))

    def test_bool_refuses(self):
        with pytest.raises(n3f.FreshRefusal, match="boolean"):
            n3f.to_epoch_ms(pd.Series([True, False]))

    def test_out_of_range_refuses(self):
        s = pd.Series(pd.to_datetime([0], unit="ms", utc=True))
        with pytest.raises(n3f.FreshRefusal, match="range"):
            n3f.to_epoch_ms(s)

    def test_strict_epoch_int_rejects_bool(self):
        with pytest.raises(n3f.FreshRefusal):
            n3f.strict_epoch_int(True, "x")


# ------------------------------------------------------------------ #
# C2 custody under umask 000                                         #
# ------------------------------------------------------------------ #

class TestCustody:

    @pytest.fixture()
    def base(self, tmp_path):
        old = os.umask(0o000)
        yield tmp_path
        os.umask(old)

    def test_create_is_0700_and_files_0600_under_umask_000(
            self, base):
        root = base / "staging"
        fd = n3f.secure_root(root, create=True)
        try:
            st = os.fstat(fd)
            assert (st.st_mode & 0o777) == 0o700
            n3f.secure_write(fd, "a.bin", b"x")
        finally:
            os.close(fd)
        assert (os.stat(root / "a.bin").st_mode & 0o777) == 0o600
        assert n3f.secure_read(root, "a.bin") == b"x"

    def test_existing_permissive_root_refused_not_repaired(
            self, base):
        root = base / "permissive"
        root.mkdir(mode=0o775)
        os.chmod(root, 0o775)
        with pytest.raises(n3f.FreshRefusal, match="already exists"):
            n3f.secure_root(root, create=True)
        with pytest.raises(n3f.FreshRefusal, match="0700"):
            n3f.secure_root(root, create=False)
        assert (os.stat(root).st_mode & 0o777) == 0o775  # untouched

    def test_symlink_root_refused(self, base):
        real = base / "real"
        real.mkdir(mode=0o700)
        os.chmod(real, 0o700)
        link = base / "link"
        link.symlink_to(real)
        with pytest.raises(n3f.FreshRefusal):
            n3f.secure_root(link, create=False)

    def test_symlinked_file_refused(self, base):
        root = base / "s2"
        fd = n3f.secure_root(root, create=True)
        try:
            n3f.secure_write(fd, "real.bin", b"y")
        finally:
            os.close(fd)
        (root / "evil.bin").symlink_to(root / "real.bin")
        with pytest.raises(n3f.FreshRefusal):
            n3f.secure_read(root, "evil.bin")

    def test_permissive_file_refused(self, base):
        root = base / "s3"
        fd = n3f.secure_root(root, create=True)
        try:
            n3f.secure_write(fd, "f.bin", b"z")
        finally:
            os.close(fd)
        os.chmod(root / "f.bin", 0o664)
        with pytest.raises(n3f.FreshRefusal, match="0600"):
            n3f.secure_read(root, "f.bin")


# ------------------------------------------------------------------ #
# C5 strict wire grammar and JSON                                    #
# ------------------------------------------------------------------ #

class TestStrictParsing:

    def test_json_nan_refused(self):
        with pytest.raises(n3f.FreshRefusal, match="non-finite"):
            n3f.strict_json(b'{"a": NaN}')

    def test_json_duplicate_key_refused(self):
        with pytest.raises(n3f.FreshRefusal, match="duplicate"):
            n3f.strict_json(b'{"a": 1, "a": 2}')

    def test_decimal_grammar(self):
        assert n3f.strict_decimal("2345.10000000", "x") == 2345.1
        for bad in ("1e5", "NaN", "Infinity", True, 5, "5", ".5"):
            with pytest.raises(n3f.FreshRefusal):
                n3f.strict_decimal(bad, "x")

    def _row(self, open_ms=1735689600000):
        return [open_ms, "100.0", "110.0", "90.0", "105.0",
                "10.5", open_ms + 14399999, "1050.0", 7,
                "5.25", "525.0", "0"]

    def test_valid_row_passes(self):
        n3f.validate_wire_rows([self._row()],
                               1735689600000 + 10 ** 9)

    def test_eleven_fields_refused(self):
        with pytest.raises(n3f.FreshRefusal, match="12"):
            n3f.validate_wire_rows([self._row()[:11]],
                                   1735689600000 + 10 ** 9)

    def test_boolean_count_refused(self):
        r = self._row()
        r[8] = True
        with pytest.raises(n3f.FreshRefusal, match="integer"):
            n3f.validate_wire_rows([r], 1735689600000 + 10 ** 9)

    def test_open_bar_refused(self):
        r = self._row()
        with pytest.raises(n3f.FreshRefusal, match="open terminal"):
            n3f.validate_wire_rows([r], r[6])


# ------------------------------------------------------------------ #
# geometry, decision, overlap exactness (carried from v1)            #
# ------------------------------------------------------------------ #

class TestGeometry:

    def test_expected_scoring_anchors_match_contract(self):
        expected = SEALED["role_ledger"][
            "expected_scoring_anchors"]
        for name, start, end, bars in n3f.BLOCKS:
            assert len(n3f.scoring_anchor_offsets(bars)) == \
                expected[name]

    def test_canonical_anchor_list_shape(self):
        canon = n3f.canonical_anchor_datetimes("B1_JanFeb")
        assert len(canon) == 86
        assert canon[0] == "2026-01-01 00:00:00"


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

    def test_representation_precedence(self):
        stats = self._stats(**{
            "bar_h6|arm2|arm1": self._good(0.02),
            "bar_h12|arm2|arm1": self._good(0.02),
            "bar_h12|arm5|arm2": self._good(0.006)})
        assert n3f.decide(stats, True, True) == \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"

    def test_not_confirmed_default(self):
        assert n3f.decide(self._stats(), True, True) == \
            "TARGET_SCALE_EFFECT_NOT_CONFIRMED"

    def test_insufficient_and_inconclusive(self):
        assert n3f.decide({}, False, True) == \
            "FRESH_CONFIRMATION_INSUFFICIENT"
        assert n3f.decide({}, True, False) == \
            "FRESH_CONFIRMATION_INCONCLUSIVE"


class TestOverlapExactness:

    def _frames(self, revise=None):
        base = 2345.123456789
        rows = [[1735689600000, str(base), str(base + 10),
                 str(base - 10), str(base + 1), "100.5",
                 1735703999999, "23451.2", 777, "50.25",
                 "11725.6", "0"]]
        if revise is not None:
            rows[0][4] = revise
        lake = pd.DataFrame({
            "open": [base], "high": [base + 10],
            "low": [base - 10], "close": [base + 1],
            "volume": [100.5], "quote_volume": [23451.2],
            "trade_count": [777],
            "taker_buy_base_volume": [50.25],
            "taker_buy_quote_volume": [11725.6]})
        return rows, lake, pd.Series([1735689600000])

    def test_sub_float32_revision_refuses(self):
        true_close = 2345.123456789 + 1
        revised = true_close * (1 + 3e-8)
        assert np.float32(revised) == np.float32(true_close)
        rows, lake, lake_ms = self._frames(revise=repr(revised))
        with pytest.raises(n3f.FreshRefusal, match="revised"):
            n3f._verify_overlap(rows, lake, lake_ms)


# ------------------------------------------------------------------ #
# synthetic v2 bundle + the verifier                                 #
# ------------------------------------------------------------------ #

def _mix(prior, y, eps, toward_truth=True):
    onehot = np.eye(3)[y]
    target = onehot if toward_truth else np.roll(onehot, 1, axis=1)
    p = (1 - eps) * prior[None, :] + eps * target
    return p / p.sum(axis=1, keepdims=True)


def _synthetic_bundle():
    rng = np.random.default_rng(21)
    hist = [900, 1000, 1500]
    prior = np.array(hist) / sum(hist)
    units = []
    for tkey in n3f.TARGETS:
        for name, start, end, bars in n3f.BLOCKS:
            anchors = n3f.canonical_anchor_datetimes(name)
            n_s = len(anchors)
            y = rng.integers(0, 3, size=n_s)
            arms = {
                "arm1": np.tile(prior, (n_s, 1)),
                "arm2": _mix(prior, y, 0.02),
                "arm3": _mix(prior, y, 0.02
                             + 0.001 * rng.standard_normal()),
                "arm4": _mix(prior, y, 0.03, toward_truth=False),
                "arm5": _mix(prior, y, 0.02, toward_truth=False)}
            payload = {
                "unit": f"{tkey}:{name}",
                "horizon": n3f.TARGETS[tkey], "block": name,
                "n_score": n_s,
                "anchor_datetimes": anchors,
                "fit_cal_label_histogram": hist,
                "labels": [int(v) for v in y],
                "class_support_score": {
                    str(c): int((y == c).sum())
                    for c in (0, 1, 2)},
                "arms": {a: {"record": {},
                             "probs": [[float(v) for v in row]
                                       for row in p],
                             "metrics": n3f.unit_metrics(y, p)}
                         for a, p in arms.items()}}
            payload["payload_sha256"] = sha_obj(payload)
            units.append(payload)
    contrasts, stats, complete = n3f._rederive(units)
    assert complete
    verdict = n3f.decide(stats, True, True)
    return {"schema": n3f.BUNDLE_SCHEMA_V2,
            "contract": n3f.CONTRACT,
            "contract_sha256": __import__(
                "agent_plugins.experiment_runtime",
                fromlist=["sha_file"]).sha_file(
                    REPO / n3f.CONTRACT),
            "role_ledger": n3f.role_ledger(),
            "digests": {"acquired_parquet": "0" * 64,
                        "model_ready_extended": "0" * 64,
                        "frozen_csv": n3f.FROZEN_SHA,
                        "lake_parquet": n3f.LAKE_SHA,
                        "code": n3f._code_digest()},
            "units": units, "contrasts": contrasts,
            "verdict": verdict, "elapsed_s": 1.0,
            "decision_constants": {}}


@pytest.fixture(scope="module")
def bundle():
    return _synthetic_bundle()


def _write(tmp_path, b):
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(b, default=float))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def _redigest(u):
    u["payload_sha256"] = sha_obj(
        {k: v for k, v in u.items() if k != "payload_sha256"})


class TestVerifierV2:

    def test_consistent_bundle_verifies_with_external_digest(
            self, bundle, tmp_path):
        p, sha = _write(tmp_path, bundle)
        out = n3f.verify(p, sha)
        assert out["verdict"] == "N3_BUNDLE_VERIFIED"
        assert out["external_digest_checked"] is True

    def test_no_external_digest_refused(self, bundle, tmp_path):
        p, _ = _write(tmp_path, bundle)
        with pytest.raises(n3f.FreshRefusal, match="EXTERNAL"):
            n3f.verify(p)

    def test_wrong_external_digest_refused(self, bundle, tmp_path):
        p, _ = _write(tmp_path, bundle)
        with pytest.raises(n3f.FreshRefusal, match="external"):
            n3f.verify(p, "ab" * 32)

    def test_internal_mode_cannot_publish_verified(self, bundle,
                                                   tmp_path):
        p, _ = _write(tmp_path, bundle)
        out = n3f.verify(p, internal_only=True)
        assert out["verdict"] == "N3_INTERNAL_CONSISTENCY_ONLY"

    # ---- the four Musashi forgeries, frozen as regressions ----

    def test_musashi_f1_zero_contract_sha(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["contract_sha256"] = "0" * 64
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal, match="contract"):
            n3f.verify(p, sha)

    def test_musashi_f2_blocks_complete_flag_is_not_authority(
            self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["blocks_complete"] = False
        b["verdict"] = "FRESH_CONFIRMATION_INSUFFICIENT"
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal):
            n3f.verify(p, sha)  # unknown field OR edited verdict

    def test_musashi_f3_absurd_unit_redigested(self, bundle,
                                               tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][0]
        u["n_score"] = 1
        u["class_support_score"] = {"0": 999, "1": 999, "2": 999}
        _redigest(u)
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal):
            n3f.verify(p, sha)

    def test_musashi_f4_coherent_fake_neural_passer(self, bundle,
                                                    tmp_path):
        """The decisive regression: a FULLY coherent forgery
        (probs, metrics, digests, contrasts and verdict all
        recomputed) is internally indistinguishable from truth.
        The EXTERNAL published digest — bound to the pushed Git
        blob — is the authority that rejects it BEFORE its
        scientific content is evaluated; and the internal-only
        mode, which cannot see the forgery, is structurally unable
        to publish N3_BUNDLE_VERIFIED."""
        _, published_sha = _write(tmp_path, bundle)
        b = copy.deepcopy(bundle)
        for u in b["units"]:
            if u["horizon"] == 6:
                y = np.asarray(u["labels"])
                fake = _mix(np.asarray(
                    u["fit_cal_label_histogram"], dtype=float)
                    / sum(u["fit_cal_label_histogram"]),
                    y, 0.05)
                u["arms"]["arm3"]["probs"] = [
                    [float(v) for v in row] for row in fake]
                u["arms"]["arm3"]["metrics"] = n3f.unit_metrics(
                    y, fake)
                _redigest(u)
        contrasts, stats, _ = n3f._rederive(b["units"])
        b["contrasts"] = contrasts
        b["verdict"] = n3f.decide(stats, True, True)
        assert b["verdict"] == \
            "INCREMENTAL_REPRESENTATION_CANDIDATE_ON_FRESH_DATA"
        forged = tmp_path / "forged.json"
        forged.write_text(json.dumps(b, default=float))
        # rejected at the byte gate, before parsing
        with pytest.raises(n3f.FreshRefusal,
                           match="external digest"):
            n3f.verify(forged, published_sha)
        # internal-only mode cannot mint publication authority
        out = n3f.verify(forged, internal_only=True)
        assert out["verdict"] == "N3_INTERNAL_CONSISTENCY_ONLY"
        assert out["verdict"] != "N3_BUNDLE_VERIFIED"

    # ---- earlier adversaries under v2 ----

    def test_anchor_forgery_fails_canonical_check(self, bundle,
                                                  tmp_path):
        b = copy.deepcopy(bundle)
        b["units"][0]["anchor_datetimes"] = \
            ["2025-12-30 04:00:00"] + \
            b["units"][0]["anchor_datetimes"][1:]
        _redigest(b["units"][0])
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal, match="canonical"):
            n3f.verify(p, sha)

    def test_moved_boundary_refused(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["role_ledger"]["blocks"]["B4_JulAug"] = [
            "2026-07-01 00:00", "2026-09-15 20:00", 372]
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal, match="boundary"):
            n3f.verify(p, sha)

    def test_metrics_must_derive_from_evidence(self, bundle,
                                               tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][2]
        u["arms"]["arm2"]["metrics"] = dict(
            u["arms"]["arm2"]["metrics"],
            multiclass_logloss_mean=0.001)
        _redigest(u)
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal, match="derive"):
            n3f.verify(p, sha)

    def test_support_derived_from_labels(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][1]
        u["class_support_score"] = {"0": 40, "1": 40, "2": 6}
        _redigest(u)
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal,
                           match="derive from the labels"):
            n3f.verify(p, sha)

    def test_prior_label_history_mismatch(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        u = b["units"][3]
        u["fit_cal_label_histogram"] = [100, 2000, 1300]
        _redigest(u)
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal,
                           match="label histories"):
            n3f.verify(p, sha)

    def test_missing_unit_refused(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["units"] = b["units"][:-1]
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal,
                           match="missing/extra"):
            n3f.verify(p, sha)

    def test_edited_verdict_refused(self, bundle, tmp_path):
        b = copy.deepcopy(bundle)
        b["verdict"] = "TARGET_SCALE_EFFECT_NOT_CONFIRMED" \
            if b["verdict"] != "TARGET_SCALE_EFFECT_NOT_CONFIRMED" \
            else "FRESH_CONFIRMATION_INCONCLUSIVE"
        p, sha = _write(tmp_path, b)
        with pytest.raises(n3f.FreshRefusal, match="edited"):
            n3f.verify(p, sha)


class TestUnitMetrics:

    def test_additive_identity_and_shapes(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 3, 40)
        p = rng.dirichlet([1, 1, 1], size=40)
        m = n3f.unit_metrics(y, p)
        assert m["additive_identity_max_abs_gap"] < 1e-9
        assert set(m["brier_components"]) == {"0", "1", "2"}

    def test_invalid_simplex_refused(self):
        y = np.array([0, 1])
        p = np.array([[0.5, 0.5, 0.5], [0.2, 0.3, 0.5]])
        with pytest.raises(n3f.FreshRefusal, match="simplex"):
            n3f.unit_metrics(y, p)

    def test_nan_probability_refused(self):
        y = np.array([0])
        p = np.array([[np.nan, 0.5, 0.5]])
        with pytest.raises(n3f.FreshRefusal, match="finite"):
            n3f.unit_metrics(y, p)

    def test_length_disagreement_refused(self):
        with pytest.raises(n3f.FreshRefusal, match="shape"):
            n3f.unit_metrics([0, 1], [[0.2, 0.3, 0.5]])

    def test_recall_typed_unavailable(self):
        y = np.array([0, 0, 1, 1])
        p = np.tile([0.4, 0.4, 0.2], (4, 1))
        m = n3f.unit_metrics(y, p)
        assert m["recall_argmax"]["2"] is None
        assert m["recall_unavailable_classes"] == [2]
