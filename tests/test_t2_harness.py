"""T2.1/T2.4 adversarial battery: leakage, bytes decoupling,
duplication, seed-as-unit, forbidden claims, cost, identity and
the confirmatory gate — frozen while the public bank is absent."""
import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
os.environ.setdefault(
    "B4_T1_PREPROCESSOR_ROOT",
    str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))

import t2_public_data_census as census_mod  # noqa: E402
import t2_assay_harness as hz  # noqa: E402

CENSUS = census_mod.build_census()
_dev_present = pytest.mark.skipif(
    not any(v["status"] == "PRESENT_DEVELOPMENT_ONLY"
            for v in CENSUS["development_only_units"].values()),
    reason="no development units in this environment")


def test_census_verdict_and_exclusions():
    assert CENSUS["verdict"] == "CONFIRMATORY_BANK_UNAVAILABLE"
    assert "financial" in CENSUS["excluded_classes"]
    assert "synthetic_own" in CENSUS["excluded_classes"]
    ded = CENSUS["deficit"]["operator_must_supply"]
    assert {"monash_subset", "etth1", "weather_jena"} <= set(ded)


@_dev_present
def test_bytes_decoupling_refuses():
    c = copy.deepcopy(CENSUS)
    c["development_only_units"]["sm_co2"]["bytes_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="decoupled"):
        hz.load_task_unit(c, "sm_co2")


def test_no_dataframe_entry_point():
    """A dataframe supplied apart from its bytes has no entry: the
    only loader takes (census, unit_id) and re-reads the bytes."""
    import inspect
    sig = inspect.signature(hz.load_task_unit)
    assert list(sig.parameters) == ["census", "unit_id"]
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "read_csv(io.BytesIO(raw))" in src


@_dev_present
def test_future_rows_cannot_reach_training(tmp_path):
    """Leakage adversary: mutating validation/score rows changes
    neither the train-role features of any arm nor the fitted
    causal transform on the train role."""
    co = hz.load_co()
    unit = hz.load_task_unit(CENSUS, "sm_co2")
    y = unit["y"]
    roles = hz.roles_of(len(y))
    lo_t, hi_t = roles["train"]
    y_mut = y.copy()
    y_mut[hi_t:] = 999.0
    den_a = hz.causal_denoise(co, y, roles, "sm_co2")
    den_b = hz.causal_denoise(co, y_mut, roles, "sm_co2")
    assert den_a["d"][:hi_t].tobytes() == \
        den_b["d"][:hi_t].tobytes()
    n_a = hz.nuisance(y, roles, "sm_co2|ewma")
    n_b = hz.nuisance(y_mut, roles, "sm_co2|ewma")
    assert all(a.tobytes() == b.tobytes()
               for a, b in zip(n_a, n_b))
    for series_pair in (([y], [y_mut]),
                        ([y, den_a["d"], y - den_a["d"]],
                         [y_mut, den_b["d"], y_mut - den_b["d"]])):
        Xa = hz._lag_matrix(series_pair[0], lo_t, hi_t, 1)
        Xb = hz._lag_matrix(series_pair[1], lo_t, hi_t, 1)
        assert Xa.tobytes() == Xb.tobytes()


@_dev_present
def test_record_schema_and_forbidden_claims():
    co = hz.load_co()
    unit = hz.load_task_unit(CENSUS, "sm_sunspots")
    rec = hz.assay_unit(co, unit)
    hz.check_record_schema(rec)
    # smuggled eligibility refuses
    bad = copy.deepcopy(rec)
    bad["publicly_eligible"] = True
    with pytest.raises(SystemExit, match="exact schema"):
        hz.check_record_schema(bad)
    # a noise/SNR claim refuses even inside nested content
    bad2 = copy.deepcopy(rec)
    bad2["rolling_origins"]["origin0"]["results"]["X"]["ridge"][
        "true_snr"] = 10.0
    bad2["record_sha256"] = hashlib.sha256(json.dumps(
        {k: bad2[k] for k in sorted(bad2)
         if k != "record_sha256"}, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    with pytest.raises(SystemExit, match="forbidden claim"):
        hz.check_record_schema(bad2)
    # per-phase cost omission refuses (C7)
    bad3 = copy.deepcopy(rec)
    bad3.pop("costs_by_phase")
    with pytest.raises(SystemExit, match="exact schema"):
        hz.check_record_schema(bad3)
    bad3b = copy.deepcopy(rec)
    for oc in bad3b["costs_by_phase"].values():
        for k in [k for k in oc if k.startswith("arm_")]:
            oc.pop(k)
    bad3b["record_sha256"] = hashlib.sha256(json.dumps(
        {k: bad3b[k] for k in sorted(bad3b)
         if k != "record_sha256"}, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    with pytest.raises(SystemExit, match="separated costs"):
        hz.check_record_schema(bad3b)
    # tampered record digest refuses
    bad4 = copy.deepcopy(rec)
    bad4["rolling_origins"]["origin0"]["results"]["X"]["ridge"][
        "mase_primary"] = 0.0
    with pytest.raises(SystemExit, match="does not re-derive"):
        hz.check_record_schema(bad4)
    # C4/C8: the SERIES is the primary unit; origins/seeds nested
    assert rec["series_is_the_primary_unit"] is True
    assert rec["origins_and_seeds_are_nested"] is True
    assert set(rec["claim_classes_only"]) == {
        "utility", "calibration", "extreme_preservation", "cost"}
    assert rec["operator"]["selection_source"] == \
        "T1_v4_record_LAB_CALIBRATED"
    o0 = rec["rolling_origins"]["origin0"]
    assert len(o0["operator_artifact_sha256"]) == 64
    assert o0["results"]["seasonal_naive"]["period_source"] == \
        "predeclared_design_constant"
    # C5: MASE primary present; raw MAE labeled per-series only
    m = o0["results"]["X"]["ridge"]
    assert "mase_primary" in m
    assert "mae_per_series_diagnostic" in m
    # C6: coverage AND width together
    assert "interval_coverage_train_q90" in m
    assert "interval_width_train_q90" in m


def test_confirmatory_gate_refuses_public_data_required(tmp_path):
    cpath = tmp_path / "census.json"
    cpath.write_text(json.dumps(CENSUS))
    rc = subprocess.run(
        [sys.executable, str(REPO / "tools/t2_assay_harness.py"),
         "--census", str(cpath),
         "--output", str(tmp_path / "out.json"),
         "--confirmatory"],
        capture_output=True, text=True,
        env={**os.environ})
    assert rc.returncode != 0
    # the gate is CLOSED at whichever stage the world is in:
    # no manifest -> PUBLIC_DATA_REQUIRED; manifest acquired ->
    # DESIGN_REQUIRED; draft sealed later -> DESIGN_REVIEW_REQUIRED.
    # In every stage: typed refusal, zero scores, no output file.
    out = rc.stderr + rc.stdout
    assert any(tok in out for tok in
               ("PUBLIC_DATA_REQUIRED", "DESIGN_REQUIRED",
                "DESIGN_REVIEW_REQUIRED"))
    assert not (tmp_path / "out.json").exists()


def test_task_duplication_refuses(tmp_path):
    c = copy.deepcopy(CENSUS)
    u = c["development_only_units"]
    if "sm_co2" in u and u["sm_co2"].get("bytes_sha256"):
        u["sm_dup"] = dict(u["sm_co2"])
        cpath = tmp_path / "census.json"
        cpath.write_text(json.dumps(c))
        rc = subprocess.run(
            [sys.executable,
             str(REPO / "tools/t2_assay_harness.py"),
             "--census", str(cpath),
             "--output", str(tmp_path / "out.json"),
             "--development-only"],
            capture_output=True, text=True, env={**os.environ})
        assert rc.returncode != 0
        assert "duplication" in rc.stderr + rc.stdout


def test_financial_and_synthetic_have_no_path():
    """Excluded classes carry no loadable unit and the census
    never lists them as consumable."""
    for uid in CENSUS["development_only_units"]:
        assert not uid.startswith(("fx_", "eth_", "synthetic_"))
    with pytest.raises(SystemExit, match="not in the census"):
        hz.load_task_unit(CENSUS, "eurusd_hour")


def test_no_authority_tokens_in_t2_tools():
    for tool in ("t2_public_data_census.py",
                 "t2_assay_harness.py"):
        text = (REPO / "tools" / tool).read_text()
        for token in ("PUBLICLY_ELIGIBLE\"", "REPRODUCED_UNDER",
                      '"AUTHORIZED"'):
            assert token not in text, (tool, token)


# ================= C1-C8 acceptance battery ========================

def test_c1_no_future_fill_and_strict_tokens():
    """C1: bfill is dead; malformed tokens refuse; changing the
    FIRST FUTURE observed value cannot change any earlier accepted
    value (frozen regression)."""
    import t2_bank as bank
    src = (REPO / "tools/t2_bank.py").read_text()
    hsrc = (REPO / "tools/t2_assay_harness.py").read_text()
    assert ".bfill(" not in src and ".bfill(" not in hsrc
    assert 'errors="coerce"' not in hsrc
    with pytest.raises(SystemExit, match="malformed numeric"):
        bank.parse_strict_numeric(["1.0", "12..5", "3"], "probe")
    y = np.array([np.nan, np.nan, 1.0, np.nan, 2.0, 3.0, 4.0])
    fixed = bank.apply_causal_missingness(y, "probe", 2)
    assert fixed["y"].tolist() == [1.0, 1.0, 2.0, 3.0, 4.0]
    assert fixed["missingness"]["leading_dropped"] == 2
    assert fixed["missingness"]["interior_filled"] == 1
    # frozen regression: mutate the first future observed value —
    # every earlier accepted value is byte-identical
    y2 = y.copy()
    cut = 5
    y2[cut:] = 999.0
    f2 = bank.apply_causal_missingness(y2, "probe", 2)
    n_prefix = cut - fixed["missingness"]["leading_dropped"]
    assert fixed["y"][:n_prefix].tobytes() == \
        f2["y"][:n_prefix].tobytes()
    # trailing missing refuses (forward fill has no later truth,
    # and backward fill is forbidden)
    with pytest.raises(SystemExit, match="trailing missing"):
        bank.apply_causal_missingness(
            np.array([1.0, 2.0, np.nan]), "probe", 2)
    # over-long interior run refuses
    with pytest.raises(SystemExit, match="exceeds the declared"):
        bank.apply_causal_missingness(
            np.array([1.0, np.nan, np.nan, np.nan, 2.0]),
            "probe", 2)


def test_c2_time_index_facts():
    import t2_bank as bank
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    facts = bank.check_time_index(ts, "probe", expected_step=1.0)
    assert facts["irregular_spacings"] == 0
    with pytest.raises(SystemExit, match="non-increasing"):
        bank.check_time_index(np.array([0.0, 2.0, 2.0, 3.0]),
                              "probe")
    with pytest.raises(SystemExit, match="differs from the "
                                         "declared frequency"):
        bank.check_time_index(np.array([0.0, 2.0, 4.0, 6.0]),
                              "probe", expected_step=1.0)
    # C2: dev units now carry parsed-or-mechanical provenance
    unit = hz.load_task_unit(CENSUS, "sm_sunspots")
    assert "time_index" in unit and "time_provenance" in unit


def test_c4_tsf_panel_and_dedup():
    import t2_bank as bank
    tsf = ("@relation test\n@frequency monthly\n"
           "@attribute series_name string\n@data\n"
           "s1:2020-01:" + ",".join(str(float(i % 7))
                                    for i in range(150)) + "\n"
           "s2:2020-01:" + ",".join(str(float(i % 7))
                                    for i in range(150)) + "\n"
           "s3:2020-01:" + ",".join(str(float(i % 5) + 1)
                                    for i in range(150)) + "\n")
    panel = bank.parse_tsf_bytes(tsf.encode(), "probe_panel")
    assert set(panel["series"]) == {"s1", "s2", "s3"}
    built = bank.build_series_units(
        panel, "probe_family", 12, "declared", min_length=100)
    # s2 duplicates s1 physically -> counted once
    assert "probe_panel::s1" in built["units"]
    assert "probe_panel::s2" not in built["units"]
    assert "DUPLICATE_OF" in built["excluded"]["probe_panel::s2"]
    assert "probe_panel::s3" in built["units"]
    # malformed .tsf refuses
    with pytest.raises(SystemExit, match="no @data"):
        bank.parse_tsf_bytes(b"@relation x\n", "bad")
    # deterministic subsampling by id hash, never by outcome
    ids = [f"s{i}" for i in range(200)]
    a1 = bank.deterministic_subsample(ids, 0.3, "salt1")
    a2 = bank.deterministic_subsample(ids, 0.3, "salt1")
    assert a1 == a2 and 30 < len(a1) < 90


def test_c5_fair_models_and_train_only_normalization():
    """C5: intercept + train-only standardization; the MASE
    denominator is train-defined (mutating score rows cannot change
    it); the MLP epoch rule uses the temporally FINAL slice of the
    fit rows only."""
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "_train_scaler" in src
    assert "never penalize the mean" in src
    y = np.cumsum(np.random.default_rng(3).normal(0, 1, 400)) + 50
    lo_t, hi_t = 0, 240
    d1 = hz._mase_denominator(y, lo_t, hi_t, 12)
    y_mut = y.copy()
    y_mut[hi_t:] = 9999.0
    d2 = hz._mase_denominator(y_mut, lo_t, hi_t, 12)
    assert d1 == d2
    # rolling origins are predeclared geometry
    origins = hz.unit_origins(400)
    assert len(origins) == hz.ROLLING_ORIGINS
    assert origins[0][0] == int(400 * hz.ORIGIN_BASE_FRAC)
    assert origins[-1][1] == 400


def test_c6_extremes_are_innovations_not_level():
    """C6: on a rising level series, 'late' must not become
    'extreme' — the extreme set follows train-scaled innovations."""
    rng = np.random.default_rng(5)
    n = 400
    trend = np.linspace(0, 100, n)
    noise = rng.normal(0, 1.0, n)
    y = trend + noise
    spikes = [300, 340, 370]
    for s in spikes:
        y[s] += 25.0
    lo_t, hi_t = 0, 240
    targets_idx = list(range(248, 399))
    mask, thresh = hz.train_innovation_extremes(
        y, lo_t, hi_t, targets_idx, 1)
    assert mask is not None
    flagged = [targets_idx[i] for i in range(len(mask))
               if mask[i]]
    # the injected innovation spikes (and their reversals) are
    # flagged; the merely-late smooth rows are NOT
    assert any(t in flagged for t in
               [s for s in spikes] + [s + 1 for s in spikes])
    late_smooth = [t for t in range(380, 395)
                   if all(abs(t - s) > 2 for s in spikes)]
    assert sum(1 for t in late_smooth if t in flagged) <= \
        len(late_smooth) // 3


def test_c3_gate_sequence_typed_refusals(tmp_path):
    """C3: the confirmatory path opens only through the ordered
    gates; each absence refuses with its own typed reason and NO
    score is ever computed in this order."""
    import t2_confirmatory as conf
    with pytest.raises(SystemExit, match="PUBLIC_DATA_REQUIRED"):
        conf.run_confirmatory(tmp_path / "none.json",
                              tmp_path / "d.json",
                              tmp_path / "l.json")
    manifest = {
        "schema": "agent_multi.t2_public_data_manifest.v1",
        "datasets": {"probe": {
            "logical_id": "probe", "family": "f1",
            "final_url": "https://example.org/x",
            "archival_record": "doi:10/x",
            "retrieved_at_utc": "2026-09-06T00:00:00Z",
            "byte_size": 10, "sha256": "a" * 64,
            "upstream_checksum": "UNAVAILABLE",
            "license_id": "cc-by-4.0",
            "license_text_sha256": "b" * 64,
            "citation": "x", "local_relpath": "x.tsf",
            "admission": "ADMISSIBLE"}}}
    mp_ = tmp_path / "manifest.json"
    mp_.write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="DESIGN_REQUIRED"):
        conf.run_confirmatory(mp_, tmp_path / "d.json",
                              tmp_path / "l.json")
    manifest_sha = hashlib.sha256(mp_.read_bytes()).hexdigest()
    design = {
        "schema": "agent_multi.t2_confirmatory_design.v1",
        "sealed_after_census_manifest_sha256": manifest_sha,
        "operator": conf.T1_ACCEPTED_OPERATOR,
        "task_population": {"series_ids": ["probe::s1"],
                            "families": ["f1"]},
        "role_geometry": {"origins": 3, "base_frac": 0.6},
        "primary_metric": "MASE_train_snaive",
        "practical_margin_mase": 0.02,
        "harm_margins": {"extreme_mase_ratio_max": 1.2,
                         "coverage_drop_max": 0.1},
        "precision_rule": {"min_series_per_family": 20,
                           "min_families": 4},
        "multiplicity_rule": {"alpha": 0.05,
                              "method": "bonferroni_by_family"},
        "missing_unit_rule": "typed refusal recorded; family "
                             "dropped below min support",
        "inconclusive_rule": "insufficient families or mixed "
                             "consistency",
        "resource_contract": {"cpu_nice": 15,
                              "max_rss_bytes": 8 << 30},
        "verifier_specification": "fresh-process re-parse of "
                                  "source bytes, splits, arms, "
                                  "metrics, costs, cardinality; "
                                  "non-authorizing label only"}
    body = {k: design[k] for k in sorted(design)}
    design["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    # note: design_review_record_sha256 is MISSING -> schema refuses
    dp = tmp_path / "design.json"
    dp.write_text(json.dumps(design))
    with pytest.raises(SystemExit, match="exact schema"):
        conf.run_confirmatory(mp_, dp, tmp_path / "l.json")
    design.pop("design_sha256")
    design["design_review_record_sha256"] = "0" * 64
    body = {k: design[k] for k in sorted(design)}
    design["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    dp.write_text(json.dumps(design))
    with pytest.raises(SystemExit,
                       match="DESIGN_REVIEW_REQUIRED"):
        conf.run_confirmatory(mp_, dp, tmp_path / "l.json")
    # ambiguous license can never be admissible
    m2 = json.loads(mp_.read_text())
    m2["datasets"]["probe"]["license_id"] = "AMBIGUOUS"
    mp2 = tmp_path / "m2.json"
    mp2.write_text(json.dumps(m2))
    with pytest.raises(SystemExit, match="ambiguous license"):
        conf.run_confirmatory(mp2, dp, tmp_path / "l.json")


def _mk_records(fam_deltas):
    """Synthetic C8 fixtures: fam_deltas maps family -> list of
    per-series MASE deltas (X - D)."""
    recs = []
    i = 0
    for fam, deltas in fam_deltas.items():
        for d in deltas:
            i += 1
            recs.append({
                "unit_id": f"u{i}", "family": fam,
                "rolling_origins": {"origin0": {"results": {
                    "X": {"ridge": {"mase_primary": 1.0},
                          "mlp_small": {}},
                    "D": {"ridge": {"mase_primary": 1.0 - d},
                          "mlp_small": {}}}}}})
    return recs


def test_c8_decision_rule_hierarchical():
    import t2_confirmatory as conf
    design = {"practical_margin_mase": 0.02,
              "precision_rule": {"min_series_per_family": 5,
                                 "min_families": 3},
              "multiplicity_rule": {"alpha": 0.05}}
    # too few families -> INCONCLUSIVE
    out = conf.adjudicate_confirmatory(
        _mk_records({"f1": [0.1] * 6}), design)
    assert out["verdict"] == "INCONCLUSIVE"
    # concentrated family harm kills a favorable grand average
    out = conf.adjudicate_confirmatory(_mk_records({
        "f1": [0.30] * 8, "f2": [0.30] * 8,
        "f3": [-0.10] * 8}), design)
    assert out["verdict"] == "PUBLICLY_INELIGIBLE"
    assert "f3" in out["reason"]
    # broad consistency with tight CIs -> eligible CANDIDATE
    out = conf.adjudicate_confirmatory(_mk_records({
        "f1": [0.10, 0.11, 0.09, 0.10, 0.12, 0.10],
        "f2": [0.08, 0.09, 0.10, 0.09, 0.08, 0.10],
        "f3": [0.11, 0.10, 0.09, 0.12, 0.10, 0.11]}), design)
    assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"
    # mixed consistency -> INCONCLUSIVE
    out = conf.adjudicate_confirmatory(_mk_records({
        "f1": [0.10] * 6, "f2": [0.10] * 6,
        "f3": [0.001, -0.001, 0.002, 0.0, 0.001, -0.002]}),
        design)
    assert out["verdict"] == "INCONCLUSIVE"


def test_old_pilot_relabeled():
    """The statsmodels pilot carries the ordered relabel and its
    gains carry no scientific authority."""
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "DEVELOPMENT_MECHANICS_ONLY_REQUIRES_" in src
    assert "DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_AUTHORITY" \
        not in src
