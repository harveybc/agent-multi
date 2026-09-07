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
    assert panel["encoding_used"] == "utf-8"
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
    # C11: EXACT top-k by id hash — order-independent, never
    # exceeds k, never by outcome
    ids = [f"s{i}" for i in range(200)]
    a1 = bank.deterministic_top_k(ids, 40, "salt1")
    a2 = bank.deterministic_top_k(list(reversed(ids)), 40,
                                  "salt1")
    assert a1 == a2 and len(a1) == 40
    assert bank.deterministic_top_k(ids[:7], 40, "s") == \
        sorted(ids[:7])


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
    """C3 (v2 world): the ordered gates refuse typed at every
    stage and no score path exists."""
    import t2_confirmatory as conf
    with pytest.raises(SystemExit, match="PUBLIC_DATA_REQUIRED"):
        conf.run_confirmatory(tmp_path / "none.json",
                              tmp_path / "d.json",
                              tmp_path / "l.json",
                              census_path=tmp_path / "c.json")
    m, raw = _mk_manifest(tmp_path)
    mp_ = tmp_path / "manifest.json"
    mp_.write_text(json.dumps(m))
    import unittest.mock as um
    with um.patch.object(conf, "validate_public_manifest",
                         lambda man, **k: {"probe": {}}):
        with pytest.raises(SystemExit, match="CENSUS_REQUIRED"):
            conf.run_confirmatory(mp_, tmp_path / "d.json",
                                  tmp_path / "l.json",
                                  census_path=tmp_path / "c.json")
        cp = tmp_path / "census.json"
        cp.write_text(json.dumps({"schema": "census"}))
        with pytest.raises(SystemExit, match="DESIGN_REQUIRED"):
            conf.run_confirmatory(mp_, tmp_path / "d.json",
                                  tmp_path / "l.json",
                                  census_path=cp)
    # ambiguous license can never be admissible (real validator)
    m2, raw2 = _mk_manifest(tmp_path / "amb")
    m2["datasets"]["probe"]["license_id"] = "AMBIGUOUS"
    m2["datasets"]["probe"]["license_id_sha256"] = \
        hashlib.sha256(b"AMBIGUOUS").hexdigest()
    with pytest.raises(SystemExit, match="concrete license"):
        conf.validate_public_manifest(m2, raw_root=raw2)


def test_c8_decision_rule_hierarchical():
    """C8 (compat name): the hierarchical rule under the COMPLETE
    C14 evidence model."""
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams, panels_per_family=3)
    good = [_rec_for(design, uid, delta=0.10)
            for uid in design["task_population"]["series_ids"]]
    out = conf.adjudicate_confirmatory(good, design)
    assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"
    # concentrated family harm defeats a favorable grand average
    mixed = [_rec_for(design, uid,
                      delta=(0.30 if not uid.startswith("f5")
                             else -0.10))
             for uid in design["task_population"]["series_ids"]]
    out = conf.adjudicate_confirmatory(mixed, design)
    assert out["verdict"] == "PUBLICLY_INELIGIBLE"
    assert "f5" in out["reason"]


def test_old_pilot_relabeled():
    """The statsmodels pilot carries the ordered relabel and its
    gains carry no scientific authority."""
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "DEVELOPMENT_MECHANICS_ONLY_REQUIRES_" in src
    assert "DEVELOPMENT_ONLY_ZERO_CONFIRMATORY_AUTHORITY" \
        not in src


# ================ C9-C16 acceptance battery ========================

def _mk_manifest(tmp_path, raw_root=None):
    import t2_confirmatory as conf
    raw_root = raw_root or tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    data = b"@frequency monthly\n@data\ns1:2020:" + \
        ",".join(str(float(i % 9)) for i in range(150)).encode() \
        + b"\n"
    f = raw_root / "probe.tsf"
    f.write_bytes(data)
    d = {"logical_id": "probe", "family": "f1",
         "final_url": "https://example.org/x",
         "archival_record": "doi:10/x",
         "record_metadata_sha256": hashlib.sha256(
             b"doi:10/x").hexdigest(),
         "retrieved_at_utc": "2026-09-06T00:00:00Z",
         "byte_size": len(data),
         "sha256": hashlib.sha256(data).hexdigest(),
         "upstream_checksum": "UNAVAILABLE",
         "license_id": "cc-by-4.0",
         "license_id_sha256": hashlib.sha256(
             b"cc-by-4.0").hexdigest(),
         "license_text_sha256": "UNAVAILABLE",
         "citation": "x", "local_relpath": "probe.tsf",
         "admission": "ADMISSIBLE"}
    return {"schema": "agent_multi.t2_public_data_manifest.v2",
            "acquired_at_utc": "2026-09-06T00:00:00Z",
            "remanifested_at_utc": "2026-09-06T00:00:00Z",
            "byte_cap": 2 * 1024 ** 3,
            "bytes_downloaded_total": len(data),
            "raw_root_note": "fixture",
            "etth1_disposition": "not present in fixture",
            "datasets": {"probe": d}}, raw_root


def test_c10_manifest_binds_physical_bytes(tmp_path):
    """C10 POST: non-hex digests, traversing paths, absent bytes,
    size and hash mismatches all refuse from the descriptor."""
    import t2_confirmatory as conf
    m, raw = _mk_manifest(tmp_path)
    assert "probe" in conf.validate_public_manifest(
        m, raw_root=raw)
    bad = copy.deepcopy(m)
    bad["datasets"]["probe"]["sha256"] = "Z" * 64
    with pytest.raises(SystemExit, match="canonical lowercase"):
        conf.validate_public_manifest(bad, raw_root=raw)
    bad = copy.deepcopy(m)
    bad["datasets"]["probe"]["local_relpath"] = "../../outside"
    with pytest.raises(SystemExit,
                       match="absolute or traversing"):
        conf.validate_public_manifest(bad, raw_root=raw)
    (raw / "probe.tsf").rename(raw / "gone.tsf")
    with pytest.raises(SystemExit, match="unopenable"):
        conf.validate_public_manifest(m, raw_root=raw)
    (raw / "gone.tsf").rename(raw / "probe.tsf")
    bad = copy.deepcopy(m)
    bad["datasets"]["probe"]["byte_size"] += 1
    with pytest.raises(SystemExit, match="size differs"):
        conf.validate_public_manifest(bad, raw_root=raw)
    data2 = (raw / "probe.tsf").read_bytes() + b"\n"
    (raw / "probe.tsf").write_bytes(data2)
    with pytest.raises(SystemExit, match="size differs"):
        conf.validate_public_manifest(m, raw_root=raw)
    # duplicate JSON keys refuse at parse
    mp_ = tmp_path / "dup.json"
    mp_.write_text('{"schema": "x", "schema": "y"}')
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        conf.strict_json_load(mp_, "probe")
    # the real remanifested v2 carries the truthful license names
    real = json.loads(
        (Path.home() / ".local/share/agent-multi/"
         "t2_public_data_manifest_20260906.json").read_text())
    for lid, dd in real["datasets"].items():
        assert "license_id_sha256" in dd
        if lid != "etth1":
            assert dd["license_text_sha256"] == "UNAVAILABLE"
    assert real["datasets"]["etth1"]["admission"] == \
        "EXCLUDED_FROM_T2_CONFIRMATORY"


def test_c14_duplicate_tsf_id_refuses():
    import t2_bank as bank
    tsf = ("@frequency monthly\n@data\n"
           "s1:2020:" + ",".join(str(float(i))
                                 for i in range(130)) + "\n"
           "s1:2020:" + ",".join(str(float(i + 500))
                                 for i in range(130)) + "\n")
    with pytest.raises(SystemExit, match="duplicate .tsf"):
        bank.parse_tsf_bytes(tsf.encode(), "dup_probe")
    # the retired truncation refuses loudly
    good = ("@frequency monthly\n@data\n"
            "s1:2020:" + ",".join(str(float(i))
                                  for i in range(130)) + "\n")
    with pytest.raises(SystemExit, match="retired"):
        bank.parse_tsf_bytes(good.encode(), "p", max_series=5)


def test_c11_panel_permutation_invariant_population():
    """C11 POST: permuting panel rows produces the SAME selected
    population and the count never exceeds k."""
    import t2_bank as bank
    rows = ["s%03d:2020:%s" % (i, ",".join(
        str(float((i * 7 + j) % 13)) for j in range(150)))
        for i in range(120)]
    head = "@frequency monthly\n@data\n"
    p1 = bank.parse_tsf_bytes(
        (head + "\n".join(rows)).encode(), "perm")
    import random
    rng = random.Random(7)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    p2 = bank.parse_tsf_bytes(
        (head + "\n".join(shuffled)).encode(), "perm")
    ids1 = bank.deterministic_top_k(p1["series"], 40, "salt")
    ids2 = bank.deterministic_top_k(p2["series"], 40, "salt")
    assert ids1 == ids2 and len(ids1) == 40


def test_c9_transplanted_or_fabricated_review_refuses(tmp_path,
                                                      monkeypatch):
    """C9 POST: the exact Musashi bypass — candidate-hashed bytes
    — dies; a transplanted/foreign-field record dies; and NO
    ledger artifact appears on any refused gate."""
    import t2_confirmatory as conf
    m, raw = _mk_manifest(tmp_path)
    mp_ = tmp_path / "manifest.json"
    mp_.write_text(json.dumps(m))
    cp = tmp_path / "census.json"
    cp.write_text(json.dumps({"schema": "census"}))
    monkeypatch.setattr(conf, "validate_public_manifest",
                        lambda man, **k: {"probe": {}})
    fake = tmp_path / "MUSASHI_T2_DESIGN_REVIEW_2026_09.json"
    fake.write_text("candidate says approved")
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", fake)
    design = {k: "x" for k in conf._DESIGN_KEYS}
    design["schema"] = "agent_multi.t2_confirmatory_design.v1"
    design["design_review_record_sha256"] = hashlib.sha256(
        b"candidate says approved").hexdigest()
    dp = tmp_path / "design.json"
    dp.write_text(json.dumps(design))
    lp = tmp_path / "ledger.json"
    with pytest.raises(SystemExit):
        conf.run_confirmatory(mp_, dp, lp, census_path=cp)
    assert not lp.exists()
    assert not (tmp_path / "ledger.json.INTENT").exists()
    # even with a schema-valid design, non-record bytes refuse on
    # strict parse; and a foreign reviewer refuses
    rec = {"schema": "agent_multi.musashi_t2_design_review.v1",
           "reviewed_at_date": "2026-09-06",
           "reviewer": "candidate-self-review",
           "decision": "SEAL_T2_CONFIRMATORY_DESIGN",
           "design_draft_sha256": "a" * 64,
           "manifest_sha256": "b" * 64,
           "census_sha256": "c" * 64}
    fake.write_text(json.dumps(rec))
    design2 = {"design_review_record_sha256":
               hashlib.sha256(fake.read_bytes()).hexdigest(),
               "supersedes_draft_sha256": "a" * 64}
    with pytest.raises(SystemExit, match="not the external "
                                         "reviewer"):
        conf.verify_design_review_record(design2, "b" * 64,
                                         "c" * 64)
    rec["reviewer"] = "General Musashi"
    rec["manifest_sha256"] = "f" * 64      # foreign binding
    fake.write_text(json.dumps(rec))
    design2["design_review_record_sha256"] = hashlib.sha256(
        fake.read_bytes()).hexdigest()
    with pytest.raises(SystemExit,
                       match="different public-data manifest"):
        conf.verify_design_review_record(design2, "b" * 64,
                                         "c" * 64)


def _complete_record(uid, fam, delta=0.10, seeds=(11, 12, 13),
                     origins=3, arms=("X", "D", "XDR",
                                      "width_control"),
                     wc_delta=0.0, ex_ratio=1.0,
                     cov_drop=0.0, width_ratio=1.0,
                     dataset=None):
    ro = {}
    costs = {}
    for i in range(origins):
        res = {}
        for arm in arms:
            base = 1.0
            if arm == "D":
                m = base - delta
            elif arm == "width_control":
                m = base - wc_delta
            elif arm == "XDR":
                m = base - delta * 0.8
            else:
                m = base
            entry = {"mase_primary": m,
                     "interval_coverage_train_q90":
                         0.9 - (cov_drop if arm == "D" else 0.0),
                     "interval_width_train_q90":
                         1.0 * (width_ratio if arm == "D"
                                else 1.0),
                     "mase_on_extreme_innovations":
                         m * (ex_ratio if arm == "D" else 1.0)}
            res[arm] = {"ridge": dict(entry),
                        "mlp_small": {f"seed{s}": dict(entry)
                                      for s in seeds}}
        res["seasonal_naive"] = {"metrics": {
            "mase_primary": 1.4,
            "interval_coverage_train_q90": 0.9,
            "interval_width_train_q90": 1.2}}
        ro[f"origin{i}"] = {"results": res}
        arm_cost = {"lag_features_s": 0.01,
                    "ridge_fit_forecast_s": 0.1}
        for s in seeds:
            arm_cost[f"mlp_fit_forecast_seed{s}_s"] = 0.2
        costs[f"origin{i}"] = {
            "denoise_fit_transform_s": 0.1,
            **{f"arm_{a}": dict(arm_cost) for a in
               ("X", "D", "XDR", "width_control")}}
    return {"unit_id": uid, "family": fam,
            "dataset": dataset or f"panel_{fam}",
            "series_numeric_sha256": "d" * 64,
            "seasonal_period": 12, "horizon": 1,
            "rolling_origins": ro, "costs_by_phase": costs}


def _design_v2_fixture(series_by_family, seeds=(11, 12, 13),
                       panels_per_family=1):
    ids = []
    unit_map = {}
    for fam, ns in series_by_family.items():
        for i in range(ns):
            uid = f"{fam}::s{i}"
            ids.append(uid)
            panel = (f"panel_{fam}" if panels_per_family == 1
                     else f"panel_{fam}_{i % panels_per_family}")
            unit_map[uid] = {"family": fam, "dataset": panel,
                             "series_numeric_sha256": "d" * 64,
                             "seasonal_period": 12, "horizon": 1}
    return {
        "task_population": {
            "series_ids": sorted(ids),
            "unit_map": unit_map,
            "primary_gate_families": list(series_by_family)[:6],
        },
        "role_geometry": {"rolling_origins": 3},
        "seed_tape": list(seeds),
        "practical_margin_mase": 0.02,
        "observed_precision_rule": {"max_ci_halfwidth": 0.05},
        "harm_margins": {"extreme_innovation_mase_ratio_max": 1.2,
                         "coverage_drop_max": 0.1,
                         "width_inflation_max": 1.5},
        "precision_rule": {"min_series_per_family": 5,
                           "min_families": 6},
        "multiplicity_rule": {"alpha": 0.05},
        "inference_method": {
            "rule": "panel_replication_or_descriptive"},
        "inference_scope": "named panels only"}


def _rec_for(design, uid, **kw):
    """Build a record consistent with the design's unit binding."""
    b = design["task_population"]["unit_map"][uid]
    return _complete_record(uid, b["family"],
                            dataset=b["dataset"], **kw)


def test_c14_incomplete_evidence_refuses():
    """C14 POST: the exact Musashi bypass — one origin, X/D only,
    ridge only, no costs — REFUSES instead of adjudicating."""
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams)
    # the bypass shape
    recs = [{"unit_id": f"f{i}::s{j}", "family": f"f{i}",
             "rolling_origins": {"origin0": {"results": {
                 "X": {"ridge": {"mase_primary": 1.0}},
                 "D": {"ridge": {"mase_primary": 0.9}}}}}}
            for i in range(6) for j in range(6)]
    with pytest.raises(SystemExit,
                       match="expected 3 rolling origins|costs"):
        conf.adjudicate_confirmatory(recs, design)
    # complete records with >=3 panels adjudicate; single panel
    # is INCONCLUSIVE by the C23 inference rule
    design3 = _design_v2_fixture(fams, panels_per_family=3)
    good = [_rec_for(design3, uid)
            for uid in design3["task_population"]["series_ids"]]
    out = conf.adjudicate_confirmatory(good, design3)
    assert out["verdict"] == "PUBLICLY_ELIGIBLE_CANDIDATE"
    single = [_rec_for(design, uid)
              for uid in design["task_population"]["series_ids"]]
    out1 = conf.adjudicate_confirmatory(single, design)
    assert out1["verdict"] == "INCONCLUSIVE"
    assert "unidentifiable" in out1["reason"] or \
        "panel" in out1["reason"]
    broken = [_rec_for(design3, uid, arms=("X", "D", "XDR"))
              for uid in design3["task_population"]["series_ids"]]
    with pytest.raises(SystemExit, match="width_control"):
        conf.adjudicate_confirmatory(broken, design3)
    # incomplete population refuses (missing unit never dropped)
    with pytest.raises(SystemExit,
                       match="differs from the sealed design"):
        conf.adjudicate_confirmatory(good[:-1], design3)
    # duplicate identity refuses
    with pytest.raises(SystemExit, match="duplicate unit"):
        conf.adjudicate_confirmatory(good + [good[0]], design3)


def test_c14_gates_bite_individually():
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams, panels_per_family=3)

    def build(**kw):
        return [_rec_for(design, uid, **kw)
                for uid in design["task_population"]["series_ids"]]
    # family absent -> INCONCLUSIVE (all six required)
    short = [r for r in build() if r["family"] != "f5"]
    design_short = _design_v2_fixture(fams, panels_per_family=3)
    design_short["task_population"]["series_ids"] = sorted(
        r["unit_id"] for r in short)
    out = conf.adjudicate_confirmatory(short, design_short)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "f5" in out["reason"]
    # unattributed gain (width control matches D) -> INCONCLUSIVE
    out = conf.adjudicate_confirmatory(
        build(wc_delta=0.10), design)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "attributable" in out["reason"]
    # extreme harm -> INELIGIBLE
    out = conf.adjudicate_confirmatory(
        build(ex_ratio=1.5), design)
    assert out["verdict"] == "PUBLICLY_INELIGIBLE"
    # coverage harm -> INELIGIBLE
    out = conf.adjudicate_confirmatory(
        build(cov_drop=0.2), design)
    assert out["verdict"] == "PUBLICLY_INELIGIBLE"
    # observed precision: high variance -> INCONCLUSIVE
    noisy = []
    import random
    rng = random.Random(3)
    for uid in design["task_population"]["series_ids"]:
        noisy.append(_rec_for(design, uid,
                              delta=rng.uniform(-0.15, 0.35)))
    out = conf.adjudicate_confirmatory(noisy, design)
    assert out["verdict"] in ("INCONCLUSIVE",
                              "PUBLICLY_INELIGIBLE")


def test_c15_ledger_durable_and_late(tmp_path):
    import t2_confirmatory as conf
    lp = tmp_path / "ledger.json"
    led = conf.open_attempt_ledger(lp)
    assert led["schema"] == "agent_multi.t2_attempt_ledger.v2"
    assert (tmp_path / "ledger.json.INTENT").exists()
    again = conf.open_attempt_ledger(lp)
    assert again["ledger_sha256"] == led["ledger_sha256"]
    # tampered ledger fails closed
    doc = json.loads(lp.read_text())
    doc["attempts"] = ["forged"]
    lp.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="does not re-derive"):
        conf.open_attempt_ledger(lp)
    # intent without ledger -> uncertain, operator disposition
    lp2 = tmp_path / "l2.json"
    (tmp_path / "l2.json.INTENT").write_text("{}")
    with pytest.raises(SystemExit, match="uncertain prior"):
        conf.open_attempt_ledger(lp2)


def test_c13_design_v2_truthful_and_structured():
    d = json.loads(
        (Path.home() / ".local/share/agent-multi/"
         "t2_confirmatory_design_DRAFT_V2_20260906.json"
         ).read_text())
    pr = d["precision_rule"]
    assert pr["min_series_per_family"] == 28
    assert "raised to 20" not in pr["note"]
    assert "IS this computed value" in pr["note"]
    assert set(d["arms"]) == {"X", "D", "XDR", "width_control"}
    assert d["primary_contrast"]["delta"] == "D_minus_X"
    assert d["seed_tape"] == [11, 12, 13]
    assert len(d["task_population"]["primary_gate_families"]) == 6
    assert "named public panels" in d["inference_scope"]
    assert d["sensitivity_rule"]["sd_grid_n_min"]["0.08"] == 112
    per = d["task_population"]["per_dataset"]
    for lid, meta in per.items():
        if meta["family"] in d["task_population"][
                "primary_gate_families"]:
            assert meta["n_series"] <= 40
    # every selected unit carries its numeric digest
    ud = d["task_population"]["unit_digests"]
    assert ud and all(len(v) == 64 for v in ud.values())


# ============= C17-C24 acceptance battery (ten kills) ==============

def _d3(fams=None, panels=3):
    return _design_v2_fixture(fams or {f"f{i}": 6
                                       for i in range(6)},
                              panels_per_family=panels)


def _recs(design, **kw):
    return [_rec_for(design, uid, **kw)
            for uid in design["task_population"]["series_ids"]]


def test_kill_1_nan_never_authorizes():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            o["results"]["D"]["ridge"]["mase_primary"] = \
                float("nan")
    with pytest.raises(SystemExit, match="not finite"):
        conf.adjudicate_confirmatory(recs, d)
    # inf / str / bool / negative each die with the field path
    for bad, msg in ((float("inf"), "not finite"),
                     ("0.9", "non-numeric"),
                     (True, "non-numeric"),
                     (-0.1, "nonnegative")):
        recs = _recs(d)
        recs[0]["rolling_origins"]["origin0"]["results"]["D"][
            "ridge"]["mase_primary"] = bad
        with pytest.raises(SystemExit, match=msg):
            conf.adjudicate_confirmatory(recs, d)
    # coverage outside [0,1] dies
    recs = _recs(d)
    recs[0]["rolling_origins"]["origin0"]["results"]["X"][
        "ridge"]["interval_coverage_train_q90"] = 1.4
    with pytest.raises(SystemExit, match="outside"):
        conf.adjudicate_confirmatory(recs, d)


def test_kill_2_family_relabel_refuses():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        i = int(r["family"][1])
        r["family"] = f"f{(i + 1) % 6}"
    with pytest.raises(SystemExit,
                       match="relabeled populations refuse"):
        conf.adjudicate_confirmatory(recs, d)
    # transplanted digest / dataset / period each refuse
    for field, val in (("series_numeric_sha256", "e" * 64),
                       ("dataset", "panel_alien"),
                       ("seasonal_period", 99)):
        recs = _recs(d)
        recs[0][field] = val
        with pytest.raises(SystemExit,
                           match="differs from the design "
                                 "binding"):
            conf.adjudicate_confirmatory(recs, d)


def test_kill_3_null_or_incomplete_costs_refuse():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["costs_by_phase"]:
            r["costs_by_phase"][o] = {"arm_X": None}
    with pytest.raises(SystemExit,
                       match="phase cost|arm costs"):
        conf.adjudicate_confirmatory(recs, d)
    # one missing MLP seed cost refuses
    recs = _recs(d)
    del recs[0]["costs_by_phase"]["origin0"]["arm_D"][
        "mlp_fit_forecast_seed12_s"]
    with pytest.raises(SystemExit, match="seed12 cost missing"):
        conf.adjudicate_confirmatory(recs, d)
    # a negative cost refuses
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"]["arm_X"][
        "ridge_fit_forecast_s"] = -1.0
    with pytest.raises(SystemExit, match="nonnegative"):
        conf.adjudicate_confirmatory(recs, d)


def test_kill_4_forged_mlp_payload_refuses():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            for arm in ("X", "D", "XDR", "width_control"):
                o["results"][arm]["mlp_small"] = {
                    f"seed{s}": "forged" for s in (11, 12, 13)}
    with pytest.raises(SystemExit, match="opaque payload"):
        conf.adjudicate_confirmatory(recs, d)
    # seasonal-naive baseline is validated too, never ignored
    recs = _recs(d)
    recs[0]["rolling_origins"]["origin0"]["results"][
        "seasonal_naive"] = {"metrics": "forged"}
    with pytest.raises(SystemExit, match="opaque payload"):
        conf.adjudicate_confirmatory(recs, d)


def test_kill_5_manifest_schema_identity_license(tmp_path):
    import t2_confirmatory as conf
    m, raw = _mk_manifest(tmp_path)
    # extra top-level key
    bad = copy.deepcopy(m)
    bad["smuggled"] = True
    with pytest.raises(SystemExit, match="top-level keys"):
        conf.validate_public_manifest(bad, raw_root=raw)
    # mapping key decoupled from logical_id
    bad = copy.deepcopy(m)
    bad["datasets"]["renamed"] = bad["datasets"].pop("probe")
    with pytest.raises(SystemExit, match="decoupled identity"):
        conf.validate_public_manifest(bad, raw_root=raw)
    # canonical-but-wrong license digest
    bad = copy.deepcopy(m)
    bad["datasets"]["probe"]["license_id_sha256"] = \
        hashlib.sha256(b"other bytes").hexdigest()
    with pytest.raises(SystemExit, match="does not re-derive "
                                         "from the license"):
        conf.validate_public_manifest(bad, raw_root=raw)
    # record metadata digest must re-derive too
    bad = copy.deepcopy(m)
    bad["datasets"]["probe"]["record_metadata_sha256"] = "a" * 64
    with pytest.raises(SystemExit,
                       match="record_metadata_sha256"):
        conf.validate_public_manifest(bad, raw_root=raw)


def test_kill_6_internal_symlink_refuses(tmp_path):
    import t2_confirmatory as conf
    m, raw = _mk_manifest(tmp_path)
    real = raw / "probe.tsf"
    os.symlink("probe.tsf", raw / "link.tsf")
    m["datasets"]["probe"]["local_relpath"] = "link.tsf"
    with pytest.raises(SystemExit,
                       match="symlinks refuse"):
        conf.validate_public_manifest(m, raw_root=raw)
    # a symlinked INTERMEDIATE directory inside the root also
    # refuses at its own component via the openat walk
    (raw / "realdir").mkdir()
    data2 = real.read_bytes()
    (raw / "realdir" / "f.tsf").write_bytes(data2)
    os.symlink("realdir", raw / "dirlink")
    m["datasets"]["probe"]["local_relpath"] = "dirlink/f.tsf"
    with pytest.raises(SystemExit, match="symlinks refuse"):
        conf.validate_public_manifest(m, raw_root=raw)
    # and one pointing OUTSIDE the root dies on containment first
    (tmp_path / "outside_dir").mkdir()
    (tmp_path / "outside_dir" / "f.tsf").write_bytes(data2)
    os.symlink(tmp_path / "outside_dir", raw / "extlink")
    m["datasets"]["probe"]["local_relpath"] = "extlink/f.tsf"
    with pytest.raises(SystemExit,
                       match="escapes the raw root|symlinks "
                             "refuse"):
        conf.validate_public_manifest(m, raw_root=raw)


def test_kill_7_census_not_derivable_refuses(tmp_path):
    """A coherent census whose units do not re-derive from the
    manifested bytes refuses in the fresh verifier."""
    import t2_fresh_verifier as fv
    import t2_confirmatory as conf
    from pathlib import Path as _P
    S = _P.home() / ".local/share/agent-multi"
    manifest = conf.strict_json_load(
        S / "t2_public_data_manifest_20260906.json", "m")
    rebuilt = fv.rebuild_population(
        manifest, S / "t2_public_raw")
    census = conf.strict_json_load(
        S / "t2_bank_census_20260906.json", "c")
    fv.verify_census_semantic(rebuilt, census)   # honest passes
    forged = json.loads(json.dumps(census))
    forged["population"]["weather"][
        "unit_numeric_digests"] = {
        k: "f" * 64 for k in
        forged["population"]["weather"][
            "unit_numeric_digests"]}
    with pytest.raises(SystemExit,
                       match="do not re-derive"):
        fv.verify_census_semantic(rebuilt, forged)
    forged2 = json.loads(json.dumps(census))
    forged2["population"]["weather"][
        "admissible_unit_ids"].append("weather::sX")
    with pytest.raises(SystemExit,
                       match="does not re-derive"):
        fv.verify_census_semantic(rebuilt, forged2)


def test_kill_8_family_cap_is_global():
    import t2_bank as bank
    ids_a = [f"pa::s{i}" for i in range(60)]
    ids_b = [f"pb::s{i}" for i in range(60)]
    sel = bank.family_top_k({"pa": ids_a, "pb": ids_b}, 40,
                            "t2_design_v3")
    assert len(sel) == 40                    # never 80
    # permutation of datasets/series cannot change the selection
    sel2 = bank.family_top_k(
        {"pb": list(reversed(ids_b)),
         "pa": list(reversed(ids_a))}, 40, "t2_design_v3")
    assert sel == sel2
    # duplicate global ids across datasets refuse
    with pytest.raises(SystemExit, match="duplicate global"):
        bank.family_top_k({"pa": ids_a, "pb": ids_a}, 40, "s")


def test_kill_9_design_duplicates_and_bool():
    import t2_confirmatory as conf
    with pytest.raises(SystemExit, match="duplicated entries"):
        conf._unique_list([11, 11, 12], "probe", elem_type=int)
    with pytest.raises(SystemExit, match="never a number"):
        conf._unique_list([11, True, 12], "probe", elem_type=int)


def test_kill_10_no_fabricated_precision():
    """C23: with a single panel per family the primary gate is
    INCONCLUSIVE by construction; with >=3 panels the t-based
    panel CI covers (committed simulation)."""
    import t2_confirmatory as conf
    d1 = _d3(panels=1)
    out = conf.adjudicate_confirmatory(_recs(d1), d1)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "unidentifiable" in out["reason"] or \
        "panel" in out["reason"]
    for st in out["families"].values():
        assert "inferential" not in st.get("ci_class", "")
    d2 = _d3(panels=2)
    out2 = conf.adjudicate_confirmatory(_recs(d2), d2)
    assert out2["verdict"] == "INCONCLUSIVE"
    sim = json.loads(
        (Path.home() / ".local/share/agent-multi/"
         "t2_coverage_sim_20260906.json").read_text())
    for row in sim["rows"]:
        assert row["panel_level_coverage"] >= 0.93
        if row["icc_true"] > 0:
            assert row["naive_single_panel_coverage"] < 0.6
            assert row["within_panel_icc_coverage"] < 0.6


def test_fresh_verifier_live_population():
    """C21 live: the fresh verifier rebuilds 4650 units from
    physical bytes and reproduces the v3 design population."""
    import subprocess
    S = Path.home() / ".local/share/agent-multi"
    rc = subprocess.run(
        [sys.executable, str(REPO / "tools/t2_fresh_verifier.py"),
         "--manifest",
         str(S / "t2_public_data_manifest_20260906.json"),
         "--census", str(S / "t2_bank_census_20260906.json"),
         "--design",
         str(S / "t2_confirmatory_design_DRAFT_V3_20260906.json")],
        capture_output=True, text=True)
    assert rc.returncode == 3
    out = json.loads(rc.stdout)
    assert out["fresh_verification"] == \
        "POPULATION_REDERIVED_NON_AUTHORIZING"
    assert out["units_rederived"] == 4650
