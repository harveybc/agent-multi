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
                "SEALED_DESIGN_REQUIRED",
                "DESIGN_REVIEW_REQUIRED",
                "T2_EXECUTION_RECORD_REQUIRED"))
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
    """C8 (compat name) under the C29 SCREEN: six panels, panel-
    level composite rule; concentrated panel harm defeats a
    favorable grand average; the confirmatory adjudicator is a
    typed refusal."""
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams)
    good = [_rec_for(design, uid, delta=0.10)
            for uid in design["task_population"]["series_ids"]]
    out = conf.adjudicate_screen(good, design)
    assert out["verdict"] == "ADVANCE_TO_DOMAIN_VALIDATION"
    assert "PUBLICLY_ELIGIBLE" not in json.dumps(out)
    # concentrated panel harm defeats a favorable grand average
    mixed = [_rec_for(design, uid,
                      delta=(0.30 if not uid.startswith("f5")
                             else -0.10))
             for uid in design["task_population"]["series_ids"]]
    out = conf.adjudicate_screen(mixed, design)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    assert "panel_f5" in out["reason"]
    # the superseded confirmatory outcome no longer exists
    with pytest.raises(SystemExit, match="superseded by"):
        conf.adjudicate_confirmatory(good, design)


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
    design = {k: "x" for k in conf._DESIGN_KEYS}
    design["schema"] = "agent_multi.t2_confirmatory_design.v1"
    dp = tmp_path / "design.json"
    dp.write_text(json.dumps(design))
    lp = tmp_path / "ledger.json"
    with pytest.raises(SystemExit):
        conf.run_confirmatory(mp_, dp, lp, census_path=cp)
    assert not lp.exists()
    assert not (tmp_path / "ledger.json.INTENT").exists()
    # C38: malformed external bytes refuse on strict parse; a
    # foreign reviewer refuses; a foreign binding refuses — all
    # through the PRIVATE custody walk
    ra = _t2_private_chain(tmp_path / "auth")
    fake = ra / "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json"
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", fake)
    fake.write_text("candidate says approved")
    os.chmod(fake, 0o600)
    with pytest.raises(SystemExit, match="never a record"):
        conf.verify_design_review_record(
            {"design_review_record_sha256": "0" * 64},
            "b" * 64, "c" * 64)
    rec = _t2_record_doc(reviewer="candidate-self-review")
    fake.write_text(json.dumps(rec))
    os.chmod(fake, 0o600)
    design2 = {"design_review_record_sha256":
               hashlib.sha256(fake.read_bytes()).hexdigest(),
               "supersedes_draft_sha256":
                   conf.T2_V6_DRAFT_FILE_SHA}
    with pytest.raises(SystemExit, match="not the external "
                                         "reviewer role"):
        conf.verify_design_review_record(
            design2, rec["manifest_sha256"],
            rec["census_sha256"])
    rec = _t2_record_doc(manifest_sha256="f" * 64)
    fake.write_text(json.dumps(rec))
    os.chmod(fake, 0o600)
    design2["design_review_record_sha256"] = hashlib.sha256(
        fake.read_bytes()).hexdigest()
    with pytest.raises(SystemExit,
                       match="different public-data manifest"):
        conf.verify_design_review_record(
            design2, "b" * 64, rec["census_sha256"])


def _t2_private_chain(base):
    am = base / "agent-multi"
    ra = am / "reviewer_authority"
    for d in (base, am, ra):
        d.mkdir(mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
    return ra


def _t2_record_doc(**over):
    import t2_confirmatory as conf
    S = Path.home() / ".local/share/agent-multi"
    rec = {"schema": "agent_multi.musashi_t2_design_review.v2",
           "reviewed_at_date": "2026-09-07",
           "reviewer": "General Musashi",
           "decision": "SEAL_T2_CONFIRMATORY_DESIGN",
           "candidate_commit": conf.T2_V6_ACCEPTED_AT_COMMIT,
           "design_draft_file_sha256": conf.T2_V6_DRAFT_FILE_SHA,
           "design_draft_self_sha256": conf.T2_V6_DRAFT_SELF_SHA,
           "manifest_sha256": conf._sha_file(
               S / "t2_public_data_manifest_20260906.json"),
           "census_sha256": conf._sha_file(
               S / "t2_bank_census_20260906.json")}
    rec.update(over)
    return rec


_FIX_OPERATOR = {"kind": "ewma", "params": {"alpha": 0.3},
                 "selection_source":
                     "T1_v4_record_LAB_CALIBRATED"}
_FIX_N_OBS = 150


def _fix_windows():
    import t2_bank as bank
    # C35: the COMMON two-origin geometry
    return bank.origin_windows_for(_FIX_N_OBS, 2, 0.6,
                                   seasonal_period=12)


def _stamp(rec):
    """Recompute record_sha256 the way the harness does — the
    adversary CAN restamp; refusals must be SEMANTIC."""
    body = {k: rec[k] for k in sorted(rec)
            if k != "record_sha256"}
    rec["record_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    return rec


def _complete_record(uid, fam, delta=0.10, seeds=(11, 12, 13),
                     arms=("X", "D", "XDR", "width_control"),
                     wc_delta=0.0, ex_ratio=1.0,
                     cov_drop=0.0, width_ratio=1.0,
                     dataset=None, windows=None, ex_support=4):
    """C26: the FULL 21-key outer record the harness emits —
    exact schema, bound geometry, extreme support, restamped
    record_sha256."""
    windows = windows or _fix_windows()
    ro = {}
    costs = {}
    for okey, wb in windows.items():
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
                         m * (ex_ratio if arm == "D" else 1.0),
                     "extreme_support": ex_support}
            res[arm] = {"ridge": dict(entry),
                        "mlp_small": {f"seed{s}": dict(entry)
                                      for s in seeds}}
        res["seasonal_naive"] = {"metrics": {
            "mase_primary": 1.4,
            "interval_coverage_train_q90": 0.9,
            "interval_width_train_q90": 1.2}}
        ro[okey] = {"train": list(wb["train"]),
                    "score": list(wb["score"]),
                    "mase_denominator_train_snaive": 0.5,
                    "extreme_innovation_threshold_train": 1.0,
                    "operator_artifact_sha256": "a" * 64,
                    "results": res}
        arm_cost = {"lag_features_s": 0.01,
                    "ridge_fit_forecast_s": 0.1}
        for s in seeds:
            arm_cost[f"mlp_fit_forecast_seed{s}_s"] = 0.2
        costs[okey] = {
            "denoise_fit_transform_s": 0.1,
            "target_construction_s": 0.02,
            "seasonal_naive_s": 0.01,
            **{f"arm_{a}": dict(arm_cost) for a in
               ("X", "D", "XDR", "width_control")}}
    rec = {"schema": "agent_multi.t2_assay_record.v3",
           "authority": "DEVELOPMENT_MECHANICS_ONLY_REQUIRES_"
                        "C1_C8_CORRECTION_CLEARED",
           "unit_id": uid, "family": fam,
           "dataset": dataset or f"panel_{fam}",
           "series_numeric_sha256": "d" * 64,
           "bytes_sha256": "b" * 64,
           "license_note": "cc-by-4.0",
           "missingness": {"n_missing": 0},
           "time_index": {"n": _FIX_N_OBS},
           "time_provenance":
               "ordinal_reconstructed_from_declared_frequency",
           "horizon": 1, "seasonal_period": 12,
           "seasonal_period_provenance":
               "monash_record_frequency",
           "operator": dict(_FIX_OPERATOR),
           "seed_tape": list(seeds),
           "series_is_the_primary_unit": True,
           "origins_and_seeds_are_nested": True,
           "claim_classes_only": ["utility", "calibration",
                                  "extreme_preservation", "cost"],
           "rolling_origins": ro, "costs_by_phase": costs,
           "peak_rss_bytes": 1 << 20}
    return _stamp(rec)


def _design_v2_fixture(series_by_family, seeds=(11, 12, 13),
                       panels_per_family=1,
                       min_series_per_panel=5):
    """v4-shaped SCREEN design fixture: six named panels, unit_map
    with bound origin windows, austere screen rule."""
    ids = []
    unit_map = {}
    windows = _fix_windows()
    for fam, ns in series_by_family.items():
        for i in range(ns):
            uid = f"{fam}::s{i}"
            ids.append(uid)
            panel = (f"panel_{fam}" if panels_per_family == 1
                     else f"panel_{fam}_{i % panels_per_family}")
            unit_map[uid] = {"family": fam, "dataset": panel,
                             "series_numeric_sha256": "d" * 64,
                             "seasonal_period": 12, "horizon": 1,
                             "n_obs": _FIX_N_OBS,
                             "time_identity_sha256": "0" * 64,
                             "origin_windows": windows}
    panels = sorted({b["dataset"] for b in unit_map.values()
                     if b["family"] in
                     list(series_by_family)[:6]})
    return {
        "schema": "agent_multi.t2_screen_design.v4_draft",
        "operator": dict(_FIX_OPERATOR),
        "task_population": {
            "series_ids": sorted(ids),
            "unit_map": unit_map,
            "primary_gate_families": list(series_by_family)[:6],
            "screen_panels": panels,
        },
        "role_geometry": {"rolling_origins": 2,
                          "origin_base_frac": 0.6,
                          "lags": 8, "horizon": 1},
        "extreme_support_rule": {
            "min_evaluable_series_absolute": 2,
            "min_evaluable_fraction": 0.25},
        "seed_tape": list(seeds),
        "practical_margin_mase": 0.02,
        "observed_precision_rule": {"max_ci_halfwidth": 0.05},
        "harm_margins": {"extreme_innovation_mase_ratio_max": 1.2,
                         "coverage_drop_max": 0.1,
                         "width_inflation_max": 1.5,
                         "non_inferiority_margin_mase": 0.02},
        "precision_rule": {
            "min_series_per_panel": min_series_per_panel,
            "min_panels": 6},
        "multiplicity_rule": {"alpha": 0.05},
        "inference_method": {
            "rule": "six_panel_screen_t_sign_lopo"},
        "inference_scope": "the six named panels only"}


def _rec_for(design, uid, **kw):
    """Build a record consistent with the design's unit binding."""
    b = design["task_population"]["unit_map"][uid]
    return _complete_record(uid, b["family"],
                            dataset=b["dataset"],
                            windows=b["origin_windows"], **kw)


def test_c14_incomplete_evidence_refuses():
    """C14 POST under C26: the exact Musashi bypass — a bare
    inner-only record — REFUSES at the OUTER schema instead of
    adjudicating anything."""
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams)
    # the bypass shape: inner fragments with no outer identity
    recs = [{"unit_id": f"f{i}::s{j}", "family": f"f{i}",
             "rolling_origins": {"origin0": {"results": {
                 "X": {"ridge": {"mase_primary": 1.0}},
                 "D": {"ridge": {"mase_primary": 0.9}}}}}}
            for i in range(6) for j in range(6)]
    with pytest.raises(SystemExit,
                       match="outer keys are not the exact"):
        conf.adjudicate_screen(recs, design)
    good = [_rec_for(design, uid)
            for uid in design["task_population"]["series_ids"]]
    out = conf.adjudicate_screen(good, design)
    assert out["verdict"] == "ADVANCE_TO_DOMAIN_VALIDATION"
    broken = [_rec_for(design, uid, arms=("X", "D", "XDR"))
              for uid in design["task_population"]["series_ids"]]
    with pytest.raises(SystemExit, match="width_control"):
        conf.adjudicate_screen(broken, design)
    # incomplete population refuses (missing unit never dropped)
    with pytest.raises(SystemExit,
                       match="differs from the sealed design"):
        conf.adjudicate_screen(good[:-1], design)
    # duplicate identity refuses
    with pytest.raises(SystemExit, match="duplicate unit"):
        conf.adjudicate_screen(good + [good[0]], design)


def test_c14_gates_bite_individually():
    import t2_confirmatory as conf
    fams = {f"f{i}": 6 for i in range(6)}
    design = _design_v2_fixture(fams)

    def build(**kw):
        return [_rec_for(design, uid, **kw)
                for uid in design["task_population"]["series_ids"]]
    # a panel below support -> INCONCLUSIVE (all six required)
    short = [r for r in build() if r["family"] != "f5"]
    design_short = _design_v2_fixture(fams)
    design_short["task_population"]["series_ids"] = sorted(
        r["unit_id"] for r in short)
    out = conf.adjudicate_screen(short, design_short)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "panel_f5" in out["reason"]
    # unattributed gain (width control matches D) -> INCONCLUSIVE
    out = conf.adjudicate_screen(build(wc_delta=0.10), design)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "attributable" in out["reason"]
    # extreme harm -> DOES_NOT_ADVANCE
    out = conf.adjudicate_screen(build(ex_ratio=1.5), design)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    # coverage harm -> DOES_NOT_ADVANCE
    out = conf.adjudicate_screen(build(cov_drop=0.2), design)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    # observed precision: high variance can never ADVANCE
    noisy = []
    import random
    rng = random.Random(3)
    for uid in design["task_population"]["series_ids"]:
        noisy.append(_rec_for(design, uid,
                              delta=rng.uniform(-0.15, 0.35)))
    out = conf.adjudicate_screen(noisy, design)
    assert out["verdict"] in ("INCONCLUSIVE", "DOES_NOT_ADVANCE")


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

def _d3(fams=None):
    """Six-panel screen design fixture (one panel per family)."""
    return _design_v2_fixture(fams or {f"f{i}": 6
                                       for i in range(6)})


def _recs(design, **kw):
    return [_rec_for(design, uid, **kw)
            for uid in design["task_population"]["series_ids"]]


def test_kill_1_nan_never_authorizes():
    import t2_confirmatory as conf
    d = _d3()
    # NaN/inf cannot even be restamped (allow_nan=False): the
    # record has NO valid identity digest and refuses TYPED
    for bad in (float("inf"), float("nan")):
        recs = _recs(d)
        recs[0]["rolling_origins"]["origin0"]["results"]["D"][
            "ridge"]["mase_primary"] = bad
        with pytest.raises(SystemExit,
                           match="non-finite numbers"):
            conf.adjudicate_screen(recs, d)
    # str / bool / negative each die with the field path
    for bad, msg in (("0.9", "non-numeric"),
                     (True, "non-numeric"),
                     (-0.1, "nonnegative")):
        recs = _recs(d)
        recs[0]["rolling_origins"]["origin0"]["results"]["D"][
            "ridge"]["mase_primary"] = bad
        _stamp(recs[0])
        with pytest.raises(SystemExit, match=msg):
            conf.adjudicate_screen(recs, d)
    # coverage outside [0,1] dies
    recs = _recs(d)
    recs[0]["rolling_origins"]["origin0"]["results"]["X"][
        "ridge"]["interval_coverage_train_q90"] = 1.4
    _stamp(recs[0])
    with pytest.raises(SystemExit, match="outside"):
        conf.adjudicate_screen(recs, d)
    # an UNRESTAMPED mutation dies even earlier, at identity
    recs = _recs(d)
    recs[0]["rolling_origins"]["origin0"]["results"]["X"][
        "ridge"]["mase_primary"] = 0.5
    with pytest.raises(SystemExit,
                       match="record_sha256 does not recompute"):
        conf.adjudicate_screen(recs, d)


def test_kill_2_family_relabel_refuses():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        i = int(r["family"][1])
        r["family"] = f"f{(i + 1) % 6}"
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="relabeled populations refuse"):
        conf.adjudicate_screen(recs, d)
    # transplanted digest / dataset / period each refuse — even
    # RESTAMPED (semantic binding, not only hash identity)
    for field, val in (("series_numeric_sha256", "e" * 64),
                       ("dataset", "panel_alien"),
                       ("seasonal_period", 99)):
        recs = _recs(d)
        recs[0][field] = val
        _stamp(recs[0])
        with pytest.raises(SystemExit,
                           match="differs from the design "
                                 "binding"):
            conf.adjudicate_screen(recs, d)


def test_kill_3_null_or_incomplete_costs_refuse():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["costs_by_phase"]:
            r["costs_by_phase"][o] = {"arm_X": None}
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)
    # one missing MLP seed cost refuses (exact per-arm set)
    recs = _recs(d)
    del recs[0]["costs_by_phase"]["origin0"]["arm_D"][
        "mlp_fit_forecast_seed12_s"]
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="arm cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)
    # a negative cost refuses
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"]["arm_X"][
        "ridge_fit_forecast_s"] = -1.0
    _stamp(recs[0])
    with pytest.raises(SystemExit, match="nonnegative"):
        conf.adjudicate_screen(recs, d)


def test_kill_4_forged_mlp_payload_refuses():
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            for arm in ("X", "D", "XDR", "width_control"):
                o["results"][arm]["mlp_small"] = {
                    f"seed{s}": "forged" for s in (11, 12, 13)}
        _stamp(r)
    with pytest.raises(SystemExit, match="opaque payload"):
        conf.adjudicate_screen(recs, d)
    # seasonal-naive baseline is validated too, never ignored
    recs = _recs(d)
    recs[0]["rolling_origins"]["origin0"]["results"][
        "seasonal_naive"] = {"metrics": "forged"}
    _stamp(recs[0])
    with pytest.raises(SystemExit, match="opaque payload"):
        conf.adjudicate_screen(recs, d)


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
    """C23→C29: series counts never become panel-level precision;
    the committed simulations ground both the panel-level t rule
    and the composite screen rule's operating characteristics."""
    sim = json.loads(
        (Path.home() / ".local/share/agent-multi/"
         "t2_coverage_sim_20260906.json").read_text())
    for row in sim["rows"]:
        assert row["panel_level_coverage"] >= 0.93
        if row["icc_true"] > 0:
            assert row["naive_single_panel_coverage"] < 0.6
            assert row["within_panel_icc_coverage"] < 0.6


def test_fresh_verifier_live_population():
    """C21/C28/C35 live: the fresh verifier rebuilds 4650 units
    from physical bytes and reproduces the v5 SCREEN population
    (242 series — hospital a full member under the common
    two-origin geometry) with every unit_map field re-derived."""
    import subprocess
    S = Path.home() / ".local/share/agent-multi"
    rc = subprocess.run(
        [sys.executable, str(REPO / "tools/t2_fresh_verifier.py"),
         "--manifest",
         str(S / "t2_public_data_manifest_20260906.json"),
         "--census", str(S / "t2_bank_census_20260906.json"),
         "--design",
         str(S / "t2_screen_design_DRAFT_V6_20260907.json")],
        capture_output=True, text=True)
    assert rc.returncode == 3, rc.stderr[-500:]
    out = json.loads(rc.stdout)
    assert out["fresh_verification"] == \
        "POPULATION_REDERIVED_NON_AUTHORIZING"
    assert out["units_rederived"] == 4650
    assert out["design_series"] == 242


# ========== C25-C30 acceptance battery (the ten kills) =============


def test_c30_kill_1_missing_extremes_refuse():
    """C25a: extreme_support > 0 with the metric absent REFUSES —
    absence never improves a gate; support absent refuses too."""
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            for arm in ("X", "D", "XDR", "width_control"):
                for ent in ([o["results"][arm]["ridge"]]
                            + list(o["results"][arm][
                                "mlp_small"].values())):
                    ent.pop("mase_on_extreme_innovations", None)
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="extreme metric is absent"):
        conf.adjudicate_screen(recs, d)
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            for arm in ("X", "D", "XDR", "width_control"):
                for ent in ([o["results"][arm]["ridge"]]
                            + list(o["results"][arm][
                                "mlp_small"].values())):
                    ent.pop("mase_on_extreme_innovations", None)
                    ent.pop("extreme_support", None)
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="extreme_support absent"):
        conf.adjudicate_screen(recs, d)


def test_c30_kill_2_zero_baseline_extreme_is_harm():
    """C25b: X extreme error 0 with D extreme error large is
    HARM_INFINITE — damage, never absence; X=0,D=0 is ratio 1.0
    with no silent division; zero support is NOT_EVALUABLE and
    never favorable."""
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            o["results"]["X"]["ridge"][
                "mase_on_extreme_innovations"] = 0.0
            o["results"]["D"]["ridge"][
                "mase_on_extreme_innovations"] = 999.0
        _stamp(r)
    out = conf.adjudicate_screen(recs, d)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    assert "infinite extreme damage" in out["reason"]
    # X=0, D=0 -> EVALUATED ratio 1.0 (no division error, no harm)
    x0 = conf.extreme_contrast(
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 0.0},
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 0.0}, "p")
    assert x0 == {"state": "EVALUATED", "ratio": 1.0}
    # zero support -> NOT_EVALUABLE -> INCONCLUSIVE, never a pass
    recs = _recs(d, ex_support=0)
    out = conf.adjudicate_screen(recs, d)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "NOT_EVALUABLE" in out["reason"]


def test_c30_kill_3_record_without_geometry_refuses():
    """C26: a record lacking train/score (however restamped)
    refuses before any metric is consumed."""
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for o in r["rolling_origins"].values():
            o.pop("train", None)
            o.pop("score", None)
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="origin keys are not the exact"):
        conf.adjudicate_screen(recs, d)


def test_c30_kill_4_shifted_window_refuses():
    """C26: mutating a single window bound by ONE row refuses
    against the design's bound geometry."""
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    tr = recs[0]["rolling_origins"]["origin1"]["train"]
    tr[1] += 1
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="windows differ from the design"):
        conf.adjudicate_screen(recs, d)
    recs = _recs(d)
    sc = recs[0]["rolling_origins"]["origin1"]["score"]
    sc[0] -= 1
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="windows differ from the design"):
        conf.adjudicate_screen(recs, d)


def test_c30_kill_5_costs_without_global_phases_refuse():
    """C27: omitting target_construction_s or seasonal_naive_s
    refuses — the cost schema is exact, null and unknown phases
    refuse too."""
    import t2_confirmatory as conf
    d = _d3()
    recs = _recs(d)
    for r in recs:
        for oc in r["costs_by_phase"].values():
            oc.pop("target_construction_s", None)
            oc.pop("seasonal_naive_s", None)
        _stamp(r)
    with pytest.raises(SystemExit,
                       match="cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)
    # null phase refuses
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"][
        "seasonal_naive_s"] = None
    _stamp(recs[0])
    with pytest.raises(SystemExit, match="finite nonnegative"):
        conf.adjudicate_screen(recs, d)
    # unknown extra phase refuses
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"]["smuggled_s"] = 0.1
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)


def test_c30_kill_6_forged_unit_map_semantics_refuse():
    """C28a: a unit_map entry with the TRUE numeric digest and
    forged family/panel/period/geometry refuses in the fresh
    verifier's full re-derivation (live, physical bytes)."""
    import t2_confirmatory as conf
    import t2_fresh_verifier as fv
    S = Path.home() / ".local/share/agent-multi"
    manifest = conf.strict_json_load(
        S / "t2_public_data_manifest_20260906.json", "m")
    census = conf.strict_json_load(
        S / "t2_bank_census_20260906.json", "c")
    design = conf.strict_json_load(
        S / "t2_screen_design_DRAFT_V4_20260906.json", "d")
    forged = json.loads(json.dumps(design))
    uid = design["task_population"]["series_ids"][0]
    b = forged["task_population"]["unit_map"][uid]
    assert len(b["series_numeric_sha256"]) == 64  # digest REAL
    b["family"] = "totally_forged_family"
    b["seasonal_period"] = 999
    with pytest.raises(SystemExit,
                       match="do not re-derive from the physical "
                             "bytes"):
        fv.fresh_verify(manifest, census, forged,
                        manifest_sha=None)
    # forged origin windows die too
    forged2 = json.loads(json.dumps(design))
    b2 = forged2["task_population"]["unit_map"][uid]
    b2["origin_windows"]["origin0"]["train"] = [0, 1]
    with pytest.raises(SystemExit,
                       match="do not re-derive from the physical "
                             "bytes"):
        fv.fresh_verify(manifest, census, forged2,
                        manifest_sha=None)


def test_c30_kill_7_symlink_root_refuses(tmp_path):
    """C28b: a raw ROOT that is itself a symlink to the true
    directory refuses — no resolve() ever precedes the open."""
    import t2_confirmatory as conf
    m, raw = _mk_manifest(tmp_path)
    os.symlink(raw, tmp_path / "rootlink")
    with pytest.raises(SystemExit, match="symlink root"):
        conf.validate_public_manifest(
            m, raw_root=tmp_path / "rootlink")
    # the honest physical root still validates
    adm = conf.validate_public_manifest(m, raw_root=raw)
    assert "probe" in adm
    # source guard: no resolve() may reappear before the root open
    src = (REPO / "tools/t2_confirmatory.py").read_text()
    seg = src[src.index("def validate_public_manifest"):
              src.index("_DESIGN_KEYS")]
    assert ".resolve()" not in seg


def test_c30_kill_8_fresh_verifier_wired_into_single_path(
        tmp_path):
    """C28c: run_confirmatory CALLS the fresh verifier before the
    review gate and the ledger — a semantically forged design dies
    at re-derivation, BEFORE any review/ledger stage, and no
    ledger artifact is created."""
    import t2_confirmatory as conf
    S = Path.home() / ".local/share/agent-multi"
    mp = S / "t2_public_data_manifest_20260906.json"
    cp = S / "t2_bank_census_20260906.json"
    dp = S / "t2_screen_design_DRAFT_V6_20260907.json"
    lp = tmp_path / "ledger.json"
    # C39 (T2): the honest DRAFT dies at the sealed-only gate —
    # draft schemas never score; no ledger artifact
    with pytest.raises(SystemExit,
                       match="SEALED_DESIGN_REQUIRED"):
        conf.run_confirmatory(mp, dp, lp, census_path=cp)
    assert not lp.exists()
    # a FORGED sealed-looking design (true digest, false
    # semantics) dies at the fresh re-derivation, BEFORE the
    # review gate
    design = conf.strict_json_load(dp, "d")
    forged = json.loads(json.dumps(design))
    forged["schema"] = "agent_multi.t2_screen_design.v6"
    forged["sealed_at_date"] = "2026-09-07"
    uid = design["task_population"]["series_ids"][0]
    forged["task_population"]["unit_map"][uid]["family"] = "alien"
    body = {k: forged[k] for k in sorted(forged)
            if k != "design_sha256"}
    forged["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    fp = tmp_path / "forged_design.json"
    fp.write_text(json.dumps(forged))
    with pytest.raises(SystemExit,
                       match="do not re-derive from the physical "
                             "bytes"):
        conf.run_confirmatory(mp, fp, lp, census_path=cp)
    assert not lp.exists()
    # source guard: inside the PURE gate sequence the fresh
    # verifier precedes the review gate, which precedes the
    # execution gate; the effectful wrapper orders gates before
    # the ledger (C53)
    src = (REPO / "tools/t2_confirmatory.py").read_text()
    seg = src[src.index("def verify_confirmatory_gates"):]
    seg = seg[:seg.index("\ndef run_confirmatory")]
    assert seg.index("fresh_verify") < seg.index(
        "verify_design_review_record") < seg.index(
        "verify_execution_record")
    assert "open_attempt_ledger" not in seg
    wrap = src[src.index("def run_confirmatory"):]
    wrap = wrap[:wrap.index("\nT2_EXECUTION_RECORD_PATH")]
    assert wrap.index("verify_confirmatory_gates") < \
        wrap.index("open_attempt_ledger")


def test_c30_kill_9_dominant_panel_fails_lopo():
    """C29: one dominant panel cannot carry the screen — the
    leave-one-panel-out rule blocks ADVANCE even when the grand
    mean clears the margin."""
    import t2_confirmatory as conf
    d = _d3()
    recs = []
    for uid in d["task_population"]["series_ids"]:
        big = uid.startswith("f0")
        recs.append(_rec_for(d, uid,
                             delta=(0.030 if big else 0.019)))
    out = conf.adjudicate_screen(recs, d)
    assert out["verdict"] != "ADVANCE_TO_DOMAIN_VALIDATION"
    assert "leave-one-panel-out" in out["reason"]
    lopo = out["leave_one_panel_out_means"]
    assert min(lopo) <= 0.02 < max(lopo)
    # a grossly dominant panel dies too (wide interval => the
    # precision gate fires first; never ADVANCE either way)
    recs2 = []
    for uid in d["task_population"]["series_ids"]:
        big = uid.startswith("f0")
        recs2.append(_rec_for(d, uid,
                              delta=(0.60 if big else 0.005)))
    out2 = conf.adjudicate_screen(recs2, d)
    assert out2["verdict"] != "ADVANCE_TO_DOMAIN_VALIDATION"
    assert min(out2["leave_one_panel_out_means"]) <= 0.02


def test_c30_kill_10_favorable_average_with_damaged_panel():
    """C29: a favorable unweighted average with ONE materially
    damaged panel is DOES_NOT_ADVANCE — sign sensitivity and the
    non-inferiority margin both block it."""
    import t2_confirmatory as conf
    d = _d3()
    recs = []
    for uid in d["task_population"]["series_ids"]:
        harmed = uid.startswith("f3")
        recs.append(_rec_for(d, uid,
                             delta=(-0.06 if harmed else 0.09)))
    out = conf.adjudicate_screen(recs, d)
    assert out[
        "primary_estimand_unweighted_mean_of_panel_effects"] > 0.02
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    assert "non-inferiority" in out["reason"]
    assert out["signs_positive"] == 5


def test_c29_screen_contract_shape():
    """C29: the adjudicator emits ONLY screen outcomes, never
    eligibility; refuses a design without the screen rule; the
    committed v4 draft carries the estimand and the committed
    screen simulation shows the composite rule's operating
    characteristics."""
    import t2_confirmatory as conf
    d = _d3()
    out = conf.adjudicate_screen(_recs(d), d)
    assert out["verdict"] in ("ADVANCE_TO_DOMAIN_VALIDATION",
                              "DOES_NOT_ADVANCE", "INCONCLUSIVE")
    assert "PUBLICLY_ELIGIBLE" not in json.dumps(out)
    assert "ONLY these six public panels" in out["scope"]
    # a v3-rule design refuses in the screen adjudicator
    d3_old = _d3()
    d3_old["inference_method"] = {
        "rule": "panel_replication_or_descriptive"}
    with pytest.raises(SystemExit,
                       match="six-panel screen rule"):
        conf.adjudicate_screen(_recs(d3_old), d3_old)
    # the live v4 draft: estimand block, outputs, hospital fact
    S = Path.home() / ".local/share/agent-multi"
    v4 = json.loads(
        (S / "t2_screen_design_DRAFT_V4_20260906.json")
        .read_text())
    assert v4["schema"] == "agent_multi.t2_screen_design.v4_draft"
    est = v4["estimand"]
    assert est["superior_unit"] == "panel"
    assert list(est["outputs"]) == [
        "ADVANCE_TO_DOMAIN_VALIDATION", "DOES_NOT_ADVANCE",
        "INCONCLUSIVE"]
    assert "PUBLICLY_ELIGIBLE" not in json.dumps(v4["estimand"])
    assert len(v4["task_population"]["screen_panels"]) == 6
    assert "hospital" in v4["task_population"][
        "geometry_limited_panels"]
    assert "t2c_successor" in est and "CONDITIONAL" in \
        est["t2c_successor"]
    # v3 superseded by digest chain
    v3_sha = hashlib.sha256(
        (S / "t2_confirmatory_design_DRAFT_V3_20260906.json")
        .read_bytes()).hexdigest()
    assert v4["supersedes_draft_sha256"] == v3_sha
    # committed screen-rule simulation: boundary type-I under
    # alpha; damaged/dominant panels never advance
    sim = json.loads(
        (S / "t2_screen_sim_20260906.json").read_text())
    for row in sim["rows"]:
        if row["scenario"] == "boundary_type_I":
            assert row["advance_rate"] < 0.05
        if row["scenario"] in ("one_damaged_panel",
                               "one_dominant_panel"):
            assert row["advance_rate"] == 0.0


# ===== C31-C36 final-screen battery (order 2026-09-06) =============


def test_c36_1_extreme_support_one_of_n_never_licenses():
    """C31: ONE evaluable extreme series among NOT_EVALUABLE
    siblings can never allow ADVANCE — the predeclared per-panel
    absolute+proportional minimum bites; X=0,D>0 stays infinite
    harm and X=0,D=0 stays ratio 1.0."""
    import t2_confirmatory as conf
    d = _d3()
    recs = []
    for uid in d["task_population"]["series_ids"]:
        only = uid.endswith("::s0")
        recs.append(_rec_for(d, uid,
                             ex_support=(4 if only else 0)))
    out = conf.adjudicate_screen(recs, d)
    assert out["verdict"] == "INCONCLUSIVE"
    assert "evaluable extreme evidence" in out["reason"]
    assert "never favorable" in out["reason"]
    # full support still adjudicates (control)
    good = _recs(d)
    assert conf.adjudicate_screen(good, d)["verdict"] == \
        "ADVANCE_TO_DOMAIN_VALIDATION"
    # the two frozen boundary cases hold under the C31 world
    hi = conf.extreme_contrast(
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 0.0},
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 5.0}, "p")
    assert hi == {"state": "HARM_INFINITE", "ratio": None}
    eq = conf.extreme_contrast(
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 0.0},
        {"extreme_support": 3,
         "mase_on_extreme_innovations": 0.0}, "p")
    assert eq == {"state": "EVALUATED", "ratio": 1.0}
    # a design without the rule refuses at validation
    d2 = json.loads(json.dumps(d))
    d2.pop("extreme_support_rule")
    with pytest.raises(SystemExit):
        conf.adjudicate_screen(_recs(d2), d2)


def test_c36_2_every_unit_map_field_forged_dies_live():
    """C34: each unit_map field falsified INDEPENDENTLY (rest
    consistent, digest real) dies in the live fresh re-derivation
    from physical bytes."""
    import t2_confirmatory as conf
    import t2_fresh_verifier as fv
    S = Path.home() / ".local/share/agent-multi"
    manifest = conf.strict_json_load(
        S / "t2_public_data_manifest_20260906.json", "m")
    census = conf.strict_json_load(
        S / "t2_bank_census_20260906.json", "c")
    design = conf.strict_json_load(
        S / "t2_screen_design_DRAFT_V6_20260907.json", "d")
    uid = design["task_population"]["series_ids"][0]
    mutations = [
        ("family", "totally_forged_family"),
        ("dataset", "alien_panel"),
        ("series_numeric_sha256", "e" * 64),
        ("seasonal_period", 999),
        ("horizon", 7),
        ("n_obs", 12345),
        ("time_identity_sha256", "f" * 64),
        ("origin_windows", {"origin0": {"train": [0, 1],
                                        "score": [1, 2]},
                            "origin1": {"train": [0, 2],
                                        "score": [2, 3]}}),
    ]
    for field, val in mutations:
        forged = json.loads(json.dumps(design))
        forged["task_population"]["unit_map"][uid][field] = val
        with pytest.raises(SystemExit,
                           match="do not re-derive|missing from"):
            fv.fresh_verify(manifest, census, forged,
                            manifest_sha=None)


def test_c36_4_each_cost_phase_omitted_refuses():
    """C33: omitting target construction, seasonal baseline, one
    arm's lag features, one arm's ridge or one MLP seed each
    refuses via the PRODUCTIVE adjudicator; an extra key refuses
    too — the schema is exact, never an open minimum."""
    import t2_confirmatory as conf
    d = _d3()
    cases = [
        ("target_construction_s", None),
        ("seasonal_naive_s", None),
        (("arm_D", "lag_features_s"), None),
        (("arm_XDR", "ridge_fit_forecast_s"), None),
        (("arm_X", "mlp_fit_forecast_seed12_s"), None),
    ]
    for key, _ in cases:
        recs = _recs(d)
        oc = recs[0]["costs_by_phase"]["origin0"]
        if isinstance(key, tuple):
            del oc[key[0]][key[1]]
            want = "arm cost phases are not the exact"
        else:
            del oc[key]
            want = "cost phases are not the exact"
        _stamp(recs[0])
        with pytest.raises(SystemExit, match=want):
            conf.adjudicate_screen(recs, d)
    # extra keys refuse at both levels
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"]["smuggled_s"] = 0.1
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)
    recs = _recs(d)
    recs[0]["costs_by_phase"]["origin0"]["arm_D"][
        "smuggled_s"] = 0.1
    _stamp(recs[0])
    with pytest.raises(SystemExit,
                       match="arm cost phases are not the exact"):
        conf.adjudicate_screen(recs, d)


def test_c36_7_hospital_two_windows_of_17():
    """C35: hospital's length-84 series mechanically produce two
    consecutive 17-observation score windows under the COMMON
    geometry, with every real model minimum intact; the harness
    delegates to the same single authority."""
    import os
    import t2_bank as bank
    os.environ.setdefault(
        "B4_T1_PREPROCESSOR_ROOT",
        str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
    import t2_assay_harness as hz
    w = bank.origin_windows_for(84, 2, 0.6, seasonal_period=12)
    assert w == {"origin0": {"train": [0, 50],
                             "score": [50, 67]},
                 "origin1": {"train": [0, 67],
                             "score": [67, 84]}}
    assert all(wb["score"][1] - wb["score"][0] == 17
               for wb in w.values())
    assert hz.ROLLING_ORIGINS == 2
    assert hz.unit_origins(84, seasonal_period=12) == \
        [(50, 67), (67, 84)]
    # real minimums: >=8 MLP fit rows after validation, >=1 scored
    for wb in w.values():
        fit = wb["train"][1] - 8 - 1
        assert fit - max(8, int(fit * 0.2)) >= 8
        assert wb["score"][1] - wb["score"][0] - 8 - 1 >= 1
    # the fixed 120 floor is retired from the productive surface
    assert not hasattr(bank, "SCREEN_MIN_LENGTH")
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "if n < 120" not in src
    # the v5 draft records hospital as a FULL member
    S = Path.home() / ".local/share/agent-multi"
    v5 = json.loads(
        (S / "t2_screen_design_DRAFT_V6_20260907.json").read_text())
    gf = v5["task_population"]["geometry_feasibility"]["hospital"]
    assert gf["n_selected"] == 40 and gf["min_score_window"] == 17
    assert gf["model_minimums_kept"] is True


def test_c36_8_every_selected_unit_geometry_admissible():
    """C35: every unit of the six panels selected by v5 is
    geometry-admissible under the common two-origin rule
    (period-aware), and the population is 242 with v4 superseded
    by digest."""
    import t2_bank as bank
    S = Path.home() / ".local/share/agent-multi"
    v5 = json.loads(
        (S / "t2_screen_design_DRAFT_V6_20260907.json").read_text())
    tp = v5["task_population"]
    assert len(tp["series_ids"]) == 242
    panels = set(tp["screen_panels"])
    n_by_panel = {}
    for uid, b in tp["unit_map"].items():
        assert bank.geometry_admissible(
            b["n_obs"], 2, 0.6,
            seasonal_period=b["seasonal_period"],
            horizon=b["horizon"]), uid
        if b["dataset"] in panels:
            n_by_panel[b["dataset"]] = \
                n_by_panel.get(b["dataset"], 0) + 1
    assert sorted(n_by_panel) == sorted(panels)
    assert all(v == 40 for v in n_by_panel.values())
    v5_sha = hashlib.sha256(
        (S / "t2_screen_design_DRAFT_V5_20260907.json")
        .read_bytes()).hexdigest()
    assert v5["supersedes_draft_sha256"] == v5_sha
    assert v5["schema"] == "agent_multi.t2_screen_design.v6_draft"


# ============ C37 estimand-polarity battery (2026-09-07) ===========


def test_c37_polarity_contract():
    """C37: the ONE estimand — mase_improvement_X_minus_D =
    MASE(X) - MASE(D). X=1.0,D=0.9 -> +0.1 beneficial;
    X=0.9,D=1.0 -> -0.1 harmful; the width-control attribution
    uses the same orientation; old/ambiguous/inverted names
    refuse in the validator."""
    import t2_confirmatory as conf
    d = _d3()
    uid = d["task_population"]["series_ids"][0]
    b = d["task_population"]["unit_map"][uid]

    def probe(x, dd):
        rec = _complete_record(uid, b["family"],
                               dataset=b["dataset"],
                               windows=b["origin_windows"])
        for o in rec["rolling_origins"].values():
            o["results"]["X"]["ridge"]["mase_primary"] = x
            o["results"]["D"]["ridge"]["mase_primary"] = dd
        _stamp(rec)
        return conf._series_stats(rec, d, "D", "ridge")["delta"]
    assert abs(probe(1.0, 0.9) - (+0.1)) < 1e-12
    assert abs(probe(0.9, 1.0) - (-0.1)) < 1e-12
    # beneficial ADVANCES, harmful DOES_NOT_ADVANCE (polarity)
    good = [_rec_for(d, u, delta=0.10)
            for u in d["task_population"]["series_ids"]]
    assert conf.adjudicate_screen(good, d)["verdict"] == \
        "ADVANCE_TO_DOMAIN_VALIDATION"
    bad = [_rec_for(d, u, delta=-0.10)
           for u in d["task_population"]["series_ids"]]
    out = conf.adjudicate_screen(bad, d)
    assert out["verdict"] == "DOES_NOT_ADVANCE"
    assert out["panel_effect_definition"].startswith(
        "mase_improvement_X_minus_D")
    # width-control attribution: same orientation — a width
    # control matching D's improvement zeroes the attribution
    wc = [_rec_for(d, u, delta=0.10, wc_delta=0.10)
          for u in d["task_population"]["series_ids"]]
    out2 = conf.adjudicate_screen(wc, d)
    assert out2["verdict"] == "INCONCLUSIVE"
    assert "attributable" in out2["reason"]
    # the validator refuses every superseded/ambiguous/inverted
    # name on a live v6 document
    import hashlib as _h
    S = Path.home() / ".local/share/agent-multi"
    mp = S / "t2_public_data_manifest_20260906.json"
    v6 = json.loads(
        (S / "t2_screen_design_DRAFT_V6_20260907.json")
        .read_text())
    for bad_name in ("D_minus_X", "delta",
                     "mase_improvement_D_minus_X", "X_minus_D"):
        f2 = json.loads(json.dumps(v6))
        f2["primary_contrast"]["delta"] = bad_name
        body = {k: f2[k] for k in sorted(f2)
                if k != "design_sha256"}
        f2["design_sha256"] = _h.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest()
        with pytest.raises(SystemExit,
                           match="estimand names never validate|"
                                 "mase_improvement_X_minus_D"):
            conf.validate_confirmatory_design(
                f2, conf._sha_file(mp))


def test_c37_v6_supersedes_v5_name_only():
    """C37: draft v6 is v5 with ONLY the estimand naming changed —
    population, unit_map, geometry, margins and rules identical;
    v6 supersedes v5 by exact digest; v2-v5 preserved."""
    S = Path.home() / ".local/share/agent-multi"
    v5 = json.loads(
        (S / "t2_screen_design_DRAFT_V5_20260907.json")
        .read_text())
    v6 = json.loads(
        (S / "t2_screen_design_DRAFT_V6_20260907.json")
        .read_text())
    tp5, tp6 = v5["task_population"], v6["task_population"]
    assert tp6["series_ids"] == tp5["series_ids"]
    assert tp6["unit_map"] == tp5["unit_map"]
    assert v6["role_geometry"] == v5["role_geometry"]
    assert v6["extreme_support_rule"] == v5["extreme_support_rule"]
    assert v6["harm_margins"] == v5["harm_margins"]
    assert v6["practical_margin_mase"] == \
        v5["practical_margin_mase"]
    assert v5["primary_contrast"]["delta"] == "D_minus_X"
    assert v6["primary_contrast"]["delta"] == \
        "mase_improvement_X_minus_D"
    assert v6["supersedes_draft_sha256"] == hashlib.sha256(
        (S / "t2_screen_design_DRAFT_V5_20260907.json")
        .read_bytes()).hexdigest()
    # the XDR FEATURE representation is untouched (not the MASE
    # contrast): [X, D, X-D] stays
    assert v6["arms"]["XDR"] == "[X, D, X-D]"


# ===== C38-C41 custody + sealing battery (order 2026-09-07) =======


def test_c40_1_repo_record_grants_nothing(tmp_path, monkeypatch):
    """C38: with the productive path at the REAL external root
    (absent) a candidate-committed record under docs/ changes
    nothing — the gate stays closed with the stop reason."""
    import t2_confirmatory as conf
    look = REPO / ("docs/audits/evidence/"
                   "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json")
    assert not look.exists()
    look.write_text(json.dumps(_t2_record_doc()))
    try:
        if conf.T2_REVIEW_RECORD_PATH.exists():
            pytest.skip("real external record present on host")
        with pytest.raises(SystemExit,
                           match="DESIGN_REVIEW_REQUIRED"):
            conf.verify_design_review_record(
                {"design_review_record_sha256": "0" * 64},
                "b" * 64, "c" * 64)
    finally:
        look.unlink()


def test_c40_2_private_custody_adversaries(tmp_path, monkeypatch):
    """C38: wrong parent mode, symlink component, wrong file mode,
    non-regular, duplicate key, non-finite and every foreign
    binding (draft file/self, manifest, census, commit) refuse."""
    import t2_confirmatory as conf
    S = Path.home() / ".local/share/agent-multi"
    man_sha = conf._sha_file(
        S / "t2_public_data_manifest_20260906.json")
    cen_sha = conf._sha_file(
        S / "t2_bank_census_20260906.json")
    ra = _t2_private_chain(tmp_path / "auth")
    fake = ra / "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json"
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", fake)

    def write(doc):
        if fake.exists():
            fake.unlink()
        fake.write_text(json.dumps(doc)
                        if isinstance(doc, dict) else doc)
        os.chmod(fake, 0o600)

    def probe():
        d2 = {"design_review_record_sha256":
              conf._sha_file(fake),
              "supersedes_draft_sha256":
                  conf.T2_V6_DRAFT_FILE_SHA}
        return conf.verify_design_review_record(
            d2, man_sha, cen_sha)
    write(_t2_record_doc())
    got = probe()
    assert got["_record_sha256"] == conf._sha_file(fake)
    os.chmod(ra, 0o755)
    with pytest.raises(SystemExit, match="not the private 0700"):
        probe()
    os.chmod(ra, 0o700)
    os.chmod(fake, 0o644)
    with pytest.raises(SystemExit, match="0600"):
        probe()
    os.chmod(fake, 0o600)
    alt = tmp_path / "alt"
    alt.mkdir(mode=0o700)
    link = ra / "link_rec.json"
    os.symlink(fake, link)
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", link)
    with pytest.raises(SystemExit,
                       match="without following links|"
                             "unopenable"):
        probe.__wrapped__() if hasattr(probe, "__wrapped__") \
            else conf.verify_design_review_record(
                {"design_review_record_sha256": "0" * 64},
                man_sha, cen_sha)
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", fake)
    write('{"schema": 1, "schema": 2}')
    with pytest.raises(SystemExit, match="duplicate JSON key"):
        probe()
    write(json.dumps(_t2_record_doc()).replace(
        '"General Musashi"', "NaN", 1))
    with pytest.raises(SystemExit):
        probe()
    for field, val, needle in (
            ("design_draft_file_sha256", "e" * 64,
             "different draft v6"),
            ("design_draft_self_sha256", "e" * 64,
             "different draft v6"),
            ("manifest_sha256", "e" * 64,
             "different public-data manifest"),
            ("census_sha256", "e" * 64, "different bank census"),
            ("candidate_commit", "f" * 40,
             "candidate commit")):
        write(_t2_record_doc(**{field: val}))
        with pytest.raises(SystemExit, match=needle):
            probe()


def test_c40_3_seal_changes_only_seal_fields(tmp_path,
                                             monkeypatch):
    """C39/C40 positive: draft-v6 -> sealed-v6 (fixture record,
    tmp output) changes ONLY the allowed seal fields; the sealed
    design validates, fresh verification re-derives 4650 units and
    the 242-series population, and run_confirmatory carries it to
    the ledger stage — while a sealed design altering ANY
    scientific field refuses inside the seal tool."""
    import t2_confirmatory as conf
    import importlib.util as ilu
    spec2 = ilu.spec_from_file_location(
        "t2seal", REPO / "tools/t2_seal_design.py")
    seal = ilu.module_from_spec(spec2)
    spec2.loader.exec_module(seal)
    S = Path.home() / ".local/share/agent-multi"
    ra = _t2_private_chain(tmp_path / "auth")
    fake = ra / "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json"
    fake.write_text(json.dumps(_t2_record_doc()))
    os.chmod(fake, 0o600)
    monkeypatch.setattr(conf, "T2_REVIEW_RECORD_PATH", fake)
    out = tmp_path / "sealed_v6.json"
    sealed = seal.seal_design(out_path=out)
    draft = json.loads(
        (S / "t2_screen_design_DRAFT_V6_20260907.json")
        .read_text())
    changed = {k for k in sealed
               if sealed.get(k) != draft.get(k)}
    changed |= set(sealed) - set(draft)
    assert changed <= set(seal.SEAL_ONLY_FIELDS), changed
    assert sealed["schema"] == "agent_multi.t2_screen_design.v6"
    assert sealed["supersedes_draft_sha256"] == \
        conf.T2_V6_DRAFT_FILE_SHA
    assert sealed["design_review_record_sha256"] == \
        conf._sha_file(fake)
    mp = S / "t2_public_data_manifest_20260906.json"
    conf.validate_confirmatory_design(sealed,
                                      conf._sha_file(mp))
    import t2_fresh_verifier as fv
    manifest = conf.strict_json_load(mp, "m")
    census = conf.strict_json_load(
        S / "t2_bank_census_20260906.json", "c")
    facts = fv.fresh_verify(manifest, census, sealed,
                            manifest_sha=conf._sha_file(mp))
    assert facts["units_rederived"] == 4650
    assert facts["design_series"] == 242
    # the single path accepts the sealed fixture through review
    # and stops at the EXECUTION-record gate (C42/C47) — no
    # ledger without the second external record
    lp = tmp_path / "ledger.json"
    with pytest.raises(SystemExit,
                       match="T2_EXECUTION_RECORD_REQUIRED"):
        conf.run_confirmatory(
            mp, out, lp,
            census_path=S / "t2_bank_census_20260906.json")
    assert not lp.exists()
    # a scientific-field mutation inside sealing refuses
    monkeypatch.setattr(seal, "SEAL_ONLY_FIELDS",
                        tuple(seal.SEAL_ONLY_FIELDS)
                        + ("practical_margin_mase",))
    def bad_seal():
        s2 = seal.seal_design(out_path=tmp_path / "x.json")
        assert s2["practical_margin_mase"] == \
            draft["practical_margin_mase"]
    bad_seal()   # widening the allowlist alone changes nothing


def test_c40_4_cli_and_api_share_the_sealed_path():
    """C39: the public CLI names the ONE current sealed identity
    (no legacy 2026-09-06 filename) and shares run_confirmatory
    with the direct API; a draft passed to scoring refuses."""
    src = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "t2_confirmatory_design_20260906.json" not in src
    assert "t2_screen_design_SEALED_V6.json" in src
    # C53: the CLI consumes the PURE gate sequence only — it can
    # never create a ledger as a side effect of a gate check
    assert "verify_confirmatory_gates" in src
    cli_seg = src[src.index("if args.confirmatory"):]
    cli_seg = cli_seg[:cli_seg.index("if not args.development")]
    assert "run_confirmatory" not in cli_seg
    csrc = (REPO / "tools/t2_confirmatory.py").read_text()
    seg = csrc[csrc.index("def verify_confirmatory_gates"):]
    seg = seg[:seg.index("\ndef run_confirmatory")]
    assert "SEALED_DESIGN_REQUIRED" in seg
    assert seg.index("SEALED_DESIGN_REQUIRED") < \
        seg.index("fresh_verify") < \
        seg.index("verify_design_review_record") < \
        seg.index("verify_execution_record")
    assert "_ACCEPTED_DESIGN_SCHEMAS" in csrc


# ===== C42-C47 confirmatory-executor battery (2026-09-07) =========


def _exec_mod():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "t2exec", REPO / "tools/t2_confirmatory_executor.py")
    m = ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m



def _priv_write(path, doc):
    """0600 fixture writer — the v2 custody walk refuses anything
    else."""
    p = Path(path)
    if p.exists():
        p.unlink()
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    os.chmod(p, 0o600)


def _git_head_tree():
    import subprocess
    h = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD"], capture_output=True,
                      text=True).stdout.strip()
    t = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                       "HEAD^{tree}"], capture_output=True,
                      text=True).stdout.strip()
    return h, t


def _v2_exec_record(conf, sealed, design, **over):
    S = Path.home() / ".local/share/agent-multi"
    head, tree = _git_head_tree()
    rec = {"schema": "agent_multi.musashi_t2_execution_record.v2",
           "reviewed_at_date": "2026-09-07",
           "reviewer": "General Musashi",
           "decision": "OPEN_T2_CONFIRMATORY_EXECUTION",
           "sealed_design_file_sha256": conf._sha_file(sealed),
           "sealed_design_self_sha256": design["design_sha256"],
           "design_review_record_sha256":
               conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
           "manifest_sha256": conf._sha_file(
               S / "t2_public_data_manifest_20260906.json"),
           "census_sha256": conf._sha_file(
               S / "t2_bank_census_20260906.json"),
           "executor_code_identity":
               conf.executor_code_identity(),
           "pinned_commit": head, "pinned_tree": tree}
    rec.update(over)
    return rec


def test_c48_1_execution_gate_v2_per_field(tmp_path, monkeypatch):
    """C48: the v2 execution record refuses per field; the P1
    bypass (an attacker-controlled nonempty candidate string) is a
    frozen regression; without the record the single path refuses
    TYPED before any ledger."""
    import t2_confirmatory as conf
    S = Path.home() / ".local/share/agent-multi"
    sealed = S / "t2_screen_design_SEALED_V6.json"
    if not sealed.exists():
        pytest.skip("sealed design absent on this host")
    mp = S / "t2_public_data_manifest_20260906.json"
    lp = tmp_path / "ledger.json"
    monkeypatch.setattr(conf, "T2_EXECUTION_RECORD_PATH",
                        tmp_path / "missing.json")
    with pytest.raises(SystemExit,
                       match="T2_EXECUTION_RECORD_REQUIRED"):
        conf.run_confirmatory(
            mp, sealed, lp,
            census_path=S / "t2_bank_census_20260906.json")
    assert not lp.exists()
    ra = _t2_private_chain(tmp_path / "auth")
    er = ra / "MUSASHI_T2_V6_EXECUTION_RECORD.json"
    monkeypatch.setattr(conf, "T2_EXECUTION_RECORD_PATH", er)
    d = json.loads(sealed.read_text())
    args = (d, conf._sha_file(sealed),
            conf._sha_file(conf.T2_REVIEW_RECORD_PATH),
            conf._sha_file(mp),
            conf._sha_file(S / "t2_bank_census_20260906.json"))
    # P1 REGRESSION with the REAL checkout verifier: the exact
    # attacker string dies on FORM before anything else
    good = _v2_exec_record(conf, sealed, d)
    _priv_write(er, {**good, "pinned_commit":
                     "attacker-controlled-nonempty-string"})
    with pytest.raises(SystemExit, match="40 lowercase hex"):
        conf.verify_execution_record(*args)
    # a v1-shaped record is a foreign schema now
    v1 = {k: good[k] for k in
          ("schema", "reviewed_at_date", "reviewer", "decision",
           "sealed_design_file_sha256",
           "sealed_design_self_sha256")}
    v1["schema"] = "agent_multi.musashi_t2_execution_record.v1"
    v1["candidate_commit"] = "attacker-controlled-nonempty-string"
    _priv_write(er, v1)
    with pytest.raises(SystemExit, match="exact v2 schema"):
        conf.verify_execution_record(*args)
    # per-field forgeries (checkout verifier stubbed so field
    # semantics are what refuses, not this dev tree's dirt)
    seen = []
    monkeypatch.setattr(conf, "verify_executor_checkout",
                        lambda c, t, repo_root=None:
                        seen.append((c, t)))
    for field, val, needle in (
            ("decision", "SOMETHING_ELSE", "does not open"),
            ("reviewer", "candidate", "external reviewer role"),
            ("reviewed_at_date", "7/9/2026", "canonical"),
            ("sealed_design_file_sha256", "e" * 64,
             "physical bytes"),
            ("sealed_design_self_sha256", "e" * 64,
             "self identity"),
            ("design_review_record_sha256", "0" * 64,
             "verified external design review record"),
            ("manifest_sha256", "1" * 64, "public-data manifest"),
            ("census_sha256", "2" * 64, "bank census"),
            ("executor_code_identity", {"attacker.py": "f" * 64},
             "physical checkout surface")):
        _priv_write(er, {**good, field: val})
        with pytest.raises(SystemExit, match=needle):
            conf.verify_execution_record(*args)
        assert not lp.exists()
    # missing and extra keys refuse
    less = dict(good)
    less.pop("pinned_tree")
    _priv_write(er, less)
    with pytest.raises(SystemExit, match="exact v2 schema"):
        conf.verify_execution_record(*args)
    _priv_write(er, {**good, "extra": "x"})
    with pytest.raises(SystemExit, match="exact v2 schema"):
        conf.verify_execution_record(*args)
    # the VALID v2 record verifies (checkout stub records the pin)
    _priv_write(er, good)
    out = conf.verify_execution_record(*args)
    assert out["_record_sha256"] == conf._sha_file(er)
    assert seen[-1] == (good["pinned_commit"],
                        good["pinned_tree"])


def test_c48_2_checkout_verifier_bites(tmp_path):
    """C48.2-4: the REAL verify_executor_checkout against a
    synthetic repository — form, existence, HEAD, tree, dirty
    tracked files and shadowing untracked sources each refuse;
    inert untracked files are tolerated."""
    import subprocess
    import t2_confirmatory as conf
    r = tmp_path / "repo"
    (r / "tools").mkdir(parents=True)
    (r / "tools/a.py").write_text("x = 1\n")

    def g(*a):
        return subprocess.run(["git", "-C", str(r), *a],
                              capture_output=True, text=True)
    g("init", "-q")
    g("config", "user.email", "t@example.invalid")
    g("config", "user.name", "t")
    g("add", "-A")
    g("commit", "-q", "-m", "one")
    head = g("rev-parse", "HEAD").stdout.strip()
    tree = g("rev-parse", "HEAD^{tree}").stdout.strip()
    conf.verify_executor_checkout(head, tree, repo_root=r)  # clean
    with pytest.raises(SystemExit, match="40 lowercase hex"):
        conf.verify_executor_checkout(
            "attacker-controlled-nonempty-string", tree,
            repo_root=r)
    with pytest.raises(SystemExit, match="existing commit"):
        conf.verify_executor_checkout("f" * 40, tree, repo_root=r)
    with pytest.raises(SystemExit, match="not the tree"):
        conf.verify_executor_checkout(head, "0" * 40, repo_root=r)
    (r / "tools/a.py").write_text("x = 2\n")
    g("add", "-A")
    g("commit", "-q", "-m", "two")
    head2 = g("rev-parse", "HEAD").stdout.strip()
    tree2 = g("rev-parse", "HEAD^{tree}").stdout.strip()
    with pytest.raises(SystemExit, match="not the.*pinned commit"):
        conf.verify_executor_checkout(head, tree, repo_root=r)
    (r / "tools/a.py").write_text("x = 3\n")     # dirty tracked
    with pytest.raises(SystemExit, match="clean checkout"):
        conf.verify_executor_checkout(head2, tree2, repo_root=r)
    g("checkout", "-q", "--", ".")
    (r / "tools/evil.py").write_text("import os\n")
    with pytest.raises(SystemExit, match="shadow"):
        conf.verify_executor_checkout(head2, tree2, repo_root=r)
    (r / "tools/evil.py").unlink()
    (r / "evil.pth").write_text("import evil\n")
    with pytest.raises(SystemExit, match="import machinery"):
        conf.verify_executor_checkout(head2, tree2, repo_root=r)
    (r / "evil.pth").unlink()
    (r / "docs").mkdir()
    (r / "docs/note.md").write_text("inert\n")   # tolerated
    conf.verify_executor_checkout(head2, tree2, repo_root=r)


@pytest.fixture(scope="module")
def rehearsal_root(tmp_path_factory):
    """ONE shared v2 mechanical rehearsal over the dev units —
    real records, real NPZ, real claims, zero sealed series."""
    os.environ.setdefault(
        "B4_T1_PREPROCESSOR_ROOT",
        str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
    ex = _exec_mod()
    if not ex.SEALED_PATH.exists():
        pytest.skip("sealed design absent on this host")
    root = tmp_path_factory.mktemp("t2reh") / "t2root"
    rc = ex.rehearse(root)
    assert rc == 0
    return root, ex


def _clone_unit(root, uid, dst):
    import shutil
    dst.mkdir(mode=0o700, exist_ok=True)
    safe = uid.replace("::", "__").replace("/", "_")
    for pref, suf in (("RECORD_", ".json"), ("ARRAYS_", ".npz"),
                      ("CLAIM_", ".json")):
        s = root / "units" / f"{pref}{safe}{suf}"
        t = dst / f"{pref}{safe}{suf}"
        shutil.copy(s, t)
        os.chmod(t, 0o600)
    return (dst / f"RECORD_{safe}.json",
            dst / f"ARRAYS_{safe}.npz")


def _restamp(ex, wrapper):
    wrapper["record_sha256"] = ex._self_sha(wrapper,
                                            "record_sha256")
    return wrapper


def test_c49_1_forged_wrapper_bypass_p2_frozen(rehearsal_root,
                                               tmp_path):
    """C49: the FULL P2 forgery (foreign unit_id, 64-zero
    execution record, attacker code_identity, all 34 extreme MASE
    values -> 999.0, self-digest repaired, NPZ intact) now refuses
    — and each component refuses on its own typed ground."""
    root, ex = rehearsal_root
    design = json.loads(ex.SEALED_PATH.read_text())
    rp, npz = _clone_unit(root, "sm_nile", tmp_path / "p2")
    base = json.loads(rp.read_text())
    # (a) the exact quadruple forgery
    w = json.loads(json.dumps(base))
    w["unit_id"] = "attacker::not_in_sealed_population"
    w["execution_record_sha256"] = "0" * 64
    w["code_identity"] = {"attacker.py": "f" * 64}
    n = 0
    for o in w["assay_record"]["rolling_origins"].values():
        for arm, entry in o["results"].items():
            pools = ([entry["metrics"]]
                     if arm == "seasonal_naive" else
                     [entry["ridge"], *entry["mlp_small"].values()])
            for m in pools:
                m["mase_on_extreme_innovations"] = 999.0
                n += 1
    assert n == 34
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit,
                       match="not a development unit"):
        ex.verify_unit_record(rp, npz, design)
    # (b) forged execution-record digest alone
    w = json.loads(json.dumps(base))
    w["execution_record_sha256"] = "0" * 64
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit, match="physical authority"):
        ex.verify_unit_record(rp, npz, design)
    # (c) forged code identity alone
    w = json.loads(json.dumps(base))
    w["code_identity"] = {"attacker.py": "f" * 64}
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit, match="reviewed checkout"):
        ex.verify_unit_record(rp, npz, design)
    # (d) all 34 extreme metrics alone — the exact path is named
    w = json.loads(json.dumps(base))
    for o in w["assay_record"]["rolling_origins"].values():
        for arm, entry in o["results"].items():
            pools = ([entry["metrics"]]
                     if arm == "seasonal_naive" else
                     [entry["ridge"], *entry["mlp_small"].values()])
            for m in pools:
                m["mase_on_extreme_innovations"] = 999.0
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit,
                       match="mase_on_extreme_innovations"):
        ex.verify_unit_record(rp, npz, design)
    # (e) foreign unit_id alone (a real dev id on foreign arrays)
    w = json.loads(json.dumps(base))
    w["unit_id"] = "sm_sunspots"
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit, match="filename|claim|binding"):
        ex.verify_unit_record(rp, npz, design)
    # (f) transplanted sealed binding
    w = json.loads(json.dumps(base))
    uid0 = design["task_population"]["series_ids"][0]
    w["unit_binding"] = \
        design["task_population"]["unit_map"][uid0]
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit, match="binding"):
        ex.verify_unit_record(rp, npz, design)
    # (g) the intact record still verifies
    _priv_write(rp, base)
    out = ex.verify_unit_record(rp, npz, design)
    assert out["verified_units"] == 1
    # (h) a rehearsal record can NEVER verify as confirmatory
    with pytest.raises(SystemExit, match="does not match the "
                                         "expected"):
        ex.verify_unit_record(rp, npz, design,
                              mode_expected="confirmatory")


def test_c50_1_arrays_bound_to_the_physical_series(rehearsal_root,
                                                   tmp_path):
    """C50.5: altering obs AND pred together (metrics recompute
    consistently!) still refuses — obs must be the exact slice of
    the physical series; the baseline prediction must be the
    seasonal-naive slice; a swapped NPZ refuses on digest."""
    import numpy as _np
    root, ex = rehearsal_root
    design = json.loads(ex.SEALED_PATH.read_text())
    rp, npz = _clone_unit(root, "sm_sunspots", tmp_path / "joint")
    base = json.loads(rp.read_text())
    with _np.load(npz, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    key = next(k for k in data if k.startswith("pred__")
               and "ridge" in k)
    okye = key.replace("pred__", "obs__")
    data[key] = data[key] + 3.7
    data[okye] = data[okye] + 3.7          # err unchanged
    mut = tmp_path / "joint" / npz.name
    mut.unlink()
    with open(mut, "wb") as f:
        _np.savez_compressed(f, **data)
    os.chmod(mut, 0o600)
    w = json.loads(json.dumps(base))
    w["arrays_npz_sha256"] = ex._sha_file(mut)
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit,
                       match="exact.*slice of the physical "
                             "series"):
        ex.verify_unit_record(rp, mut, design)
    # baseline prediction must derive from the series itself
    rp2, npz2 = _clone_unit(root, "sm_sunspots", tmp_path / "bl")
    with _np.load(npz2, allow_pickle=False) as z:
        data2 = {k: z[k] for k in z.files}
    bkey = next(k for k in data2 if k.startswith("pred__")
                and k.endswith("__baseline"))
    data2[bkey] = data2[bkey] + 1.0
    npz2.unlink()
    with open(npz2, "wb") as f:
        _np.savez_compressed(f, **data2)
    os.chmod(npz2, 0o600)
    w2 = json.loads(json.dumps(base))
    w2["arrays_npz_sha256"] = ex._sha_file(npz2)
    _priv_write(rp2, _restamp(ex, w2))
    with pytest.raises(SystemExit, match="seasonal-naive slice"):
        ex.verify_unit_record(rp2, npz2, design)
    # a swapped NPZ (another unit's arrays under THIS unit's
    # filename) dies on the digest
    import shutil
    rp3, npz3 = _clone_unit(root, "sm_nile", tmp_path / "swap")
    npz3.unlink()
    shutil.copy(root / "units" / "ARRAYS_sm_co2.npz", npz3)
    os.chmod(npz3, 0o600)
    with pytest.raises(SystemExit, match="swapped or edited"):
        ex.verify_unit_record(rp3, npz3, design)


def test_c50_2_inventory_shape_dtype_finiteness(rehearsal_root,
                                                tmp_path):
    """C50.3-4: missing arrays, extra arrays, NaN, wrong dtype and
    wrong length each refuse typed (with the record's digest
    repaired, so the refusals are semantic)."""
    import numpy as _np
    root, ex = rehearsal_root
    design = json.loads(ex.SEALED_PATH.read_text())

    def _mutated(name, fn):
        rp, npz = _clone_unit(root, "sm_nile", tmp_path / name)
        base = json.loads(rp.read_text())
        with _np.load(npz, allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        fn(data)
        npz.unlink()
        with open(npz, "wb") as f:
            _np.savez_compressed(f, **data)
        os.chmod(npz, 0o600)
        w = json.loads(json.dumps(base))
        w["arrays_npz_sha256"] = ex._sha_file(npz)
        _priv_write(rp, _restamp(ex, w))
        return rp, npz

    fitk = None
    with _np.load(root / "units" / "ARRAYS_sm_nile.npz",
                  allow_pickle=False) as z:
        fitk = next(k for k in z.files if k.startswith("fit__"))
        predk = next(k for k in z.files
                     if k.startswith("pred__") and "ridge" in k)
    rp, npz = _mutated("miss", lambda d: d.pop(fitk))
    with pytest.raises(SystemExit, match="inventory is not exact"):
        ex.verify_unit_record(rp, npz, design)
    rp, npz = _mutated("extra", lambda d: d.update(
        {"smuggled": _np.zeros(3)}))
    with pytest.raises(SystemExit, match="inventory is not exact"):
        ex.verify_unit_record(rp, npz, design)

    def _nan(d):
        a = d[predk].copy()
        a[0] = _np.nan
        d[predk] = a
    rp, npz = _mutated("nan", _nan)
    with pytest.raises(SystemExit, match="finite float64"):
        ex.verify_unit_record(rp, npz, design)
    rp, npz = _mutated("f32", lambda d: d.update(
        {predk: d[predk].astype(_np.float32)}))
    with pytest.raises(SystemExit, match="finite float64"):
        ex.verify_unit_record(rp, npz, design)
    rp, npz = _mutated("short", lambda d: d.update(
        {predk: d[predk][:-1]}))
    with pytest.raises(SystemExit, match="length"):
        ex.verify_unit_record(rp, npz, design)


def test_c50_3_descriptor_custody_of_evidence(rehearsal_root,
                                              tmp_path):
    """C50.1-2: wrong mode refuses (never chmodded); a symlinked
    evidence object refuses; pickled/object arrays refuse; the
    producer NPZ path is exclusive-create (source)."""
    import numpy as _np
    root, ex = rehearsal_root
    design = json.loads(ex.SEALED_PATH.read_text())
    rp, npz = _clone_unit(root, "sm_nile", tmp_path / "mode")
    os.chmod(rp, 0o644)
    with pytest.raises(SystemExit, match="exact private 0600"):
        ex.verify_unit_record(rp, npz, design)
    os.chmod(rp, 0o600)
    link = tmp_path / "mode" / "link.npz"
    link.symlink_to(npz)
    with pytest.raises(SystemExit, match="unopenable|filename"):
        ex.verify_unit_record(rp, link, design)
    rp2, npz2 = _clone_unit(root, "sm_nile", tmp_path / "pick")
    npz2.unlink()
    with open(npz2, "wb") as f:
        _np.savez(f, y=_np.array([{"a": 1}], dtype=object))
    os.chmod(npz2, 0o600)
    base = json.loads(rp2.read_text())
    base["arrays_npz_sha256"] = ex._sha_file(npz2)
    _priv_write(rp2, _restamp(ex, base))
    with pytest.raises(SystemExit, match="pickle-free"):
        ex.verify_unit_record(rp2, npz2, design)
    esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    assert 'with open(npz_p, "wb")' not in esrc
    assert "_excl_write_npz(npz_p, arrays)" in esrc
    assert "O_EXCL" in esrc and "allow_pickle=False" in esrc


def test_c51_1_every_metric_recomputes_or_refuses(rehearsal_root,
                                                  tmp_path):
    """C51: falsifying ANY consumed metric of any entry refuses
    naming its exact path — MASE, MAE, RMSE, coverage, width,
    extreme support and the MASE denominator itself."""
    root, ex = rehearsal_root
    design = json.loads(ex.SEALED_PATH.read_text())
    rp, npz = _clone_unit(root, "sm_co2", tmp_path / "m")
    base = json.loads(rp.read_text())
    cases = [
        ("mase_primary", 0.0001, "mase_primary"),
        ("mae_per_series_diagnostic", 0.0001,
         "mae_per_series_diagnostic"),
        ("rmse_per_series_diagnostic", 0.0001,
         "rmse_per_series_diagnostic"),
        ("interval_coverage_train_q90", 0.123456,
         "interval_coverage_train_q90"),
        ("interval_width_train_q90", 0.0001,
         "interval_width_train_q90"),
        ("extreme_support", 999, "extreme_support"),
    ]
    for field, val, needle in cases:
        w = json.loads(json.dumps(base))
        entry = w["assay_record"]["rolling_origins"]["origin0"][
            "results"]["D"]["ridge"]
        entry[field] = val
        _priv_write(rp, _restamp(ex, w))
        with pytest.raises(SystemExit, match=needle) as ei:
            ex.verify_unit_record(rp, npz, design)
        assert "origin0" in str(ei.value)
        assert "ridge" in str(ei.value)
    # the denominator itself is re-derived from the series
    w = json.loads(json.dumps(base))
    w["assay_record"]["rolling_origins"]["origin1"][
        "mase_denominator_train_snaive"] = 0.5
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit,
                       match="mase_denominator_train_snaive"):
        ex.verify_unit_record(rp, npz, design)
    # and a seed-level MLP entry names its full path
    w = json.loads(json.dumps(base))
    w["assay_record"]["rolling_origins"]["origin0"]["results"][
        "XDR"]["mlp_small"]["seed12"]["mase_primary"] = 0.0001
    _priv_write(rp, _restamp(ex, w))
    with pytest.raises(SystemExit,
                       match=r"mlp_small\.seed12\.mase_primary"):
        ex.verify_unit_record(rp, npz, design)


def test_c52_1_budget_stop_inside_a_unit_blocks_typed(tmp_path):
    """C52/C54.3: a budget stop MID-UNIT re-raises typed WITHOUT a
    terminal; the claim then adjudicates UNCERTAIN and blocks
    until the EXPLICIT recorded operator disposition converts it
    to TERMINAL_FAILED; a second disposition refuses."""
    os.environ.setdefault(
        "B4_T1_PREPROCESSOR_ROOT",
        str(Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"))
    ex = _exec_mod()
    if not ex.SEALED_PATH.exists():
        pytest.skip("sealed design absent on this host")
    design = json.loads(ex.SEALED_PATH.read_text())
    import t2_assay_harness as hz
    import t2_public_data_census as dc
    co = hz.load_co()
    unit = hz.load_task_unit(dc.build_census(), "sm_nile")
    authority = ex.physical_authority(design,
                                      "mechanical_rehearsal")
    root = tmp_path / "t2stop"
    root.mkdir(mode=0o700)
    calls = []

    def tripping_guard(label):
        calls.append(label)
        if len(calls) == 7:
            raise ex.T2BudgetStop("test bound", label)

    with pytest.raises(SystemExit, match="T2_BUDGET_STOP"):
        ex.run_unit(hz, co, unit, design, authority, root,
                    "mechanical_rehearsal", guard=tripping_guard)
    u = root / "units"
    assert (u / "CLAIM_sm_nile.json").exists()
    assert not (u / "RECORD_sm_nile.json").exists()
    assert not (u / "TERMINAL_sm_nile.json").exists()
    st, why = ex.adjudicate_unit_shallow(u, "sm_nile")
    assert st == "UNCERTAIN" and "claim without" in why
    # the explicit recorded operator disposition
    ex.declare_attempt_failed(root, "sm_nile",
                              "test disposition: budget-stopped "
                              "attempt adjudicated failed")
    st, _ = ex.adjudicate_unit_shallow(u, "sm_nile")
    assert st == "TERMINAL_FAILED"
    term = json.loads((u / "TERMINAL_sm_nile.json").read_text())
    assert term["operator_disposition"] is True
    assert term["failure_class"] == "OPERATOR_DISPOSITION"
    with pytest.raises(SystemExit, match="applies only to"):
        ex.declare_attempt_failed(root, "sm_nile", "again")
    # in-assay checkpoints reached origins/arms before the trip
    assert calls[0] == "origin0:start"
    assert any("epoch_candidate" in c for c in calls)
    # C55: the stop also fires BETWEEN ORIGINS and BETWEEN SEEDS
    for trip_label, sub in (("origin1:start", "between origins"),
                            ("mlp_seed12:start",
                             "between seeds")):
        subroot = tmp_path / f"t2stop_{trip_label.split(':')[0]}"
        subroot.mkdir(mode=0o700)

        def boundary_guard(label, _t=trip_label):
            if label.endswith(_t):
                raise ex.T2BudgetStop(f"stop {_t}", label)

        with pytest.raises(SystemExit, match="T2_BUDGET_STOP"):
            ex.run_unit(hz, co, unit, design, authority, subroot,
                        "mechanical_rehearsal",
                        guard=boundary_guard)
        st, why = ex.adjudicate_unit_shallow(subroot / "units",
                                             "sm_nile")
        assert st == "UNCERTAIN" and "claim without" in why, sub
        assert not (subroot / "units"
                    / "RECORD_sm_nile.json").exists()
    # a NON-budget in-unit failure DOES write a typed terminal
    root2 = tmp_path / "t2fail"
    root2.mkdir(mode=0o700)

    def broken_guard(label):
        if "ridge_done" in label:
            raise ValueError("synthetic in-unit defect")

    with pytest.raises(ValueError):
        ex.run_unit(hz, co, unit, design, authority, root2,
                    "mechanical_rehearsal", guard=broken_guard)
    st, _ = ex.adjudicate_unit_shallow(root2 / "units", "sm_nile")
    assert st == "TERMINAL_FAILED"


def test_c52_2_budget_mechanics_bite(tmp_path, monkeypatch):
    """C52: the durable accumulated wall (resume NEVER renews the
    4 h), the RSS bound, the design-declared stop-file location and
    the supervised worker's typed harvests."""
    ex = _exec_mod()
    ledger = tmp_path / "wall.jsonl"
    ledger.write_text(json.dumps(
        {"session": "prior", "elapsed_seconds": 100.0}) + "\n")
    stop = tmp_path / "T2_STOP"
    g = ex.BudgetGuard({"max_wall_seconds": 50,
                        "max_rss_bytes": 8 << 30}, stop, ledger,
                       "fresh")
    assert g.prior_wall == 100.0
    with pytest.raises(SystemExit, match="never renews"):
        g.check("between_units:probe")
    g.close()
    # RSS bound
    g2 = ex.BudgetGuard({"max_wall_seconds": 10 ** 6,
                         "max_rss_bytes": 1}, stop,
                        tmp_path / "w2.jsonl", "s2")
    with pytest.raises(SystemExit, match="RSS"):
        g2.check("x")
    g2.close()
    # stop-file at the DESIGN-DECLARED state root
    g3 = ex.BudgetGuard({"max_wall_seconds": 10 ** 6,
                         "max_rss_bytes": 8 << 30}, stop,
                        tmp_path / "w3.jsonl", "s3")
    g3.check("pre")
    stop.write_text("halt")
    with pytest.raises(SystemExit, match="state root"):
        g3.check("post")
    g3.close()
    if ex.SEALED_PATH.exists():
        design = json.loads(ex.SEALED_PATH.read_text())
        assert ex.resolve_stop_file(design) == \
            ex.STATE / "T2_STOP"
        assert design["resource_contract"]["stop_file"] == \
            "<state_root>/T2_STOP"
    # supervised worker: typed WALL_KILLED and CRASH harvests
    monkeypatch.setattr(ex, "PER_FIT_WALL_SECONDS", 1.0)
    sup = ex.make_fit_supervisor({"max_rss_bytes": 8 << 30})

    class _Hang:
        def __call__(self):
            import time as _t
            _t.sleep(30)

    class _Boom:
        def __call__(self):
            raise ValueError("synthetic fit crash")

    class _Ok:
        def __call__(self):
            return 41 + 1

    assert sup(_Ok(), "ok") == 42
    with pytest.raises(SystemExit, match="WALL_KILLED"):
        sup(_Hang(), "hang")
    with pytest.raises(SystemExit, match="CRASH.*synthetic fit"):
        sup(_Boom(), "boom")
    # the harness passes the guard INTO epoch candidates (source)
    hsrc = (REPO / "tools/t2_assay_harness.py").read_text()
    assert "guard(f\"{label}:epoch_candidate_{epochs}\")" in hsrc
    assert "fit_supervisor" in hsrc
    esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    assert "guard=guard.check" in esrc
    assert "fit_supervisor=supervisor" in esrc


def test_c53_1_plan_is_pure_and_gates_precede_effects(tmp_path,
                                                      monkeypatch):
    """C53: --plan and a failed gate leave ZERO writes (the
    out_root is not even created); with gates stubbed open the
    plan still writes nothing; durable effects exist only in
    --execute after all gates."""
    import t2_confirmatory as conf
    ex = _exec_mod()
    if not ex.SEALED_PATH.exists():
        pytest.skip("sealed design absent on this host")
    monkeypatch.setattr(conf, "T2_EXECUTION_RECORD_PATH",
                        tmp_path / "missing.json")
    t = tmp_path / "planroot"
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(SystemExit,
                       match="T2_EXECUTION_RECORD_REQUIRED"):
        ex.main(["--plan", "--out-root", str(t)])
    assert not t.exists()
    with pytest.raises(SystemExit,
                       match="T2_EXECUTION_RECORD_REQUIRED"):
        ex.main(["--execute", "--out-root", str(t)])
    assert not t.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    # gates stubbed OPEN: the plan prints and still writes nothing
    fake_facts = {"gates": "ALL_OPEN",
                  "execution_record_sha256": "e" * 64,
                  "review_record_sha256": "r" * 64,
                  "design_file_sha256": "d" * 64,
                  "manifest_sha256": "m" * 64,
                  "census_sha256": "c" * 64,
                  "pinned_commit": "0" * 40}
    monkeypatch.setattr(conf, "verify_confirmatory_gates",
                        lambda *a, **k: fake_facts)
    rc = ex.main(["--plan", "--out-root", str(t)])
    assert rc == 0
    assert not t.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == before
    # source: in main, gates run BEFORE mkdir/lock/ledger; the
    # ledger exists only on the --execute path
    esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    seg = esrc[esrc.index("def main"):esrc.index("def rehearse")]
    assert seg.index("verify_confirmatory_gates") < \
        seg.index("if args.plan") < \
        seg.index("out_root.mkdir") < \
        seg.index("acquire_lock") < \
        seg.index("open_attempt_ledger")
    plan_seg = seg[seg.index("if args.plan"):
                   seg.index("out_root.mkdir")]
    for effect in ("mkdir", "acquire_lock", "open_attempt_ledger",
                   "_excl_write", "_heartbeat"):
        assert effect not in plan_seg


def test_c54_1_monotonic_lock_and_crash_boundaries(tmp_path):
    """C54: locks are never unlinked — release is a durable
    record; a live pid refuses; a provably dead pid requires the
    EXPLICIT recorded takeover; crash remnants adjudicate
    UNCERTAIN with typed causes."""
    ex = _exec_mod()
    root = tmp_path / "lockroot"
    root.mkdir(mode=0o700)
    n1 = ex.acquire_lock(root, "sess-one")
    assert n1 == 1
    with pytest.raises(SystemExit, match="alive"):
        ex.acquire_lock(root, "sess-two")
    ex.release_lock(root, 1, "sess-one")
    assert (root / "locks" / "RELEASE_000001.json").exists()
    assert (root / "locks" / "SESSION_000001.json").exists()
    n2 = ex.acquire_lock(root, "sess-two")
    assert n2 == 2
    ex.release_lock(root, 2, "sess-two")
    # a dead-pid session without release: stale, never stolen
    import subprocess
    p = subprocess.Popen(["true"])
    p.wait()
    doc = {"schema": "agent_multi.t2_lock_session.v1",
           "session": 3, "session_uuid": "ghost",
           "pid": p.pid, "started_wall": 0.0}
    doc["session_sha256"] = ex._self_sha(doc, "session_sha256")
    sp = root / "locks" / "SESSION_000003.json"
    sp.write_text(json.dumps(doc))
    os.chmod(sp, 0o600)
    with pytest.raises(SystemExit,
                       match="explicit recorded takeover"):
        ex.acquire_lock(root, "sess-four")
    n4 = ex.acquire_lock(root, "sess-four", takeover_stale=True)
    assert n4 == 4
    assert (root / "locks" / "TAKEOVER_000003.json").exists()
    ex.release_lock(root, 4, "sess-four")
    # locks are never unlinked in the productive source
    esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    assert "lock.unlink" not in esrc
    assert ".unlink(missing_ok=True)" not in esrc
    # crash boundaries adjudicate UNCERTAIN with typed causes
    u = root / "units"
    u.mkdir(mode=0o700)
    (u / "ARRAYS_probe.npz").write_bytes(b"partial")
    st, why = ex.adjudicate_unit_shallow(u, "probe")
    assert st == "UNCERTAIN" and "arrays without a record" in why
    (u / "ARRAYS_probe.npz").unlink()
    bad = {"schema": "agent_multi.t2_unit_terminal.v2",
           "unit_id": "probe", "terminal": "FAILED",
           "terminal_sha256": "0" * 64}
    tp = u / "TERMINAL_probe.json"
    tp.write_text(json.dumps(bad))
    os.chmod(tp, 0o600)
    st, why = ex.adjudicate_unit_shallow(u, "probe")
    assert st == "UNCERTAIN" and "does not re-derive" in why


def test_c55_resume_order_census_and_separated_counts():
    """C55/C56: resume verifies COMPLETED records (deep, under
    current authority) before counting terminals, before budget/
    load; the census is exact; every reported count separates
    done / failed_preserved / resumed_verified — a skip is never
    published as a pass."""
    ex = _exec_mod()
    esrc = (REPO / "tools/t2_confirmatory_executor.py").read_text()
    seg = esrc[esrc.index("def main"):esrc.index("def rehearse")]
    i_adj = seg.index("adjudicate_unit_shallow")
    i_completed = seg.index('== "COMPLETED"')
    i_term = seg.index('== "TERMINAL_FAILED"')
    i_load_new = seg.rindex("load_bank_unit(")
    assert i_adj < i_completed < i_term < i_load_new
    assert '"resumed_verified"' in seg
    assert '"failed_preserved"' in seg
    assert "UNCERTAIN units block" in seg
    if ex.SEALED_PATH.exists():
        d = json.loads(ex.SEALED_PATH.read_text())
        w = ex.census_of_work(d)
        assert w["units"] == 242 and w["origins_per_unit"] == 2
        assert w["model_fits"] == 242 * 2 * 4 * 4 == 7744
        assert w["baseline_evals"] == 484
    # rehearsal output separates its facts too
    rseg = esrc[esrc.index("def rehearse"):]
    assert '"records_verified_from_persisted_arrays"' in rseg
    assert '"sealed_bank_series_touched": 0' in rseg
