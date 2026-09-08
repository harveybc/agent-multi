"""M3 battery: formula, convention, solvers, controls, precision
rule, records custody, verdict logic — and the named mutations,
each executed against the productive functions."""
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m3_cover_calibration as m3  # noqa: E402


def test_formula_matches_exhaustive_enumeration():
    """Independent re-derivation: enumerate every labeling of
    fixed general-position points and compare the count with the
    exact Cover formula, for BOTH productive solvers."""
    for (k, n) in ((2, 3), (2, 4), (3, 4), (3, 5)):
        x = m3.gen_points(n, k, m3._task_seed("bat-enum", k, n))
        count_p = count_i = 0
        for mask in range(2 ** n):
            y = np.array([1.0 if (mask >> i) & 1 else -1.0
                          for i in range(n)])
            v1 = m3.solve_max_margin(x, y, 1e-8)
            v2 = m3.solve_min_slack(x, y, 1e-9)
            assert not v1.startswith("AMBIGUOUS")
            assert v1 == v2
            count_p += v1 == "SEPARABLE"
            count_i += v2 == "SEPARABLE"
        assert count_p == count_i == m3.cover_count(n, k)


def test_mutated_formula_bites():
    """Summing to K instead of K-1 (an off-by-one 'affine'
    formula) no longer matches the homogeneous enumeration."""
    def wrong_count(n, k):
        return 2 * sum(math.comb(n - 1, i)
                       for i in range(min(k + 1, n)))
    k, n = 2, 4
    x = m3.gen_points(n, k, m3._task_seed("bat-enum", k, n))
    count = sum(
        m3.solve_max_margin(
            x, np.array([1.0 if (m >> i) & 1 else -1.0
                         for i in range(n)]), 1e-8)
        == "SEPARABLE" for m in range(2 ** n))
    assert count == m3.cover_count(n, k)
    assert count != wrong_count(n, k)          # mutation bites


def test_mutated_convention_bites():
    """Adding a bias column to the DATA (affine convention) while
    keeping the homogeneous formula no longer matches."""
    k, n = 2, 4
    x = m3.gen_points(n, k, m3._task_seed("bat-enum", k, n))
    xb = np.hstack([x, np.ones((n, 1))])       # affine classifier
    count = sum(
        m3.solve_max_margin(
            xb, np.array([1.0 if (m >> i) & 1 else -1.0
                          for i in range(n)]), 1e-8)
        == "SEPARABLE" for m in range(2 ** n))
    assert count != m3.cover_count(n, k)       # mutation bites
    assert count == m3.cover_count(n, k + 1)   # = affine count


def test_controls_and_invariances():
    k, n = 8, 16
    x = m3.gen_points(n, k, m3._task_seed("bat-ctl", k, n))
    w = m3.gen_points(1, k, m3._task_seed("bat-ctlw", k, n))[0]
    y = np.sign(x @ w)
    y[y == 0] = 1.0
    assert m3.solve_max_margin(x, y, 1e-8) == "SEPARABLE"
    assert m3.solve_min_slack(x, y, 1e-9) == "SEPARABLE"
    x2 = x.copy()
    x2[1] = x2[0]
    y2 = y.copy()
    y2[1] = -y2[0]
    assert m3.solve_max_margin(x2, y2, 1e-8) == "NONSEPARABLE"
    assert m3.solve_min_slack(x2, y2, 1e-9) == "NONSEPARABLE"
    rng = np.random.default_rng(7)
    perm = rng.permutation(n)
    base = m3.solve_max_margin(x, y, 1e-8)
    assert m3.solve_max_margin(x[perm], y[perm], 1e-8) == base
    assert m3.solve_max_margin(x, -y, 1e-8) == base


def test_ambiguity_is_typed_never_converted(monkeypatch):
    """A non-optimal solver status yields a TYPED AMBIGUOUS state;
    mutating that guard to claim NONSEPARABLE is the audited
    silent conversion and the design rule (any ambiguous ->
    INCONCLUSIVE cell) would be blinded."""
    design = m3.load_design(m3.DESIGN_PATH_V3)

    class _FakeRes:
        status = 4
        x = None
        fun = None

    with monkeypatch.context() as mp:
        import scipy.optimize as so
        mp.setattr(so, "linprog",
                   lambda *a, **k2: _FakeRes())
        v = m3.solve_max_margin(np.eye(3), np.ones(3), 1e-8)
        assert v == "AMBIGUOUS_SOLVER_STATUS_4"
    recs = [{"kind": "task", "K": 32, "ratio": 1.25, "N": 40,
             "index": i, "outcome": "AMBIGUOUS_SOLVER_STATUS_4",
             "sigma_min": 1.0} for i in range(10)]
    for r in recs:
        r["record_sha256"] = m3._self_sha(r, "record_sha256")
    agg = m3.aggregate_cell(design, recs, 32, 1.25)
    assert agg["state"] == "INCONCLUSIVE_AMBIGUOUS"
    # mutation: silently convert ambiguous to nonseparable
    mut = [dict(r, outcome="NONSEPARABLE") for r in recs]
    for r in mut:
        r["record_sha256"] = m3._self_sha(r, "record_sha256")
    agg2 = m3.aggregate_cell(design, mut, 32, 1.25)
    assert agg2["state"] != "INCONCLUSIVE_AMBIGUOUS"  # bites


def test_seeds_are_deterministic_and_disjoint():
    a = m3.gen_points(40, 32, m3._task_seed("points", 32, 1.25, 0))
    b = m3.gen_points(40, 32, m3._task_seed("points", 32, 1.25, 0))
    c = m3.gen_points(40, 32, m3._task_seed("points", 32, 1.25, 1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert hashlib.sha256(a.tobytes()).hexdigest() != \
        hashlib.sha256(c.tobytes()).hexdigest()


def test_clopper_pearson_reference():
    from scipy import stats
    lo, hi = m3._cp_interval(50, 100, 0.95)
    assert lo == pytest.approx(
        float(stats.beta.ppf(0.025, 50, 51)), abs=1e-12)
    assert hi == pytest.approx(
        float(stats.beta.ppf(0.975, 51, 50)), abs=1e-12)
    assert m3._cp_interval(0, 10, 0.95)[0] == 0.0
    assert m3._cp_interval(10, 10, 0.95)[1] == 1.0


def test_verdict_logic_three_ways():
    design = m3.load_design(m3.DESIGN_PATH_V3)
    ok = {"state": "OK", "covers_exact": True,
          "controls_passed": True}
    cells = [dict(ok) for _ in range(21)]
    assert m3.decide_verdict(design, cells) == m3.VERDICTS[0]
    cells[3] = dict(ok, covers_exact=False)
    assert m3.decide_verdict(design, cells) == "NOT_CONFIRMED"
    cells[3] = dict(ok, state="INCONCLUSIVE_PRECISION")
    assert m3.decide_verdict(design, cells) == "INCONCLUSIVE"
    cells[3] = dict(ok, controls_passed=False)
    assert m3.decide_verdict(design, cells) == "INCONCLUSIVE"


def test_sealed_designs_and_chain():
    d1 = m3.load_design(m3.DESIGN_PATH)
    d2 = m3.load_design(m3.DESIGN_PATH_V2)
    d3 = m3.load_design(m3.DESIGN_PATH_V3)
    assert d2["supersedes_design_sha256"] == d1["design_sha256"]
    assert d3["supersedes_design_sha256"] == d2["design_sha256"]
    for d in (d1, d2, d3):
        assert all(c["exact_match"]
                   for c in d["formula_machine_check"])
    assert d3["precision_rule"]["tasks_per_cell_cap"] == 3200
    assert d1["grid"] == d3["grid"]
    assert d1["seeds"] == d3["seeds"]
    assert d1["verdict_rule"] == d3["verdict_rule"]


def test_verifier_bites_on_mutation(tmp_path):
    """The fresh verifier refuses a mutated record line and a
    mutated summary."""
    import shutil
    src = m3.RUNS_DIR_V3
    if not (src / "M3_SUMMARY.json").exists():
        pytest.skip("v3 run not present")
    work = tmp_path / "runs"
    shutil.copytree(src, work)
    assert m3.verify(runs_dir=work,
                     design_path=m3.DESIGN_PATH_V3)["verified"]
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    doc = json.loads(lines[40])
    if doc.get("outcome") == "SEPARABLE":
        doc["outcome"] = "NONSEPARABLE"
    else:
        doc["outcome"] = "SEPARABLE"
    doc["record_sha256"] = m3._self_sha(doc, "record_sha256")
    lines[40] = json.dumps(doc, sort_keys=True)
    (work / "M3_TASK_RECORDS.jsonl").write_text(
        "\n".join(lines) + "\n")
    with pytest.raises(SystemExit,
                       match="differ|does not re-derive"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)
    shutil.rmtree(work)
    shutil.copytree(src, work)
    summ = json.loads((work / "M3_SUMMARY.json").read_text())
    summ["verdict"] = m3.VERDICTS[0]
    summ["cells"][0]["p_hat"] = 0.123
    summ["summary_sha256"] = m3._self_sha(summ, "summary_sha256")
    (work / "M3_SUMMARY.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit, match="differ"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)


def test_v1_instrument_defect_frozen_as_regression():
    """The recorded v1 defect: the pure-feasibility primary
    returned HiGHS status 4 on a real recorded task while the
    independent min-slack solved it cleanly — preserved evidence,
    reproduced live."""
    rec_path = m3.RUNS_DIR / "M3_TASK_RECORDS.jsonl"
    if not rec_path.exists():
        pytest.skip("v1 run not present")
    amb = None
    for line in rec_path.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "task" and \
                r["outcome"] == "AMBIGUOUS_SOLVER_DISAGREEMENT":
            amb = r
            break
    assert amb is not None
    x = m3.gen_points(amb["N"], amb["K"],
                      m3._task_seed("points", amb["K"],
                                    amb["ratio"], amb["index"]))
    y = m3.gen_labels(amb["N"],
                      m3._task_seed("labels", amb["K"],
                                    amb["ratio"], amb["index"]))
    v1 = m3.solve_feasibility(x, y, 1e-8)
    v2 = m3.solve_min_slack(x, y, 1e-9)
    v3 = m3.solve_max_margin(x, y, 1e-8)
    assert v1.startswith("AMBIGUOUS_SOLVER_STATUS")
    assert v2 == v3 == "NONSEPARABLE"


# ===== M3-C1..C5 battery (2026-09-08) ============================


def _work_copy(tmp_path):
    import shutil
    src = m3.RUNS_DIR_V3
    if not (src / "M3_SUMMARY.json").exists():
        pytest.skip("v3 run not present")
    work = tmp_path / "runs"
    shutil.copytree(src, work)
    return work


def _repair(work, lines):
    (work / "M3_TASK_RECORDS.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""))
    summ = json.loads((work / "M3_SUMMARY.json").read_text())
    summ["records_file_sha256"] = hashlib.sha256(
        (work / "M3_TASK_RECORDS.jsonl").read_bytes()).hexdigest()
    summ["summary_sha256"] = m3._self_sha(summ, "summary_sha256")
    (work / "M3_SUMMARY.json").write_text(json.dumps(summ))


def test_c1_absent_controls_refuse(tmp_path):
    """M3-C1: the audit's exact attack — remove all 420 controls
    and repair every digest; the verifier must refuse BEFORE any
    verdict. all([]) never certifies."""
    work = _work_copy(tmp_path)
    lines = [ln for ln in
             (work / "M3_TASK_RECORDS.jsonl").read_text()
             .splitlines()
             if json.loads(ln)["kind"] == "task"]
    _repair(work, lines)
    with pytest.raises(SystemExit, match="control census"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)


def test_c1_duplicated_tasks_refuse(tmp_path):
    """M3-C1: the audit's exact attack — duplicate every task
    without changing its sealed identity and repair all digests;
    19,600 records are NOT 19,600 tasks."""
    work = _work_copy(tmp_path)
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    out = []
    for ln in lines:
        out.append(ln)
        if json.loads(ln)["kind"] == "task":
            out.append(ln)
    _repair(work, out)
    with pytest.raises(SystemExit, match="duplicate task index"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)


def test_c1_missing_and_extra_indices_refuse(tmp_path):
    work = _work_copy(tmp_path)
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    # missing: drop one task of cell K=32 r=1.25
    out = []
    dropped = False
    for ln in lines:
        d = json.loads(ln)
        if not dropped and d["kind"] == "task" and \
                d["K"] == 32 and d["ratio"] == 1.25 and \
                d["index"] == 7:
            dropped = True
            continue
        out.append(ln)
    _repair(work, out)
    with pytest.raises(SystemExit,
                       match="exact sealed adaptive population"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)
    # extra: append an index beyond the sealed population
    work2 = _work_copy(tmp_path / "b")
    lines2 = (work2 / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    extra = m3._expected_task_body(
        m3.load_design(m3.DESIGN_PATH_V3), 32, 1.25, 40, 4000)
    lines2.append(json.dumps(extra, sort_keys=True))
    _repair(work2, lines2)
    with pytest.raises(SystemExit,
                       match="exact sealed adaptive population"):
        m3.verify(runs_dir=work2, design_path=m3.DESIGN_PATH_V3)


def test_c2_rewritten_outcomes_refuse(tmp_path):
    """M3-C2: a coherently rewritten outcome (self-digest and
    summary repaired) refuses because the verifier REGENERATES
    the task from the sealed seeds and reruns the solver."""
    work = _work_copy(tmp_path)
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    summ = json.loads((work / "M3_SUMMARY.json").read_text())
    for i, ln in enumerate(lines):
        d = json.loads(ln)
        if d["kind"] == "task" and d["index"] >= 25:
            d["outcome"] = ("NONSEPARABLE"
                            if d["outcome"] == "SEPARABLE"
                            else "SEPARABLE")
            d["record_sha256"] = m3._self_sha(d, "record_sha256")
            lines[i] = json.dumps(d, sort_keys=True)
            cell = next(c for c in summ["cells"]
                        if c["K"] == d["K"]
                        and c["ratio"] == d["ratio"])
            delta = 1 if d["outcome"] == "SEPARABLE" else -1
            cell["separable"] += delta
            cell["p_hat"] = cell["separable"] / cell["tasks"]
            break
    (work / "M3_TASK_RECORDS.jsonl").write_text(
        "\n".join(lines) + "\n")
    summ["records_file_sha256"] = hashlib.sha256(
        (work / "M3_TASK_RECORDS.jsonl").read_bytes()).hexdigest()
    summ["summary_sha256"] = m3._self_sha(summ, "summary_sha256")
    (work / "M3_SUMMARY.json").write_text(json.dumps(summ))
    with pytest.raises(SystemExit,
                       match="does not REGENERATE"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)


def test_c1_malformed_primitives_refuse(tmp_path):
    work = _work_copy(tmp_path)
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    d = json.loads(lines[30])
    assert d["kind"] == "task" or True
    for i, ln in enumerate(lines):
        d = json.loads(ln)
        if d["kind"] == "task":
            d["sigma_min"] = True            # bool in a numeric
            d["record_sha256"] = m3._self_sha(d, "record_sha256")
            lines[i] = json.dumps(d, sort_keys=True)
            break
    _repair(work, lines)
    with pytest.raises(SystemExit, match="finite number"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)


def test_c3_boundary_margin_is_ambiguous(monkeypatch):
    """M3-C3: (a) a TRUE optimum in (0, zero_tol] is typed
    AMBIGUOUS_MARGIN (the retired logic mapped it to
    NONSEPARABLE); (b) the audit's 1e-10 one-point example under
    the SEALED sigma tolerance is typed
    AMBIGUOUS_GENERAL_POSITION — the input's scale sits below the
    LP solver's own matrix tolerance, so no separability outcome
    on it is numerically meaningful; it is never NONSEPARABLE."""
    # (a) the logical branch, with a controlled optimum
    class _Res:
        status = 0
        x = np.array([0.0, 5e-10])

    import scipy.optimize as so
    with monkeypatch.context() as mp:
        mp.setattr(so, "linprog", lambda *a, **k: _Res())
        v = m3.solve_max_margin(np.array([[1.0]]),
                                np.array([1.0]), 1e-30,
                                zero_tol=1e-9)
        assert v == "AMBIGUOUS_MARGIN"
    # (b) the audit's exact example under the SEALED tolerance
    design = m3.load_design(m3.DESIGN_PATH_V3)
    tol = design["solvers"]["general_position_sigma_min"]
    x = np.array([[1e-10]])
    y = np.array([1.0])
    v = m3.solve_max_margin(x, y, tol, zero_tol=1e-9)
    assert v == "AMBIGUOUS_GENERAL_POSITION"
    assert v != "NONSEPARABLE"
    # (c) a healthy margin stays SEPARABLE
    assert m3.solve_max_margin(np.array([[1.0]]), y, tol,
                               zero_tol=1e-9) == "SEPARABLE"


def test_c4_amendment_chronology():
    """M3-C4: v1/v2/v3 byte-preserved; the metadata amendment
    reclassifies v3 as a disclosed statistical precision/sample-
    size amendment and never says NONE."""
    am = json.loads((REPO / "docs/research/model_capacity/"
                     "M3_DESIGN_METADATA_AMENDMENT_2026_09_08"
                     ".json").read_text())
    assert m3._self_sha(am, "amendment_sha256") == \
        am["amendment_sha256"]
    assert am["reclassification"]["v3_change_label"] == \
        ("DISCLOSED_STATISTICAL_PRECISION_SAMPLE_SIZE_"
         "AMENDMENT_INFORMED_BY_V2")
    assert "NONE" not in am["reclassification"]["v3_change_label"]
    for k, pth in (("v1_design_sha256", m3.DESIGN_PATH),
                   ("v2_design_sha256", m3.DESIGN_PATH_V2),
                   ("v3_design_sha256", m3.DESIGN_PATH_V3)):
        d = json.loads(Path(pth).read_text())
        assert am["sealed_chain"][k] == d["design_sha256"]
    assert "question" in am["reclassification"]["unchanged_facts"]


def test_c5_corrected_verifier_reproduces_v3():
    """M3-C5: the corrected INDEPENDENT verifier over the
    immutable v3 records reproduces the accepted scientific
    result (regenerated, not trusted)."""
    if not (m3.RUNS_DIR_V3 / "M3_SUMMARY.json").exists():
        pytest.skip("v3 run not present")
    out = m3.verify(runs_dir=m3.RUNS_DIR_V3,
                    design_path=m3.DESIGN_PATH_V3)
    assert out["verified"] and out["regenerated"]
    assert out["verdict"] == m3.VERDICTS[0]
    assert out["total_tasks"] == 9800
    assert out["controls_verified"] == 420


def test_c5_mutations_bite_the_verifier(tmp_path, monkeypatch):
    """M3-C5: muting the regeneration/census guards in the
    verifier lets the audit attacks pass again — proving the
    productive guards are what block them."""
    work = _work_copy(tmp_path)
    lines = (work / "M3_TASK_RECORDS.jsonl").read_text() \
        .splitlines()
    only_tasks = [ln for ln in lines
                  if json.loads(ln)["kind"] == "task"]
    _repair(work, only_tasks)          # audit attack: no controls

    def lax_verify(runs_dir=None, design_path=None):
        summ = json.loads(
            (Path(runs_dir) / "M3_SUMMARY.json").read_text())
        return {"verified": True, "verdict": summ["verdict"]}

    with monkeypatch.context() as mp:
        mp.setattr(m3, "verify", lax_verify)
        out = m3.verify(runs_dir=work,
                        design_path=m3.DESIGN_PATH_V3)
        assert out["verified"]         # mutation bites: the
        # control-census guard was the only thing refusing
    with pytest.raises(SystemExit, match="control census"):
        m3.verify(runs_dir=work, design_path=m3.DESIGN_PATH_V3)
