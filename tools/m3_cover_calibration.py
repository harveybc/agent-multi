#!/usr/bin/env python3
"""M3: confirmatory Cover/MacKay single-neuron calibration (order
T2 C57-C65 + M3, stage C1 of the M3-M6 draft).

M3.0 pre-result seal: `--seal-design` turns stage C1 into an
executable IMMUTABLE design BEFORE any separability outcome is
computed. The seal embeds and machine-checks the EXACT finite-N
Cover counting formula on small fully ENUMERATED cases (every
labeling of fixed general-position point sets) so the formula,
the data-generation convention and the feasibility solver are
proven to share ONE convention. The classifier is HOMOGENEOUS
(linear through the origin, no bias): data carries no ones
column, the formula counts homogeneous dichotomies, and the
solver searches w only.

    C(N, K) = 2 * sum_{i=0}^{K-1} binom(N-1, i)
    P(N, K) = C(N, K) / 2**N        (Cover 1965, Theorem 1)

Primary citation: T. M. Cover, "Geometrical and Statistical
Properties of Systems of Linear Inequalities with Applications in
Pattern Recognition", IEEE Transactions on Electronic Computers
EC-14(3):326-334, 1965. Secondary: D. J. C. MacKay, "Information
Theory, Inference, and Learning Algorithms", CUP 2003, chapter 40
(Capacity of a Single Neuron). The 2K statement is the asymptotic
transition; the sealed reference is the exact finite-N count.

M3.1: the primary outcome is DETERMINISTIC linear feasibility
(HiGHS LP: find w with y_i * <w, x_i> >= 1), never perceptron
training. An independently FORMULATED solver (minimum-slack LP:
min 1's subject to y_i <w, x_i> >= 1 - s_i, s >= 0; separable iff
optimum == 0) cross-checks a frozen subset and every enumerated
case. Controls per cell: planted separable positives, deliberately
inconsistent negatives (a duplicated point with opposite labels),
permutation and sign-symmetry invariance on a frozen subset, and a
numerical general-position diagnostic (sigma_min of X). Solver
disagreement, numerical ambiguity or resource exhaustion is TYPED
and stays in the denominator — it is never silently converted to
nonseparable; any typed-ambiguous task makes its cell
INCONCLUSIVE under the sealed rule.

M3.2: CPU-only with CUDA hidden, ONE logical worker, six
wall-hours maximum, observable heartbeat and stop-file. Every
task writes one immutable self-digested record line; `--verify`
reconstructs every aggregate and the verdict from those records
in a FRESH process. Allowed verdicts:
COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_PRECISION,
NOT_CONFIRMED, INCONCLUSIVE. This stage estimates no MLP
intelligence, no residual capacity, no Kolmogorov complexity and
grants no DOIN feature."""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DESIGN_PATH = (REPO / "docs/research/model_capacity/"
               "M3_SEALED_DESIGN_2026_09_08.json")
DESIGN_PATH_V2 = (REPO / "docs/research/model_capacity/"
                  "M3_SEALED_DESIGN_V2_2026_09_08.json")
DESIGN_PATH_V3 = (REPO / "docs/research/model_capacity/"
                  "M3_SEALED_DESIGN_V3_2026_09_08.json")
RUNS_DIR = REPO / "docs/audits/evidence/m3_runs"
RUNS_DIR_V2 = REPO / "docs/audits/evidence/m3_runs_v2"
RUNS_DIR_V3 = REPO / "docs/audits/evidence/m3_runs_v3"
MASTER_SEED_PHRASE = "m3_cover_calibration_2026_09_08"

VERDICTS = ("COVER_CALIBRATION_CONFIRMED_WITHIN_DECLARED_"
            "PRECISION", "NOT_CONFIRMED", "INCONCLUSIVE")


class M3Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


def cover_count(n: int, k: int) -> int:
    """Exact finite-N count of homogeneously linearly separable
    dichotomies of n points in general position in R^k."""
    if n <= 0 or k <= 0:
        raise M3Refusal("cover_count needs positive N, K")
    return 2 * sum(math.comb(n - 1, i) for i in range(min(k, n)))


def cover_probability(n: int, k: int):
    """Exact P(separable) for random independent binary labels."""
    from fractions import Fraction
    return Fraction(cover_count(n, k), 2 ** n)


def _task_seed(*parts) -> int:
    h = hashlib.sha256(("|".join(
        [MASTER_SEED_PHRASE, *map(str, parts)])).encode())
    return int.from_bytes(h.digest()[:8], "big")


def gen_points(n: int, k: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, k))


def gen_labels(n: int, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=n)


def general_position_sigma(x) -> float:
    """A FULL-RANK numerical diagnostic (smallest singular value
    of the whole matrix). It does NOT prove that every required
    subset of points is in general position; Gaussian generation
    supplies that almost surely, and the exact enumerated cases
    remain the finite check."""
    import numpy as np
    s = np.linalg.svd(x, compute_uv=False)
    return float(s[min(x.shape) - 1])


def solve_max_margin(x, y, tol_sigma: float,
                     zero_tol: float = 1e-9):
    """PRIMARY (design v2): deterministic HiGHS Chebyshev
    max-margin LP — maximize d subject to y_i <w, x_i> >= d,
    -1 <= w_j <= 1, d <= 1. Always feasible (w=0, d=0), so a
    non-optimal status is a true anomaly. Separable iff the
    optimum margin d* > zero_tol; d* in [0, zero_tol] is typed
    AMBIGUOUS_MARGIN (a measure-zero boundary under the sealed
    general-position diagnostic). Never a silent conversion."""
    import numpy as np
    from scipy.optimize import linprog
    if general_position_sigma(x) < tol_sigma and \
            x.shape[0] >= x.shape[1]:
        return "AMBIGUOUS_GENERAL_POSITION"
    n, k = x.shape
    c = np.zeros(k + 1)
    c[-1] = -1.0                       # maximize d
    a_ub = np.hstack([-(y[:, None] * x), np.ones((n, 1))])
    b_ub = np.zeros(n)
    bounds = [(-1, 1)] * k + [(None, 1)]
    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                  method="highs")
    if res.status != 0:
        return f"AMBIGUOUS_SOLVER_STATUS_{res.status}"
    d_star = float(res.x[-1])
    if d_star > zero_tol:
        return "SEPARABLE"
    if d_star <= 0.0:
        return "NONSEPARABLE"
    # 0 < d* <= zero_tol: exactly what the sealed design types as
    # AMBIGUOUS_MARGIN. Data whose scale sits BELOW the solver's
    # own matrix tolerance (the audit's 1e-10 one-point example)
    # never reaches this branch honestly: the sealed sigma_min
    # diagnostic (1e-8) types it AMBIGUOUS_GENERAL_POSITION
    # first, because no LP outcome on such input is numerically
    # meaningful.
    return "AMBIGUOUS_MARGIN"


def solve_feasibility(x, y, tol_sigma: float):
    """PRIMARY of design v1 (RETIRED for v2 after the recorded
    instrument defect: HiGHS status 4 on ~5% of infeasible pure
    feasibility systems; the v1 run is preserved INCONCLUSIVE).
    Deterministic HiGHS feasibility for homogeneous strict
    separability. Returns 'SEPARABLE', 'NONSEPARABLE' or a typed
    'AMBIGUOUS_*' state — never a silent conversion."""
    import numpy as np
    from scipy.optimize import linprog
    if general_position_sigma(x) < tol_sigma and \
            x.shape[0] >= x.shape[1]:
        return "AMBIGUOUS_GENERAL_POSITION"
    a_ub = -(y[:, None] * x)
    b_ub = -np.ones(x.shape[0])
    res = linprog(c=np.zeros(x.shape[1]), A_ub=a_ub, b_ub=b_ub,
                  bounds=[(None, None)] * x.shape[1],
                  method="highs")
    if res.status == 0:
        return "SEPARABLE"
    if res.status == 2:
        return "NONSEPARABLE"
    return f"AMBIGUOUS_SOLVER_STATUS_{res.status}"


def solve_min_slack(x, y, zero_tol: float):
    """INDEPENDENT formulation: minimize total slack; separable
    iff the optimum is exactly zero (within the sealed
    tolerance)."""
    import numpy as np
    from scipy.optimize import linprog
    n, k = x.shape
    c = np.concatenate([np.zeros(k), np.ones(n)])
    a_ub = np.hstack([-(y[:, None] * x), -np.eye(n)])
    b_ub = -np.ones(n)
    bounds = [(None, None)] * k + [(0, None)] * n
    res = linprog(c=c, A_ub=a_ub, b_ub=b_ub, bounds=bounds,
                  method="highs")
    if res.status != 0:
        return f"AMBIGUOUS_SOLVER_STATUS_{res.status}"
    return "SEPARABLE" if res.fun <= zero_tol else "NONSEPARABLE"


def enumerate_check(k: int, n: int, seed: int,
                    tol_sigma: float, zero_tol: float,
                    primary=None) -> dict:
    """Machine check of formula+convention+solver: enumerate ALL
    2^n labelings of one fixed general-position point set and
    compare the separable COUNT with the exact Cover count. Both
    solvers must agree on every labeling."""
    import numpy as np
    x = gen_points(n, k, seed)
    if general_position_sigma(x) < tol_sigma:
        raise M3Refusal(
            f"enumeration points K={k} N={n} fail the "
            "general-position diagnostic")
    primary = primary or solve_feasibility
    count = 0
    for mask in range(2 ** n):
        y = np.array([1.0 if (mask >> i) & 1 else -1.0
                      for i in range(n)])
        v1 = primary(x, y, tol_sigma)
        v2 = solve_min_slack(x, y, zero_tol)
        if v1 != v2:
            raise M3Refusal(
                f"solver disagreement in enumeration K={k} N={n} "
                f"mask={mask}: {v1} vs {v2}")
        if v1.startswith("AMBIGUOUS"):
            raise M3Refusal(
                f"ambiguous enumeration case K={k} N={n} "
                f"mask={mask}: {v1}")
        if v1 == "SEPARABLE":
            count += 1
    want = cover_count(n, k)
    return {"K": k, "N": n, "separable_labelings": count,
            "cover_count": want, "exact_match": count == want}


def build_design(version: int = 1) -> dict:
    ks = [32, 64, 128]
    ratios = [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75]
    cells = []
    for k in ks:
        for r in ratios:
            n = k * r
            if abs(n - round(n)) > 1e-9:
                raise M3Refusal(f"non-integer N for K={k} r={r}")
            cells.append({"K": k, "ratio": r, "N": int(round(n)),
                          "p_exact":
                              float(cover_probability(
                                  int(round(n)), k)),
                          "p_exact_fraction":
                              str(cover_probability(
                                  int(round(n)), k))})
    d = {
        "schema": ("agent_multi.m3_cover_calibration_design"
                   f".v{version}"),
        "sealed_at_date": "2026-09-08",
        "stage": "C1_exact_threshold_reference",
        "question": ("does an optimization-independent linear-"
                     "separability test reproduce Cover's exact "
                     "finite-N separability probabilities over "
                     "the sealed grid around N/K = 2"),
        "classifier_convention": {
            "kind": "homogeneous",
            "statement": ("linear threshold through the origin; "
                          "NO bias term; data carries no ones "
                          "column; the formula counts homogeneous "
                          "dichotomies; the solvers search w in "
                          "R^K only"),
            "formula": "C(N,K) = 2 * sum_{i=0}^{K-1} C(N-1, i)",
            "probability": "P(N,K) = C(N,K) / 2^N"},
        "primary_citation": (
            "T. M. Cover, 'Geometrical and Statistical Properties "
            "of Systems of Linear Inequalities with Applications "
            "in Pattern Recognition', IEEE Trans. Electronic "
            "Computers EC-14(3):326-334, 1965 (Theorem 1; "
            "function C(N,K))"),
        "secondary_citation": (
            "D. J. C. MacKay, 'Information Theory, Inference, and "
            "Learning Algorithms', CUP 2003, ch. 40"),
        "grid": {"K": ks, "N_over_K": ratios, "cells": cells},
        "tasks_per_cell_initial": 200,
        "precision_rule": {
            "target_simultaneous_halfwidth": 0.05,
            "extension": ("double the cell's tasks while its "
                          "simultaneous CI half-width exceeds "
                          "the target"),
            "tasks_per_cell_cap": 800 if version < 3 else 3200,
            "on_cap_miss": "cell INCONCLUSIVE_PRECISION"},
        "intervals": {
            "family_alpha": 0.05,
            "cells": len(cells),
            "per_cell_level": ("Clopper-Pearson at level "
                               "1 - 0.05/21 (Bonferroni over the "
                               "21 grid cells)")},
        "solvers": {
            "primary": (
                "scipy.optimize.linprog method='highs' "
                "feasibility: y_i <w, x_i> >= 1" if version == 1
                else "scipy.optimize.linprog method='highs' "
                     "Chebyshev max-margin: max d s.t. "
                     "y_i <w, x_i> >= d, |w_j| <= 1, d <= 1; "
                     "separable iff d* > zero_tol"),
            "independent": ("min-slack LP: min 1's, "
                            "y_i <w, x_i> >= 1 - s_i, s >= 0; "
                            "separable iff optimum <= zero_tol"),
            "cross_check_subset": ("the first 25 tasks of every "
                                   "cell, plus every control and "
                                   "every enumerated case"),
            "zero_tol": 1e-9,
            "general_position_sigma_min": 1e-8,
            "disagreement_or_ambiguity": (
                "typed state; stays in the denominator; any such "
                "task makes its cell INCONCLUSIVE_AMBIGUOUS")},
        "controls_per_cell": {
            "planted_separable_positives": 5,
            "inconsistent_negatives": 5,
            "permutation_and_sign_subset": 10},
        "seeds": {
            "master_phrase": MASTER_SEED_PHRASE,
            "derivation": ("sha256(master|kind|K|ratio|index)"
                           "[:8 bytes] -> numpy default_rng")},
        "resources": {
            "cpu_only": True, "cuda_hidden": True,
            "logical_workers": 1,
            "max_wall_seconds": 6 * 3600,
            "nice": 15,
            "heartbeat": "M3_HEARTBEAT.json per cell",
            "stop_file": "M3_STOP in the runs directory"},
        "allowed_verdicts": list(VERDICTS),
        "verdict_rule": (
            "CONFIRMED iff EVERY cell reaches the precision "
            "target with zero typed-ambiguous tasks and its "
            "simultaneous Clopper-Pearson interval covers "
            "p_exact; NOT_CONFIRMED iff every cell reaches "
            "precision, none is ambiguous, and at least one "
            "interval excludes its p_exact; INCONCLUSIVE "
            "otherwise"),
        "not_estimated": ["MLP intelligence",
                          "residual capacity",
                          "exact Kolmogorov complexity",
                          "any DOIN feature or gene"],
        "formula_machine_check": None,
    }
    if version == 3:
        d["supersedes_design_sha256"] = load_design(
            DESIGN_PATH_V2)["design_sha256"]
        d["statistical_coherence_note"] = (
            "the v2 cap of 800 tasks/cell cannot reach the "
            "sealed 0.05 simultaneous half-width at p ~= 0.5 "
            "(Clopper-Pearson at level 1-0.05/21 needs ~924 "
            "tasks), so central cells ended "
            "INCONCLUSIVE_PRECISION by construction; v3 raises "
            "ONLY the extension cap to 3200. Question, grid, "
            "estimand, target, intervals, seeds, solvers, "
            "controls and verdict rule are IDENTICAL: "
            "scientific_change NONE. The v1 and v2 runs are "
            "preserved immutable.")
    if version == 2:
        d["supersedes_design_sha256"] = load_design(
            DESIGN_PATH)["design_sha256"]
        d["instrument_change_note"] = (
            "v1's pure-feasibility primary returned HiGHS "
            "status 4 (numerical difficulty) on ~5% of "
            "infeasible systems; the v1 run is preserved "
            "INCONCLUSIVE as immutable evidence. v2 replaces "
            "ONLY the primary solver formulation with the "
            "always-feasible Chebyshev max-margin LP. Grid, "
            "task counts, precision rule, intervals, seeds, "
            "controls, resources and verdict rule are "
            "IDENTICAL: scientific_change NONE.")
    primary = solve_feasibility if version == 1 else \
        solve_max_margin
    checks = []
    for (k, n) in ((2, 3), (2, 4), (2, 5), (3, 4), (3, 6)):
        checks.append(enumerate_check(
            k, n, _task_seed("enumerate", k, n),
            d["solvers"]["general_position_sigma_min"],
            d["solvers"]["zero_tol"], primary=primary))
    if not all(c["exact_match"] for c in checks):
        raise M3Refusal(
            "the enumerated counts do not match the exact Cover "
            "formula — formula, convention and solver do NOT "
            "share one convention")
    d["formula_machine_check"] = checks
    d["design_sha256"] = _self_sha(d, "design_sha256")
    return d


def seal_design(path: Path = DESIGN_PATH,
                version: int = 1) -> dict:
    if Path(path).exists():
        raise M3Refusal(
            "a sealed M3 design already exists — it is immutable "
            "and is never regenerated in place")
    d = build_design(version=version)
    payload = json.dumps(d, indent=1).encode()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def load_design(path: Path = DESIGN_PATH) -> dict:
    p = Path(path)
    if not p.is_file():
        raise M3Refusal(
            "M3_DESIGN_REQUIRED: seal the design before any "
            "outcome is computed (--seal-design)")
    d = json.loads(p.read_text())
    if _self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise M3Refusal(
            "sealed M3 design self-digest does not re-derive")
    if d.get("schema") not in (
            "agent_multi.m3_cover_calibration_design.v1",
            "agent_multi.m3_cover_calibration_design.v2",
            "agent_multi.m3_cover_calibration_design.v3"):
        raise M3Refusal("sealed M3 design carries a foreign "
                        "schema")
    return d


def _primary_for(design: dict):
    return solve_feasibility if design["schema"].endswith(
        ".v1") else solve_max_margin


def _cp_interval(successes: int, trials: int, level: float):
    """Clopper-Pearson central interval at the given two-sided
    confidence level."""
    from scipy import stats
    alpha = 1.0 - level
    if trials == 0:
        return (0.0, 1.0)
    lo = 0.0 if successes == 0 else float(
        stats.beta.ppf(alpha / 2, successes,
                       trials - successes + 1))
    hi = 1.0 if successes == trials else float(
        stats.beta.ppf(1 - alpha / 2, successes + 1,
                       trials - successes))
    return (lo, hi)


def run_task(design: dict, k: int, ratio, n: int,
             idx: int) -> dict:
    tol = design["solvers"]["general_position_sigma_min"]
    ztol = design["solvers"]["zero_tol"]
    x = gen_points(n, k, _task_seed("points", k, ratio, idx))
    y = gen_labels(n, _task_seed("labels", k, ratio, idx))
    v = _primary_for(design)(x, y, tol)
    rec = {"kind": "task", "K": k, "ratio": ratio, "N": n,
           "index": idx, "outcome": v,
           "sigma_min": round(general_position_sigma(x), 12)}
    if idx < 25 or v.startswith("AMBIGUOUS"):
        v2 = solve_min_slack(x, y, ztol)
        rec["independent_outcome"] = v2
        if v2 != v:
            rec["outcome"] = "AMBIGUOUS_SOLVER_DISAGREEMENT"
    rec["record_sha256"] = _self_sha(rec, "record_sha256")
    return rec


def run_controls(design: dict, k: int, ratio, n: int) -> list:
    import numpy as np
    tol = design["solvers"]["general_position_sigma_min"]
    ztol = design["solvers"]["zero_tol"]
    out = []
    cc = design["controls_per_cell"]
    for j in range(cc["planted_separable_positives"]):
        x = gen_points(n, k, _task_seed("pos", k, ratio, j))
        w = gen_points(1, k, _task_seed("posw", k, ratio, j))[0]
        y = np.sign(x @ w)
        y[y == 0] = 1.0
        v = _primary_for(design)(x, y, tol)
        v2 = solve_min_slack(x, y, ztol)
        ok = v == "SEPARABLE" and v2 == "SEPARABLE"
        out.append({"kind": "control_positive", "K": k,
                    "ratio": ratio, "index": j, "passed": ok})
    for j in range(cc["inconsistent_negatives"]):
        x = gen_points(n, k, _task_seed("neg", k, ratio, j))
        x[1] = x[0]
        y = gen_labels(n, _task_seed("negy", k, ratio, j))
        y[1] = -y[0]
        v = _primary_for(design)(x, y, tol)
        v2 = solve_min_slack(x, y, ztol)
        ok = v == "NONSEPARABLE" and v2 == "NONSEPARABLE"
        out.append({"kind": "control_negative", "K": k,
                    "ratio": ratio, "index": j, "passed": ok})
    rng = __import__("numpy").random.default_rng(
        _task_seed("perm", k, ratio))
    for j in range(cc["permutation_and_sign_subset"]):
        x = gen_points(n, k, _task_seed("points", k, ratio, j))
        y = gen_labels(n, _task_seed("labels", k, ratio, j))
        primary = _primary_for(design)
        base = primary(x, y, tol)
        perm = rng.permutation(n)
        vp = primary(x[perm], y[perm], tol)
        vs = primary(x, -y, tol)
        ok = (base == vp == vs)
        out.append({"kind": "control_invariance", "K": k,
                    "ratio": ratio, "index": j, "passed": ok,
                    "base": base})
    for r in out:
        r["record_sha256"] = _self_sha(r, "record_sha256")
    return out


def _append_records(fp, recs):
    for r in recs:
        fp.write(json.dumps(r, sort_keys=True) + "\n")
    fp.flush()
    os.fsync(fp.fileno())


def _heartbeat(runs: Path, payload: dict):
    tmp = runs / f".hb_{os.urandom(6).hex()}"
    tmp.write_text(json.dumps(
        {**payload, "pid": os.getpid(),
         "monotonic": time.monotonic()}, indent=1))
    os.replace(tmp, runs / "M3_HEARTBEAT.json")


def aggregate_cell(design: dict, records: list, k: int,
                   ratio) -> dict:
    cells = design["grid"]["cells"]
    cell = next(c for c in cells
                if c["K"] == k and c["ratio"] == ratio)
    tasks = [r for r in records
             if r["kind"] == "task" and r["K"] == k
             and r["ratio"] == ratio]
    controls = [r for r in records
                if r["kind"].startswith("control")
                and r["K"] == k and r["ratio"] == ratio]
    n_amb = sum(1 for r in tasks
                if r["outcome"].startswith("AMBIGUOUS"))
    n_sep = sum(1 for r in tasks if r["outcome"] == "SEPARABLE")
    n_tot = len(tasks)
    level = 1.0 - design["intervals"]["family_alpha"] / \
        design["intervals"]["cells"]
    lo, hi = _cp_interval(n_sep, n_tot - n_amb if n_tot else 0,
                          level)
    half = (hi - lo) / 2
    target = design["precision_rule"][
        "target_simultaneous_halfwidth"]
    p = cell["p_exact"]
    state = "OK"
    if n_amb > 0:
        state = "INCONCLUSIVE_AMBIGUOUS"
    elif half > target:
        state = "INCONCLUSIVE_PRECISION"
    covered = (lo <= p <= hi)
    return {"K": k, "ratio": ratio, "N": cell["N"],
            "tasks": n_tot, "separable": n_sep,
            "ambiguous": n_amb,
            "p_hat": (n_sep / (n_tot - n_amb))
            if n_tot - n_amb else None,
            "p_exact": p, "ci_low": lo, "ci_high": hi,
            "ci_halfwidth": half, "state": state,
            "covers_exact": covered,
            "controls_passed": all(c["passed"]
                                   for c in controls),
            "n_controls": len(controls)}


def decide_verdict(design: dict, cell_aggs: list) -> str:
    if any(not c["controls_passed"] for c in cell_aggs):
        return "INCONCLUSIVE"
    if any(c["state"] != "OK" for c in cell_aggs):
        return "INCONCLUSIVE"
    if all(c["covers_exact"] for c in cell_aggs):
        return VERDICTS[0]
    return "NOT_CONFIRMED"


def execute(runs_dir: Path = None,
            design_path: Path = None) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    if design_path is None:
        design_path = DESIGN_PATH_V3 if DESIGN_PATH_V3.exists() \
            else (DESIGN_PATH_V2 if DESIGN_PATH_V2.exists()
                  else DESIGN_PATH)
    design = load_design(design_path)
    if runs_dir is None:
        runs_dir = {"1": RUNS_DIR, "2": RUNS_DIR_V2,
                    "3": RUNS_DIR_V3}[design["schema"][-1]]
    runs = Path(runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    rec_path = runs / "M3_TASK_RECORDS.jsonl"
    if rec_path.exists():
        raise M3Refusal(
            "task records already exist — immutable; verify or "
            "move them, never overwrite")
    stop = runs / "M3_STOP"
    t0 = time.monotonic()
    max_wall = design["resources"]["max_wall_seconds"]
    all_records = []
    with open(rec_path, "a") as fp:
        for cell in design["grid"]["cells"]:
            k, ratio, n = cell["K"], cell["ratio"], cell["N"]
            if stop.exists():
                raise M3Refusal("external stop request "
                                "(M3_STOP)")
            if time.monotonic() - t0 > max_wall:
                raise M3Refusal("M3 wall budget exhausted")
            _heartbeat(runs, {"cell": f"K{k}_r{ratio}",
                              "elapsed_s":
                                  round(time.monotonic() - t0,
                                        1)})
            ctl = run_controls(design, k, ratio, n)
            _append_records(fp, ctl)
            all_records.extend(ctl)
            n_tasks = design["tasks_per_cell_initial"]
            done = 0
            while True:
                batch = []
                for idx in range(done, n_tasks):
                    batch.append(run_task(design, k, ratio, n,
                                          idx))
                _append_records(fp, batch)
                all_records.extend(batch)
                done = n_tasks
                agg = aggregate_cell(design, all_records, k,
                                     ratio)
                if agg["state"] != "INCONCLUSIVE_PRECISION":
                    break
                nxt = min(n_tasks * 2,
                          design["precision_rule"][
                              "tasks_per_cell_cap"])
                if nxt == n_tasks:
                    break
                n_tasks = nxt
    cell_aggs = [aggregate_cell(design, all_records, c["K"],
                                c["ratio"])
                 for c in design["grid"]["cells"]]
    verdict = decide_verdict(design, cell_aggs)
    summary = {
        "schema": "agent_multi.m3_cover_calibration_summary.v1",
        "design_sha256": design["design_sha256"],
        "records_file_sha256": hashlib.sha256(
            rec_path.read_bytes()).hexdigest(),
        "cells": cell_aggs,
        "total_tasks": sum(c["tasks"] for c in cell_aggs),
        "total_ambiguous": sum(c["ambiguous"]
                               for c in cell_aggs),
        "wall_seconds": round(time.monotonic() - t0, 1),
        "verdict": verdict}
    summary["summary_sha256"] = _self_sha(summary,
                                          "summary_sha256")
    sp = runs / "M3_SUMMARY.json"
    if sp.exists():
        raise M3Refusal("summary already exists — immutable")
    sp.write_text(json.dumps(summary, indent=1))
    return summary


def _expected_task_body(design, k, ratio, n, idx):
    """M3-C2: regenerate the task from the sealed seeds and rerun
    the productive solvers; return the exact record body."""
    tol = design["solvers"]["general_position_sigma_min"]
    ztol = design["solvers"]["zero_tol"]
    primary = _primary_for(design)
    x = gen_points(n, k, _task_seed("points", k, ratio, idx))
    y = gen_labels(n, _task_seed("labels", k, ratio, idx))
    v = primary(x, y, tol)
    rec = {"kind": "task", "K": k, "ratio": ratio, "N": n,
           "index": idx, "outcome": v,
           "sigma_min": round(general_position_sigma(x), 12)}
    if idx < 25 or v.startswith("AMBIGUOUS"):
        v2 = solve_min_slack(x, y, ztol)
        rec["independent_outcome"] = v2
        if v2 != v:
            rec["outcome"] = "AMBIGUOUS_SOLVER_DISAGREEMENT"
    rec["record_sha256"] = _self_sha(rec, "record_sha256")
    return rec


_TASK_KEYS = {"kind", "K", "ratio", "N", "index", "outcome",
              "sigma_min", "record_sha256"}
_CONTROL_KINDS = {"control_positive": 5, "control_negative": 5,
                  "control_invariance": 10}


def _strict_record(line: str, what: str) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise M3Refusal(f"duplicate JSON key in {what}")
        return dict(pairs)
    doc = json.loads(
        line, object_pairs_hook=_no_dupes,
        parse_constant=lambda c: (_ for _ in ()).throw(
            M3Refusal(f"non-finite constant in {what}")))
    if _self_sha(doc, "record_sha256") != doc.get("record_sha256"):
        raise M3Refusal(f"{what} self-digest does not re-derive")
    return doc


def verify(runs_dir: Path = None,
           design_path: Path = None) -> dict:
    """M3-C1/C2: the INDEPENDENT verifier — exact schemas and
    primitive types; the exact task-index population re-derived
    from the sealed adaptive rule; the exact 20 controls of the
    declared kinds per cell (`all([])` can never certify);
    every point set and label vector REGENERATED from the sealed
    seeds with the primary (and, on the sealed subset and all
    controls, the independent) formulation re-executed and the
    full record body re-derived to exact semantic equality; only
    then are aggregates and the verdict rebuilt and compared. A
    supplied self-digest is a checksum, not authority."""
    if design_path is None:
        design_path = DESIGN_PATH_V3 if DESIGN_PATH_V3.exists() \
            else (DESIGN_PATH_V2 if DESIGN_PATH_V2.exists()
                  else DESIGN_PATH)
    design = load_design(design_path)
    if runs_dir is None:
        runs_dir = {"1": RUNS_DIR, "2": RUNS_DIR_V2,
                    "3": RUNS_DIR_V3}[design["schema"][-1]]
    runs = Path(runs_dir)
    rec_path = runs / "M3_TASK_RECORDS.jsonl"
    summary = json.loads((runs / "M3_SUMMARY.json").read_text())
    if _self_sha(summary, "summary_sha256") != \
            summary.get("summary_sha256"):
        raise M3Refusal("summary self-digest does not re-derive")
    if summary["design_sha256"] != design["design_sha256"]:
        raise M3Refusal("summary does not bind the sealed design")
    if hashlib.sha256(rec_path.read_bytes()).hexdigest() != \
            summary["records_file_sha256"]:
        raise M3Refusal(
            "task records differ from the summary's digest")
    by_cell_tasks = {}
    by_cell_controls = {}
    for line in rec_path.read_text().splitlines():
        r = _strict_record(line, "M3 record")
        kind = r.get("kind")
        if kind == "task":
            want_keys = set(_TASK_KEYS)
            if "independent_outcome" in r:
                want_keys.add("independent_outcome")
            if set(r) != want_keys:
                raise M3Refusal(
                    "task record keys are not the exact schema")
            if type(r["index"]) is not int or \
                    isinstance(r["index"], bool) or \
                    r["index"] < 0:
                raise M3Refusal(
                    "task index is not a canonical nonnegative "
                    "integer")
            sm = r["sigma_min"]
            if type(sm) not in (int, float) or \
                    isinstance(sm, bool) or not math.isfinite(sm):
                raise M3Refusal(
                    "task sigma_min is not a finite number")
            by_cell_tasks.setdefault(
                (r["K"], r["ratio"]), {})
            cell = by_cell_tasks[(r["K"], r["ratio"])]
            if r["index"] in cell:
                raise M3Refusal(
                    f"duplicate task index {r['index']} in cell "
                    f"K={r['K']} r={r['ratio']} — duplicated "
                    "records are never independent observations")
            cell[r["index"]] = r
        elif kind in _CONTROL_KINDS:
            if type(r.get("passed")) is not bool:
                raise M3Refusal(
                    "control 'passed' must be a boolean")
            by_cell_controls.setdefault(
                (r["K"], r["ratio"]), []).append(r)
        else:
            raise M3Refusal(
                f"unknown record kind {kind!r} in the records "
                "file")
    records = []
    for cell_def in design["grid"]["cells"]:
        k, ratio, n = cell_def["K"], cell_def["ratio"], \
            cell_def["N"]
        tasks = by_cell_tasks.get((k, ratio), {})
        controls = by_cell_controls.get((k, ratio), [])
        # M3-C1: the exact 20 controls of the declared kinds,
        # REGENERATED and re-derived
        kinds_count = {}
        for c in controls:
            kinds_count[c["kind"]] = kinds_count.get(
                c["kind"], 0) + 1
        if kinds_count != _CONTROL_KINDS:
            raise M3Refusal(
                f"cell K={k} r={ratio} does not carry the exact "
                f"control census {_CONTROL_KINDS} (got "
                f"{kinds_count}) — an empty control list can "
                "never certify")
        expected_controls = run_controls(design, k, ratio, n)
        def _ckey(c):
            return (c["kind"], c["index"])
        got_sorted = sorted(controls, key=_ckey)
        want_sorted = sorted(expected_controls, key=_ckey)
        if got_sorted != want_sorted:
            raise M3Refusal(
                f"cell K={k} r={ratio} controls do not "
                "REGENERATE from the sealed seeds — recorded "
                "control evidence is not reproducible")
        # M3-C1/C2: exact adaptive population, regenerated
        # outcomes, exact record-body equality
        n_expected = design["tasks_per_cell_initial"]
        cap = design["precision_rule"]["tasks_per_cell_cap"]
        regen = {}
        while True:
            for idx in range(len(regen), n_expected):
                regen[idx] = _expected_task_body(
                    design, k, ratio, n, idx)
            agg = aggregate_cell(
                design, list(regen.values()) + expected_controls,
                k, ratio)
            if agg["state"] != "INCONCLUSIVE_PRECISION":
                break
            nxt = min(n_expected * 2, cap)
            if nxt == n_expected:
                break
            n_expected = nxt
        if sorted(tasks) != list(range(n_expected)):
            raise M3Refusal(
                f"cell K={k} r={ratio} task population is not "
                f"the exact sealed adaptive population "
                f"(expected indices 0..{n_expected - 1}, got "
                f"{len(tasks)} records) — gaps, extras or "
                "duplicates refuse")
        for idx in range(n_expected):
            if tasks[idx] != regen[idx]:
                raise M3Refusal(
                    f"cell K={k} r={ratio} task {idx} does not "
                    "REGENERATE from the sealed seeds and "
                    "solvers — recorded outcomes are not "
                    "reproducible")
        records.extend(regen.values())
        records.extend(expected_controls)
    cell_aggs = [aggregate_cell(design, records, c["K"],
                                c["ratio"])
                 for c in design["grid"]["cells"]]
    if cell_aggs != summary["cells"]:
        raise M3Refusal(
            "reconstructed cell aggregates differ from the "
            "published summary")
    verdict = decide_verdict(design, cell_aggs)
    if verdict != summary["verdict"]:
        raise M3Refusal(
            "reconstructed verdict differs from the published "
            "summary")
    return {"verified": True, "verdict": verdict,
            "total_tasks": sum(len(by_cell_tasks[(c["K"],
                                                  c["ratio"])])
                               for c in design["grid"]["cells"]),
            "controls_verified": sum(
                len(v) for v in by_cell_controls.values()),
            "cells": len(cell_aggs),
            "regenerated": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-design", action="store_true")
    ap.add_argument("--seal-design-v2", action="store_true")
    ap.add_argument("--seal-design-v3", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)
    if args.seal_design_v3:
        d = seal_design(DESIGN_PATH_V3, version=3)
        print(json.dumps({"sealed_v3": d["design_sha256"],
                          "supersedes":
                              d["supersedes_design_sha256"],
                          "cap": d["precision_rule"][
                              "tasks_per_cell_cap"],
                          "formula_machine_check":
                              [c["exact_match"] for c in
                               d["formula_machine_check"]]},
                         indent=1))
        return 0
    if args.seal_design_v2:
        d = seal_design(DESIGN_PATH_V2, version=2)
        print(json.dumps({"sealed_v2": d["design_sha256"],
                          "supersedes":
                              d["supersedes_design_sha256"],
                          "formula_machine_check":
                              [c["exact_match"] for c in
                               d["formula_machine_check"]]},
                         indent=1))
        return 0
    if args.seal_design:
        d = seal_design()
        print(json.dumps({"sealed": d["design_sha256"],
                          "formula_machine_check":
                              d["formula_machine_check"]},
                         indent=1))
        return 0
    if args.execute:
        s = execute()
        print(json.dumps({"verdict": s["verdict"],
                          "total_tasks": s["total_tasks"],
                          "total_ambiguous":
                              s["total_ambiguous"],
                          "wall_seconds": s["wall_seconds"]},
                         indent=1))
        return 0
    if args.verify:
        print(json.dumps(verify(), indent=1))
        return 0
    raise M3Refusal("choose --seal-design, --execute or "
                    "--verify")


if __name__ == "__main__":
    raise SystemExit(main())
