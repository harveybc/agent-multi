"""M4-C17/C18/C21: the intervention-foundation design v4.

Supersedes the accepted mechanics v3 WITHOUT changing any
accepted mechanic. Before any developmental score it enumerates
the complete candidate population with an exact unit and update
census, freezes the estimands and attribution analyses, freezes
the learnability-gate rule, freezes the confirmatory contrast
family with its multiplicity correction and censoring analysis,
runs the a-priori precision simulation (assumption-driven, no
outcome data), and states the deterministic reduction and
four-unit selection rules — all BEFORE outcomes exist.

scientific_outcome: NONE. CPU only. Grants nothing.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402

DESIGN_PATH_V4 = (REPO / "docs/research/model_capacity/"
                  "M4_SEALED_DESIGN_V4_2026_09_09.json")

WIDTHS = (8, 16, 32, 64)
CHECKPOINTS = ("initialization", "pre_stop", "calibration_stop",
               "post_stop_bounded")
MODEL_SEEDS = 3
G_DEV = 2                     # DEVELOPMENT generators per cell
G_CAL = 2                     # CALIBRATION generators per cell
FIT_UPDATES = 2000            # frozen learnability fit budget
FIT_MINIBATCH = 16
CPU_RESOURCE_CEILING_H = 48.0     # declared CPU ceiling for the
#                                   future confirmatory campaign
SMALLEST_EFFECT_ASSOC = 4.0   # smallest effect of interest
#                               (associations, paired difference)
PRECISION_TARGET_POWER = 0.8
PRECISION_ALPHA = 0.05
BETWEEN_GEN_SD_RANGE = (2.0, 8.0)   # assumed SD grid (assoc)
ATTRITION_ALLOWANCE = 0.2
CENSOR_DOMINANT_FRACTION = 0.2


class DesignRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def structured_cells():
    cells = []
    for fam in gb.BOOL_FAMILIES:
        cells.append((fam, "clean"))
    for fam in gb.TEMPORAL_FAMILIES:
        for nz in gb.NOISE_REGIMES:
            cells.append((fam, nz))
    return cells


def confirmatory_cells():
    """Deterministic pre-outcome REDUCTION rule: the confirmatory
    grid keeps every primary family, restricts widths to {16, 64}
    (smallest informative + largest), temporal noise to
    {clean, white} (the two regimes every hypothesis needs), and
    the primary contrast to the checkpoint pair
    (initialization, calibration_stop). Written BEFORE any
    developmental outcome; a family enters only if the
    learnability gate returns LEARNABLE_UNDER_FROZEN_BUDGET."""
    cells = []
    for fam in gb.BOOL_FAMILIES:
        cells.append((fam, "clean"))
    for fam in gb.TEMPORAL_FAMILIES:
        for nz in ("clean", "white"):
            cells.append((fam, nz))
    return cells


def population_census() -> dict:
    sc = structured_cells()
    cc = confirmatory_cells()
    n_struct = len(sc)
    # learnability screen units (DEVELOPMENT only)
    screen_fits = n_struct * len(WIDTHS) * G_DEV
    control_fits = (len(WIDTHS) * G_DEV) * 2   # random_label +
    #                                            easy_constant
    screen_units = screen_fits + control_fits
    screen_updates = screen_units * FIT_UPDATES
    # intervention run cost model (updates per run)
    run_updates = (FIT_UPDATES              # pretrain to stop
                   + 64 * m4.UPDATES_PER_BATCH)   # batch cap
    conf_runs_upper = (len(cc) * 2          # widths {16,64}
                       * MODEL_SEEDS
                       * 2)                 # treatment + control
    return {
        "structured_family_noise_cells": n_struct,
        "meaningful_products_rule": (
            "Boolean and control families admit only `clean`; "
            "the five noise regimes apply to the five temporal "
            "observation processes — 4 + 5x5 = 29 structured "
            "cells; nonsensical products REFUSE in the "
            "generator bank"),
        "widths": list(WIDTHS),
        "checkpoints": list(CHECKPOINTS),
        "model_seeds_nested": MODEL_SEEDS,
        "development_generators_per_cell": G_DEV,
        "calibration_generators_per_cell": G_CAL,
        "learnability_screen_units": screen_units,
        "learnability_screen_fits_structured": screen_fits,
        "learnability_screen_fits_controls": control_fits,
        "learnability_screen_updates": screen_updates,
        "intervention_updates_per_run": run_updates,
        "confirmatory_cells_after_reduction": len(cc),
        "confirmatory_runs_upper_bound_per_generator":
            conf_runs_upper,
        "confirmatory_updates_upper_bound_per_generator":
            conf_runs_upper * run_updates,
        "note": ("the full Cartesian product is NOT assumed "
                 "affordable; the reduction rule above is frozen "
                 "before outcomes and preserves every primary "
                 "family, both informative widths and the "
                 "primary checkpoint pair"),
    }


def precision_simulation(seed_phrase="m4_precision_2026_09_09",
                         n_sims=4000) -> dict:
    """C21 a-priori simulation: minimum CONFIRMATION generators
    for the primary paired effect at the smallest effect of
    interest, under the assumed between-generator SD grid. Pure
    assumption-driven Monte Carlo — no outcome data enters."""
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
        seed_phrase.encode()).digest()[:8], "big"))
    k_contrasts = len(confirmatory_cells()) + 2
    alpha_adj = PRECISION_ALPHA / k_contrasts   # Bonferroni bound
    table = {}
    n_required = {}
    for sd in np.arange(BETWEEN_GEN_SD_RANGE[0],
                        BETWEEN_GEN_SD_RANGE[1] + 0.1, 2.0):
        for n_gen in (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48):
            eff_n = max(3, int(np.floor(
                n_gen * (1 - ATTRITION_ALLOWANCE))))
            hits = 0
            for _ in range(n_sims):
                d = rng.standard_normal(eff_n) * sd + \
                    SMALLEST_EFFECT_ASSOC
                t = d.mean() / (d.std(ddof=1) /
                                np.sqrt(eff_n) + 1e-12)
                from math import lgamma
                # two-sided t p-value via survival approx: use
                # normal approx guarded by df>=8 exact-t table
                df = eff_n - 1
                # Student-t two-sided p via incomplete beta
                x = df / (df + t * t)
                a_, b_ = df / 2.0, 0.5
                # regularized incomplete beta by continued
                # fraction (Lentz), sufficient for p-values
                def betacf(a, b, xx):
                    qab, qap, qam = a + b, a + 1.0, a - 1.0
                    c, dd = 1.0, 1.0 - qab * xx / qap
                    if abs(dd) < 1e-30:
                        dd = 1e-30
                    dd = 1.0 / dd
                    h = dd
                    for m_ in range(1, 200):
                        m2 = 2 * m_
                        aa = m_ * (b - m_) * xx / (
                            (qam + m2) * (a + m2))
                        dd = 1.0 + aa * dd
                        if abs(dd) < 1e-30:
                            dd = 1e-30
                        c = 1.0 + aa / c
                        if abs(c) < 1e-30:
                            c = 1e-30
                        dd = 1.0 / dd
                        h *= dd * c
                        aa = -(a + m_) * (qab + m_) * xx / (
                            (a + m2) * (qap + m2))
                        dd = 1.0 + aa * dd
                        if abs(dd) < 1e-30:
                            dd = 1e-30
                        c = 1.0 + aa / c
                        if abs(c) < 1e-30:
                            c = 1e-30
                        dd = 1.0 / dd
                        delc = dd * c
                        h *= delc
                        if abs(delc - 1.0) < 1e-9:
                            break
                    return h
                lbeta = (lgamma(a_) + lgamma(b_)
                         - lgamma(a_ + b_))
                if x in (0.0, 1.0):
                    ib = x
                else:
                    front = np.exp(a_ * np.log(x) + b_ *
                                   np.log(1 - x) - lbeta) / a_
                    ib = front * betacf(a_, b_, x) \
                        if x < (a_ + 1) / (a_ + b_ + 2) else \
                        1.0 - np.exp(
                            b_ * np.log(1 - x) + a_ * np.log(x)
                            - lbeta) / b_ * betacf(b_, a_, 1 - x)
                p = min(max(ib, 0.0), 1.0)
                if p < alpha_adj:
                    hits += 1
            table[f"sd{sd:.0f}_n{n_gen}"] = round(
                hits / n_sims, 4)
    for sd in np.arange(BETWEEN_GEN_SD_RANGE[0],
                        BETWEEN_GEN_SD_RANGE[1] + 0.1, 2.0):
        n_star = None
        for n_gen in (4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48):
            if table[f"sd{sd:.0f}_n{n_gen}"] >= \
                    PRECISION_TARGET_POWER:
                n_star = n_gen
                break
        n_required[f"sd{sd:.0f}"] = n_star
    return {
        "method": ("paired one-sample t on generator-level "
                   "differences, Bonferroni-adjusted alpha, "
                   "Monte Carlo with Lentz incomplete-beta "
                   "p-values"),
        "assumptions": {
            "smallest_effect_associations":
                SMALLEST_EFFECT_ASSOC,
            "between_generator_sd_grid":
                list(np.arange(BETWEEN_GEN_SD_RANGE[0],
                               BETWEEN_GEN_SD_RANGE[1] + 0.1,
                               2.0)),
            "alpha": PRECISION_ALPHA,
            "contrast_family_size": k_contrasts,
            "attrition_allowance": ATTRITION_ALLOWANCE,
            "power_target": PRECISION_TARGET_POWER,
            "n_sims": n_sims,
        },
        "power_table": table,
        "minimum_confirmation_generators_by_sd": n_required,
    }


def build_design_v4() -> dict:
    v3 = m4._strict_json_file(m4.DESIGN_PATH_V3,
                              "M4 sealed design v3")
    if m4._self_sha(v3, "design_sha256") != v3["design_sha256"]:
        raise DesignRefusal("v3 self-digest does not re-derive")
    m4.verify_design_supersession_v3(v3)
    census = population_census()
    precision = precision_simulation()
    run_updates = census["intervention_updates_per_run"]
    cc = confirmatory_cells()
    d = json.loads(json.dumps(v3))
    d["schema"] = "agent_multi.m4_intervention_design.v4"
    d["sealed_at_date"] = "2026-09-09"
    d["supersedes_design_sha256"] = v3["design_sha256"]
    d["supersession_classification"] = (
        "INTERVENTION_FOUNDATION_DESIGN_BEFORE_ANY_OUTCOME")
    d["scientific_outcome"] = "NONE"
    d["question_v4"] = (
        "Conditional on a learned structured task, architecture "
        "and checkpoint, how many blinded random associations "
        "remain jointly acquired before the frozen retention "
        "rule fails, and do training measurements improve "
        "prediction of that endpoint beyond parameter count and "
        "checkpoint loss? The endpoint is an empirical "
        "intervention result under this protocol — never unused "
        "intelligence, free bits, exact information content or "
        "exact Kolmogorov complexity.")
    d["candidate_population"] = {
        "structured_boolean_families": list(gb.BOOL_FAMILIES),
        "temporal_families": list(gb.TEMPORAL_FAMILIES),
        "negative_control": "random_label (never a learnable "
                            "structured family)",
        "positive_control": "easy_constant (deliberately easy)",
        "noise_regimes_temporal": list(gb.NOISE_REGIMES),
        "hidden_widths": list(WIDTHS),
        "checkpoint_taxonomy": list(CHECKPOINTS),
        "generator_roles": {
            "DEVELOPMENT": "screen + mechanics; thresholds "
                           "estimated here",
            "CALIBRATION": "threshold freeze; outcomes not "
                           "inspected in this order",
            "CONFIRMATION": "untouched until the confirmatory "
                            "order"},
        "unit_definition": (
            "the task GENERATOR is the independent unit; model "
            "seeds and checkpoints are nested repetitions"),
        "cell_record_fields": [
            "family", "noise_or_NOT_APPLICABLE", "width",
            "checkpoint", "generator_role", "generator_id",
            "model_seed", "optimizer_budget", "unit_role"],
    }
    d["population_census"] = census
    d["estimands"] = {
        "primary_endpoint": (
            "cumulative count of associations jointly acquired "
            "at the LAST PASSING evaluation; both stopping "
            "causes published (acquisition failure; two "
            "consecutive retention failures); MAX_BATCHES is "
            "RIGHT-CENSORED, never evidence the endpoint equals "
            "the cap"),
        "analysis_1_intervention_effect": (
            "within-generator paired difference between the "
            "calibration_stop checkpoint and a matched random "
            "initialization under identical association batches "
            "and compute"),
        "analysis_2_checkpoint_effect": (
            "paired contrasts among the four checkpoints within "
            "(generator, width, model seed); checkpoints are "
            "NEVER treated as independent"),
        "analysis_3_incremental_prediction": (
            "out-of-generator prediction of log1p(endpoint): "
            "(M0) parameter count; (M1) M0 + checkpoint loss + "
            "elapsed updates; (M2) M1 + frozen trajectory and "
            "description measurements — M2 advances only if it "
            "improves unseen-generator error AND calibration "
            "after its measurement cost is included"),
        "descriptor_status": (
            "compressed length, pruning and spectral rank are "
            "candidate DESCRIPTORS, not ground truth; repeated "
            "weights do not imply remaining capacity"),
        "m3_status": ("prior evidence only — never a training "
                      "sample for M4"),
        "censoring_analysis": (
            "primary paired analysis uses min(endpoint, cap) "
            "with a published censoring flag per unit; if the "
            f"censored fraction exceeds "
            f"{CENSOR_DOMINANT_FRACTION} in any confirmatory "
            "cell the cell reports CENSORING_DOMINANT and a "
            "rank-based survival comparison (Gehan) becomes the "
            "cell's analysis; defined BEFORE any occurrence"),
    }
    d["learnability_gate"] = {
        "scope": "DEVELOPMENT-only in this order",
        "optimizer": {"updates": FIT_UPDATES,
                      "minibatch": FIT_MINIBATCH,
                      "learning_rate": m4.LEARNING_RATE,
                      "identical_to_intervention": True},
        "criterion": {
            "boolean": "held-out accuracy",
            "temporal": "held-out MSE skill vs frozen baseline"},
        "baselines": {
            "boolean": "train majority class",
            "temporal": "persistence (last window value)"},
        "threshold_rule": (
            "margin = max(0.05, p95 of the random_label "
            "control's improvement-over-baseline across widths "
            "and DEVELOPMENT generators); estimated on "
            "DEVELOPMENT, to be FROZEN on CALIBRATION before "
            "any confirmation outcome"),
        "outcomes": ["LEARNABLE_UNDER_FROZEN_BUDGET",
                     "OPTIMIZATION_LIMITED",
                     "NUMERICALLY_INVALID"],
        "controls": ["random_label negative control",
                     "no_learning (zero-update evaluation)",
                     "random_initialization band (3 extra "
                     "inits evaluated without updates)",
                     "easy_constant positive control"],
        "gate_rule": (
            "only LEARNABLE_UNDER_FROZEN_BUDGET enters the "
            "capacity intervention; missing/failed cells stay "
            "in the denominator; a primary family failing at "
            "every width is reported as a limitation and never "
            "reinterpreted; budgets are never increased after "
            "seeing confirmation behavior"),
    }
    d["confirmatory_contrast_family"] = {
        "contrasts": (
            [f"intervention_effect::{fam}::{nz}"
             for fam, nz in cc]
            + ["checkpoint_effect::primary_pair",
               "incremental_prediction::M2_vs_M1"]),
        "count": len(cc) + 2,
        "multiplicity": "Bonferroni-bounded design basis; Holm "
                        "step-down at analysis time",
        "heterogeneity_rule": (
            "family-specific effects reported ALWAYS; pooling "
            "only if the family-effect interquartile range is "
            "below half the smallest effect of interest"),
    }
    d["precision"] = precision
    d["reduction_rule"] = {
        "rule": ("confirmatory grid = every primary family x "
                 "widths {16, 64} x temporal noise {clean, "
                 "white} x checkpoint pair (initialization, "
                 "calibration_stop) x 3 nested seeds; frozen "
                 "BEFORE developmental outcomes"),
        "cells_after_reduction": len(cc),
        "runs_per_generator":
            census["confirmatory_runs_upper_bound_per_generator"],
        "updates_per_run": run_updates,
        "cpu_resource_ceiling_hours": CPU_RESOURCE_CEILING_H,
        "affordability_check": (
            "runtime per run is MEASURED by the four development "
            "units of this order; N* generators from the "
            "precision table times measured runtime must fit "
            "the declared ceiling or the return is "
            "M4_CONFIRMATORY_POPULATION_INSUFFICIENT"),
    }
    d["four_unit_rule"] = {
        "rule": ("exactly four end-to-end DEVELOPMENT "
                 "intervention units, chosen pre-outcome for "
                 "family/width/noise diversity; runtime and "
                 "mechanics only — endpoints never tune the "
                 "protocol"),
        "units": [
            {"family": "identity", "noise": "clean",
             "width": 16, "generator_index": 0, "model_seed": 0},
            {"family": "parity4", "noise": "clean",
             "width": 64, "generator_index": 0, "model_seed": 0},
            {"family": "sine", "noise": "white",
             "width": 16, "generator_index": 0, "model_seed": 0},
            {"family": "state_space",
             "noise": "heteroscedastic", "width": 64,
             "generator_index": 0, "model_seed": 0}],
        "checkpoint": "calibration_stop vs initialization "
                      "control + one matched diagnostic batch",
    }
    d["grants_nothing_v4"] = [
        "confirmation outcomes", "DOIN genes", "production "
        "gates", "GPU use", "financial data", "venue access",
        "B4/T2 alteration", "exact capacity in bits"]
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


V4_ALLOWED_NEW_KEYS = {
    "question_v4", "candidate_population", "population_census",
    "estimands", "learnability_gate",
    "confirmatory_contrast_family", "precision",
    "reduction_rule", "four_unit_rule", "grants_nothing_v4"}


def verify_design_supersession_v4(v4: dict) -> None:
    """v4 must supersede EXACTLY the sealed v3, carry
    scientific_outcome NONE, and change NOTHING of the accepted
    v3 mechanics — only the frozen v4 foundation keys plus the
    supersession surface may differ."""
    v3 = m4._strict_json_file(m4.DESIGN_PATH_V3,
                              "M4 sealed design v3")
    if m4._self_sha(v3, "design_sha256") != v3["design_sha256"]:
        raise DesignRefusal("v3 self-digest does not re-derive")
    m4.verify_design_supersession_v3(v3)
    if v4.get("supersedes_design_sha256") != v3["design_sha256"]:
        raise DesignRefusal(
            "v4 does not supersede the sealed v3 identity")
    if v4.get("scientific_outcome") != "NONE":
        raise DesignRefusal("v4 must declare scientific_outcome "
                            "NONE")
    deltas = m4._design_delta_paths(v3, v4)
    allowed_roots = V4_ALLOWED_NEW_KEYS | {
        "schema", "sealed_at_date", "supersedes_design_sha256",
        "supersession_classification", "design_sha256"}
    illegal = {dd for dd in deltas if dd[0] not in allowed_roots}
    if illegal:
        raise DesignRefusal(
            "v4 changes an accepted v3 mechanic: "
            f"{sorted('.'.join(map(str, dd)) for dd in illegal)[:3]}")


def seal_design_v4(path: Path = None) -> dict:
    path = Path(path or DESIGN_PATH_V4)
    if path.exists():
        raise DesignRefusal("a sealed M4 v4 design already "
                            "exists — immutable")
    d = build_design_v4()
    verify_design_supersession_v4(d)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(d, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def load_design_v4(path: Path = None) -> dict:
    d = m4._strict_json_file(Path(path or DESIGN_PATH_V4),
                             "M4 sealed design v4")
    if m4._self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise DesignRefusal("v4 self-digest does not re-derive")
    if d.get("schema") != "agent_multi.m4_intervention_design.v4":
        raise DesignRefusal("design is not the v4 intervention "
                            "foundation")
    verify_design_supersession_v4(d)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--census", action="store_true")
    a = ap.parse_args()
    if a.seal:
        d = seal_design_v4()
        print(json.dumps({
            "sealed_v4": d["design_sha256"],
            "supersedes": d["supersedes_design_sha256"],
            "confirmatory_contrasts":
                d["confirmatory_contrast_family"]["count"],
            "n_star_by_sd":
                d["precision"]
                ["minimum_confirmation_generators_by_sd"]},
            indent=1))
    elif a.census:
        print(json.dumps(population_census(), indent=1))
    else:
        raise DesignRefusal("choose --seal or --census")
