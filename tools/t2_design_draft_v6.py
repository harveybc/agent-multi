#!/usr/bin/env python3
"""T2-S step 7 (order C37): DRAFT v6 — ONE unambiguous estimand
name. Supersedes v5 by digest; v2-v5 stay on disk as immutable
history. The ONLY change vs v5 is the estimand naming contract:
the primary contrast is `mase_improvement_X_minus_D` = MASE(X) -
MASE(D), so POSITIVE means D reduces error — the arithmetic the
adjudicator always implemented. Population, geometry, selection
salt, margins and rules are byte-identical in meaning to v5.

What changed vs v4 (C31/C35):
- ONE common geometry for ALL six panels: initial fit fraction
  0.60, TWO rolling origins, lags 8, horizon 1, consecutive score
  windows over the final 40%. No panel exception. Hospital's 767
  series of length 84 mechanically produce two 17-observation
  windows; every model keeps its OWN fit/score minimums (verified
  per unit; any real MLP requirement failure makes the draft
  GEOMETRY_INFEASIBLE and it is returned, never executed).
- Per-panel EXTREME-SUPPORT rule (absolute + proportional),
  derived before results: a panel below it is INCONCLUSIVE; one
  evaluable series among many NOT_EVALUABLE can never license.
- Selection salt t2_design_v5 over geometry-admissible units
  (period-aware feasibility via the ONE productive rule that the
  harness itself delegates to).

No design is sealed, no ledger exists, no score is computed, no
data is downloaded."""
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402
import t2_fresh_verifier as fv  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"

ROLLING_ORIGINS = 2
ORIGIN_BASE_FRAC = 0.6
LAGS = 8
HORIZON = 1
K = 40
SALT = "t2_design_v5"
MARGIN = 0.02
FAMILIES_PRIMARY = ["tourism", "urban_pedestrian",
                    "health_hospital", "solar_energy",
                    "electricity_demand", "weather"]
# C31: predeclared per-panel extreme-support minimum — absolute
# AND proportional of the panel's selected series.
EXTREME_SUPPORT_RULE = {"min_evaluable_series_absolute": 8,
                        "min_evaluable_fraction": 0.25}


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    manifest_path = STATE / "t2_public_data_manifest_20260906.json"
    manifest = conf.strict_json_load(manifest_path, "manifest")
    census = conf.strict_json_load(
        STATE / "t2_bank_census_20260906.json", "census")
    manifest_sha = census["manifest_sha256"]
    rebuilt = fv.rebuild_population(manifest,
                                    STATE / "t2_public_raw")
    fv.verify_census_semantic(rebuilt, census)

    sd_planning = 0.04
    alpha = 0.05
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    n_min_panel = math.ceil((z * sd_planning / MARGIN) ** 2)
    sensitivity = {str(s): math.ceil((z * s / MARGIN) ** 2)
                   for s in (0.02, 0.04, 0.08)}

    by_family = {}
    for lid, r in rebuilt.items():
        admissible = [uid for uid in r["admissible_unit_ids"]
                      if bank.geometry_admissible(
                          r["unit_meta"][uid]["n"],
                          ROLLING_ORIGINS, ORIGIN_BASE_FRAC,
                          seasonal_period=r["seasonal_period"],
                          horizon=HORIZON)]
        by_family.setdefault(r["family"], {})[lid] = admissible
    selected_by_family = {}
    for fam, ds_map in by_family.items():
        if fam in FAMILIES_PRIMARY:
            selected_by_family[fam] = set(
                bank.family_top_k(ds_map, K, salt=SALT))
        else:
            selected_by_family[fam] = set(
                sid for ids in ds_map.values() for sid in ids)

    unit_map = {}
    unit_digests = {}
    per_dataset = {}
    screen_panels = []
    geometry_feasibility = {}
    for lid, r in sorted(rebuilt.items()):
        fam = r["family"]
        geo_ids = set(by_family[fam][lid])
        chosen = sorted(geo_ids & selected_by_family[fam])
        n_total = len(r["admissible_unit_ids"])
        per_dataset[lid] = {
            "family": fam,
            "n_series_selected": len(chosen),
            "n_admissible_census": n_total,
            "n_geometry_admissible": len(geo_ids),
            "seasonal_period": r["seasonal_period"]}
        if fam in FAMILIES_PRIMARY and n_total > 0:
            screen_panels.append(lid)
            # C35: the panel must carry its support under the
            # COMMON geometry — a shortfall is GEOMETRY_INFEASIBLE
            # and the draft is returned, never executed.
            if len(chosen) < n_min_panel:
                raise SystemExit(
                    f"REFUSED: GEOMETRY_INFEASIBLE — panel {lid} "
                    f"selects {len(chosen)} series < the "
                    f"predeclared per-panel minimum {n_min_panel} "
                    "under the common two-origin geometry; the "
                    "draft is returned for external design review")
        min_fit_after_val = None
        min_scored = None
        min_score_w = None
        for uid in chosen:
            meta = r["unit_meta"][uid]
            wins = bank.origin_windows_for(
                meta["n"], ROLLING_ORIGINS, ORIGIN_BASE_FRAC,
                seasonal_period=r["seasonal_period"],
                horizon=HORIZON)
            unit_digests[uid] = r["unit_numeric_digests"][uid]
            unit_map[uid] = {
                "family": fam, "dataset": lid,
                "series_numeric_sha256":
                    r["unit_numeric_digests"][uid],
                "seasonal_period": r["seasonal_period"],
                "horizon": HORIZON,
                "n_obs": meta["n"],
                "time_identity_sha256":
                    meta["time_identity_sha256"],
                "origin_windows": wins}
            for wb in wins.values():
                fit = wb["train"][1] - LAGS - HORIZON
                n_val = max(8, int(fit * 0.2))
                scored = (wb["score"][1] - wb["score"][0]
                          - LAGS - HORIZON)
                sw = wb["score"][1] - wb["score"][0]
                min_fit_after_val = (fit - n_val
                                     if min_fit_after_val is None
                                     else min(min_fit_after_val,
                                              fit - n_val))
                min_scored = (scored if min_scored is None
                              else min(min_scored, scored))
                min_score_w = (sw if min_score_w is None
                               else min(min_score_w, sw))
        if chosen:
            geometry_feasibility[lid] = {
                "n_selected": len(chosen),
                "min_score_window": min_score_w,
                "min_scored_rows_per_origin": min_scored,
                "min_mlp_fit_rows_after_validation":
                    min_fit_after_val,
                "model_minimums_kept":
                    bool(min_fit_after_val >= 8
                         and min_scored >= 1)}
            if not geometry_feasibility[lid][
                    "model_minimums_kept"]:
                raise SystemExit(
                    f"REFUSED: GEOMETRY_INFEASIBLE — panel {lid} "
                    "violates a real model minimum; the draft is "
                    "returned, models are never weakened")
    screen_panels = sorted(screen_panels)
    if len(screen_panels) != 6:
        raise SystemExit(
            f"REFUSED: expected exactly six screen panels, got "
            f"{screen_panels}")
    series_ids = sorted(unit_map)

    draft = {
        "schema": "agent_multi.t2_screen_design.v6_draft",
        "sealed_after_census_manifest_sha256": manifest_sha,
        "supersedes_draft_sha256": sha_file(
            STATE / "t2_screen_design_DRAFT_V5_20260907.json"),
        "operator": {"kind": "ewma", "params": {"alpha": 0.3},
                     "selection_source":
                         "T1_v4_record_LAB_CALIBRATED"},
        "task_population": {
            "series_ids": series_ids,
            "unit_digests": unit_digests,
            "unit_map": unit_map,
            "families": sorted(by_family),
            "primary_gate_families": FAMILIES_PRIMARY,
            "sensitivity_only_families": ["hydrology",
                                          "demography"],
            "screen_panels": screen_panels,
            "geometry_feasibility": geometry_feasibility,
            "per_dataset": per_dataset,
            "selection": {
                "rule": "family_top_k_geometry_admissible",
                "k": K, "salt": SALT},
            "selection_rule": (
                f"exact top-k by lowest sha256 of '{SALT}|<id>' "
                f"over the WHOLE family (datasets pooled), "
                f"k=min({K}, n), restricted to units the COMMON "
                "two-origin geometry can window under the real "
                "model minimums (period-aware); global ids "
                "unique; order-independent")},
        "role_geometry": {"rolling_origins": ROLLING_ORIGINS,
                          "origin_base_frac": ORIGIN_BASE_FRAC,
                          "lags": LAGS, "horizon": HORIZON},
        "arms": {"X": "identity",
                 "D": "causal ewma(alpha=0.3) via T0 contract",
                 "XDR": "[X, D, X-D]",
                 "width_control": "train-frozen independent "
                                  "channels matching XDR width"},
        "models": {
            "ridge": {"lags": LAGS, "lambda": 1.0,
                      "intercept": "unpenalized",
                      "scaling": "train-only standardization"},
            "mlp_small": {"hidden": [16], "tol": 1e-4,
                          "scaling": "train-only",
                          "epoch_rule": "temporally final 20% of "
                                        "fit rows (min 8); grid "
                                        "[40,80,120,200]"},
            "seasonal_naive": {"period": "per-unit predeclared"}},
        "seed_tape": [11, 12, 13],
        "primary_contrast": {
            "delta": "mase_improvement_X_minus_D",
            "model": "ridge",
            "statistic": "paired per-series improvement "
                         "MASE(X) - MASE(D) — POSITIVE means D "
                         "reduces error; origins averaged within "
                         "series"},
        "secondary_gates": {
            "attribution": "panel mean of (D-X) minus "
                           "(width_control-X) must be positive",
            "XDR": "secondary descriptive contrast, predeclared, "
                   "never promoted post hoc",
            "preservation": "extreme-innovation gate on explicit "
                            "states (C25) bound to the C31 "
                            "per-panel evaluable-support minimum",
            "calibration": "coverage-drop and width-inflation "
                           "gates"},
        "primary_metric": "MASE (train-defined seasonal-naive "
                          "denominator); raw MAE/RMSE per-series "
                          "diagnostics only, never pooled",
        "estimand": {
            "population": "the six named public primary panels "
                          "(screen_panels), exactly as acquired",
            "superior_unit": "panel",
            "panel_effect": "paired mean of "
                            "mase_improvement_X_minus_D "
                            "(= MASE(X) - MASE(D); positive = D "
                            "reduces error) over the panel's "
                            "selected series (origins averaged "
                            "within series first)",
            "primary": "UNWEIGHTED mean of the six panel effects",
            "scope": "ONLY these six panels and the series "
                     "admitted by this contract — no family-"
                     "level generalization, no public-"
                     "eligibility claim",
            "outputs": ["ADVANCE_TO_DOMAIN_VALIDATION",
                        "DOES_NOT_ADVANCE", "INCONCLUSIVE"],
            "decision_rule": (
                "ADVANCE requires SIMULTANEOUSLY: (1) t-interval "
                "over the six panel effects (df=5) with lower "
                "bound > practical margin; (2) exact sign "
                "sensitivity compatible with alpha 0.05 — all "
                "six panel effects positive (two-sided exact "
                "p=2/64=0.03125); (3) leave-one-panel-out mean > "
                "margin in ALL six omissions; (4) no panel with "
                "harm beyond the non-inferiority margin, no "
                "HARM_INFINITE extreme state, no calibration "
                "harm; (5) preservation (incl. the per-panel "
                "extreme-support minimum), calibration, cost and "
                "support gates complete. Insufficient power or "
                "precision => INCONCLUSIVE; margins are FROZEN "
                "before any score and never adjusted after "
                "seeing results"),
            "t2c_successor": (
                "T2-C (cross-panel confirmation, >=3 independent "
                "panels per family) is a CONDITIONAL successor "
                "only: it will be designed with its own "
                "acquisition and review IF T2-S advances and a "
                "family-level claim is wanted. Nothing here "
                "authorizes that acquisition.")},
        "extreme_support_rule": EXTREME_SUPPORT_RULE,
        "practical_margin_mase": MARGIN,
        "observed_precision_rule": {
            "max_ci_halfwidth": MARGIN,
            "rule": "if the panel-level t CI half-width exceeds "
                    "this, the screen is INCONCLUSIVE even with "
                    "a favorable mean"},
        "harm_margins": {
            "extreme_innovation_mase_ratio_max": 1.20,
            "coverage_drop_max": 0.10,
            "width_inflation_max": 1.50,
            "non_inferiority_margin_mase": MARGIN,
            "frozen_before_any_confirmatory_score": True},
        "precision_rule": {
            "sd_planning_source": "development pilot per-origin "
                                  "delta scale (mechanics only)",
            "sd_planning": sd_planning,
            "z_two_sided": round(z, 4),
            "min_series_per_panel": n_min_panel,
            "min_panels": 6,
            "note": f"per-panel support n >= (z*sd/margin)^2 = "
                    f"{n_min_panel}; the executable minimum IS "
                    "this computed value; a panel below it makes "
                    "the screen INCONCLUSIVE"},
        "sensitivity_rule": {
            "sd_grid_n_min": sensitivity,
            "note": "planning sd comes from a small mechanics "
                    "pilot; the observed-precision rule governs "
                    "the final call; the sign/LOPO rules are the "
                    "small-K sensitivity analyses"},
        "inference_method": {
            "rule": "six_panel_screen_t_sign_lopo",
            "detail": ("the PANEL is the inferential unit; six "
                       "panel effects; t-based CI with df=5 PLUS "
                       "exact sign sensitivity PLUS leave-one-"
                       "panel-out — all three must clear the "
                       "frozen margin for ADVANCE. Origins and "
                       "seeds stay nested measurements inside "
                       "each series; series counts never become "
                       "panel-level precision"),
            "coverage_simulation":
                "tools/t2_coverage_sim.py (panel-level t "
                "intervals) + tools/t2_screen_sim.py (composite "
                "six-panel screen rule) — both operate on "
                "synthetic panel effects and are geometry-"
                "independent; re-executed for this draft"},
        "inference_scope": (
            "conclusions are limited to the six named public "
            "panels; the hospital panel is a FULL member under "
            "the common two-origin geometry (767 admissible "
            "series of length 84 -> two consecutive "
            "17-observation score windows; no exception was "
            "created and no model minimum was weakened)"),
        "multiplicity_rule": {
            "alpha": 0.05,
            "method": "single_primary_estimand_no_split"},
        "missing_unit_rule": "a refused/absent unit is a typed "
                             "PREFLIGHT fact; the adjudicator "
                             "refuses population inequality — "
                             "nothing is silently dropped",
        "inconclusive_rule": "any screen panel absent/below "
                             "support, extreme evaluable support "
                             "below the predeclared minimum, "
                             "observed precision unmet, or "
                             "unattributed gain",
        "resource_contract": {"cpu_nice": 15,
                              "max_rss_bytes": 8 << 30,
                              "max_wall_seconds": 14400,
                              "stop_file":
                                  "<state_root>/T2_STOP"},
        "verifier_specification": (
            "fresh-process verifier (t2_fresh_verifier."
            "fresh_verify), CALLED BY run_confirmatory before "
            "the review gate and again before ledger creation: "
            "validates census and design via the productive "
            "parsers, reopens manifested bytes by descriptor, "
            "rebuilds every unit through the C1/C2/C4 loaders, "
            "and re-derives EVERY unit_map field — family, "
            "panel, digest, period, horizon, length, temporal "
            "identity and the exact two-origin windows (the "
            "generator's emitted windows are never the "
            "authority); emits ONLY a non-authorizing "
            "consistency label"),
        "design_review_record_sha256":
            "PENDING_MUSASHI_DESIGN_REVIEW",
    }
    body = {k: draft[k] for k in sorted(draft)}
    draft["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    out = STATE / "t2_screen_design_DRAFT_V6_20260907.json"
    out.write_text(json.dumps(draft, indent=1))
    counts = {}
    for uid, b in unit_map.items():
        counts[b["dataset"]] = counts.get(b["dataset"], 0) + 1
    print(json.dumps({
        "draft_v6_sha256": draft["design_sha256"],
        "series_total": len(series_ids),
        "per_panel_selected": counts,
        "screen_panels": screen_panels,
        "geometry_feasibility": geometry_feasibility,
        "extreme_support_rule": EXTREME_SUPPORT_RULE,
        "min_series_per_panel": n_min_panel}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
