#!/usr/bin/env python3
"""T2-S chronology step 5 (order C25-C30): DRAFT v4 — the PUBLIC
SCREEN design. Supersedes draft v3 by digest; v3 stays on disk as
immutable history.

What changed and why (C29):
- The estimand is the T2-S screen: population = the six named
  public primary panels; superior unit = PANEL; panel effect =
  paired D-X mean of its selected series; primary = the UNWEIGHTED
  mean of the six panel effects. Outputs are ONLY
  ADVANCE_TO_DOMAIN_VALIDATION / DOES_NOT_ADVANCE / INCONCLUSIVE —
  the screen never grants public eligibility and never names
  PUBLICLY_ELIGIBLE_CANDIDATE.
- Every selected unit binds its exact causal geometry BEFORE any
  result: train/score windows derived from the series length and
  the frozen origin contract (C26), plus length and temporal
  identity, all re-derivable from physical bytes (C28).
- Selection now filters by GEOMETRY ADMISSIBILITY (the frozen
  harness contract: n >= 120 and score window >= lags+4) before
  the family top-k. This surfaces a PHYSICAL FACT: the hospital
  panel (all series length 84) admits ZERO units under the frozen
  geometry. It is declared GEOMETRY_LIMITED here, not papered
  over; with it below support the screen is INCONCLUSIVE by
  construction, and the resolution (5-panel redesign, geometry
  amendment, or replacement panel) is an EXTERNAL design decision
  for the Musashi review.
- T2-C (cross-panel confirmation) is documented ONLY as a
  conditional successor; no new panels are acquired.

No design is sealed, no ledger exists, no score is computed."""
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

ROLLING_ORIGINS = 3
ORIGIN_BASE_FRAC = 0.6
K = 40
SALT = "t2_design_v4"
MARGIN = 0.02
FAMILIES_PRIMARY = ["tourism", "urban_pedestrian",
                    "health_hospital", "solar_energy",
                    "electricity_demand", "weather"]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    manifest_path = STATE / "t2_public_data_manifest_20260906.json"
    manifest = conf.strict_json_load(manifest_path, "manifest")
    census = conf.strict_json_load(
        STATE / "t2_bank_census_20260906.json", "census")
    manifest_sha = census["manifest_sha256"]
    # C28: lengths and temporal identity come from PHYSICAL bytes
    # through the same loaders the fresh verifier uses — the
    # windows are therefore re-derivable, never asserted.
    rebuilt = fv.rebuild_population(manifest,
                                    STATE / "t2_public_raw")
    fv.verify_census_semantic(rebuilt, census)

    # ---- disclosed precision planning (single primary estimand,
    # alpha 0.05, no family split => no Bonferroni) ----
    sd_planning = 0.04
    alpha = 0.05
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    n_min_panel = math.ceil((z * sd_planning / MARGIN) ** 2)
    sd_sensitivity = [0.02, 0.04, 0.08]
    sensitivity = {str(s): math.ceil((z * s / MARGIN) ** 2)
                   for s in sd_sensitivity}

    # ---- geometry-admissible selection (C26/C29) ----
    by_family = {}
    for lid, r in rebuilt.items():
        admissible = [uid for uid in r["admissible_unit_ids"]
                      if bank.geometry_admissible(
                          r["unit_meta"][uid]["n"],
                          ROLLING_ORIGINS, ORIGIN_BASE_FRAC)]
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
    geometry_limited = {}
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
        # the six NAMED panels: per primary family, the dataset
        # that carries the family's census-admissible series
        # (solar_weekly has zero census-admissible units and was
        # never a panel). Hospital STAYS named even with zero
        # geometry-admissible units — a named panel's emptiness
        # is a declared fact, not a silent drop.
        if fam in FAMILIES_PRIMARY and n_total > 0:
            screen_panels.append(lid)
            if len(chosen) < n_min_panel:
                geometry_limited[lid] = {
                    "n_geometry_admissible": len(geo_ids),
                    "n_census_admissible": n_total,
                    "fact": "the frozen origin/lag contract "
                            "cannot window enough of this "
                            "panel's series (hospital: every "
                            "series has length 84 < the harness "
                            "minimum 120) — below the "
                            "predeclared per-panel support"}
        for uid in chosen:
            meta = r["unit_meta"][uid]
            unit_digests[uid] = r["unit_numeric_digests"][uid]
            unit_map[uid] = {
                "family": fam, "dataset": lid,
                "series_numeric_sha256":
                    r["unit_numeric_digests"][uid],
                "seasonal_period": r["seasonal_period"],
                "horizon": 1,
                "n_obs": meta["n"],
                "time_identity_sha256":
                    meta["time_identity_sha256"],
                "origin_windows": bank.origin_windows_for(
                    meta["n"], ROLLING_ORIGINS,
                    ORIGIN_BASE_FRAC)}
    screen_panels = sorted(screen_panels)
    if len(screen_panels) != 6:
        raise SystemExit(
            f"REFUSED: expected exactly six screen panels, got "
            f"{screen_panels}")
    series_ids = sorted(unit_map)

    draft = {
        "schema": "agent_multi.t2_screen_design.v4_draft",
        "sealed_after_census_manifest_sha256": manifest_sha,
        "supersedes_draft_sha256": sha_file(
            STATE / "t2_confirmatory_design_DRAFT_V3_20260906"
                    ".json"),
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
            "geometry_limited_panels": geometry_limited,
            "per_dataset": per_dataset,
            "selection": {
                "rule": "family_top_k_geometry_admissible",
                "k": K, "salt": SALT},
            "selection_rule": (
                f"exact top-k by lowest sha256 of '{SALT}|<id>' "
                f"over the WHOLE family (datasets pooled), "
                f"k=min({K}, n), restricted to units the FROZEN "
                "origin/lag contract can window (n>=120 and "
                "score window >= lags+4); global ids unique; "
                "order-independent")},
        "role_geometry": {"rolling_origins": ROLLING_ORIGINS,
                          "origin_base_frac": ORIGIN_BASE_FRAC,
                          "lags": 8, "horizon": 1},
        "arms": {"X": "identity",
                 "D": "causal ewma(alpha=0.3) via T0 contract",
                 "XDR": "[X, D, X-D]",
                 "width_control": "train-frozen independent "
                                  "channels matching XDR width"},
        "models": {
            "ridge": {"lags": 8, "lambda": 1.0,
                      "intercept": "unpenalized",
                      "scaling": "train-only standardization"},
            "mlp_small": {"hidden": [16], "tol": 1e-4,
                          "scaling": "train-only",
                          "epoch_rule": "temporally final 20% of "
                                        "fit rows; grid "
                                        "[40,80,120,200]"},
            "seasonal_naive": {"period": "per-unit predeclared"}},
        "seed_tape": [11, 12, 13],
        "primary_contrast": {
            "delta": "D_minus_X", "model": "ridge",
            "statistic": "paired per-series MASE delta, origins "
                         "averaged within series"},
        "secondary_gates": {
            "attribution": "panel mean of (D-X) minus "
                           "(width_control-X) must be positive",
            "XDR": "secondary descriptive contrast, predeclared, "
                   "never promoted post hoc",
            "preservation": "extreme-innovation gate on explicit "
                            "states (C25): support-bound, "
                            "HARM_INFINITE on X=0&D>0, "
                            "NOT_EVALUABLE never favorable",
            "calibration": "coverage-drop and width-inflation "
                           "gates"},
        "primary_metric": "MASE (train-defined seasonal-naive "
                          "denominator); raw MAE/RMSE per-series "
                          "diagnostics only, never pooled",
        "estimand": {
            "population": "the six named public primary panels "
                          "(screen_panels), exactly as acquired",
            "superior_unit": "panel",
            "panel_effect": "paired D-X mean MASE delta over the "
                            "panel's selected series (origins "
                            "averaged within series first)",
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
                "harm; (5) preservation/calibration/cost/support "
                "gates complete. Insufficient power or precision "
                "=> INCONCLUSIVE; margins are FROZEN before any "
                "score and never adjusted after seeing results"),
            "t2c_successor": (
                "T2-C (cross-panel confirmation, >=3 independent "
                "panels per family) is a CONDITIONAL successor "
                "only: it will be designed with its own "
                "acquisition and review IF T2-S advances and a "
                "family-level claim is wanted. Nothing here "
                "authorizes that acquisition.")},
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
                       "frozen margin for ADVANCE. Chosen because "
                       "my own coverage simulation shows within-"
                       "panel dependence is unidentifiable from "
                       "inside one panel; the screen therefore "
                       "never converts series counts into panel-"
                       "level precision"),
            "coverage_simulation":
                "tools/t2_coverage_sim.py (panel-level t "
                "intervals) + tools/t2_screen_sim.py (composite "
                "six-panel screen rule: boundary type-I, power, "
                "damaged-panel sensitivity)"},
        "inference_scope": (
            "conclusions are limited to the six named public "
            "panels; GEOMETRY_LIMITED FACT: the hospital panel "
            "admits ZERO units under the frozen origin/lag "
            "contract (all series length 84 < 120), so with THIS "
            "bank and THIS geometry the screen is INCONCLUSIVE "
            "by construction at that panel — the resolution "
            "(5-panel redesign, geometry amendment, or "
            "replacement panel) is an explicit open question "
            "for the external design review, not a candidate "
            "decision"),
        "multiplicity_rule": {
            "alpha": 0.05,
            "method": "single_primary_estimand_no_split"},
        "missing_unit_rule": "a refused/absent unit is a typed "
                             "PREFLIGHT fact; the adjudicator "
                             "refuses population inequality — "
                             "nothing is silently dropped",
        "inconclusive_rule": "any screen panel absent/below "
                             "support, observed precision unmet, "
                             "unattributed gain, or extreme "
                             "evidence NOT_EVALUABLE",
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
            "identity and origin windows; emits ONLY a non-"
            "authorizing consistency label"),
        "design_review_record_sha256":
            "PENDING_MUSASHI_DESIGN_REVIEW",
    }
    body = {k: draft[k] for k in sorted(draft)}
    draft["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    out = STATE / "t2_screen_design_DRAFT_V4_20260906.json"
    out.write_text(json.dumps(draft, indent=1))
    counts = {}
    for uid, b in unit_map.items():
        counts[b["dataset"]] = counts.get(b["dataset"], 0) + 1
    print(json.dumps({
        "draft_v4_sha256": draft["design_sha256"],
        "series_total": len(series_ids),
        "per_panel_selected": counts,
        "screen_panels": screen_panels,
        "geometry_limited": geometry_limited,
        "min_series_per_panel": n_min_panel}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
