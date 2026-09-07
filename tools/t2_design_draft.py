#!/usr/bin/env python3
"""T2 chronology step 4: DRAFT of the immutable confirmatory
design + the predeclared precision calculation. This is a
CANDIDATE for the Musashi design review — it is NOT sealed, it
carries no review record, and no confirmatory score exists. The
practical margin frozen here comes from the development pilot's
variance scale, not from any confirmatory outcome."""
import hashlib
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_bank as bank  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    census = json.loads(
        (STATE / "t2_bank_census_20260906.json").read_text())
    manifest_sha = census["manifest_sha256"]
    # ---- precision calculation (C8/C13), fully disclosed ----
    sd_planning = 0.04
    sd_sensitivity = [0.02, 0.04, 0.08]
    margin = 0.02
    families_primary = ["tourism", "urban_pedestrian",
                        "health_hospital", "solar_energy",
                        "electricity_demand", "weather"]
    alpha = 0.05 / len(families_primary)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    n_min = math.ceil((z * sd_planning / margin) ** 2)
    sensitivity = {str(s): math.ceil((z * s / margin) ** 2)
                   for s in sd_sensitivity}
    # C11: EXACT top-k selection by lowest id hash, k=min(40,n)
    K = 40
    population = {}
    unit_digests = {}
    for lid, p_ in census["population"].items():
        ids = p_["admissible_unit_ids"]
        if p_["family"] in families_primary:
            ids = bank.deterministic_top_k(ids, K,
                                           salt="t2_design_v2")
        population[lid] = {"family": p_["family"],
                           "series_ids": ids,
                           "seasonal_period":
                               p_["seasonal_period"]}
        for uid in ids:
            unit_digests[uid] = p_.get(
                "unit_numeric_digests", {}).get(uid, "PENDING")
    series_ids = [sid for p_ in population.values()
                  for sid in p_["series_ids"]]
    draft = {
        "schema": "agent_multi.t2_confirmatory_design.v2_draft",
        "sealed_after_census_manifest_sha256": manifest_sha,
        "supersedes_draft_sha256": "GENESIS_V2_DRAFT",
        "operator": {"kind": "ewma", "params": {"alpha": 0.3},
                     "selection_source":
                         "T1_v4_record_LAB_CALIBRATED"},
        "task_population": {
            "series_ids": sorted(series_ids),
            "unit_digests": unit_digests,
            "families": sorted({p_["family"]
                                for p_ in population.values()}),
            "primary_gate_families": families_primary,
            "sensitivity_only_families": ["hydrology",
                                          "demography"],
            "per_dataset": {k: {"family": v["family"],
                                "n_series":
                                    len(v["series_ids"]),
                                "seasonal_period":
                                    v["seasonal_period"]}
                            for k, v in population.items()},
            "selection_rule": f"exact top-k by lowest sha256 of "
                              f"'t2_design_v2|<id>', k=min({K},"
                              "n_admissible) per primary family "
                              "(order-independent, never exceeds "
                              "k)"},
        "role_geometry": {"rolling_origins": 3,
                          "origin_base_frac": 0.6,
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
            "attribution": "family mean of (D-X) minus "
                           "(width_control-X) must be positive",
            "XDR": "secondary descriptive contrast, predeclared, "
                   "never promoted post hoc",
            "preservation": "extreme-innovation MASE ratio gate",
            "calibration": "coverage-drop and width-inflation "
                           "gates"},
        "primary_metric": "MASE (train-defined seasonal-naive "
                          "denominator); raw MAE/RMSE per-series "
                          "diagnostics only, never pooled",
        "practical_margin_mase": margin,
        "observed_precision_rule": {
            "max_ci_halfwidth": margin,
            "rule": "if any primary family's multiplicity-"
                    "corrected CI half-width exceeds this, the "
                    "result is INCONCLUSIVE even with a favorable "
                    "mean"},
        "harm_margins": {
            "extreme_innovation_mase_ratio_max": 1.20,
            "coverage_drop_max": 0.10,
            "width_inflation_max": 1.50,
            "frozen_before_any_confirmatory_score": True},
        "precision_rule": {
            "sd_planning_source": "development pilot per-origin "
                                  "delta scale (mechanics only)",
            "sd_planning": sd_planning,
            "z_two_sided_bonferroni": round(z, 4),
            "min_series_per_family": n_min,
            "min_families": 6,
            "note": f"n >= (z*sd/margin)^2 = {n_min}; the "
                    "executable minimum IS this computed value"},
        "sensitivity_rule": {
            "sd_grid_n_min": sensitivity,
            "note": "planning sd comes from a small mechanics "
                    "pilot; the observed-precision rule governs "
                    "the final call, and this grid discloses how "
                    "n_min moves with sd"},
        "inference_scope": (
            "conclusions are limited to the named public panels; "
            "generalization to whole series families is NOT "
            "claimed (one panel per family carries no "
            "between-dataset replication)"),
        "multiplicity_rule": {"alpha": 0.05,
                              "method": "bonferroni_by_family"},
        "missing_unit_rule": "a refused/absent unit is a typed "
                             "PREFLIGHT fact; the adjudicator "
                             "refuses population inequality — "
                             "nothing is silently dropped",
        "inconclusive_rule": "any primary family absent/below "
                             "support, observed precision "
                             "unmet, or unattributed gain",
        "resource_contract": {"cpu_nice": 15,
                              "max_rss_bytes": 8 << 30,
                              "max_wall_seconds": 14400,
                              "stop_file":
                                  "<state_root>/T2_STOP"},
        "verifier_specification": (
            "fresh-process verifier: re-parse manifested source "
            "bytes, rebuild units through the C1/C2/C4 loaders, "
            "reconstruct splits/arms/seed tape, recompute every "
            "metric and per-phase cost, check exact population "
            "cardinality and family clusters, and emit ONLY a "
            "non-authorizing consistency label"),
        "design_review_record_sha256":
            "PENDING_MUSASHI_DESIGN_REVIEW",
    }
    body = {k: draft[k] for k in sorted(draft)}
    draft["design_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    out = STATE / "t2_confirmatory_design_DRAFT_V2_20260906.json"
    out.write_text(json.dumps(draft, indent=1))
    counts = {p_["family"]: 0 for p_ in population.values()}
    for p_ in population.values():
        counts[p_["family"]] += len(p_["series_ids"])
    print(json.dumps({
        "draft_v2_sha256": draft["design_sha256"],
        "series_total": len(series_ids),
        "per_family": counts,
        "n_min": n_min,
        "sensitivity_n_min": sensitivity}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
