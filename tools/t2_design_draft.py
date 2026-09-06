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
    # ---- precision calculation (C8), fully disclosed ----
    # sd of per-series paired MASE deltas: development-pilot scale
    # (co2/sunspots per-origin deltas ~0.02-0.08) -> sd ~= 0.04 as
    # the planning value. CI half-width must beat the practical
    # margin under Bonferroni across families.
    sd_planning = 0.04
    margin = 0.02
    families_primary = ["tourism", "urban_pedestrian",
                        "health_hospital", "solar_energy",
                        "electricity_demand", "weather"]
    alpha = 0.05 / len(families_primary)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    n_min = math.ceil((z * sd_planning / margin) ** 2)
    # deterministic per-family series subsample (id hash, never
    # outcome) to bound confirmatory compute
    per_family_cap = max(n_min + 10, 40)
    population = {}
    for lid, p in census["population"].items():
        ids = p["admissible_unit_ids"]
        if p["family"] in families_primary and \
                len(ids) > per_family_cap:
            frac = per_family_cap / len(ids)
            ids = bank.deterministic_subsample(
                ids, frac, salt="t2_design_v1")
        population[lid] = {"family": p["family"],
                           "series_ids": ids,
                           "seasonal_period":
                               p["seasonal_period"]}
    series_ids = [sid for p in population.values()
                  for sid in p["series_ids"]]
    draft = {
        "schema": "agent_multi.t2_confirmatory_design.v1",
        "sealed_after_census_manifest_sha256": manifest_sha,
        "operator": {"kind": "ewma", "params": {"alpha": 0.3},
                     "selection_source":
                         "T1_v4_record_LAB_CALIBRATED"},
        "task_population": {
            "series_ids": sorted(series_ids),
            "families": sorted({p["family"]
                                for p in population.values()}),
            "primary_gate_families": families_primary,
            "sensitivity_only_families": ["hydrology",
                                          "demography"],
            "per_dataset": {k: {"family": v["family"],
                                "n_series":
                                    len(v["series_ids"]),
                                "seasonal_period":
                                    v["seasonal_period"]}
                            for k, v in population.items()},
            "subsample_rule": "deterministic id-hash, salt "
                              "t2_design_v1, per-family cap "
                              f"{per_family_cap}"},
        "role_geometry": {"rolling_origins": 3,
                          "origin_base_frac": 0.6,
                          "lags": 8, "horizon": 1},
        "primary_metric": "MASE (train-defined seasonal-naive "
                          "denominator); raw MAE/RMSE per-series "
                          "diagnostics only, never pooled",
        "practical_margin_mase": margin,
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
            "min_series_per_family": max(20, n_min),
            "min_families": 4,
            "note": f"n >= (z*sd/margin)^2 = {n_min}; declared "
                    "minimum raised to 20 for safety"},
        "multiplicity_rule": {"alpha": 0.05,
                              "method": "bonferroni_by_family"},
        "missing_unit_rule": "typed refusal recorded per unit; a "
                             "family falling below "
                             "min_series_per_family leaves the "
                             "primary gate (disclosed), never "
                             "silently imputed",
        "inconclusive_rule": "fewer than min_families usable, or "
                             "mixed family consistency, or "
                             "planning-precision not met",
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
    out = STATE / "t2_confirmatory_design_DRAFT_20260906.json"
    out.write_text(json.dumps(draft, indent=1))
    print(json.dumps({
        "draft_sha256": draft["design_sha256"],
        "series_total": len(series_ids),
        "primary_families": families_primary,
        "min_series_per_family":
            draft["precision_rule"]["min_series_per_family"],
        "n_min_from_precision": n_min}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
