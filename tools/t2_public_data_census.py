#!/usr/bin/env python3
"""T2.0 census: locally available public NON-FINANCIAL time-series
tasks, with physical data authority (bytes digest, logical source,
license, schema, frequency, missingness policy).

The census is the AUTHORITY for what T2 may consume. Financial
series and self-generated synthetic replicas are inventoried only
to be excluded. If the minimum confirmatory out-of-family bank
cannot be built from lawful, verifiable local inputs, the census
verdict is CONFIRMATORY_BANK_UNAVAILABLE and the caller must
return PUBLIC_DATA_REQUIRED with the exact deficit."""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# Preferred confirmatory sources (order T2.0.1). All verified
# ABSENT from local storage on 2026-09-06; network retrieval is
# outside my standing constraints, so the operator must supply
# the bytes.
PREFERRED_CONFIRMATORY = {
    "monash_subset": {
        "logical_source": "Monash Time Series Forecasting "
                          "Repository (zenodo .tsf archives)",
        "license": "CC-BY-4.0 (per-dataset; verify on receipt)",
        "want": "a bounded subset of >=4 independent task "
                "families (e.g. tourism, hospital, solar, "
                "pedestrian) as .tsf bytes",
        "status": "ABSENT_LOCALLY"},
    "etth1": {
        "logical_source": "ETDataset ETTh1.csv (electricity "
                          "transformer temperature, hourly)",
        "license": "CC-BY-ND 4.0 (verify on receipt)",
        "want": "ETTh1.csv exact bytes",
        "status": "ABSENT_LOCALLY"},
    "weather_jena": {
        "logical_source": "Jena climate / Weather (Max Planck "
                          "Institute) 10-minute multivariate",
        "license": "public/citation-required (verify on receipt)",
        "want": "jena_climate_2009_2016.csv or the Autoformer "
                "weather.csv exact bytes",
        "status": "ABSENT_LOCALLY"},
    "m4_subset": {
        "logical_source": "M4 competition subset",
        "license": "open (verify on receipt)",
        "want": "only if it adds an independent task family",
        "status": "ABSENT_LOCALLY_OPTIONAL"},
}

# Development-only units physically present inside the installed
# statsmodels package (BSD-3-Clause; the embedded datasets are
# public-domain/attribution series). ZERO confirmatory authority:
# univariate, short, and too few independent families for the
# sealed T2.2/T2.3 gate.
_SM_UNITS = {
    "sm_co2": {
        "module": "co2", "column": "co2",
        "family": "atmospheric_chemistry",
        "frequency": "weekly", "license_note":
            "public domain (NOAA/Scripps via statsmodels, "
            "package BSD-3-Clause)",
        "missingness_policy": "forward-fill declared at load; "
                              "raw NaN count recorded"},
    "sm_sunspots": {
        "module": "sunspots", "column": "SUNACTIVITY",
        "family": "solar_activity",
        "frequency": "yearly", "license_note":
            "public domain (SILSO via statsmodels)",
        "missingness_policy": "none expected; refuse on NaN"},
    "sm_nile": {
        "module": "nile", "column": "volume",
        "family": "hydrology",
        "frequency": "yearly", "license_note":
            "public domain (classic Nile series via statsmodels)",
        "missingness_policy": "none expected; refuse on NaN"},
}


def census_development_units() -> dict:
    """Physical authority for the development-only units: the
    packaged csv BYTES are digested; the dataframe is reconstructed
    from those exact bytes at consumption time."""
    import importlib
    units = {}
    for uid, meta in _SM_UNITS.items():
        mod = importlib.import_module(
            f"statsmodels.datasets.{meta['module']}")
        csv = Path(mod.__file__).parent / f"{meta['module']}.csv"
        if not csv.is_file():
            units[uid] = {"status": "ABSENT", **meta}
            continue
        units[uid] = {
            "status": "PRESENT_DEVELOPMENT_ONLY",
            "bytes_sha256": _sha_file(csv),
            "logical_source":
                f"python-env:statsmodels.datasets.{meta['module']}"
                f"/{meta['module']}.csv",
            **meta}
    return units


EXCLUDED_CLASSES = {
    "financial": "order T2.0.3: financial data may never replace "
                 "missing public tasks (FX/equity/VIX inventories "
                 "under feature-eng/tests/data and examples/data "
                 "are excluded)",
    "synthetic_own": "order T2.0.3: synthetic replicas may never "
                     "replace missing public tasks "
                     "(synthetic-datagen outputs, T1 bank units)",
    "macro_economic": "statsmodels macrodata is economic — "
                      "excluded under the non-financial rule",
}


def build_census() -> dict:
    dev = census_development_units()
    present_dev = [u for u, v in dev.items()
                   if v["status"] == "PRESENT_DEVELOPMENT_ONLY"]
    families = {dev[u]["family"] for u in present_dev}
    verdict = "CONFIRMATORY_BANK_UNAVAILABLE"
    deficit = {
        "reason": ("the preferred confirmatory sources are absent "
                   "from local storage and network retrieval is "
                   "outside standing constraints; the local "
                   "development units are univariate, short and "
                   f"span only {len(families)} independent "
                   "families — insufficient task support for the "
                   "sealed confirmatory gate (whole-family "
                   "holdouts + multiplicity correction)"),
        "operator_must_supply": {
            k: v for k, v in PREFERRED_CONFIRMATORY.items()},
        "minimum": ("bytes for >=4 independent non-financial task "
                    "families with >=1000 observations per series "
                    "or >=20 series per family, licenses included"),
    }
    return {"schema": "agent_multi.t2_public_data_census.v1",
            "verdict": verdict,
            "preferred_confirmatory": PREFERRED_CONFIRMATORY,
            "development_only_units": dev,
            "excluded_classes": EXCLUDED_CLASSES,
            "deficit": deficit}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    census = build_census()
    args.output.write_text(json.dumps(census, indent=1))
    print(json.dumps({"verdict": census["verdict"],
                      "development_units_present": sorted(
                          u for u, v in
                          census["development_only_units"].items()
                          if v["status"] ==
                          "PRESENT_DEVELOPMENT_ONLY")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
