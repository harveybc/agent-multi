#!/usr/bin/env python3
"""T2 chronology step 3: census of the ACQUIRED public bank — the
exact admissible/excluded population, built from the manifested
bytes through the C1/C2/C4 loaders with GLOBAL cross-archive
deduplication. ETTh1 stays REVIEW_REQUIRED (license) and is
inventoried, never admitted here. No scores are computed."""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_bank as bank  # noqa: E402

RAW = Path.home() / ".local/share/agent-multi/t2_public_raw"

# Predeclared per-dataset loading contract (period provenance =
# dataset frequency documented by the Monash record).
CONTRACTS = {
    "tourism_monthly": {"period": 12, "min_length": 120,
                        "max_gap": 0},
    "pedestrian_counts": {"period": 24, "min_length": 500,
                          "max_gap": 24},
    "hospital": {"period": 12, "min_length": 60, "max_gap": 0},
    "solar_weekly": {"period": 52, "min_length": 100,
                     "max_gap": 0},
    "solar_10_minutes": {"period": 144, "min_length": 1000,
                         "max_gap": 144},
    "saugeenday": {"period": 365, "min_length": 1000,
                   "max_gap": 0},
    "us_births": {"period": 7, "min_length": 1000, "max_gap": 0},
    "electricity_weekly": {"period": 52, "min_length": 100,
                           "max_gap": 0},
    "weather": {"period": 365, "min_length": 1000, "max_gap": 30},
}
MAX_SERIES_PER_PANEL = 300     # bounded census parse per archive


def main() -> int:
    manifest = json.loads(
        (Path.home() / ".local/share/agent-multi/"
         "t2_public_data_manifest_20260906.json").read_text())
    seen = {}
    population = {}
    excluded_summary = {}
    for lid, contract in CONTRACTS.items():
        d = manifest["datasets"][lid]
        raw_p = RAW / d["local_relpath"]
        raw = raw_p.read_bytes()
        if hashlib.sha256(raw).hexdigest() != d["sha256"]:
            raise SystemExit(
                f"REFUSED: {lid} bytes differ from the manifest")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tsf_names = [n for n in zf.namelist()
                         if n.endswith(".tsf")]
            if len(tsf_names) != 1:
                raise SystemExit(
                    f"REFUSED: {lid} zip does not contain exactly "
                    "one .tsf")
            tsf_bytes = zf.read(tsf_names[0])
        panel = bank.parse_tsf_bytes(tsf_bytes, lid,
                                     max_series=
                                     MAX_SERIES_PER_PANEL)
        built = bank.build_series_units(
            panel, d["family"], contract["period"],
            "monash_record_frequency", contract["max_gap"],
            contract["min_length"], seen_digests=seen)
        lengths = sorted(u["n"] for u in built["units"].values())
        population[lid] = {
            "family": d["family"],
            "zip_sha256": d["sha256"],
            "tsf_member_sha256": hashlib.sha256(
                tsf_bytes).hexdigest(),
            "frequency_declared": panel["frequency"],
            "seasonal_period": contract["period"],
            "series_parsed": len(panel["series"]),
            "series_admissible": len(built["units"]),
            "series_excluded": len(built["excluded"]),
            "admissible_unit_ids": sorted(built["units"]),
            "length_min_median_max": (
                [lengths[0], lengths[len(lengths) // 2],
                 lengths[-1]] if lengths else None),
        }
        reasons = {}
        for uid, why in built["excluded"].items():
            key = why.split(":")[0][:40]
            reasons[key] = reasons.get(key, 0) + 1
        excluded_summary[lid] = reasons
        print(f"{lid}: parsed={len(panel['series'])} "
              f"admissible={len(built['units'])} "
              f"excluded={len(built['excluded'])}")
    families = {}
    for lid, p in population.items():
        families.setdefault(p["family"], 0)
        families[p["family"]] += p["series_admissible"]
    out = {
        "schema": "agent_multi.t2_bank_census.v1",
        "manifest_sha256": hashlib.sha256(
            (Path.home() / ".local/share/agent-multi/"
             "t2_public_data_manifest_20260906.json"
             ).read_bytes()).hexdigest(),
        "bounded_parse_note": f"census parsed at most "
                              f"{MAX_SERIES_PER_PANEL} series per "
                              "archive (disclosed bound; the "
                              "sealed design fixes the exact "
                              "scored population)",
        "population": population,
        "excluded_reasons": excluded_summary,
        "admissible_series_by_family": families,
        "etth1": {"status": "REVIEW_REQUIRED",
                  "license_id":
                      manifest["datasets"]["etth1"]["license_id"],
                  "license_text_sha256":
                      manifest["datasets"]["etth1"][
                          "license_text_sha256"],
                  "note": "CC BY-ND 4.0 recorded exactly; not "
                          "admitted and no transformed bytes "
                          "redistributed pending review"}}
    outp = (Path.home() / ".local/share/agent-multi/"
            "t2_bank_census_20260906.json")
    outp.write_text(json.dumps(out, indent=1))
    print(json.dumps({"families": families}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
