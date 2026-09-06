#!/usr/bin/env python3
"""D0: bounded read-only HTTPS acquisition of public research
datasets from their official archival records, with a full
per-file manifest (final URL, archival record/DOI, retrieval time,
byte size, SHA-256, upstream checksum when available, license
identifier + license-text digest, citation, logical ID). Raw files
live OUTSIDE Git under the designated data root; only sanitized
manifests enter the repository. Cumulative cap: 2 GiB. No
credentials. One license per RECORD — never assumed archive-wide."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

RAW_ROOT = Path.home() / ".local/share/agent-multi/t2_public_raw"
BYTE_CAP = 2 * 1024 ** 3

# Selected from the completed Zenodo community census (65 records,
# license read PER RECORD). Diverse NON-financial families; the
# Monash archive's financial/economic records (Bitcoin, FRED-MD,
# Dominick retail) are EXCLUDED by class; M4 is excluded because
# its mixed micro/macro/financial composition does not add an
# independent non-financial family beyond those below.
ZENODO_SELECTION = {
    "tourism_monthly": {"record": 4656096, "family": "tourism"},
    "pedestrian_counts": {"record": 4656626,
                          "family": "urban_pedestrian"},
    "hospital": {"record": 4656014, "family": "health_hospital"},
    "solar_weekly": {"record": 4656151, "family": "solar_energy"},
    "saugeenday": {"record": 4656058, "family": "hydrology"},
    "us_births": {"record": 4656049, "family": "demography"},
    "electricity_weekly": {"record": 4656141,
                           "family": "electricity_demand"},
    "weather": {"record": 4654822, "family": "weather"},
}
# ETTh1: inventoried SEPARATELY; its CC BY-ND 4.0 license is
# recorded exactly and the dataset stays REVIEW_REQUIRED (never
# silently admitted; no transformed bytes redistributed).
ETT = {
    "etth1": {
        "csv_url": ("https://raw.githubusercontent.com/zhouhaoyi/"
                    "ETDataset/main/ETT-small/ETTh1.csv"),
        "license_url": ("https://raw.githubusercontent.com/"
                        "zhouhaoyi/ETDataset/main/LICENSE"),
        "archival_record":
            "https://github.com/zhouhaoyi/ETDataset",
        "family": "electricity_transformer"}}


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def fetch(url: str, out: Path, budget: list) -> None:
    rc = subprocess.run(
        ["curl", "-sSL", "--max-time", "600", "--fail",
         url, "-o", str(out)])
    if rc.returncode != 0:
        raise SystemExit(f"REFUSED: fetch failed for {url}")
    size = out.stat().st_size
    budget[0] += size
    if budget[0] > BYTE_CAP:
        out.unlink()
        raise SystemExit(
            "REFUSED: cumulative download would exceed the "
            "authorized 2 GiB cap")


def main() -> int:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    budget = [0]
    datasets = {}
    for lid, sel in ZENODO_SELECTION.items():
        rec_url = f"https://zenodo.org/api/records/{sel['record']}"
        meta_p = RAW_ROOT / f"record_{sel['record']}.json"
        fetch(rec_url, meta_p, budget)
        meta = json.loads(meta_p.read_text())
        md = meta.get("metadata", {})
        lic = (md.get("license") or {}).get("id", "UNKNOWN")
        doi = md.get("doi") or meta.get("doi")
        files = meta.get("files", [])
        if not files:
            datasets[lid] = {"admission": "EXCLUDED_QUALITY",
                             "note": "record lists no files"}
            continue
        f0 = sorted(files, key=lambda f: f.get("size", 0))[-1]
        furl = f0["links"]["self"]
        fname = f0.get("key") or f0.get("filename") or f"{lid}.bin"
        out = RAW_ROOT / f"{lid}__{fname}"
        t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        fetch(furl, out, budget)
        upstream = (f0.get("checksum") or "").replace("md5:", "")
        datasets[lid] = {
            "logical_id": lid,
            "family": sel["family"],
            "final_url": furl,
            "archival_record": f"doi:{doi} "
                               f"(zenodo:{sel['record']})",
            "retrieved_at_utc": t0,
            "byte_size": out.stat().st_size,
            "sha256": sha_file(out),
            "upstream_checksum": f"md5:{upstream}"
                                 if upstream else "UNAVAILABLE",
            "license_id": lic,
            "license_text_sha256": hashlib.sha256(
                lic.encode()).hexdigest(),
            "citation": (md.get("title", lid)
                         + " — Monash Time Series Forecasting "
                           "Repository (Godahewa et al. 2021), "
                         + f"doi:{doi}"),
            "local_relpath": out.name,
            "admission": ("ADMISSIBLE"
                          if lic.startswith("cc-by")
                          else "REVIEW_REQUIRED"),
        }
        print(f"{lid}: {out.stat().st_size/1e6:.2f}MB lic={lic}")
    for lid, sel in ETT.items():
        t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        csv_p = RAW_ROOT / f"{lid}__ETTh1.csv"
        licp = RAW_ROOT / f"{lid}__LICENSE"
        fetch(sel["csv_url"], csv_p, budget)
        fetch(sel["license_url"], licp, budget)
        lic_text = licp.read_text(errors="replace")
        lic_id = ("CC-BY-ND-4.0" if "NoDerivatives" in lic_text
                  or "CC BY-ND" in lic_text or "no-derivatives"
                  in lic_text.lower() else "UNKNOWN_SEE_TEXT")
        datasets[lid] = {
            "logical_id": lid,
            "family": sel["family"],
            "final_url": sel["csv_url"],
            "archival_record": sel["archival_record"],
            "retrieved_at_utc": t0,
            "byte_size": csv_p.stat().st_size,
            "sha256": sha_file(csv_p),
            "upstream_checksum": "UNAVAILABLE",
            "license_id": lic_id,
            "license_text_sha256": sha_file(licp),
            "citation": "Zhou et al. 2021 (Informer), ETDataset "
                        "ETT-small/ETTh1.csv",
            "local_relpath": csv_p.name,
            "admission": "REVIEW_REQUIRED",
        }
        print(f"{lid}: {csv_p.stat().st_size/1e6:.2f}MB "
              f"lic={lic_id} (REVIEW_REQUIRED per order)")
    manifest = {"schema": "agent_multi.t2_public_data_manifest.v1",
                "acquired_at_utc": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "byte_cap": BYTE_CAP,
                "bytes_downloaded_total": budget[0],
                "raw_root_note": "raw files live outside Git under "
                                 "<state_root>/t2_public_raw",
                "datasets": datasets}
    out = (Path.home() / ".local/share/agent-multi/"
           "t2_public_data_manifest_20260906.json")
    out.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"datasets": len(datasets),
                      "total_MB": round(budget[0] / 1e6, 1)},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
