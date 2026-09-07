#!/usr/bin/env python3
"""C21: the fresh-process population verifier — runs BEFORE the
seal and again before any score. From nothing but the manifest and
the physical bytes it: (1) validates the manifest
(descriptor-bound), (2) reopens every admitted dataset by
descriptor, (3) reconstructs every unit and its numeric digest
through the C1/C2/C4 loaders, (4) reproduces the COMPLETE census,
and (5) reproduces the exact design population (family-level
top-k). Equality is SEMANTIC and byte-bound — never a SHA of a
candidate-produced JSON. Emits only a non-authorizing consistency
label."""
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import t2_bank as bank  # noqa: E402
import t2_confirmatory as conf  # noqa: E402
import t2_bank_census as census_mod  # noqa: E402

STATE = Path.home() / ".local/share/agent-multi"


class FreshVerifierRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def rebuild_population(manifest: dict, raw_root: Path) -> dict:
    """Steps 2-4: reopen bytes by descriptor and rebuild every
    unit exactly as the census loaders do."""
    admissible = conf.validate_public_manifest(
        manifest, raw_root=raw_root)
    seen = {}
    rebuilt = {}
    for lid, contract in census_mod.CONTRACTS.items():
        if lid not in admissible:
            continue
        d = admissible[lid]
        fd = conf._open_nofollow_under(raw_root,
                                       Path(d["local_relpath"]))
        try:
            chunks = []
            while True:
                b = os.read(fd, 1 << 20)
                if not b:
                    break
                chunks.append(b)
        finally:
            os.close(fd)
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != d["sha256"]:
            raise FreshVerifierRefusal(
                f"{lid}: bytes drifted between manifest checks")
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tsf = [n for n in zf.namelist()
                   if n.endswith(".tsf")]
            if len(tsf) != 1:
                raise FreshVerifierRefusal(
                    f"{lid}: zip does not contain exactly one "
                    ".tsf")
            tsf_bytes = zf.read(tsf[0])
        panel = bank.parse_tsf_bytes(tsf_bytes, lid)
        built = bank.build_series_units(
            panel, d["family"], contract["period"],
            "monash_record_frequency", contract["max_gap"],
            contract["min_length"], seen_digests=seen)
        rebuilt[lid] = {
            "family": d["family"],
            "admissible_unit_ids": sorted(built["units"]),
            "unit_numeric_digests": {
                uid: u["series_numeric_sha256"]
                for uid, u in built["units"].items()},
            # C28: full per-unit semantics rebuilt from bytes —
            # length and canonical temporal identity — so EVERY
            # unit_map field can be re-derived, not just the
            # numeric digest.
            "unit_meta": {
                uid: {"n": int(u["n"]),
                      "time_identity_sha256":
                          bank.unit_time_identity(u)}
                for uid, u in built["units"].items()},
            "seasonal_period": contract["period"]}
    return rebuilt


def verify_census_semantic(rebuilt: dict, census: dict) -> None:
    """Step 4: the candidate census must equal the rebuilt truth
    unit by unit and digest by digest."""
    pop = census.get("population", {})
    for lid, r in rebuilt.items():
        c = pop.get(lid)
        if c is None:
            raise FreshVerifierRefusal(
                f"census lacks dataset {lid}")
        if sorted(c.get("admissible_unit_ids", [])) != \
                r["admissible_unit_ids"]:
            raise FreshVerifierRefusal(
                f"{lid}: census unit population does not "
                "re-derive from the physical bytes")
        if c.get("unit_numeric_digests") != \
                r["unit_numeric_digests"]:
            raise FreshVerifierRefusal(
                f"{lid}: census unit digests do not re-derive "
                "from the physical bytes")
    extra = set(pop) - set(rebuilt) - {"solar_weekly"}
    ex_ok = {lid for lid in extra
             if pop[lid].get("series_admissible") == 0}
    if extra - ex_ok:
        raise FreshVerifierRefusal(
            f"census carries datasets not derivable from the "
            f"manifest bytes: {sorted(extra - ex_ok)}")


def verify_design_population(rebuilt: dict, design: dict) -> None:
    """Step 5 (C28): the design's exact population re-derives via
    the DECLARED structured selection (family top-k over
    geometry-admissible units), and EVERY unit_map field is
    re-derived from the physical bytes — family, dataset/panel,
    numeric digest, seasonal period, horizon, length, temporal
    identity and the exact origin windows. A map with one true
    digest and forged semantics refuses."""
    tp = design["task_population"]
    fams_primary = set(tp["primary_gate_families"])
    rg = design["role_geometry"]
    n_origins = int(rg["rolling_origins"])
    base_frac = float(rg["origin_base_frac"])
    horizon = int(rg["horizon"])
    sel = tp.get("selection")
    if not isinstance(sel, dict) or \
            sel.get("rule") != "family_top_k_geometry_admissible":
        raise FreshVerifierRefusal(
            "design lacks the structured selection contract — "
            "the verifier re-derives only declared rules")
    k_sel = int(sel["k"])
    salt = str(sel["salt"])
    by_family = {}
    unit_home = {}
    meta_all = {}
    dig_all = {}
    period_by_lid = {}
    for lid, r in rebuilt.items():
        admissible = [uid for uid in r["admissible_unit_ids"]
                      if bank.geometry_admissible(
                          r["unit_meta"][uid]["n"], n_origins,
                          base_frac)]
        by_family.setdefault(r["family"], {})[lid] = admissible
        for uid in r["admissible_unit_ids"]:
            unit_home[uid] = lid
        meta_all.update(r["unit_meta"])
        dig_all.update(r["unit_numeric_digests"])
        period_by_lid[lid] = r["seasonal_period"]
    want = set()
    for fam, ds_map in by_family.items():
        if fam in fams_primary:
            want.update(bank.family_top_k(ds_map, k_sel,
                                          salt=salt))
        else:
            for ids in ds_map.values():
                want.update(ids)
    got = set(tp["series_ids"])
    if got != want:
        raise FreshVerifierRefusal(
            f"design population does not re-derive from the "
            f"bytes (missing {sorted(want - got)[:3]}, extra "
            f"{sorted(got - want)[:3]})")
    umap = tp["unit_map"]
    fam_of_lid = {lid: r["family"] for lid, r in rebuilt.items()}
    for uid in sorted(want):
        b = umap.get(uid)
        if b is None:
            raise FreshVerifierRefusal(
                f"{uid}: selected unit missing from unit_map")
        lid = unit_home.get(uid)
        truth = {
            "family": fam_of_lid.get(lid),
            "dataset": lid,
            "series_numeric_sha256": dig_all.get(uid),
            "seasonal_period": period_by_lid.get(lid),
            "horizon": horizon,
            "n_obs": meta_all[uid]["n"],
            "time_identity_sha256":
                meta_all[uid]["time_identity_sha256"],
            "origin_windows": bank.origin_windows_for(
                meta_all[uid]["n"], n_origins, base_frac)}
        if b != truth:
            diff = sorted(k for k in set(b) | set(truth)
                          if b.get(k) != truth.get(k))
            raise FreshVerifierRefusal(
                f"{uid}: unit_map fields do not re-derive from "
                f"the physical bytes (forged/stale: {diff}) — a "
                "true digest with false semantics never passes")


def fresh_verify(manifest: dict, census: dict, design: dict,
                 raw_root: Path = None,
                 manifest_sha: str = None) -> dict:
    """C28: the ONE fresh-verification entry consumed by the
    executing path (run_confirmatory) and by the CLI alike —
    census and design schemas via the productive parsers, then
    the full byte-level re-derivation. Returns only a
    non-authorizing summary."""
    raw_root = raw_root or (STATE / "t2_public_raw")
    census_mod.validate_census_schema(census)
    if manifest_sha is not None:
        conf.validate_confirmatory_design(design, manifest_sha)
    rebuilt = rebuild_population(manifest, raw_root)
    verify_census_semantic(rebuilt, census)
    verify_design_population(rebuilt, design)
    return {"fresh_verification":
            "POPULATION_REDERIVED_NON_AUTHORIZING",
            "datasets_rebuilt": len(rebuilt),
            "units_rederived": sum(
                len(r["admissible_unit_ids"])
                for r in rebuilt.values()),
            "design_series": len(
                design["task_population"]["series_ids"])}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, default=None)
    args = ap.parse_args()
    raw_root = args.raw_root or (STATE / "t2_public_raw")
    manifest = conf.strict_json_load(args.manifest, "manifest")
    census = conf.strict_json_load(args.census, "census")
    design = conf.strict_json_load(args.design, "design")
    out = fresh_verify(manifest, census, design,
                       raw_root=raw_root,
                       manifest_sha=conf._sha_file(args.manifest))
    print(json.dumps(out, indent=1))
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
