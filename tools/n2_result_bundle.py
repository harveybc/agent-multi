#!/usr/bin/env python3
"""Self-contained N2 result bundle + offline verifier (order
agent-multi@4c1f1532 §3 C4).

`build` packs, for every one of the 60 original N2 units: identity,
terminal state, the VERBATIM result payload text and its sha256 —
plus the verbatim ledger — into one sanitized JSON bundle (no
absolute paths, no topology). `verify` authenticates the bundle
WITHOUT the private run directory: it rejects missing, extra,
duplicate or altered units; recomputes every unit id and both result
digests (file-byte sha256 and the runtime's canonical self-digest);
reaggregates the verdict through the shared science_aggregate; and
requires semantic equality with the committed N2 verdict trace. The
original local run directory is evidence input, never the only
durable authority."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from agent_plugins.experiment_runtime import (  # noqa: E402
    sha_obj, unit_id)

_spec = importlib.util.spec_from_file_location(
    "target_horizon_census_n2",
    REPO / "tools" / "target_horizon_census_n2.py")
_tcn2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tcn2)
science_aggregate = _tcn2.science_aggregate

BUNDLE_SCHEMA = "agent_multi.n2_result_bundle.v1"

# semantic core compared against the committed trace: additive
# fields introduced by later corrections (available_blocks,
# p_resolution, bootstrap_p_reported, sealed-rule causes) are NOT
# part of the frozen scientific content
CANDIDATE_KEYS = ("outcome", "pooled_skill_vs_strongest",
                  "strongest_baseline", "bootstrap_p_one_sided",
                  "all_windows_positive_vs_every_baseline",
                  "holm_p", "licensed")


class BundleRefusal(ValueError):
    """Typed refusal: the bundle failed authentication."""


def build(run_root: Path, out_path: Path) -> dict:
    units_dir = run_root / "census" / "units"
    ledger = json.loads(
        (run_root / "census" / "ledger.json").read_text())
    entries = []
    for u in ledger["units"]:
        uid = u["unit_id"]
        state = json.loads(
            (units_dir / f"{uid}.state.json").read_text())
        raw = (units_dir / f"{uid}.result.json").read_text()
        entries.append({
            "unit_id": uid,
            "identity": u["identity"],
            "state": state["state"],
            "attempt": state.get("attempt"),
            "result_text": raw,
            "result_sha256": hashlib.sha256(
                raw.encode()).hexdigest()})
    bundle = {"schema": BUNDLE_SCHEMA,
              "experiment": ledger["experiment"],
              "ledger": ledger,
              "units": entries}
    out_path.write_text(json.dumps(bundle, indent=1) + "\n")
    return {"units": len(entries), "path": str(out_path)}


def verify(bundle_path: Path, trace_path: Path) -> dict:
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise BundleRefusal("unknown bundle schema")
    ledger = bundle["ledger"]
    expected = [u["unit_id"] for u in ledger["units"]]
    identities = {u["unit_id"]: u["identity"]
                  for u in ledger["units"]}
    seen = [e["unit_id"] for e in bundle["units"]]
    if len(seen) != len(set(seen)):
        raise BundleRefusal("duplicate units in bundle")
    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    if missing:
        raise BundleRefusal(f"missing units: {missing[:3]}")
    if extra:
        raise BundleRefusal(f"extra units not in ledger: "
                            f"{extra[:3]}")
    results = {}
    for e in bundle["units"]:
        uid = e["unit_id"]
        if e["state"] != "COMPLETED":
            raise BundleRefusal(
                f"unit {uid} is {e['state']}, not COMPLETED")
        if e["identity"] != identities[uid]:
            raise BundleRefusal(
                f"unit {uid}: bundled identity differs from ledger")
        if unit_id(e["identity"]) != uid:
            raise BundleRefusal(
                f"unit {uid}: identity does not hash to its id")
        raw = e["result_text"]
        digest = hashlib.sha256(raw.encode()).hexdigest()
        if digest != e["result_sha256"]:
            raise BundleRefusal(
                f"unit {uid}: result bytes altered "
                f"({digest[:12]} != {e['result_sha256'][:12]})")
        payload = json.loads(raw)
        if payload.get("unit_id") != uid:
            raise BundleRefusal(
                f"unit {uid}: payload claims unit "
                f"{payload.get('unit_id')}")
        self_digest = sha_obj({k: v for k, v in payload.items()
                               if k != "result_digest"})
        if payload.get("result_digest") != self_digest:
            raise BundleRefusal(
                f"unit {uid}: runtime self-digest mismatch")
        results[uid] = payload
    fresh = science_aggregate(ledger, results, [])
    committed = json.loads(trace_path.read_text())
    drift = []
    for field in ("verdict", "holm_pvalues",
                  "inconclusive_candidates"):
        if field in committed and \
                committed[field] != fresh.get(field):
            drift.append(field)
    for key, a in committed.get("candidates", {}).items():
        f = fresh["candidates"].get(key, {})
        for ck in CANDIDATE_KEYS:
            if ck in a and a[ck] != f.get(ck):
                drift.append(f"candidates.{key}.{ck}")
        for wk, w in a.get("windows", {}).items():
            for sk, sv in w.items():
                if sk.startswith("skill_vs_") and \
                        f.get("windows", {}).get(wk, {}) \
                        .get(sk) != sv:
                    drift.append(f"candidates.{key}.{wk}.{sk}")
    if "selection" in committed:
        if committed["selection"].get("selected") != \
                fresh.get("selection", {}).get("selected"):
            drift.append("selection.selected")
    for key, c in committed.get("negative_controls", {}).items():
        fc = fresh.get("negative_controls", {}).get(key, {})
        for flag in ("detects_leakage", "falsely_passes"):
            if flag in c and c[flag] != fc.get(flag):
                drift.append(f"controls.{key}.{flag}")
    if drift:
        raise BundleRefusal(
            f"semantic drift vs committed trace: {drift[:6]}")
    return {"verdict": "BUNDLE_VERIFIED_SEMANTICALLY_EQUAL",
            "units_verified": len(results),
            "reaggregated_verdict": fresh["verdict"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--run-root", required=True)
    b.add_argument("--out", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--bundle", required=True)
    v.add_argument("--trace", required=True)
    args = parser.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(Path(args.run_root),
                               Path(args.out))))
        return 0
    try:
        print(json.dumps(verify(Path(args.bundle),
                                Path(args.trace)), indent=1))
        return 0
    except BundleRefusal as refusal:
        print(json.dumps({"verdict": "BUNDLE_REFUSED",
                          "reason": str(refusal)}, indent=1))
        return 1


if __name__ == "__main__":
    sys.exit(main())
