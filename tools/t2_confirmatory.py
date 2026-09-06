#!/usr/bin/env python3
"""T2 confirmatory path (orders C3, C8).

C3: `--confirmatory` is no longer an unconditional refusal — it is
a REAL path that can open only when ALL of these independently
validate: (1) a complete public-data manifest, (2) an immutable
design sealed after acquisition/census and before any score,
(3) the accepted T1 operator identity, (4) the exact task
population and role geometry, (5) an attempt ledger and resource
contract, and (6) a fresh-process verifier specification. For the
present order the path stays CLOSED at the final gate: the design
must additionally carry a Musashi design-review record, which does
not exist yet — the next review seals or rejects the design. No
confirmatory score is computed here.

C8: the decision rule is predeclared and hierarchical — dataset/
family is the outer cluster, the series is the inner paired unit,
rolling origins and seeds are nested repeated measurements that
never inflate the independent sample count. PUBLICLY_ELIGIBLE
requires broad-family consistency, a confidence bound beyond the
frozen practical margin, no material preservation/calibration
harm, and complete cost accounting; a favorable grand average with
concentrated family harm does not pass."""
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

T1_ACCEPTED_OPERATOR = {"kind": "ewma", "params": {"alpha": 0.3},
                        "selection_source":
                            "T1_v4_record_LAB_CALIBRATED"}
MANIFEST_BYTE_LIMIT = 2 * 1024 ** 3          # D0: 2 GiB compressed


class ConfirmatoryRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


_DATASET_KEYS = {
    "logical_id", "family", "final_url", "archival_record",
    "retrieved_at_utc", "byte_size", "sha256",
    "upstream_checksum", "license_id", "license_text_sha256",
    "citation", "local_relpath", "admission"}


def validate_public_manifest(manifest: dict) -> dict:
    """C3.1/D0: every consumed dataset carries its full physical
    and legal identity; ambiguous licenses are excluded, never
    assumed; the cumulative compressed size respects the D0 cap."""
    if not isinstance(manifest, dict) or manifest.get("schema") != \
            "agent_multi.t2_public_data_manifest.v1":
        raise ConfirmatoryRefusal(
            "public-data manifest absent or foreign schema")
    ds = manifest.get("datasets")
    if not isinstance(ds, dict) or not ds:
        raise ConfirmatoryRefusal("manifest lists no datasets")
    total = 0
    admissible = {}
    for lid, d in ds.items():
        if not isinstance(d, dict) or set(d) != _DATASET_KEYS:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} keys are not the exact manifest "
                "schema")
        if type(d["byte_size"]) is not int or d["byte_size"] <= 0:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} byte size mistyped")
        if len(d["sha256"]) != 64:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} digest is not 64-hex")
        total += d["byte_size"]
        if d["admission"] not in ("ADMISSIBLE",
                                  "EXCLUDED_LICENSE",
                                  "EXCLUDED_DUPLICATE",
                                  "EXCLUDED_QUALITY",
                                  "REVIEW_REQUIRED"):
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} admission label invalid")
        if d["admission"] == "ADMISSIBLE":
            if not d["license_id"] or d["license_id"] in (
                    "UNKNOWN", "AMBIGUOUS"):
                raise ConfirmatoryRefusal(
                    f"dataset {lid!r} admissible without a "
                    "concrete license — an absent or ambiguous "
                    "license excludes the candidate")
            admissible[lid] = d
    if total > MANIFEST_BYTE_LIMIT:
        raise ConfirmatoryRefusal(
            "manifest exceeds the authorized 2 GiB cumulative cap")
    if not admissible:
        raise ConfirmatoryRefusal(
            "manifest carries no ADMISSIBLE dataset")
    return admissible


_DESIGN_KEYS = {
    "schema", "sealed_after_census_manifest_sha256",
    "operator", "task_population", "role_geometry",
    "primary_metric", "practical_margin_mase",
    "harm_margins", "precision_rule", "multiplicity_rule",
    "missing_unit_rule", "inconclusive_rule",
    "resource_contract", "verifier_specification",
    "design_review_record_sha256", "design_sha256"}


def validate_confirmatory_design(design: dict,
                                 manifest_sha: str) -> None:
    """C3.2-C3.6 + C8: the immutable design, sealed after the
    census/acquisition and before any score."""
    if not isinstance(design, dict) or set(design) != _DESIGN_KEYS:
        raise ConfirmatoryRefusal(
            "confirmatory design absent or not the exact schema")
    body = {k: design[k] for k in sorted(design)
            if k != "design_sha256"}
    if hashlib.sha256(json.dumps(
            body, sort_keys=True).encode()).hexdigest() != \
            design["design_sha256"]:
        raise ConfirmatoryRefusal(
            "design self-integrity digest does not re-derive")
    if design["sealed_after_census_manifest_sha256"] != \
            manifest_sha:
        raise ConfirmatoryRefusal(
            "design was not sealed over THIS public-data manifest")
    if design["operator"] != T1_ACCEPTED_OPERATOR:
        raise ConfirmatoryRefusal(
            "design operator differs from the accepted T1 v4 "
            "identity")
    tp = design["task_population"]
    if not isinstance(tp, dict) or not tp.get("series_ids") or \
            not tp.get("families"):
        raise ConfirmatoryRefusal(
            "design lacks the exact task population")
    for req in ("practical_margin_mase", "harm_margins",
                "precision_rule", "multiplicity_rule",
                "missing_unit_rule", "inconclusive_rule"):
        if not design[req]:
            raise ConfirmatoryRefusal(f"design lacks {req}")
    if not design["resource_contract"] or \
            not design["verifier_specification"]:
        raise ConfirmatoryRefusal(
            "design lacks the resource contract or the "
            "fresh-process verifier specification")


def open_attempt_ledger(path: Path) -> dict:
    """C3.5: one append-only attempt ledger; existing attempts are
    facts, never retried away."""
    p = Path(path)
    if p.exists():
        led = json.loads(p.read_text())
        if led.get("schema") != \
                "agent_multi.t2_attempt_ledger.v1":
            raise ConfirmatoryRefusal(
                "attempt ledger foreign schema")
        return led
    led = {"schema": "agent_multi.t2_attempt_ledger.v1",
           "attempts": []}
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(led, indent=1).encode())
    finally:
        os.close(fd)
    return led


def run_confirmatory(manifest_path: Path, design_path: Path,
                     ledger_path: Path) -> None:
    """C3: the ordered gate sequence. Every missing element refuses
    with its own typed reason; when everything else validates, the
    final gate still requires the Musashi design-review record —
    absent today, so the path is CLOSED and no score is computed."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise ConfirmatoryRefusal(
            "PUBLIC_DATA_REQUIRED: no public-data manifest exists "
            "yet — acquire and census the bank first")
    manifest = json.loads(mp.read_text())
    validate_public_manifest(manifest)
    manifest_sha = _sha_file(mp)
    dp = Path(design_path)
    if not dp.is_file():
        raise ConfirmatoryRefusal(
            "DESIGN_REQUIRED: no immutable confirmatory design is "
            "sealed over the acquired bank")
    design = json.loads(dp.read_text())
    validate_confirmatory_design(design, manifest_sha)
    open_attempt_ledger(Path(ledger_path))
    rr = design.get("design_review_record_sha256")
    rr_path = REPO / ("docs/audits/evidence/"
                      "MUSASHI_T2_DESIGN_REVIEW_2026_09.json")
    if not rr or not rr_path.is_file() or \
            _sha_file(rr_path) != rr:
        raise ConfirmatoryRefusal(
            "DESIGN_REVIEW_REQUIRED: the sealed design must carry "
            "the Musashi design-review record digest and that "
            "record must exist — the next review seals or rejects "
            "the design; no confirmatory score is computed")
    raise ConfirmatoryRefusal(
        "CONFIRMATORY_EXECUTION_NOT_IMPLEMENTED_IN_THIS_ORDER: "
        "scoring begins only after the design review")


# ---------------------- C8: decision rule -------------------------

def paired_series_deltas(records: list, arm: str,
                         baseline: str = "X") -> dict:
    """Per-SERIES paired MASE deltas (baseline − arm; positive =
    improvement), averaging nested origins/seeds WITHIN the series
    first — origins and seeds never inflate n."""
    out = {}
    for rec in records:
        vals = []
        for o in rec["rolling_origins"].values():
            res = o["results"]
            b = res[baseline]["ridge"]["mase_primary"]
            a_ = res[arm]["ridge"]["mase_primary"]
            if b is None or a_ is None:
                continue
            vals.append(b - a_)
            for seed_m in res[arm]["mlp_small"].values():
                pass          # seeds live inside the model family
        if vals:
            out[rec["unit_id"]] = {
                "delta": float(np.mean(vals)),
                "family": rec["family"]}
    return out


def adjudicate_confirmatory(records: list, design: dict) -> dict:
    """C8: hierarchical, predeclared. INCONCLUSIVE and
    INELIGIBLE paths are first-class outcomes."""
    margin = float(design["practical_margin_mase"])
    prec = design["precision_rule"]
    min_series = int(prec["min_series_per_family"])
    min_families = int(prec["min_families"])
    deltas = paired_series_deltas(records, "D")
    fams = {}
    for uid, d in deltas.items():
        fams.setdefault(d["family"], []).append(d["delta"])
    usable = {f: v for f, v in fams.items()
              if len(v) >= min_series}
    if len(usable) < min_families:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"only {len(usable)} families with "
                           f">={min_series} series; the "
                           f"predeclared minimum is "
                           f"{min_families}")}
    alpha = float(design["multiplicity_rule"]["alpha"]) / \
        max(1, len(usable))
    fam_stats = {}
    harmed = []
    for fam, v in usable.items():
        arr = np.array(v, dtype=float)
        n = len(arr)
        se = float(arr.std(ddof=1) / math.sqrt(n)) if n > 1 else \
            float("inf")
        from statistics import NormalDist
        z = NormalDist().inv_cdf(1 - alpha / 2)
        lo = float(arr.mean() - z * se)
        fam_stats[fam] = {"n_series": n,
                          "mean_delta": float(arr.mean()),
                          "ci_low": lo}
        if arr.mean() < -margin:
            harmed.append(fam)
    if harmed:
        return {"verdict": "PUBLICLY_INELIGIBLE",
                "reason": ("concentrated family harm in "
                           f"{sorted(harmed)} — a favorable grand "
                           "average cannot pass over it"),
                "families": fam_stats}
    passing = [f for f, s in fam_stats.items()
               if s["ci_low"] > margin]
    if len(passing) == len(usable):
        return {"verdict": "PUBLICLY_ELIGIBLE_CANDIDATE",
                "reason": ("every family's multiplicity-corrected "
                           "confidence bound clears the frozen "
                           "practical margin — still subject to "
                           "harm-margin, cost and width-control "
                           "attribution gates"),
                "families": fam_stats}
    return {"verdict": "INCONCLUSIVE",
            "reason": (f"{len(passing)}/{len(usable)} families "
                       "clear the margin — broad-family "
                       "consistency is required"),
            "families": fam_stats}
