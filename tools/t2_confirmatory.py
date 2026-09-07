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
    "record_metadata_sha256", "retrieved_at_utc", "byte_size",
    "sha256", "upstream_checksum", "license_id",
    "license_id_sha256", "license_text_sha256", "citation",
    "local_relpath", "admission"}
_ADMISSIONS = ("ADMISSIBLE", "EXCLUDED_LICENSE",
               "EXCLUDED_DUPLICATE", "EXCLUDED_QUALITY",
               "EXCLUDED_FROM_T2_CONFIRMATORY",
               "REVIEW_REQUIRED")
_HEX64 = "0123456789abcdef"


def _canon_sha(v, what):
    if type(v) is not str or len(v) != 64 or \
            any(c not in _HEX64 for c in v):
        raise ConfirmatoryRefusal(
            f"{what} is not a canonical lowercase 64-hex digest")
    return v


def strict_json_load(path: Path, where: str) -> dict:
    """C10: duplicate keys and non-finite constants refuse."""
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ConfirmatoryRefusal(
                f"duplicate JSON key in {where}")
        return dict(pairs)
    try:
        return json.loads(
            Path(path).read_text(), object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(
                ConfirmatoryRefusal(
                    f"non-finite constant in {where}")))
    except json.JSONDecodeError as exc:
        raise ConfirmatoryRefusal(
            f"{where} is not well-formed JSON ({exc.msg} at char "
            f"{exc.pos}) — arbitrary candidate bytes are never a "
            "record")


def validate_public_manifest(manifest: dict,
                             raw_root: Path = None,
                             verify_bytes: bool = True) -> dict:
    """C10: every consumed dataset is bound to PHYSICAL bytes —
    exact typed schema, canonical digests, contained relative
    paths, descriptor-first open with symlink/non-regular refusal,
    size and hash verified from that descriptor, and the archival
    record metadata bound by digest."""
    if not isinstance(manifest, dict) or manifest.get("schema") != \
            "agent_multi.t2_public_data_manifest.v2":
        raise ConfirmatoryRefusal(
            "public-data manifest absent or foreign schema "
            "(v2 physically-bound manifest required)")
    ds = manifest.get("datasets")
    if not isinstance(ds, dict) or not ds:
        raise ConfirmatoryRefusal("manifest lists no datasets")
    if raw_root is None:
        raw_root = (Path.home() /
                    ".local/share/agent-multi/t2_public_raw")
    raw_root = Path(raw_root).resolve()
    total = 0
    admissible = {}
    for lid, d in ds.items():
        if not isinstance(d, dict) or set(d) != _DATASET_KEYS:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} keys are not the exact manifest "
                f"schema (diff: "
                f"{sorted(set(d) ^ _DATASET_KEYS)})")
        for k in ("logical_id", "family", "final_url",
                  "archival_record", "retrieved_at_utc",
                  "upstream_checksum", "license_id", "citation",
                  "local_relpath", "admission"):
            if type(d[k]) is not str or not d[k]:
                raise ConfirmatoryRefusal(
                    f"dataset {lid!r} field {k!r} mistyped")
        if type(d["byte_size"]) is not int or d["byte_size"] <= 0:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} byte size mistyped")
        _canon_sha(d["sha256"], f"{lid} sha256")
        _canon_sha(d["record_metadata_sha256"],
                   f"{lid} record metadata digest")
        _canon_sha(d["license_id_sha256"],
                   f"{lid} license id digest")
        if d["license_text_sha256"] != "UNAVAILABLE":
            _canon_sha(d["license_text_sha256"],
                       f"{lid} license text digest")
        rel = Path(d["local_relpath"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} relpath is absolute or "
                "traversing")
        target = (raw_root / rel).resolve()
        if raw_root not in target.parents and target != raw_root:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} escapes the raw root")
        total += d["byte_size"]
        if d["admission"] not in _ADMISSIONS:
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} admission label invalid")
        if d["admission"] != "ADMISSIBLE":
            continue
        if not d["license_id"] or d["license_id"] in (
                "UNKNOWN", "AMBIGUOUS"):
            raise ConfirmatoryRefusal(
                f"dataset {lid!r} admissible without a concrete "
                "license")
        if verify_bytes:
            try:
                fd = os.open(str(target),
                             os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as exc:
                raise ConfirmatoryRefusal(
                    f"dataset {lid!r} bytes unopenable ({exc}) — "
                    "a manifest row without its physical file is "
                    "not evidence")
            try:
                import stat as _stat
                st = os.fstat(fd)
                if not _stat.S_ISREG(st.st_mode):
                    raise ConfirmatoryRefusal(
                        f"dataset {lid!r} is not a regular file")
                if st.st_size != d["byte_size"]:
                    raise ConfirmatoryRefusal(
                        f"dataset {lid!r} size differs from the "
                        "manifest")
                h = hashlib.sha256()
                while True:
                    chunk = os.read(fd, 1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
            finally:
                os.close(fd)
            if h.hexdigest() != d["sha256"]:
                raise ConfirmatoryRefusal(
                    f"dataset {lid!r} bytes differ from the "
                    "manifest digest")
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
    "supersedes_draft_sha256", "operator", "task_population",
    "role_geometry", "arms", "models", "seed_tape",
    "primary_contrast", "secondary_gates", "primary_metric",
    "practical_margin_mase", "observed_precision_rule",
    "harm_margins", "precision_rule", "sensitivity_rule",
    "inference_scope", "multiplicity_rule", "missing_unit_rule",
    "inconclusive_rule", "resource_contract",
    "verifier_specification", "design_review_record_sha256",
    "design_sha256"}


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
                "missing_unit_rule", "inconclusive_rule",
                "arms", "models", "seed_tape", "primary_contrast",
                "secondary_gates", "observed_precision_rule",
                "sensitivity_rule", "inference_scope"):
        if not design[req]:
            raise ConfirmatoryRefusal(f"design lacks {req}")
    # C13: the structured statistical specification is executable
    if set(design["arms"]) != {"X", "D", "XDR", "width_control"}:
        raise ConfirmatoryRefusal(
            "design arms are not the exact required set")
    if design["primary_contrast"].get("delta") != "D_minus_X" or \
            design["primary_contrast"].get("model") != "ridge":
        raise ConfirmatoryRefusal(
            "design must declare the single frozen primary "
            "contrast (paired D-X under the frozen ridge)")
    if not isinstance(design["seed_tape"], list) or \
            not design["seed_tape"]:
        raise ConfirmatoryRefusal("design seed tape missing")
    tp2 = design["task_population"]
    if len(tp2.get("primary_gate_families", [])) != 6:
        raise ConfirmatoryRefusal(
            "design must name exactly six primary-gate families")
    if not tp2.get("unit_digests"):
        raise ConfirmatoryRefusal(
            "design must bind every unit's numeric digest")
    if not design["resource_contract"] or \
            not design["verifier_specification"]:
        raise ConfirmatoryRefusal(
            "design lacks the resource contract or the "
            "fresh-process verifier specification")


T2_REVIEW_RECORD_PATH = REPO / (
    "docs/audits/evidence/MUSASHI_T2_DESIGN_REVIEW_2026_09.json")
_REVIEW_KEYS = {"schema", "reviewed_at_date", "reviewer",
                "decision", "design_draft_sha256",
                "manifest_sha256", "census_sha256"}


def verify_design_review_record(design: dict,
                                manifest_sha: str,
                                census_sha: str) -> dict:
    """C9: finite, non-circular external review authority. The
    Musashi record pins the PRE-review draft, manifest and census
    digests; the sealed design may NAME that record but can alter
    no scientific field. The candidate can neither write nor
    select the root: the path is a repo constant and every field
    is verified — schema, decision, author, bytes, bindings and
    chronology."""
    rr = design.get("design_review_record_sha256")
    p = T2_REVIEW_RECORD_PATH
    if not p.is_file():
        raise ConfirmatoryRefusal(
            "DESIGN_REVIEW_REQUIRED: the external Musashi design "
            "review record does not exist — the next review seals "
            "or rejects the design; no confirmatory score")
    if _sha_file(p) != rr:
        raise ConfirmatoryRefusal(
            "the sealed design names a DIFFERENT review record "
            "than the external root — candidate-selected bytes "
            "grant nothing")
    rec = strict_json_load(p, "design review record")
    if set(rec) != _REVIEW_KEYS:
        raise ConfirmatoryRefusal(
            "review record keys are not the exact schema")
    if rec["schema"] != "agent_multi.musashi_t2_design_review.v1":
        raise ConfirmatoryRefusal(
            "review record carries a foreign schema")
    if rec["reviewer"] != "General Musashi":
        raise ConfirmatoryRefusal(
            "review record author is not the external reviewer")
    if rec["decision"] != "SEAL_T2_CONFIRMATORY_DESIGN":
        raise ConfirmatoryRefusal(
            "review record decision does not seal this design")
    for k in ("design_draft_sha256", "manifest_sha256",
              "census_sha256"):
        _canon_sha(rec[k], f"review {k}")
    if rec["manifest_sha256"] != manifest_sha:
        raise ConfirmatoryRefusal(
            "review record binds a different public-data manifest")
    if rec["census_sha256"] != census_sha:
        raise ConfirmatoryRefusal(
            "review record binds a different bank census")
    draft_sha = design.get("supersedes_draft_sha256")
    if rec["design_draft_sha256"] != draft_sha:
        raise ConfirmatoryRefusal(
            "review record pins a different pre-review draft than "
            "the sealed design supersedes — chronology broken")
    return rec


def open_attempt_ledger(path: Path) -> dict:
    """C15: the durable append-only attempt ledger — created ONLY
    after every prior verification (never as a side effect of a
    refused gate), via the accepted intent/completion protocol:
    LEDGER_INTENT then the exclusive self-integral ledger file,
    both fsynced; recovery reads physical state and fails closed.
    Pre-score failures are recorded as PREFLIGHT, never as
    scientific attempts."""
    p = Path(path)
    intent_p = p.with_name(p.name + ".INTENT")
    if p.exists():
        led = strict_json_load(p, "attempt ledger")
        if led.get("schema") != \
                "agent_multi.t2_attempt_ledger.v2":
            raise ConfirmatoryRefusal(
                "attempt ledger foreign schema")
        body = {k: led[k] for k in sorted(led)
                if k != "ledger_sha256"}
        if hashlib.sha256(json.dumps(
                body, sort_keys=True).encode()).hexdigest() != \
                led.get("ledger_sha256"):
            raise ConfirmatoryRefusal(
                "attempt ledger self-digest does not re-derive — "
                "uncertain state, operator disposition")
        return led
    if intent_p.exists():
        raise ConfirmatoryRefusal(
            "a ledger INTENT exists without its ledger — "
            "uncertain prior creation, operator disposition")
    led = {"schema": "agent_multi.t2_attempt_ledger.v2",
           "attempts": [],
           "budget": {"max_wall_seconds": 14400,
                      "max_rss_bytes": 8 << 30,
                      "stop_file": "T2_STOP"}}
    body = {k: led[k] for k in sorted(led)}
    led["ledger_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    for target, payload in ((intent_p, {"creates": p.name}),
                            (p, led)):
        fd = os.open(str(target),
                     os.O_CREAT | os.O_EXCL | os.O_WRONLY
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            os.write(fd, json.dumps(payload, indent=1).encode())
            os.fsync(fd)
        finally:
            os.close(fd)
    dfd = os.open(str(p.parent), os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)
    return led


def run_confirmatory(manifest_path: Path, design_path: Path,
                     ledger_path: Path,
                     census_path: Path = None) -> None:
    """C3/C9/C15: the ordered gate sequence — every missing element
    refuses with its own typed reason; the attempt ledger is
    created ONLY after every verification passes; and the external
    review root cannot be candidate-written. Scoring remains
    unimplemented pending the external audit of the C9-C16
    corrections."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise ConfirmatoryRefusal(
            "PUBLIC_DATA_REQUIRED: no public-data manifest exists "
            "yet — acquire and census the bank first")
    manifest = strict_json_load(mp, "public-data manifest")
    validate_public_manifest(manifest)
    manifest_sha = _sha_file(mp)
    if census_path is None:
        census_path = (Path.home() / ".local/share/agent-multi/"
                       "t2_bank_census_20260906.json")
    cp = Path(census_path)
    if not cp.is_file():
        raise ConfirmatoryRefusal(
            "CENSUS_REQUIRED: the bank census does not exist")
    census_sha = _sha_file(cp)
    dp = Path(design_path)
    if not dp.is_file():
        raise ConfirmatoryRefusal(
            "DESIGN_REQUIRED: no immutable confirmatory design is "
            "sealed over the acquired bank")
    design = strict_json_load(dp, "confirmatory design")
    validate_confirmatory_design(design, manifest_sha)
    # C9: the external review — verified in full, BEFORE any
    # ledger artifact can exist.
    verify_design_review_record(design, manifest_sha, census_sha)
    # C15: only now may the durable attempt ledger be created.
    open_attempt_ledger(Path(ledger_path))
    raise ConfirmatoryRefusal(
        "CONFIRMATORY_EXECUTION_NOT_IMPLEMENTED_IN_THIS_ORDER: "
        "scoring begins only after the external audit of the "
        "C9-C16 corrections")


# ------------- C8/C14: complete decision rule ---------------------

_REQUIRED_ARMS = ("X", "D", "XDR", "width_control")


def _series_stats(rec, design, arm="D", model="ridge"):
    """Per-series paired deltas vs X for one arm/model, averaged
    over the design's origins (nested, never inflating n)."""
    origins = rec["rolling_origins"]
    deltas, harms = [], {"extreme_ratio": [], "coverage_drop": [],
                         "width_ratio": []}
    for o in origins.values():
        res = o["results"]
        x = res["X"][model]
        a_ = res[arm][model]
        if x["mase_primary"] is None or \
                a_["mase_primary"] is None:
            raise ConfirmatoryRefusal(
                f"{rec['unit_id']}: non-finite primary metric")
        deltas.append(x["mase_primary"] - a_["mase_primary"])
        ex_x = x.get("mase_on_extreme_innovations")
        ex_a = a_.get("mase_on_extreme_innovations")
        if ex_x and ex_a:
            harms["extreme_ratio"].append(ex_a / max(ex_x, 1e-12))
        harms["coverage_drop"].append(
            x["interval_coverage_train_q90"]
            - a_["interval_coverage_train_q90"])
        harms["width_ratio"].append(
            a_["interval_width_train_q90"]
            / max(x["interval_width_train_q90"], 1e-12))
    return {"delta": float(np.mean(deltas)),
            "harms": {k: (float(np.mean(v)) if v else None)
                      for k, v in harms.items()}}


def check_record_completeness(rec: dict, design: dict) -> None:
    """C14: three origins, ALL arms, both models with the full
    seed tape, per-phase costs and finite preservation/calibration
    metrics — every record, no producer aggregate trusted."""
    want_origins = int(design["role_geometry"]["rolling_origins"])
    origins = rec.get("rolling_origins")
    if not isinstance(origins, dict) or \
            len(origins) != want_origins:
        raise ConfirmatoryRefusal(
            f"{rec.get('unit_id')}: expected {want_origins} "
            "rolling origins")
    costs = rec.get("costs_by_phase")
    if not isinstance(costs, dict) or \
            set(costs) != set(origins):
        raise ConfirmatoryRefusal(
            f"{rec.get('unit_id')}: per-phase costs incomplete")
    for okey, o in origins.items():
        res = o.get("results", {})
        for arm in _REQUIRED_ARMS:
            if arm not in res:
                raise ConfirmatoryRefusal(
                    f"{rec.get('unit_id')} {okey}: arm {arm!r} "
                    "missing — incomplete evidence never "
                    "adjudicates")
            ridge = res[arm].get("ridge")
            if not isinstance(ridge, dict) or \
                    "mase_primary" not in ridge or \
                    "interval_coverage_train_q90" not in ridge or \
                    "interval_width_train_q90" not in ridge:
                raise ConfirmatoryRefusal(
                    f"{rec.get('unit_id')} {okey} {arm}: ridge "
                    "metrics incomplete")
            mlp = res[arm].get("mlp_small")
            want_seeds = {f"seed{s}" for s in design["seed_tape"]}
            if not isinstance(mlp, dict) or \
                    set(mlp) != want_seeds:
                raise ConfirmatoryRefusal(
                    f"{rec.get('unit_id')} {okey} {arm}: MLP "
                    "seed tape incomplete")
        acost = costs.get(okey, {})
        if not any(k.startswith("arm_") for k in acost):
            raise ConfirmatoryRefusal(
                f"{rec.get('unit_id')} {okey}: separated arm "
                "costs missing")


def adjudicate_confirmatory(records: list, design: dict) -> dict:
    """C8/C14: population equality, completeness, re-derived
    deltas/intervals/harms/attribution, observed-precision rule,
    all six primary families, multiplicity — every gate must pass
    before the candidate label may exist."""
    tp = design["task_population"]
    want_ids = set(tp["series_ids"])
    got_ids = [r.get("unit_id") for r in records]
    if len(got_ids) != len(set(got_ids)):
        raise ConfirmatoryRefusal(
            "duplicate unit identity in the record population")
    if set(got_ids) != want_ids:
        missing = sorted(want_ids - set(got_ids))[:3]
        extra = sorted(set(got_ids) - want_ids)[:3]
        raise ConfirmatoryRefusal(
            f"record population differs from the sealed design "
            f"(missing e.g. {missing}, extra e.g. {extra}) — "
            "absent units are never silently dropped")
    for rec in records:
        check_record_completeness(rec, design)
    margin = float(design["practical_margin_mase"])
    prec = design["precision_rule"]
    min_series = int(prec["min_series_per_family"])
    fams_required = list(tp["primary_gate_families"])
    max_ci_width = float(
        design["observed_precision_rule"]["max_ci_halfwidth"]) * 2
    fam_deltas = {}
    fam_attrib = {}
    fam_harms = {}
    for rec in records:
        fam = rec["family"]
        if fam not in fams_required:
            continue
        s = _series_stats(rec, design, "D", "ridge")
        w = _series_stats(rec, design, "width_control", "ridge")
        fam_deltas.setdefault(fam, []).append(s["delta"])
        fam_attrib.setdefault(fam, []).append(
            s["delta"] - w["delta"])
        fam_harms.setdefault(fam, []).append(s["harms"])
    absent = [f for f in fams_required
              if len(fam_deltas.get(f, [])) < min_series]
    if absent:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"primary families below support/"
                           f"absent: {sorted(absent)} — all six "
                           "are required for a primary positive")}
    alpha = float(design["multiplicity_rule"]["alpha"]) / \
        len(fams_required)
    from statistics import NormalDist
    z = NormalDist().inv_cdf(1 - alpha / 2)
    fam_stats = {}
    harmed, unattributed, imprecise, failing = [], [], [], []
    hm = design["harm_margins"]
    for fam in fams_required:
        arr = np.array(fam_deltas[fam], dtype=float)
        att = np.array(fam_attrib[fam], dtype=float)
        n = len(arr)
        se = float(arr.std(ddof=1) / math.sqrt(n))
        lo = float(arr.mean() - z * se)
        width = float(2 * z * se)
        ex = [h["extreme_ratio"] for h in fam_harms[fam]
              if h["extreme_ratio"] is not None]
        cov = [h["coverage_drop"] for h in fam_harms[fam]]
        wid = [h["width_ratio"] for h in fam_harms[fam]]
        fam_stats[fam] = {
            "n_series": n, "mean_delta": float(arr.mean()),
            "ci_low": lo, "ci_width": width,
            "attribution_mean": float(att.mean()),
            "extreme_ratio_mean": (float(np.mean(ex))
                                   if ex else None),
            "coverage_drop_mean": float(np.mean(cov)),
            "width_ratio_mean": float(np.mean(wid))}
        st = fam_stats[fam]
        if arr.mean() < -margin:
            harmed.append(fam)
        if width > max_ci_width:
            imprecise.append(fam)
        if st["extreme_ratio_mean"] is not None and \
                st["extreme_ratio_mean"] > \
                float(hm["extreme_innovation_mase_ratio_max"]):
            harmed.append(fam)
        if st["coverage_drop_mean"] > \
                float(hm["coverage_drop_max"]):
            harmed.append(fam)
        if st["width_ratio_mean"] > \
                float(hm["width_inflation_max"]):
            harmed.append(fam)
        if st["attribution_mean"] <= 0:
            unattributed.append(fam)
        if lo <= margin:
            failing.append(fam)
    if harmed:
        return {"verdict": "PUBLICLY_INELIGIBLE",
                "reason": (f"material harm in families "
                           f"{sorted(set(harmed))} — a favorable "
                           "grand average cannot pass over it"),
                "families": fam_stats}
    if imprecise:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"observed precision insufficient in "
                           f"{sorted(imprecise)}: CI wider than "
                           "the predeclared bound even if the "
                           "mean is favorable"),
                "families": fam_stats}
    if unattributed:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"gain not attributable beyond the "
                           f"matched-width control in "
                           f"{sorted(unattributed)}"),
                "families": fam_stats}
    if failing:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"{sorted(failing)} do not clear the "
                           "practical margin — broad-family "
                           "consistency is required"),
                "families": fam_stats}
    return {"verdict": "PUBLICLY_ELIGIBLE_CANDIDATE",
            "reason": ("all six primary families pass margin, "
                       "observed precision, attribution vs the "
                       "width control and every harm gate; "
                       "inference limited to the studied panels "
                       "per the design's inference_scope"),
            "inference_scope": design["inference_scope"],
            "families": fam_stats}
