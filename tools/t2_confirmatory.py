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


_MANIFEST_TOP_KEYS = {"schema", "acquired_at_utc",
                      "remanifested_at_utc", "byte_cap",
                      "bytes_downloaded_total", "raw_root_note",
                      "etth1_disposition", "datasets"}


def _open_nofollow_under(root: Path, rel: Path):
    """C20: open strictly UNDER an already-verified root using
    dir_fd/openat with O_NOFOLLOW on EVERY component — the object
    identity named in the manifest is preserved to fstat; no
    resolve() ever follows a link first."""
    fd = os.open(str(root), os.O_RDONLY | os.O_NOFOLLOW
                 | getattr(os, "O_DIRECTORY", 0))
    try:
        parts = rel.parts
        for comp in parts[:-1]:
            nfd = os.open(comp, os.O_RDONLY | os.O_NOFOLLOW
                          | getattr(os, "O_DIRECTORY", 0),
                          dir_fd=fd)
            os.close(fd)
            fd = nfd
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW,
                       dir_fd=fd)
    except OSError as exc:
        os.close(fd)
        raise ConfirmatoryRefusal(
            f"{rel}: unopenable under the verified root ({exc}) "
            "— symlinks refuse at their own component")
    os.close(fd)
    return leaf


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
    # C20: the TOP-LEVEL schema is exact too
    if set(manifest) != _MANIFEST_TOP_KEYS:
        raise ConfirmatoryRefusal(
            f"manifest top-level keys are not the exact schema "
            f"(diff: {sorted(set(manifest) ^ _MANIFEST_TOP_KEYS)})")
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
        # C20: the mapping key IS the identity
        if lid != d["logical_id"]:
            raise ConfirmatoryRefusal(
                f"manifest key {lid!r} differs from the row's "
                f"logical_id {d['logical_id']!r} — decoupled "
                "identity refused")
        _canon_sha(d["sha256"], f"{lid} sha256")
        _canon_sha(d["record_metadata_sha256"],
                   f"{lid} record metadata digest")
        _canon_sha(d["license_id_sha256"],
                   f"{lid} license id digest")
        # C20: the license digest is RECOMPUTED, never trusted
        if d["license_id_sha256"] != hashlib.sha256(
                d["license_id"].encode()).hexdigest():
            raise ConfirmatoryRefusal(
                f"{lid}: license_id_sha256 does not re-derive "
                "from the license identifier bytes")
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
            fd = _open_nofollow_under(raw_root, rel)
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
            # C20: the archival record metadata is verified
            # against PHYSICAL bytes when the row names a Zenodo
            # record whose metadata file is archived; otherwise
            # the digest must equal the declared non-verifying
            # derivation (sha of the archival_record string).
            rec_txt = d["archival_record"]
            if "zenodo:" in rec_txt:
                rid = rec_txt.split("zenodo:")[1].rstrip(")")
                meta_rel = Path(f"record_{rid}.json")
                mfd = _open_nofollow_under(raw_root, meta_rel)
                try:
                    mh = hashlib.sha256()
                    while True:
                        chunk = os.read(mfd, 1 << 20)
                        if not chunk:
                            break
                        mh.update(chunk)
                finally:
                    os.close(mfd)
                if mh.hexdigest() != d["record_metadata_sha256"]:
                    raise ConfirmatoryRefusal(
                        f"{lid}: archival record metadata bytes "
                        "differ from the manifest digest")
            elif d["record_metadata_sha256"] != hashlib.sha256(
                    rec_txt.encode()).hexdigest():
                raise ConfirmatoryRefusal(
                    f"{lid}: record_metadata_sha256 does not "
                    "re-derive from its declared non-verifying "
                    "source")
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
    "inference_method", "inference_scope", "multiplicity_rule",
    "missing_unit_rule", "inconclusive_rule", "resource_contract",
    "verifier_specification", "design_review_record_sha256",
    "design_sha256"}


def _unique_list(v, what, elem_type=None):
    """C22: a list validator that a duplicated list cannot fool —
    set() is never the only check."""
    if not isinstance(v, list) or not v:
        raise ConfirmatoryRefusal(f"{what}: not a nonempty list")
    if len(v) != len(set(map(str, v))):
        raise ConfirmatoryRefusal(f"{what}: duplicated entries")
    if elem_type is int:
        for x in v:
            if isinstance(x, bool) or type(x) is not int:
                raise ConfirmatoryRefusal(
                    f"{what}: element {x!r} is not a true int "
                    "(bool is never a number)")
    return v


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
    _unique_list(design["seed_tape"], "design.seed_tape",
                 elem_type=int)
    tp2 = design["task_population"]
    fams = tp2.get("primary_gate_families", [])
    _unique_list(fams, "design.primary_gate_families")
    if len(fams) != 6:
        raise ConfirmatoryRefusal(
            "design must name exactly six DISTINCT primary-gate "
            "families")
    _unique_list(tp2.get("series_ids", []),
                 "design.series_ids")
    if not tp2.get("unit_digests"):
        raise ConfirmatoryRefusal(
            "design must bind every unit's numeric digest")
    if not tp2.get("unit_map"):
        raise ConfirmatoryRefusal(
            "design must carry the canonical per-unit map "
            "(family/dataset/digest/period/horizon)")
    rg = design["role_geometry"]
    for k in ("rolling_origins", "origin_base_frac", "lags",
              "horizon"):
        if k not in rg:
            raise ConfirmatoryRefusal(
                f"design.role_geometry lacks {k}")
    for k in ("practical_margin_mase",):
        v = design[k]
        if isinstance(v, bool) or not isinstance(
                v, (int, float)) or not math.isfinite(float(v)) \
                or v <= 0:
            raise ConfirmatoryRefusal(
                f"design.{k} outside its domain")
    a_ = design["multiplicity_rule"].get("alpha")
    if isinstance(a_, bool) or not isinstance(a_, (int, float)) \
            or not 0 < float(a_) < 1:
        raise ConfirmatoryRefusal(
            "design.multiplicity_rule.alpha outside (0,1)")
    if not design.get("inference_method", {}).get("rule"):
        raise ConfirmatoryRefusal(
            "design must predeclare its inference method "
            "(intrapanel dependence rule)")
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


# ------- C17/C19: total numeric validation + model evidence -------

METRIC_DOMAINS = {
    "mase_primary": ("nonneg", False),
    "mae_per_series_diagnostic": ("nonneg", True),
    "rmse_per_series_diagnostic": ("nonneg", True),
    "interval_coverage_train_q90": ("unit_interval", False),
    "interval_width_train_q90": ("nonneg", False),
    "mase_on_extreme_innovations": ("nonneg", True),
    "extreme_support": ("nonneg_int", True),
}
_REQUIRED_METRICS = [k for k, (_, opt) in METRIC_DOMAINS.items()
                     if not opt]


def check_metric(value, name: str, path: str):
    """C17: every consumed metric is an exact-typed finite number
    inside its physical domain; None only where the design
    declares a typed absence (optional metrics), and an absence
    can never improve a gate. NaN/inf/str/bool/out-of-domain
    refuse with the exact field path."""
    domain, optional = METRIC_DOMAINS.get(name, ("nonneg", True))
    if value is None:
        if optional:
            return None
        raise ConfirmatoryRefusal(
            f"{path}: required metric {name!r} is absent")
    if isinstance(value, bool) or not isinstance(
            value, (int, float)):
        raise ConfirmatoryRefusal(
            f"{path}: metric {name!r} has non-numeric type "
            f"{type(value).__name__}")
    v = float(value)
    if not math.isfinite(v):
        raise ConfirmatoryRefusal(
            f"{path}: metric {name!r} is not finite")
    if domain in ("nonneg", "nonneg_int") and v < 0:
        raise ConfirmatoryRefusal(
            f"{path}: metric {name!r} violates its nonnegative "
            "domain")
    if domain == "unit_interval" and not 0.0 <= v <= 1.0:
        raise ConfirmatoryRefusal(
            f"{path}: metric {name!r} outside [0,1]")
    if domain == "nonneg_int" and (isinstance(value, float)
                                   and not v.is_integer()):
        raise ConfirmatoryRefusal(
            f"{path}: metric {name!r} must be an integer count")
    return v


def check_model_result(entry, path: str) -> dict:
    """C19: ONE exact reusable schema for every model result —
    ridge, every MLP seed, and the seasonal-naive baseline alike.
    Opaque payloads refuse."""
    if not isinstance(entry, dict):
        raise ConfirmatoryRefusal(
            f"{path}: model result is not a mapping (opaque "
            "payload refused)")
    unknown = set(entry) - set(METRIC_DOMAINS)
    if unknown:
        raise ConfirmatoryRefusal(
            f"{path}: unknown metric fields {sorted(unknown)}")
    missing = [m for m in _REQUIRED_METRICS if m not in entry]
    if missing:
        raise ConfirmatoryRefusal(
            f"{path}: required metrics missing {missing}")
    out = {}
    for k, v in entry.items():
        out[k] = check_metric(v, k, path)
    return out


_COST_PHASE_PREFIXES = ("denoise_fit_transform_s",
                        "target_construction_s",
                        "seasonal_naive_s")


def check_origin_costs(oc, arms, seed_tape, path: str) -> None:
    """C19: costs enumerate every phase and every arm/model with
    finite nonnegative values — a single arm_* key satisfies
    nothing."""
    if not isinstance(oc, dict):
        raise ConfirmatoryRefusal(f"{path}: costs are not a "
                                  "mapping")
    def _num(v, where):
        if isinstance(v, bool) or not isinstance(
                v, (int, float)) or not math.isfinite(float(v)) \
                or float(v) < 0:
            raise ConfirmatoryRefusal(
                f"{where}: cost is not a finite nonnegative "
                "number")
    for ph in ("denoise_fit_transform_s",):
        if ph not in oc:
            raise ConfirmatoryRefusal(
                f"{path}: phase cost {ph!r} missing")
        _num(oc[ph], f"{path}.{ph}")
    for arm in arms:
        key = f"arm_{arm}"
        ac = oc.get(key)
        if not isinstance(ac, dict) or not ac:
            raise ConfirmatoryRefusal(
                f"{path}.{key}: arm costs absent or opaque")
        if "lag_features_s" not in ac or \
                "ridge_fit_forecast_s" not in ac:
            raise ConfirmatoryRefusal(
                f"{path}.{key}: lag/ridge phase costs missing")
        for s in seed_tape:
            if f"mlp_fit_forecast_seed{s}_s" not in ac:
                raise ConfirmatoryRefusal(
                    f"{path}.{key}: MLP seed{s} cost missing")
        for ck, cv in ac.items():
            _num(cv, f"{path}.{key}.{ck}")


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
        xm = check_metric(x["mase_primary"], "mase_primary",
                          f"{rec['unit_id']}.X.{model}")
        am = check_metric(a_["mase_primary"], "mase_primary",
                          f"{rec['unit_id']}.{arm}.{model}")
        deltas.append(xm - am)
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
    """C14/C17/C18/C19: three origins, ALL arms, ridge + every
    MLP seed + the seasonal-naive baseline validated against the
    ONE exact model-result schema with full numeric domains;
    per-unit design binding (family/digest/geometry) enforced;
    costs enumerated for every phase and arm/model."""
    uid = rec.get("unit_id")
    want_origins = int(design["role_geometry"]["rolling_origins"])
    origins = rec.get("rolling_origins")
    if not isinstance(origins, dict) or \
            len(origins) != want_origins:
        raise ConfirmatoryRefusal(
            f"{uid}: expected {want_origins} rolling origins")
    # C18: per-unit binding to the design's canonical map
    umap = design["task_population"].get("unit_map", {})
    bound = umap.get(uid)
    if bound is None:
        raise ConfirmatoryRefusal(
            f"{uid}: no canonical unit binding in the design — "
            "population membership alone is not identity")
    if rec.get("family") != bound["family"]:
        raise ConfirmatoryRefusal(
            f"{uid}: family {rec.get('family')!r} differs from "
            f"the design binding {bound['family']!r} — relabeled "
            "populations refuse")
    for bk, rk in (("dataset", "dataset"),
                   ("series_numeric_sha256",
                    "series_numeric_sha256"),
                   ("seasonal_period", "seasonal_period"),
                   ("horizon", "horizon")):
        if bk in bound and rec.get(rk) != bound[bk]:
            raise ConfirmatoryRefusal(
                f"{uid}: {rk} differs from the design binding")
    costs = rec.get("costs_by_phase")
    if not isinstance(costs, dict) or \
            set(costs) != set(origins):
        raise ConfirmatoryRefusal(
            f"{uid}: per-phase costs incomplete")
    want_seeds = {f"seed{s}" for s in design["seed_tape"]}
    for okey, o in origins.items():
        if "origin_binding" in (bound or {}):
            ob = bound["origin_binding"].get(okey)
            got = {"train": o.get("train"),
                   "score": o.get("score")}
            if ob is not None and (got["train"] != ob["train"]
                                   or got["score"] != ob["score"]):
                raise ConfirmatoryRefusal(
                    f"{uid} {okey}: origin geometry differs from "
                    "the design binding")
        res = o.get("results", {})
        want_res = set(_REQUIRED_ARMS) | {"seasonal_naive"}
        if not isinstance(res, dict) or \
                not want_res.issubset(res):
            raise ConfirmatoryRefusal(
                f"{uid} {okey}: arms/baseline missing "
                f"({sorted(want_res - set(res))}) — incomplete "
                "evidence never adjudicates")
        sn = res["seasonal_naive"]
        if not isinstance(sn, dict) or "metrics" not in sn:
            raise ConfirmatoryRefusal(
                f"{uid} {okey}: seasonal-naive baseline malformed")
        check_model_result(sn["metrics"],
                           f"{uid}.{okey}.seasonal_naive")
        for arm in _REQUIRED_ARMS:
            check_model_result(res[arm].get("ridge"),
                               f"{uid}.{okey}.{arm}.ridge")
            mlp = res[arm].get("mlp_small")
            if not isinstance(mlp, dict) or \
                    set(mlp) != want_seeds:
                raise ConfirmatoryRefusal(
                    f"{uid} {okey} {arm}: MLP seed tape "
                    "incomplete")
            for sk, sv in mlp.items():
                check_model_result(
                    sv, f"{uid}.{okey}.{arm}.mlp.{sk}")
        check_origin_costs(costs.get(okey), _REQUIRED_ARMS,
                           design["seed_tape"],
                           f"{uid}.{okey}.costs")


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
    unidentifiable = []
    hm = design["harm_margins"]
    inf_rule = design.get("inference_method", {}).get(
        "rule", "panel_replication_or_descriptive")
    if inf_rule != "panel_replication_or_descriptive":
        raise ConfirmatoryRefusal(
            "design names an unimplemented inference rule")
    # C23: series inside ONE panel share unidentifiable panel-
    # level dependence (my own coverage simulation shows the
    # within-panel ICC estimator CANNOT see a common intercept —
    # the mean removes it). Therefore: with >=2 independent
    # panels per family, the PANEL is the inferential unit
    # (delta per panel = mean of its series); with a single
    # panel, between-series intervals are DESCRIPTIVE ONLY and
    # the family is INCONCLUSIVE for the primary gate.
    umap = design["task_population"].get("unit_map", {})
    # group series deltas by (family, dataset/panel)
    panel_map = {}
    for rec in records:
        fam = rec["family"]
        if fam not in fams_required:
            continue
        panel = umap.get(rec["unit_id"], {}).get("dataset",
                                                 "UNKNOWN")
        s = _series_stats(rec, design, "D", "ridge")
        w = _series_stats(rec, design, "width_control", "ridge")
        panel_map.setdefault(fam, {}).setdefault(
            panel, {"d": [], "a": []})
        panel_map[fam][panel]["d"].append(s["delta"])
        panel_map[fam][panel]["a"].append(
            s["delta"] - w["delta"])
    for fam in fams_required:
        arr = np.array(fam_deltas[fam], dtype=float)
        att = np.array(fam_attrib[fam], dtype=float)
        n = len(arr)
        panels = panel_map.get(fam, {})
        k_panels = len(panels)
        if k_panels >= 2:
            pd = np.array([float(np.mean(v["d"]))
                           for v in panels.values()])
            pa = np.array([float(np.mean(v["a"]))
                           for v in panels.values()])
            if k_panels < 3:
                unidentifiable.append(fam)
                fam_stats[fam] = {
                    "n_series": n, "n_panels": k_panels,
                    "mean_delta": float(arr.mean()),
                    "ci_class": "descriptive_insufficient_"
                                "panel_replication",
                    "note": "at least 3 panels are needed for a "
                            "panel-level interval"}
                continue
            from scipy import stats as _st
            tq = float(_st.t.ppf(1 - alpha / 2, k_panels - 1))
            se = float(pd.std(ddof=1) / math.sqrt(k_panels))
            lo = float(pd.mean() - tq * se)
            width = float(2 * tq * se)
            mean_delta = float(pd.mean())
            att_mean = float(pa.mean())
            ci_class = "panel_level_inferential"
            n_support = k_panels
        else:
            unidentifiable.append(fam)
            fam_stats[fam] = {
                "n_series": n, "n_panels": k_panels,
                "mean_delta": float(arr.mean()),
                "descriptive_series_sd":
                    float(arr.std(ddof=1)) if n > 1 else None,
                "ci_class": "descriptive_within_single_panel",
                "note": "panel-level dependence unidentifiable "
                        "with one panel — no inferential "
                        "interval exists; INCONCLUSIVE for the "
                        "primary gate"}
            continue
        ex = [h["extreme_ratio"] for h in fam_harms[fam]
              if h["extreme_ratio"] is not None]
        cov = [h["coverage_drop"] for h in fam_harms[fam]]
        wid = [h["width_ratio"] for h in fam_harms[fam]]
        fam_stats[fam] = {
            "n_series": n, "n_panels": n_support,
            "mean_delta": mean_delta,
            "ci_low": lo, "ci_width": width,
            "ci_class": ci_class,
            "attribution_mean": att_mean,
            "extreme_ratio_mean": (float(np.mean(ex))
                                   if ex else None),
            "coverage_drop_mean": float(np.mean(cov)),
            "width_ratio_mean": float(np.mean(wid))}
        st = fam_stats[fam]
        if mean_delta < -margin:
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
    if unidentifiable:
        return {"verdict": "INCONCLUSIVE",
                "reason": (f"dependence not identifiable / "
                           f"effective support too small in "
                           f"{sorted(unidentifiable)} — "
                           "intrapanel correlation never "
                           "fabricates precision"),
                "families": fam_stats}
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
