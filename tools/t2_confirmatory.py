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
    # C28: the PHYSICAL root is opened and verified WITHOUT
    # resolving symlinks first — os.path.abspath is lexical only,
    # and the O_NOFOLLOW open refuses a root that is itself a
    # link, exactly like leaves and intermediate components.
    raw_root = Path(os.path.abspath(raw_root))
    try:
        rfd = os.open(str(raw_root),
                      os.O_RDONLY | os.O_NOFOLLOW
                      | getattr(os, "O_DIRECTORY", 0))
    except OSError as exc:
        raise ConfirmatoryRefusal(
            f"raw root unopenable without following links "
            f"(errno {exc.errno}: {exc.strerror}) — a symlink "
            "root refuses like any other symlinked component")
    try:
        import stat as _stat
        if not _stat.S_ISDIR(os.fstat(rfd).st_mode):
            raise ConfirmatoryRefusal(
                "raw root is not a physical directory")
    finally:
        os.close(rfd)
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
        # C28: containment is by CONSTRUCTION — rel is relative
        # with no '..' (checked above) and every component is
        # opened via openat/O_NOFOLLOW from the verified root
        # descriptor; no resolve() ever names the target.
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
    "estimand", "extreme_support_rule",
    "practical_margin_mase", "observed_precision_rule",
    "harm_margins", "precision_rule", "sensitivity_rule",
    "inference_method", "inference_scope", "multiplicity_rule",
    "missing_unit_rule", "inconclusive_rule", "resource_contract",
    "verifier_specification", "design_review_record_sha256",
    "design_sha256"}

_ACCEPTED_DESIGN_SCHEMAS = {"agent_multi.t2_screen_design.v6_draft",
                      "agent_multi.t2_screen_design.v6"}
SCREEN_OUTCOMES = ("ADVANCE_TO_DOMAIN_VALIDATION",
                   "DOES_NOT_ADVANCE", "INCONCLUSIVE")
_UNIT_MAP_KEYS = {"family", "dataset", "series_numeric_sha256",
                  "seasonal_period", "horizon", "n_obs",
                  "time_identity_sha256", "origin_windows"}


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
    """C3.2-C3.6 + C8 + C26/C29: the immutable T2-S SCREEN design
    (v4), sealed after the census/acquisition and before any
    score. v3 and earlier are superseded history and refuse
    here."""
    _want_keys = _DESIGN_KEYS
    if isinstance(design, dict) and not str(
            design.get("schema", "")).endswith("_draft"):
        # C39: the SEALED object carries exactly one extra field —
        # the review chronology; drafts never carry it.
        _want_keys = _DESIGN_KEYS | {"sealed_at_date"}
    if not isinstance(design, dict) or set(design) != _want_keys:
        raise ConfirmatoryRefusal(
            "confirmatory design absent or not the exact schema")
    if design.get("schema") not in _ACCEPTED_DESIGN_SCHEMAS:
        raise ConfirmatoryRefusal(
            "design schema is not the v4 T2-S screen contract — "
            "superseded drafts (v3 and earlier) never validate")
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
    # C37: ONE unambiguous estimand name. The implemented and
    # intended arithmetic is MASE(X) - MASE(D): positive means D
    # REDUCES error. The old inverted name, a bare ambiguous
    # name, and opposite-polarity names all refuse typed.
    _delta_name = design["primary_contrast"].get("delta")
    if _delta_name in ("D_minus_X", "delta",
                       "mase_improvement_D_minus_X",
                       "X_minus_D"):
        raise ConfirmatoryRefusal(
            f"design names the primary contrast {_delta_name!r} "
            "— superseded, ambiguous or polarity-inverted "
            "estimand names never validate; the ONE name is "
            "'mase_improvement_X_minus_D' (positive = D reduces "
            "error)")
    if _delta_name != "mase_improvement_X_minus_D" or \
            design["primary_contrast"].get("model") != "ridge":
        raise ConfirmatoryRefusal(
            "design must declare the single frozen primary "
            "contrast mase_improvement_X_minus_D (MASE(X) - "
            "MASE(D), positive = improvement) under the frozen "
            "ridge")
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
    # ---- C29: the T2-S screen estimand, exact and austere ----
    est = design["estimand"]
    if not isinstance(est, dict):
        raise ConfirmatoryRefusal("design.estimand malformed")
    for k in ("population", "superior_unit", "panel_effect",
              "primary", "scope", "outputs", "decision_rule",
              "t2c_successor"):
        if not est.get(k):
            raise ConfirmatoryRefusal(f"design.estimand lacks {k}")
    if tuple(est["outputs"]) != SCREEN_OUTCOMES:
        raise ConfirmatoryRefusal(
            "design.estimand.outputs must be exactly the three "
            "screen outcomes")
    if "PUBLICLY_ELIGIBLE" in json.dumps(est["outputs"]):
        raise ConfirmatoryRefusal(
            "PUBLICLY_ELIGIBLE_CANDIDATE is not a screen outcome "
            "— the screen never grants public eligibility")
    if est["superior_unit"] != "panel":
        raise ConfirmatoryRefusal(
            "the screen's superior unit is the PANEL")
    panels = tp2.get("screen_panels")
    if not isinstance(panels, list) or len(panels) != 6 or \
            len(set(panels)) != 6:
        raise ConfirmatoryRefusal(
            "design must name exactly six distinct screen panels")
    sel = tp2.get("selection")
    if not isinstance(sel, dict) or \
            sel.get("rule") != "family_top_k_geometry_admissible" \
            or type(sel.get("k")) is not int or \
            not sel.get("salt"):
        raise ConfirmatoryRefusal(
            "design must carry the structured selection contract "
            "(rule/k/salt) for the fresh verifier to re-derive")
    hm_ = design["harm_margins"]
    ni = hm_.get("non_inferiority_margin_mase")
    if isinstance(ni, bool) or not isinstance(ni, (int, float)) \
            or not math.isfinite(float(ni)) or ni <= 0:
        raise ConfirmatoryRefusal(
            "design.harm_margins lacks a positive "
            "non_inferiority_margin_mase")
    msp = design["precision_rule"].get("min_series_per_panel")
    if type(msp) is not int or msp <= 0:
        raise ConfirmatoryRefusal(
            "design.precision_rule lacks min_series_per_panel")
    # C31: the per-panel EXTREME support minimum — absolute AND
    # proportional, fixed in the design before any result.
    esr = design["extreme_support_rule"]
    if not isinstance(esr, dict) or set(esr) != {
            "min_evaluable_series_absolute",
            "min_evaluable_fraction"}:
        raise ConfirmatoryRefusal(
            "design.extreme_support_rule is not the exact "
            "absolute+proportional schema")
    ab = esr["min_evaluable_series_absolute"]
    fr = esr["min_evaluable_fraction"]
    if isinstance(ab, bool) or type(ab) is not int or ab < 1:
        raise ConfirmatoryRefusal(
            "extreme_support_rule.min_evaluable_series_absolute "
            "must be a positive int")
    if isinstance(fr, bool) or not isinstance(fr, (int, float)) \
            or not 0 < float(fr) <= 1:
        raise ConfirmatoryRefusal(
            "extreme_support_rule.min_evaluable_fraction outside "
            "(0, 1]")
    # ---- C26: every unit binds its exact causal geometry ----
    import t2_bank as _bank
    n_origins = int(rg["rolling_origins"])
    base_frac = float(rg["origin_base_frac"])
    for uid, b in tp2["unit_map"].items():
        if not isinstance(b, dict) or set(b) != _UNIT_MAP_KEYS:
            raise ConfirmatoryRefusal(
                f"{uid}: unit binding keys are not the exact v4 "
                f"schema (diff: "
                f"{sorted(set(b) ^ _UNIT_MAP_KEYS)})")
        _canon_sha(b["series_numeric_sha256"], f"{uid} digest")
        _canon_sha(b["time_identity_sha256"],
                   f"{uid} time identity")
        if b["horizon"] != int(rg["horizon"]):
            raise ConfirmatoryRefusal(
                f"{uid}: horizon differs from the role geometry")
        want_w = _bank.origin_windows_for(
            b["n_obs"], n_origins, base_frac,
            seasonal_period=b["seasonal_period"],
            horizon=int(rg["horizon"]))
        if b["origin_windows"] != want_w:
            raise ConfirmatoryRefusal(
                f"{uid}: origin_windows do not re-derive from the "
                "series length and the origin contract — geometry "
                "is bound before results, never free-floating")
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


# C38: the PRODUCTIVE review record lives OUTSIDE the candidate
# repository at ONE fixed path under the private reviewer-
# authority root — never CLI/env-selected, never created/chmodded
# by candidate code. A record committed under docs/ grants
# nothing (the repo may carry a non-authorizing template only).
# These checks establish custody facts and exact bytes; they do
# NOT cryptographically identify an author.
AUTHORITY_ROOT = (Path.home() /
                  ".config/agent-multi/reviewer_authority")
T2_REVIEW_RECORD_PATH = (
    AUTHORITY_ROOT / "MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json")
# The commit at which Musashi ACCEPTED scientific design v6, and
# the exact draft identities his record must pin (audit
# MUSASHI_AUDIT_B4_C35_C38_AND_T2_C37_2026_09_07).
T2_V6_ACCEPTED_AT_COMMIT = (
    "2ecd7915fe4f5636fe11368ec4a4087acd94eb59")
T2_V6_DRAFT_FILE_SHA = (
    "a68fccefd00e2e20f1dfb071980d2a51ee296f934bc39c404dbb8d0baf"
    "36aec0")
T2_V6_DRAFT_SELF_SHA = (
    "96cde8b17176358e5721919a3c8f14ebaf5b5b51bba0f072d9de4875c3"
    "3ddd5b")
_REVIEW_KEYS = {"schema", "reviewed_at_date", "reviewer",
                "decision", "candidate_commit",
                "design_draft_file_sha256",
                "design_draft_self_sha256",
                "manifest_sha256", "census_sha256"}


def _open_private_authority_file(path: Path,
                                 missing_msg: str = None):
    """C38 (mirrors the B4 rule): descriptor-first open of ONE
    private-authority object — every component walked O_NOFOLLOW;
    the final two directories owned by the executing uid, exact
    mode 0700; the file regular, same uid, exact 0600; hash and
    parse consume the same descriptor stream. Custody facts only —
    nothing here identifies an author cryptographically."""
    import stat as _stat
    parts = Path(path).parts
    if parts[0] != os.sep:
        raise ConfirmatoryRefusal(
            "the authority path must be absolute")
    fd = os.open("/", os.O_RDONLY
                 | getattr(os, "O_DIRECTORY", 0))
    try:
        for i, comp in enumerate(parts[1:-1], start=1):
            nfd = os.open(comp, os.O_RDONLY | os.O_NOFOLLOW
                          | getattr(os, "O_DIRECTORY", 0),
                          dir_fd=fd)
            os.close(fd)
            fd = nfd
            if len(parts) - 1 - i <= 2:
                st = os.fstat(fd)
                if st.st_uid != os.getuid():
                    raise ConfirmatoryRefusal(
                        f"authority directory {comp!r} has a "
                        "foreign owner")
                if _stat.S_IMODE(st.st_mode) != 0o700:
                    raise ConfirmatoryRefusal(
                        f"authority directory {comp!r} mode "
                        f"{oct(_stat.S_IMODE(st.st_mode))} is "
                        "not the private 0700 — refused, never "
                        "chmodded")
        leaf = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW,
                       dir_fd=fd)
    except FileNotFoundError:
        os.close(fd)
        raise ConfirmatoryRefusal(
            missing_msg or
            "DESIGN_REVIEW_REQUIRED: the EXTERNAL Musashi design "
            "review record does not exist under the private "
            "reviewer-authority root — no seal, no score")
    except OSError as exc:
        os.close(fd)
        raise ConfirmatoryRefusal(
            f"authority path unopenable without following links "
            f"(errno {exc.errno}: {exc.strerror})")
    os.close(fd)
    st = os.fstat(leaf)
    if not _stat.S_ISREG(st.st_mode):
        os.close(leaf)
        raise ConfirmatoryRefusal(
            "the authority record is not a regular file")
    if st.st_uid != os.getuid():
        os.close(leaf)
        raise ConfirmatoryRefusal(
            "the authority record has a foreign owner")
    if _stat.S_IMODE(st.st_mode) != 0o600:
        os.close(leaf)
        raise ConfirmatoryRefusal(
            f"the authority record mode "
            f"{oct(_stat.S_IMODE(st.st_mode))} is not the exact "
            "private 0600")
    return leaf


def verify_design_review_record(design: dict,
                                manifest_sha: str,
                                census_sha: str) -> dict:
    """C9/C38: finite, non-circular EXTERNAL review authority.
    The record is read descriptor-first from the fixed private
    reviewer-authority path (a repository copy grants nothing),
    strict-parsed from the same byte stream, and must pin: the
    reviewer role and decision, a canonical review date, the
    candidate commit at which design v6 was accepted, the physical
    draft-v6 SHA AND its self identity, and the exact manifest and
    census. The sealed design must name this record's exact
    bytes. Custody facts and exact bytes only — no claim of
    cryptographic authorship."""
    fd = _open_private_authority_file(T2_REVIEW_RECORD_PATH)
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
    record_sha = hashlib.sha256(raw).hexdigest()
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ConfirmatoryRefusal(
                "duplicate JSON key in the review record")
        return dict(pairs)
    try:
        rec = json.loads(raw.decode("utf-8"),
                         object_pairs_hook=_no_dupes,
                         parse_constant=lambda c: (
                             _ for _ in ()).throw(
                             ConfirmatoryRefusal(
                                 "non-finite constant in the "
                                 "review record")))
    except json.JSONDecodeError as exc:
        raise ConfirmatoryRefusal(
            f"review record is not well-formed JSON ({exc.msg}) "
            "— arbitrary bytes are never a record")
    if set(rec) != _REVIEW_KEYS:
        raise ConfirmatoryRefusal(
            "review record keys are not the exact v2 schema")
    for k in _REVIEW_KEYS:
        if type(rec[k]) is not str or not rec[k]:
            raise ConfirmatoryRefusal(
                f"review record field {k!r} must be a nonempty "
                "string")
    if rec["schema"] != "agent_multi.musashi_t2_design_review.v2":
        raise ConfirmatoryRefusal(
            "review record carries a foreign schema")
    import datetime as _dt
    try:
        d_ = _dt.date.fromisoformat(rec["reviewed_at_date"])
    except ValueError:
        raise ConfirmatoryRefusal(
            "review reviewed_at_date is not a canonical ISO date")
    if d_.isoformat() != rec["reviewed_at_date"]:
        raise ConfirmatoryRefusal(
            "review reviewed_at_date is not canonical")
    if rec["reviewer"] != "General Musashi":
        raise ConfirmatoryRefusal(
            "review record author field is not the external "
            "reviewer role")
    if rec["decision"] != "SEAL_T2_CONFIRMATORY_DESIGN":
        raise ConfirmatoryRefusal(
            "review record decision does not seal this design")
    if rec["candidate_commit"] != T2_V6_ACCEPTED_AT_COMMIT:
        raise ConfirmatoryRefusal(
            "review record does not pin the candidate commit at "
            "which design v6 was accepted")
    for k in ("design_draft_file_sha256",
              "design_draft_self_sha256",
              "manifest_sha256", "census_sha256"):
        _canon_sha(rec[k], f"review {k}")
    if rec["design_draft_file_sha256"] != T2_V6_DRAFT_FILE_SHA or \
            rec["design_draft_self_sha256"] != \
            T2_V6_DRAFT_SELF_SHA:
        raise ConfirmatoryRefusal(
            "review record pins a different draft v6 (file or "
            "self identity) than the accepted design")
    if rec["manifest_sha256"] != manifest_sha:
        raise ConfirmatoryRefusal(
            "review record binds a different public-data manifest")
    if rec["census_sha256"] != census_sha:
        raise ConfirmatoryRefusal(
            "review record binds a different bank census")
    if design.get("design_review_record_sha256") != record_sha:
        raise ConfirmatoryRefusal(
            "the sealed design does not name THIS external review "
            "record's exact bytes — candidate-selected records "
            "grant nothing")
    if design.get("supersedes_draft_sha256") != \
            rec["design_draft_file_sha256"]:
        raise ConfirmatoryRefusal(
            "the sealed design does not supersede the exact "
            "reviewed draft the record pins — chronology broken")
    rec["_record_sha256"] = record_sha
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


def verify_confirmatory_gates(manifest_path: Path,
                              design_path: Path,
                              census_path: Path = None,
                              raw_root: Path = None,
                              repo_root: Path = REPO) -> dict:
    """C3/C9/C28/C53: the ordered gate sequence, PURE — every
    missing element refuses with its own typed reason; the FRESH
    VERIFIER is an executing precondition of THIS single path (a
    separate script satisfies nothing); the external review root
    cannot be candidate-written; and NOTHING durable is created
    here — no ledger, no directory, no lock. Effects belong to the
    executor's --execute step, strictly after these gates."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise ConfirmatoryRefusal(
            "PUBLIC_DATA_REQUIRED: no public-data manifest exists "
            "yet — acquire and census the bank first")
    manifest = strict_json_load(mp, "public-data manifest")
    validate_public_manifest(manifest, raw_root=raw_root)
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
    # C39 (T2): draft schemas can be validated for review but can
    # NEVER enter scoring — only the sealed v6 identity proceeds
    # toward review-record verification and the ledger.
    if design.get("schema", "").endswith("_draft"):
        raise ConfirmatoryRefusal(
            "SEALED_DESIGN_REQUIRED: a draft schema never scores "
            "— seal the accepted draft v6 through the external "
            "review record first")
    # C28: the fresh-process re-derivation runs INSIDE the single
    # path, immediately before any durable artifact — census and
    # design schemas via the productive parsers, population,
    # digests and EVERY unit_map field rebuilt from physical
    # bytes. Its output is a precondition, never an authority.
    import t2_fresh_verifier as _fv
    census = strict_json_load(cp, "bank census")
    _fv.fresh_verify(manifest, census, design, raw_root=raw_root,
                     manifest_sha=manifest_sha)
    # C9: the external review — verified in full, BEFORE any
    # ledger artifact can exist.
    review = verify_design_review_record(design, manifest_sha,
                                         census_sha)
    # C48 (execution-custody order): confirmatory EXECUTION is
    # structurally closed by a SECOND external record — the
    # Musashi v2 execution record at the private authority root,
    # which pins the sealed design, the review record, manifest,
    # census, the physical executor code identity and the FULL
    # executing checkout.
    exec_rec = verify_execution_record(
        design, _sha_file(dp), review["_record_sha256"],
        manifest_sha, census_sha, repo_root=repo_root)
    return {"gates": "ALL_OPEN",
            "execution_record_sha256": exec_rec["_record_sha256"],
            "review_record_sha256": review["_record_sha256"],
            "design_file_sha256": _sha_file(dp),
            "manifest_sha256": manifest_sha,
            "census_sha256": census_sha,
            "pinned_commit": exec_rec["pinned_commit"],
            "note": "gates only — durable effects (out_root, "
                    "lock, ledger, claims) are created by the "
                    "executor's --execute step alone"}


def run_confirmatory(manifest_path: Path, design_path: Path,
                     ledger_path: Path,
                     census_path: Path = None,
                     raw_root: Path = None,
                     repo_root: Path = REPO) -> dict:
    """C15/C53: the effectful entry — the PURE gate sequence first,
    then (and only then) the durable attempt ledger. `--plan` and
    every read-only consumer must call verify_confirmatory_gates
    directly; this path exists for the executor's --execute step."""
    facts = verify_confirmatory_gates(
        manifest_path, design_path, census_path=census_path,
        raw_root=raw_root, repo_root=repo_root)
    # C15: only now may the durable attempt ledger be created.
    open_attempt_ledger(Path(ledger_path))
    return facts


T2_EXECUTION_RECORD_PATH = (
    AUTHORITY_ROOT / "MUSASHI_T2_V6_EXECUTION_RECORD.json")
# C48: the exact code surface the execution record pins — every
# module the confirmatory executor imports to fit, score or verify.
T2_EXECUTOR_CODE_SURFACE = (
    "tools/t2_confirmatory.py",
    "tools/t2_confirmatory_executor.py",
    "tools/t2_assay_harness.py",
    "tools/t2_bank.py",
    "tools/t2_bank_census.py",
    "tools/t2_fresh_verifier.py",
    "tools/t2_public_data_census.py",
)
# Untracked/ignored sources under these roots (or the repo root)
# can shadow executor imports — the checkout gate refuses them.
_IMPORT_ROOTS = ("agent_plugins", "app", "pipeline_plugins",
                 "tools", "tests")
_EXEC_KEYS = {"schema", "reviewed_at_date", "reviewer",
              "decision", "sealed_design_file_sha256",
              "sealed_design_self_sha256",
              "design_review_record_sha256", "manifest_sha256",
              "census_sha256", "executor_code_identity",
              "pinned_commit", "pinned_tree"}


def executor_code_identity(repo_root: Path = REPO) -> dict:
    """C48.5: the physical identity of the executor code surface,
    derived from checkout bytes — never from a declared value."""
    return {rel: _sha_file(Path(repo_root) / rel)
            for rel in T2_EXECUTOR_CODE_SURFACE}


def _git(repo_root, *args) -> str:
    import subprocess
    r = subprocess.run(["git", "-C", str(repo_root), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise ConfirmatoryRefusal(
            f"git {' '.join(args[:2])} failed "
            f"({r.stderr.strip()[:120]}) — the executor identity "
            "cannot be established")
    return r.stdout


def verify_executor_checkout(pinned_commit: str,
                             pinned_tree: str,
                             repo_root: Path = REPO) -> None:
    """C48.2-C48.4: the execution record names the FULL checkout —
    a real existing commit that IS the executing HEAD, the exact
    tree of that commit, a clean index and tracked worktree, and no
    untracked or ignored sources/configurations able to alter
    imports or entry points."""
    import re as _re
    from pathlib import PurePosixPath
    for name, v in (("pinned_commit", pinned_commit),
                    ("pinned_tree", pinned_tree)):
        if type(v) is not str or not _re.fullmatch(
                r"[0-9a-f]{40}", v):
            raise ConfirmatoryRefusal(
                f"execution record {name} must be 40 lowercase hex "
                "— an arbitrary string never names the executor")
    import subprocess as _sp
    typ = _sp.run(["git", "-C", str(repo_root), "cat-file", "-t",
                   pinned_commit], capture_output=True, text=True)
    if typ.returncode != 0 or typ.stdout.strip() != "commit":
        raise ConfirmatoryRefusal(
            "execution record pinned_commit does not name an "
            "existing commit object")
    head = _git(repo_root, "rev-parse", "HEAD").strip()
    if head != pinned_commit:
        raise ConfirmatoryRefusal(
            f"the executing checkout HEAD ({head[:12]}) is not the "
            f"pinned commit ({pinned_commit[:12]}) — the record "
            "authorizes exactly one executor identity")
    tree = _git(repo_root, "rev-parse",
                pinned_commit + "^{tree}").strip()
    if tree != pinned_tree:
        raise ConfirmatoryRefusal(
            "execution record pinned_tree is not the tree of the "
            "pinned commit")
    status = _git(repo_root, "status", "--porcelain=v1",
                  "--untracked-files=all", "--ignored=matching")
    for line in status.splitlines():
        code, path = line[:2], line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if code not in ("??", "!!"):
            raise ConfirmatoryRefusal(
                f"tracked path {path!r} is modified — the executor "
                "runs only a clean checkout of the pinned commit")
        p = PurePosixPath(path.strip('"'))
        base = p.name
        if base.endswith(".pth") or base == "entry_points.txt" or \
                any(part.endswith((".dist-info", ".egg-info"))
                    for part in p.parts):
            raise ConfirmatoryRefusal(
                f"untracked/ignored import machinery {path!r} can "
                "alter entry points — refused")
        if base.endswith((".py", ".so", ".pyd")) and (
                len(p.parts) == 1 or p.parts[0] in _IMPORT_ROOTS):
            raise ConfirmatoryRefusal(
                f"untracked/ignored source {path!r} can shadow "
                "executor imports — refused")


def verify_execution_record(design: dict,
                            design_file_sha: str,
                            review_record_sha: str,
                            manifest_sha: str,
                            census_sha: str,
                            repo_root: Path = REPO) -> dict:
    """C48: the v2 EXECUTION record — a second, separate external
    Musashi record at the private authority root; the review
    record seals the design, THIS one opens scoring AND pins the
    executor: the sealed design (physical + self), the verified
    review record, manifest, census, the physical code identity of
    the executor surface, and the FULL checkout (existing commit ==
    executing HEAD, exact tree, clean worktree, no shadowing
    sources). Custody facts and exact bytes only."""
    fd = _open_private_authority_file(
        T2_EXECUTION_RECORD_PATH,
        missing_msg=(
            "T2_EXECUTION_RECORD_REQUIRED: confirmatory scoring "
            "stays STRUCTURALLY CLOSED — the external Musashi "
            "execution record does not exist under the private "
            "reviewer-authority root; the sealed design alone "
            "never scores"))
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
    rec_sha = hashlib.sha256(raw).hexdigest()
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise ConfirmatoryRefusal(
                "duplicate JSON key in the execution record")
        return dict(pairs)
    try:
        rec = json.loads(raw.decode("utf-8"),
                         object_pairs_hook=_no_dupes)
    except json.JSONDecodeError as exc:
        raise ConfirmatoryRefusal(
            f"execution record is not well-formed JSON "
            f"({exc.msg})")
    if set(rec) != _EXEC_KEYS:
        raise ConfirmatoryRefusal(
            "execution record keys are not the exact v2 schema")
    for k in _EXEC_KEYS - {"executor_code_identity"}:
        if type(rec[k]) is not str or not rec[k]:
            raise ConfirmatoryRefusal(
                f"execution record field {k!r} must be a "
                "nonempty string")
    if rec["schema"] != \
            "agent_multi.musashi_t2_execution_record.v2":
        raise ConfirmatoryRefusal(
            "execution record carries a foreign schema")
    import datetime as _dt
    try:
        d_ = _dt.date.fromisoformat(rec["reviewed_at_date"])
    except ValueError:
        raise ConfirmatoryRefusal(
            "execution reviewed_at_date is not a canonical ISO "
            "date")
    if d_.isoformat() != rec["reviewed_at_date"]:
        raise ConfirmatoryRefusal(
            "execution reviewed_at_date is not canonical")
    if rec["reviewer"] != "General Musashi":
        raise ConfirmatoryRefusal(
            "execution record author field is not the external "
            "reviewer role")
    if rec["decision"] != "OPEN_T2_CONFIRMATORY_EXECUTION":
        raise ConfirmatoryRefusal(
            "execution record decision does not open scoring")
    body = {k: design[k] for k in sorted(design)
            if k != "design_sha256"}
    self_sha = hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()
    if rec["sealed_design_self_sha256"] != self_sha or \
            rec["sealed_design_self_sha256"] != \
            design.get("design_sha256"):
        raise ConfirmatoryRefusal(
            "execution record does not pin THIS sealed design's "
            "self identity")
    _canon_sha(rec["sealed_design_file_sha256"],
               "execution sealed file digest")
    if rec["sealed_design_file_sha256"] != design_file_sha:
        raise ConfirmatoryRefusal(
            "execution record does not pin THIS sealed design's "
            "physical bytes")
    # C48.5: the record binds the verified review record, the exact
    # manifest and census, and the PHYSICAL executor code identity.
    for k, want, what in (
            ("design_review_record_sha256", review_record_sha,
             "the verified external design review record"),
            ("manifest_sha256", manifest_sha,
             "the public-data manifest"),
            ("census_sha256", census_sha, "the bank census")):
        _canon_sha(rec[k], f"execution {k}")
        if rec[k] != want:
            raise ConfirmatoryRefusal(
                f"execution record does not pin {what}")
    ci = rec["executor_code_identity"]
    if type(ci) is not dict or not ci:
        raise ConfirmatoryRefusal(
            "execution record executor_code_identity must be a "
            "nonempty object")
    for k, v in ci.items():
        if type(k) is not str or type(v) is not str:
            raise ConfirmatoryRefusal(
                "execution record executor_code_identity entries "
                "must be string -> string")
        _canon_sha(v, f"executor_code_identity[{k}]")
    physical_ci = executor_code_identity(repo_root)
    if ci != physical_ci:
        diff = sorted(set(ci) ^ set(physical_ci)) or sorted(
            k for k in ci if ci[k] != physical_ci[k])
        raise ConfirmatoryRefusal(
            "execution record executor_code_identity does not "
            f"match the physical checkout surface (diff: {diff}) "
            "— a declared identity never substitutes for bytes")
    # C48.2-4: the FULL checkout — commit form/existence/HEAD,
    # exact tree, clean worktree, no shadowing sources.
    verify_executor_checkout(rec["pinned_commit"],
                             rec["pinned_tree"], repo_root)
    return {**rec, "_record_sha256": rec_sha}


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


GLOBAL_COST_PHASES = ("denoise_fit_transform_s",
                      "target_construction_s",
                      "seasonal_naive_s")


def check_origin_costs(oc, arms, seed_tape, path: str) -> None:
    """C27: the per-origin cost schema is EXACT — every declared
    global phase, every arm with its lag/ridge/MLP-per-seed
    phases, all values finite nonnegative non-bool. Unknown,
    missing, null or extra phases refuse."""
    if not isinstance(oc, dict):
        raise ConfirmatoryRefusal(f"{path}: costs are not a "
                                  "mapping")

    def _num(v, where):
        if v is None or isinstance(v, bool) or not isinstance(
                v, (int, float)) or not math.isfinite(float(v)) \
                or float(v) < 0:
            raise ConfirmatoryRefusal(
                f"{where}: cost is not a finite nonnegative "
                "number")
    want_top = set(GLOBAL_COST_PHASES) | {
        f"arm_{a}" for a in arms}
    if set(oc) != want_top:
        raise ConfirmatoryRefusal(
            f"{path}: cost phases are not the exact schema "
            f"(diff: {sorted(set(oc) ^ want_top)})")
    for ph in GLOBAL_COST_PHASES:
        _num(oc[ph], f"{path}.{ph}")
    want_arm = {"lag_features_s", "ridge_fit_forecast_s"} | {
        f"mlp_fit_forecast_seed{s}_s" for s in seed_tape}
    for arm in arms:
        key = f"arm_{arm}"
        ac = oc[key]
        if not isinstance(ac, dict) or set(ac) != want_arm:
            raise ConfirmatoryRefusal(
                f"{path}.{key}: arm cost phases are not the "
                "exact schema")
        for ck in want_arm:
            _num(ac[ck], f"{path}.{key}.{ck}")


# ------------- C8/C14: complete decision rule ---------------------

_REQUIRED_ARMS = ("X", "D", "XDR", "width_control")


def extreme_contrast(x_entry, a_entry, path: str):
    """C25: EXPLICIT states, never boolean truth. The evidence
    binds extreme_support to its metric:
    - support > 0: both metrics are MANDATORY and compared,
      including zero; X=0 & D=0 -> ratio 1.0 (no silent
      division); X=0 & D>0 -> HARM_INFINITE (damage, not
      absence);
    - support == 0: NOT_EVALUABLE, never favorable."""
    sup_x = x_entry.get("extreme_support")
    sup_a = a_entry.get("extreme_support")
    sup = sup_x if sup_x is not None else sup_a
    if sup is None:
        raise ConfirmatoryRefusal(
            f"{path}: extreme_support absent — extreme evidence "
            "must carry its support state")
    sup = int(check_metric(sup, "extreme_support", path))
    if sup == 0:
        return {"state": "NOT_EVALUABLE", "ratio": None}
    ex_x = x_entry.get("mase_on_extreme_innovations")
    ex_a = a_entry.get("mase_on_extreme_innovations")
    if ex_x is None or ex_a is None:
        raise ConfirmatoryRefusal(
            f"{path}: extreme_support {sup} > 0 but the extreme "
            "metric is absent — absence never improves a gate")
    ex_x = check_metric(ex_x, "mase_on_extreme_innovations",
                        f"{path}.X")
    ex_a = check_metric(ex_a, "mase_on_extreme_innovations",
                        f"{path}.{'arm'}")
    if ex_x == 0.0 and ex_a == 0.0:
        return {"state": "EVALUATED", "ratio": 1.0}
    if ex_x == 0.0 and ex_a > 0.0:
        return {"state": "HARM_INFINITE", "ratio": None}
    return {"state": "EVALUATED", "ratio": ex_a / ex_x}


def _series_stats(rec, design, arm="D", model="ridge"):
    """C37: per-series paired improvements
    mase_improvement_X_minus_D = MASE(X) - MASE(arm) — POSITIVE
    means the arm reduces error vs X. Averaged over the design's
    origins (nested, never inflating n)."""
    origins = rec["rolling_origins"]
    deltas = []
    harms = {"coverage_drop": [], "width_ratio": []}
    ex_states = []
    for okey, o in origins.items():
        res = o["results"]
        x = res["X"][model]
        a_ = res[arm][model]
        xm = check_metric(x["mase_primary"], "mase_primary",
                          f"{rec['unit_id']}.X.{model}")
        am = check_metric(a_["mase_primary"], "mase_primary",
                          f"{rec['unit_id']}.{arm}.{model}")
        deltas.append(xm - am)
        ex_states.append(extreme_contrast(
            x, a_, f"{rec['unit_id']}.{okey}.{arm}"))
        harms["coverage_drop"].append(
            check_metric(x["interval_coverage_train_q90"],
                         "interval_coverage_train_q90",
                         f"{rec['unit_id']}.X")
            - check_metric(a_["interval_coverage_train_q90"],
                           "interval_coverage_train_q90",
                           f"{rec['unit_id']}.{arm}"))
        wx = check_metric(x["interval_width_train_q90"],
                          "interval_width_train_q90",
                          f"{rec['unit_id']}.X")
        wa = check_metric(a_["interval_width_train_q90"],
                          "interval_width_train_q90",
                          f"{rec['unit_id']}.{arm}")
        harms["width_ratio"].append(wa / max(wx, 1e-12))
    if any(s["state"] == "HARM_INFINITE" for s in ex_states):
        ex_summary = {"state": "HARM_INFINITE", "ratio": None}
    elif all(s["state"] == "NOT_EVALUABLE" for s in ex_states):
        ex_summary = {"state": "NOT_EVALUABLE", "ratio": None}
    else:
        ratios = [s["ratio"] for s in ex_states
                  if s["state"] == "EVALUATED"]
        ex_summary = {"state": "EVALUATED",
                      "ratio": float(np.mean(ratios))}
    return {"delta": float(np.mean(deltas)),
            "extreme": ex_summary,
            "harms": {k: float(np.mean(v))
                      for k, v in harms.items()}}


RECORD_OUTER_KEYS = {
    "schema", "authority", "unit_id", "family", "dataset",
    "series_numeric_sha256", "bytes_sha256",
    "license_note", "missingness", "time_index",
    "time_provenance", "horizon", "seasonal_period",
    "seasonal_period_provenance", "operator", "seed_tape",
    "series_is_the_primary_unit", "origins_and_seeds_are_nested",
    "claim_classes_only", "rolling_origins", "costs_by_phase",
    "peak_rss_bytes", "record_sha256"}


def check_record_outer(rec: dict, design: dict) -> None:
    """C26: the OUTER record is consumed whole — exact v3 schema
    (identity fields physical, under the digest), recomputed
    record_sha256, operator/seed-tape/claims equal to the design;
    no extra or missing fields."""
    uid = rec.get("unit_id")
    if not isinstance(rec, dict) or set(rec) != RECORD_OUTER_KEYS:
        raise ConfirmatoryRefusal(
            f"{uid}: record outer keys are not the exact schema "
            f"(diff: {sorted(set(rec) ^ RECORD_OUTER_KEYS)})")
    if rec["schema"] != "agent_multi.t2_assay_record.v3":
        raise ConfirmatoryRefusal(
            f"{uid}: record schema is not the v3 identity-"
            "bearing contract")
    body = {k: rec[k] for k in sorted(rec)
            if k != "record_sha256"}
    try:
        blob = json.dumps(body, sort_keys=True, allow_nan=False)
    except ValueError:
        raise ConfirmatoryRefusal(
            f"{uid}: record contains non-finite numbers — no "
            "valid identity digest exists (NaN/inf are not "
            "evidence)")
    if hashlib.sha256(blob.encode()).hexdigest() != \
            rec["record_sha256"]:
        raise ConfirmatoryRefusal(
            f"{uid}: record_sha256 does not recompute — identity "
            "and provenance unconsumed evidence never "
            "adjudicates")
    dop = design["operator"]
    rop = rec["operator"]
    if not isinstance(rop, dict) or \
            rop.get("kind") != dop["kind"] or \
            rop.get("params") != dop["params"]:
        raise ConfirmatoryRefusal(
            f"{uid}: operator identity differs from the design")
    if list(rec["seed_tape"]) != list(design["seed_tape"]):
        raise ConfirmatoryRefusal(
            f"{uid}: seed tape differs from the design")
    if set(rec["claim_classes_only"]) != {
            "utility", "calibration", "extreme_preservation",
            "cost"}:
        raise ConfirmatoryRefusal(
            f"{uid}: claim classes differ from the contract")


def check_record_completeness(rec: dict, design: dict) -> None:
    """C14/C17/C18/C19: three origins, ALL arms, ridge + every
    MLP seed + the seasonal-naive baseline validated against the
    ONE exact model-result schema with full numeric domains;
    per-unit design binding (family/digest/geometry) enforced;
    costs enumerated for every phase and arm/model."""
    uid = rec.get("unit_id")
    check_record_outer(rec, design)
    want_origins = int(design["role_geometry"]["rolling_origins"])
    origins = rec.get("rolling_origins")
    if not isinstance(origins, dict) or \
            set(origins) != {f"origin{i}"
                             for i in range(want_origins)}:
        raise ConfirmatoryRefusal(
            f"{uid}: expected exactly the {want_origins} named "
            "rolling origins")
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
    ow = bound.get("origin_windows")
    if not isinstance(ow, dict) or \
            set(ow) != set(origins):
        raise ConfirmatoryRefusal(
            f"{uid}: the design binds no exact origin windows — "
            "causal geometry must be bound before results")
    _ORIGIN_KEYS = {"train", "score",
                    "mase_denominator_train_snaive",
                    "extreme_innovation_threshold_train",
                    "operator_artifact_sha256", "results"}
    for okey, o in origins.items():
        if not isinstance(o, dict) or set(o) != _ORIGIN_KEYS:
            raise ConfirmatoryRefusal(
                f"{uid} {okey}: origin keys are not the exact "
                f"schema (diff: "
                f"{sorted(set(o) ^ _ORIGIN_KEYS)}) — a record "
                "without its causal geometry never adjudicates")
        wb = ow[okey]
        if list(o["train"]) != list(wb["train"]) or \
                list(o["score"]) != list(wb["score"]):
            raise ConfirmatoryRefusal(
                f"{uid} {okey}: train/score windows differ from "
                "the design's bound geometry — a record without "
                "or with shifted windows never adjudicates")
        den = o["mase_denominator_train_snaive"]
        thr = o["extreme_innovation_threshold_train"]
        for nm, v in (("mase_denominator_train_snaive", den),
                      ("extreme_innovation_threshold_train",
                       thr)):
            if isinstance(v, bool) or not isinstance(
                    v, (int, float)) or not math.isfinite(
                        float(v)) or float(v) < 0:
                raise ConfirmatoryRefusal(
                    f"{uid} {okey}: {nm} outside its domain")
        _canon_sha(o["operator_artifact_sha256"],
                   f"{uid} {okey} operator artifact")
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
    """C29: SUPERSEDED. The T2 target is now the T2-S public
    screen (panel-level, six panels, ADVANCE/DOES_NOT_ADVANCE/
    INCONCLUSIVE). PUBLICLY_ELIGIBLE_CANDIDATE no longer exists as
    an outcome anywhere."""
    raise ConfirmatoryRefusal(
        "adjudicate_confirmatory is superseded by "
        "adjudicate_screen (T2-S); the screen decides only "
        "advancement to domain validation, never public "
        "eligibility")


def adjudicate_screen(records: list, design: dict) -> dict:
    """C29 — the T2-S public screen, austere by construction:

    population: the six named public primary panels;
    superior unit: the PANEL; panel effect = paired mean of
    mase_improvement_X_minus_D (= MASE(X) - MASE(D), positive =
    D reduces error) over its selected series; primary estimand =
    the UNWEIGHTED mean of the six panel effects; scope = only
    these panels and their admitted series.

    ADVANCE_TO_DOMAIN_VALIDATION requires SIMULTANEOUSLY:
    t lower bound (df=5) above the practical margin; the exact
    sign test compatible with alpha 0.05 (all six panel effects
    positive); every leave-one-panel-out mean above the margin;
    no panel harmed beyond the non-inferiority margin; and the
    preservation/calibration/cost/support gates complete.
    Anything under-powered or under-precise is INCONCLUSIVE —
    margins are never adjusted after seeing results."""
    if design.get("inference_method", {}).get("rule") != \
            "six_panel_screen_t_sign_lopo":
        raise ConfirmatoryRefusal(
            "design does not predeclare the six-panel screen "
            "rule — the screen adjudicates only its own contract")
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
    ni_margin = float(design["harm_margins"].get(
        "non_inferiority_margin_mase", margin))
    hm = design["harm_margins"]
    umap = tp["unit_map"]
    panels_named = list(tp["screen_panels"])
    if len(panels_named) != 6 or \
            len(set(panels_named)) != 6:
        raise ConfirmatoryRefusal(
            "the screen requires exactly six distinct named "
            "panels")
    min_series = int(design["precision_rule"][
        "min_series_per_panel"])
    per_panel = {p_: {"d": [], "a": [], "ex": [],
                      "cov": [], "wid": []}
                 for p_ in panels_named}
    for rec in records:
        panel = umap[rec["unit_id"]]["dataset"]
        if panel not in per_panel:
            continue          # sensitivity-only units
        s = _series_stats(rec, design, "D", "ridge")
        w = _series_stats(rec, design, "width_control", "ridge")
        pp = per_panel[panel]
        pp["d"].append(s["delta"])
        pp["a"].append(s["delta"] - w["delta"])
        pp["ex"].append(s["extreme"])
        pp["cov"].append(s["harms"]["coverage_drop"])
        pp["wid"].append(s["harms"]["width_ratio"])
    panel_stats = {}
    inconclusive_reasons = []
    harmed = []
    esr = design.get("extreme_support_rule")
    if not isinstance(esr, dict) or \
            "min_evaluable_series_absolute" not in esr or \
            "min_evaluable_fraction" not in esr:
        raise ConfirmatoryRefusal(
            "design lacks the predeclared per-panel "
            "extreme_support_rule — the screen adjudicates only "
            "its own contract")
    for p_, pp in per_panel.items():
        n = len(pp["d"])
        if n < min_series:
            inconclusive_reasons.append(
                f"{p_}: {n} series < declared minimum "
                f"{min_series}")
            continue
        # C31: the panel's EXTREME evidence needs a predeclared
        # minimum of EVALUABLE series — absolute AND proportional;
        # one evaluable series among many NOT_EVALUABLE can never
        # license the panel (HARM_INFINITE counts as evaluable
        # evidence of damage, never as absence).
        evaluable = sum(1 for e in pp["ex"]
                        if e["state"] in ("EVALUATED",
                                          "HARM_INFINITE"))
        need = max(int(esr["min_evaluable_series_absolute"]),
                   math.ceil(float(esr["min_evaluable_fraction"])
                             * n))
        if evaluable < need:
            inconclusive_reasons.append(
                f"{p_}: only {evaluable}/{n} series carry "
                f"evaluable extreme evidence < predeclared "
                f"minimum {need} — never favorable")
        eff = float(np.mean(pp["d"]))
        att = float(np.mean(pp["a"]))
        if any(e["state"] == "HARM_INFINITE" for e in pp["ex"]):
            ex_state, ex_ratio = "HARM_INFINITE", None
        elif all(e["state"] == "NOT_EVALUABLE"
                 for e in pp["ex"]):
            ex_state, ex_ratio = "NOT_EVALUABLE", None
        else:
            rr = [e["ratio"] for e in pp["ex"]
                  if e["state"] == "EVALUATED"]
            ex_state, ex_ratio = "EVALUATED", float(np.mean(rr))
        panel_stats[p_] = {
            "n_series": n, "effect": eff,
            "attribution": att,
            "extreme_state": ex_state,
            "extreme_ratio": ex_ratio,
            "coverage_drop": float(np.mean(pp["cov"])),
            "width_ratio": float(np.mean(pp["wid"]))}
        st = panel_stats[p_]
        if eff < -ni_margin:
            harmed.append(f"{p_}: effect {eff:.4f} beyond the "
                          "non-inferiority margin")
        if ex_state == "HARM_INFINITE":
            harmed.append(f"{p_}: infinite extreme damage "
                          "(X extreme error 0, D > 0)")
        if ex_state == "EVALUATED" and ex_ratio > float(
                hm["extreme_innovation_mase_ratio_max"]):
            harmed.append(f"{p_}: extreme ratio {ex_ratio:.3f}")
        if ex_state == "NOT_EVALUABLE":
            inconclusive_reasons.append(
                f"{p_}: extreme evidence NOT_EVALUABLE (zero "
                "support) — never favorable")
        if st["coverage_drop"] > float(hm["coverage_drop_max"]):
            harmed.append(f"{p_}: coverage drop "
                          f"{st['coverage_drop']:.3f}")
        if st["width_ratio"] > float(hm["width_inflation_max"]):
            harmed.append(f"{p_}: width inflation "
                          f"{st['width_ratio']:.3f}")
        if att <= 0:
            inconclusive_reasons.append(
                f"{p_}: gain not attributable beyond the width "
                "control")
    if len(panel_stats) < 6:
        return {"verdict": "INCONCLUSIVE",
                "reason": "; ".join(inconclusive_reasons[:4]),
                "panels": panel_stats}
    effects = np.array([panel_stats[p_]["effect"]
                        for p_ in panels_named], dtype=float)
    grand = float(effects.mean())
    from scipy import stats as _st
    tq = float(_st.t.ppf(0.975, 5))
    se = float(effects.std(ddof=1) / math.sqrt(6))
    ci_low = grand - tq * se
    # C29: the OBSERVED precision gate — an interval too wide is
    # INCONCLUSIVE even with a favorable mean; the threshold is
    # frozen in the design, never adjusted after seeing it.
    max_hw = float(design["observed_precision_rule"][
        "max_ci_halfwidth"])
    if tq * se > max_hw:
        inconclusive_reasons.append(
            f"observed precision insufficient: t CI half-width "
            f"{tq * se:.4f} > frozen maximum {max_hw}")
    signs_positive = int((effects > 0).sum())
    sign_ok = signs_positive == 6      # exact p = 2/64 = 0.03125
    lopo = [float(np.delete(effects, i).mean())
            for i in range(6)]
    lopo_ok = all(v > margin for v in lopo)
    out = {"panels": panel_stats,
           "primary_estimand_unweighted_mean_of_panel_effects":
               grand,
           "t_ci_low_df5": ci_low,
           "signs_positive": signs_positive,
           "sign_test_exact_p_two_sided":
               round(2 * (0.5 ** 6) * sum(
                   math.comb(6, k) for k in
                   range(signs_positive, 7)), 5)
               if signs_positive >= 3 else None,
           "leave_one_panel_out_means": lopo,
           "panel_effect_definition":
               "mase_improvement_X_minus_D = MASE(X) - MASE(D); "
               "positive = D reduces error",
           "scope": "ONLY these six public panels and their "
                    "admitted series — no family-level or "
                    "public-eligibility claim"}
    if harmed:
        out.update({"verdict": "DOES_NOT_ADVANCE",
                    "reason": "; ".join(sorted(set(harmed))[:4])})
        return out
    if inconclusive_reasons:
        out.update({"verdict": "INCONCLUSIVE",
                    "reason": "; ".join(
                        inconclusive_reasons[:4])})
        return out
    if ci_low > margin and sign_ok and lopo_ok:
        out.update({"verdict": "ADVANCE_TO_DOMAIN_VALIDATION",
                    "reason": ("t lower bound, exact sign test "
                              "(6/6, p=0.03125) and all six "
                              "leave-one-panel-out means clear "
                              "the frozen margin; no harm gate "
                              "fired")})
        return out
    why = []
    if ci_low <= margin:
        why.append(f"t lower bound {ci_low:.4f} <= margin")
    if not sign_ok:
        why.append(f"only {signs_positive}/6 positive panels "
                   "(exact sign test incompatible with alpha "
                   "0.05)")
    if not lopo_ok:
        why.append("a leave-one-panel-out mean falls below the "
                   "margin (single-panel dominance)")
    out.update({"verdict": "INCONCLUSIVE"
                if grand > 0 else "DOES_NOT_ADVANCE",
                "reason": "; ".join(why)})
    return out
