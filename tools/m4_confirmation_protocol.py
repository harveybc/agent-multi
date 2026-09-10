"""M4 C32-C34 + C36: the CONFIRMATION protocol authority.

C32  bind_calibration_evidence() consumes the accepted
     calibration evidence by exact bytes: the four order-pinned
     identities (reviewed tip, sealed design v5 self, numeric
     amendment self, governing adjudication self) verify by
     recomputation, and the order facts re-derive from the
     adjudication STRUCTURES (21/28 eligible slots, exactly two
     incomplete generators, zero calibration-incomplete cells,
     M2 gain -0.41982887). Any altered attempt, adjudication,
     rule or identity refuses BEFORE a confirmation ledger can
     exist.

C33  build/verify_confirmation_successor(): an append-only
     confirmation successor labeled CALIBRATION_DERIVED_AND_
     REVIEWED (never predeclared), classified SCIENTIFIC_
     ANALYSIS_FREEZE (explicitly NOT ``scientific_change:
     NONE``): eligibility iff >=12/16 CALIBRATION generators
     LEARNABLE_UNDER_FROZEN_BUDGET and zero NUMERICALLY_INVALID;
     the exact 21 eligible and 7 typed-ineligible slots copied
     by identity from the accepted adjudication; 48 CONFIRMATION
     generators per eligible slot; the existing 20% attrition
     allowance with the existing minimum-complete formula
     (=> 39 of 48); M2 = DOES_NOT_ADVANCE_FROM_CALIBRATION.

C34  sixteen_contrasts(): the executable confirmatory analysis.
     14 family/noise intervention contrasts (generator-level
     paired effect, averaged EQUALLY across frozen-eligible
     widths; exactly one eligible width is used and named; zero
     eligible widths => NOT_EVALUABLE with non-rejecting p=1),
     checkpoint_effect::primary_pair as the 15th, and
     incremental_prediction::M2_vs_M1 as the 16th non-rejecting
     p=1 placeholder (M2 failed CALIBRATION). Holm over ALL 16
     slots including placeholders — the successor DECLARES this
     supersedes the design-v5 Bonferroni line (both control
     FWER; ordered by Musashi's C34.7; frozen before any
     CONFIRMATION outcome). Width-specific effects and attrition
     are secondary heterogeneity, never extra primaries. The
     generator is the unit; seeds and widths are nested/paired.

C36  strict consuming APIs for the two still-uninstalled
     external records (Musashi design review + owner execution)
     and their non-authorizing templates. Templates grant
     nothing; this code never creates or installs a real record.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

import m4_residual_capacity as m4  # noqa: E402
import m4_v5_runner as rn  # noqa: E402


class ConfirmationProtocolRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


# ---- C32: the order-pinned identities (order @889320ee §C32,
# copied into this branch at the PRE commit) ----
REVIEWED_TIP = "5e7a8fd430c8231a049baf03f00e720ba24ec994"
DESIGN_SHA = ("d7280a92047d98898418fb7cd750b22c506a621e"
              "b381d9847e0fe926b7df69b9")
AMENDMENT_SHA = ("43e0804e1e6e583b10ddbe46b7d4cd752838b047"
                 "3ccbc6496f0e458c49aedd4b")
ADJUDICATION_SHA = ("b35b6fd969aa162047bdfb55b8f9fcce01aa7686"
                    "4c388d29a1c36642ab051ade")
ORDER_FACTS = {
    "eligible_slots": 21,
    "total_slots": 28,
    "incomplete_generators": 2,
    "calibration_incomplete_cells": 0,
    "m2_gain": -0.41982887,
}

DESIGN_PATH = ("docs/research/model_capacity/"
               "M4_SEALED_DESIGN_V5_2026_09_09.json")
AMENDMENT_PATH = ("docs/research/model_capacity/"
                  "M4_V5_NUMERIC_VALIDITY_AMENDMENT_1_"
                  "2026_09_09.json")
ADJUDICATION_PATH = ("docs/audits/evidence/"
                     "M4_V5_CALIBRATION_ADJUDICATION_ATTEMPT3_"
                     "GOVERNING_2026_09_09.json")
SUCCESSOR_PATH = ("docs/research/model_capacity/"
                  "M4_CONFIRMATION_SUCCESSOR_2026_09_10.json")

# C33 frozen policy numbers
ELIGIBILITY_MIN_LEARNABLE = 12
ELIGIBILITY_OF = 16
CONFIRMATION_PER_SLOT = 48
ATTRITION_ALLOWANCE = 0.20
CONFIRMATION_SEEDS = 3


def _selfsha(doc, key):
    body = {k: doc[k] for k in sorted(doc) if k != key}
    return hashlib.sha256(json.dumps(
        body, sort_keys=True).encode()).hexdigest()


def _read_pinned(repo_root, rel, self_key, pinned):
    p = Path(repo_root) / rel
    fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, 64 * 1024 * 1024)
    finally:
        os.close(fd)
    try:
        doc = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfirmationProtocolRefusal(
            f"{rel}: not strict JSON ({e.__class__.__name__}) — "
            "malformed evidence never binds")
    if doc.get(self_key) != pinned:
        raise ConfirmationProtocolRefusal(
            f"{rel}: declared {self_key} is not the order-pinned "
            "identity — an altered document never binds")
    if _selfsha(doc, self_key) != pinned:
        raise ConfirmationProtocolRefusal(
            f"{rel}: {self_key} does not re-derive from the "
            "bytes — a repaired declaration never converts "
            "altered content into the accepted evidence")
    return doc


def bind_calibration_evidence(repo_root=REPO) -> dict:
    """C32: consume the accepted calibration evidence by bytes;
    refuse any altered attempt, adjudication, rule or identity
    BEFORE a confirmation ledger can exist."""
    repo_root = Path(repo_root)
    design = _read_pinned(repo_root, DESIGN_PATH,
                          "design_sha256", DESIGN_SHA)
    amend = _read_pinned(repo_root, AMENDMENT_PATH,
                         "amendment_sha256", AMENDMENT_SHA)
    if amend.get("amends_design_sha256") != DESIGN_SHA:
        raise ConfirmationProtocolRefusal(
            "numeric amendment does not amend the pinned design")
    adj = _read_pinned(repo_root, ADJUDICATION_PATH,
                       "record_sha256", ADJUDICATION_SHA)
    if adj.get("design_sha256") != DESIGN_SHA:
        raise ConfirmationProtocolRefusal(
            "governing adjudication does not bind the pinned "
            "design")
    if adj.get("authority") != \
            "CANDIDATE_FOR_MUSASHI_REVIEW_NO_CONFIRMATION_" \
            "AUTHORITY":
        raise ConfirmationProtocolRefusal(
            "adjudication authority field is not the accepted "
            "candidate declaration — the record itself claims "
            "no confirmation authority; what makes it governing "
            "is the ORDER pinning its exact self-identity, and "
            "a record declaring anything else is substituted")
    # ---- re-derive the order facts from STRUCTURES ----
    slots = adj["confirmation_slots"]
    elig = [s for s in slots if s["typed_status"]
            == "ELIGIBLE_UNDER_PROPOSED_RULE"]
    inelig = [s for s in slots if s["typed_status"]
              != "ELIGIBLE_UNDER_PROPOSED_RULE"]
    if len(slots) != ORDER_FACTS["total_slots"] or \
            len(elig) != ORDER_FACTS["eligible_slots"]:
        raise ConfirmationProtocolRefusal(
            f"slot census {len(elig)}/{len(slots)} is not the "
            "accepted 21/28 — a changed population never enters "
            "confirmation")
    inc_gens = sorted({u.rsplit("::", 1)[0] for u in
                       adj["incomplete_units_in_denominator"]})
    if len(inc_gens) != ORDER_FACTS["incomplete_generators"]:
        raise ConfirmationProtocolRefusal(
            "incomplete-generator census differs from the "
            "accepted two")
    disp = adj["dispersion"]
    bad = [c for c, d in disp.items()
           if d.get("status") == "CALIBRATION_INCOMPLETE"]
    if len(bad) != ORDER_FACTS["calibration_incomplete_cells"]:
        raise ConfirmationProtocolRefusal(
            "calibration-incomplete census differs from the "
            "accepted zero")
    gain = adj["ladder"]["m2_minus_m1_paired_gain"]
    if gain != ORDER_FACTS["m2_gain"]:
        raise ConfirmationProtocolRefusal(
            f"M2 gain {gain!r} is not the accepted "
            f"{ORDER_FACTS['m2_gain']!r} — a changed ladder "
            "outcome never rehabilitates M2")
    head = subprocess.run(
        ["git", "rev-parse", REVIEWED_TIP + "^{commit}"],
        cwd=repo_root, capture_output=True, text=True)
    if head.returncode != 0:
        raise ConfirmationProtocolRefusal(
            "the reviewed tip does not exist in this repository")
    return {"design": design, "amendment": amend,
            "adjudication": adj,
            "eligible_slots": elig,
            "ineligible_slots": inelig,
            "incomplete_generators": inc_gens}


# ---- C33: the confirmation successor ----

def _successor_body(auth) -> dict:
    adj = auth["adjudication"]
    design = auth["design"]
    contrasts = design["confirmatory_contrast_family"][
        "contrasts"]
    if len(contrasts) != 16:
        raise ConfirmationProtocolRefusal(
            "sealed contrast family is not 16 slots")
    planned = CONFIRMATION_PER_SLOT
    min_complete = max(3, int(np.ceil(
        planned * (1 - ATTRITION_ALLOWANCE))))
    elig_slots = []
    for s in auth["eligible_slots"]:
        if s["reserved_generators"] != CONFIRMATION_PER_SLOT:
            raise ConfirmationProtocolRefusal(
                f"{s['cell']}: reserved generators "
                f"{s['reserved_generators']} != 48")
        elig_slots.append({
            "cell": s["cell"],
            "confirmation_generators": CONFIRMATION_PER_SLOT,
            "calibration_dispersion": s["dispersion"]})
    inelig_slots = [{"cell": s["cell"],
                     "typed_status": s["typed_status"]}
                    for s in auth["ineligible_slots"]]
    return {
        "schema": "m4_confirmation_successor.v1",
        "created_date": "2026-09-10",
        "supersedes_design_sha256": DESIGN_SHA,
        "binds_numeric_amendment_sha256": AMENDMENT_SHA,
        "binds_governing_adjudication_sha256": ADJUDICATION_SHA,
        "binds_reviewed_tip": REVIEWED_TIP,
        "selection_rule_label":
            "CALIBRATION_DERIVED_AND_REVIEWED",
        "selection_rule_provenance":
            "derived from the accepted CALIBRATION adjudication "
            "after external audit — NEVER predeclared; this "
            "successor records a calibration decision made "
            "before any CONFIRMATION data exists",
        "classification": "SCIENTIFIC_ANALYSIS_FREEZE",
        "classification_note":
            "this successor freezes the confirmatory analysis "
            "derived from calibration; it is NOT "
            "'scientific_change: NONE' — the eligibility rule, "
            "slot population, M2 exclusion and Holm multiplicity "
            "are scientific analysis decisions frozen here",
        "eligibility_rule": {
            "min_learnable_under_frozen_budget":
                ELIGIBILITY_MIN_LEARNABLE,
            "of_calibration_generators": ELIGIBILITY_OF,
            "max_numerically_invalid": 0,
            "statement":
                "a slot is eligible iff at least 12 of its 16 "
                "CALIBRATION generators are "
                "LEARNABLE_UNDER_FROZEN_BUDGET and none is "
                "NUMERICALLY_INVALID"},
        "eligible_slots": elig_slots,
        "ineligible_slots": inelig_slots,
        "confirmation_generators_per_eligible_slot":
            CONFIRMATION_PER_SLOT,
        "nested_seeds_per_generator": CONFIRMATION_SEEDS,
        "attrition": {
            "allowance": ATTRITION_ALLOWANCE,
            "planned_generators_per_slot": planned,
            "min_complete_required": min_complete,
            "rule": "max(3, ceil(planned*(1-allowance))) — the "
                    "existing calibration formula applied to "
                    "the 48-generator confirmation population; "
                    "below the floor the slot is "
                    "CONFIRMATION_INCOMPLETE, never favorable"},
        "m2_status": {
            "status": "DOES_NOT_ADVANCE_FROM_CALIBRATION",
            "calibration_gain": ORDER_FACTS["m2_gain"],
            "consequence":
                "incremental_prediction::M2_vs_M1 is carried as "
                "a non-rejecting p=1 placeholder; M2 is never "
                "fitted or scored on CONFIRMATION"},
        "contrast_family_16": contrasts,
        "analysis_freeze": {
            "unit": "the task GENERATOR; seeds and widths are "
                    "nested or paired repetitions, never "
                    "independent observations",
            "per_contrast_effect":
                "generator-level paired restricted-endpoint "
                "difference (calibration_stop minus matched "
                "initialization), mean over the exactly-3 "
                "nested seeds, averaged EQUALLY across widths "
                "frozen eligible by this successor",
            "single_width_rule":
                "if exactly one width is eligible for a "
                "contrast, it is used and that fact is named "
                "in the result",
            "no_width_rule":
                "if no width is eligible, the contrast is "
                "NOT_EVALUABLE with a non-rejecting p=1",
            "per_contrast_test":
                "two-sided one-sample t on the generator-level "
                "effects, df = n_complete_generators - 1; "
                "fewer than 2 complete generators => "
                "NOT_EVALUABLE p=1",
            "multiplicity":
                "Holm step-down over ALL 16 slots including "
                "placeholders, alpha 0.05",
            "multiplicity_supersession":
                "the sealed design v5 line 'Bonferroni over the "
                "frozen confirmatory contrast family' is "
                "SUPERSEDED by Holm per order @889320ee C34.7; "
                "both control FWER at alpha; Holm is uniformly "
                "at least as powerful; frozen here BEFORE any "
                "CONFIRMATION outcome and named for external "
                "review",
            "width_heterogeneity":
                "width-specific effects and attrition are "
                "published as SECONDARY heterogeneity results, "
                "never as additional primary hypotheses"},
        "grants_nothing":
            "this successor authorizes NO execution; "
            "CONFIRMATION arrays, scores and ledgers require "
            "the two external records (Musashi design review + "
            "owner execution) consumed by "
            "m4_confirmation_runner",
    }


def build_confirmation_successor(repo_root=REPO) -> Path:
    repo_root = Path(repo_root)
    out = repo_root / SUCCESSOR_PATH
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", SUCCESSOR_PATH],
        cwd=repo_root, capture_output=True)
    if tracked.returncode == 0:
        raise ConfirmationProtocolRefusal(
            "the confirmation successor is already published — "
            "append-only: a published successor is never "
            "regenerated in place (a9/a15 lesson)")
    auth = bind_calibration_evidence(repo_root)
    body = _successor_body(auth)
    body["successor_sha256"] = _selfsha(body, "successor_sha256")
    if out.exists():
        out.unlink()
    fd = os.open(str(out), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(body, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return out


def verify_confirmation_successor(repo_root=REPO) -> dict:
    """Consume the successor by bytes; refuse any scientific
    mutation against the LIVE re-derivation from the bound
    calibration evidence."""
    repo_root = Path(repo_root)
    p = repo_root / SUCCESSOR_PATH
    if not p.is_file():
        raise ConfirmationProtocolRefusal(
            "no confirmation successor exists")
    fd = os.open(str(p), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        raw = os.read(fd, 16 * 1024 * 1024)
    finally:
        os.close(fd)
    doc = json.loads(raw.decode())
    if _selfsha(doc, "successor_sha256") != \
            doc.get("successor_sha256"):
        raise ConfirmationProtocolRefusal(
            "successor self-identity does not re-derive")
    auth = bind_calibration_evidence(repo_root)
    expected = _successor_body(auth)
    got = {k: v for k, v in doc.items()
           if k != "successor_sha256"}
    if got != expected:
        diff = sorted(set(got) ^ set(expected)) or sorted(
            k for k in expected if got.get(k) != expected[k])
        raise ConfirmationProtocolRefusal(
            f"successor content diverges from the live "
            f"re-derivation (fields: {diff}) — a mutated "
            "successor never governs")
    if doc["selection_rule_label"] != \
            "CALIBRATION_DERIVED_AND_REVIEWED":
        raise ConfirmationProtocolRefusal(
            "selection rule label is not "
            "CALIBRATION_DERIVED_AND_REVIEWED")
    return doc


# ---- C34: the executable 16-contrast family ----

def holm(pvals: dict, alpha: float = 0.05) -> dict:
    """Frozen Holm step-down over ALL slots (placeholders
    included). Returns per-slot adjusted p and rejection."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    rejecting = True
    for i, (k, p) in enumerate(items):
        adj = (m - i) * p
        running = max(running, min(1.0, adj))
        if rejecting and running > alpha:
            rejecting = False
        out[k] = {"p_raw": p, "p_holm": running,
                  "reject_at_alpha": bool(
                      rejecting and running <= alpha)}
    return {k: out[k] for k in pvals}


def _contrast_widths(successor, cell_prefix):
    """Widths frozen eligible for a fam::nz contrast."""
    return sorted(
        int(s["cell"].rsplit("::w", 1)[1])
        for s in successor["eligible_slots"]
        if s["cell"].rsplit("::w", 1)[0] == cell_prefix)


def sixteen_contrasts(successor, per_generator_effects,
                      checkpoint_pair_effects,
                      alpha=0.05) -> dict:
    """The confirmatory analysis, executable and pure.

    per_generator_effects: {f"{fam}::{nz}": {width:
        {generator_id: mean-paired-effect-over-3-seeds}}}
    checkpoint_pair_effects: {generator_id: paired effect} for
        checkpoint_effect::primary_pair.
    """
    contrasts = successor["contrast_family_16"]
    results = {}
    pvals = {}
    for key in contrasts:
        if key.startswith("intervention_effect::"):
            cell_prefix = key.split("intervention_effect::")[1]
            widths = _contrast_widths(successor, cell_prefix)
            fam_effects = per_generator_effects.get(
                cell_prefix, {})
            if not widths:
                results[key] = {
                    "status": "NOT_EVALUABLE",
                    "reason": "no width is frozen eligible",
                    "eligible_widths": []}
                pvals[key] = 1.0
                continue
            per_gen = {}
            for g in sorted(set().union(*(
                    fam_effects.get(w, {}) for w in widths))):
                vals = [fam_effects[w][g] for w in widths
                        if g in fam_effects.get(w, {})]
                if len(vals) == len(widths):
                    per_gen[g] = float(np.mean(vals))
            eff = np.array(list(per_gen.values()))
            if eff.size < 2:
                results[key] = {
                    "status": "NOT_EVALUABLE",
                    "reason": "fewer than 2 complete "
                              "generators",
                    "eligible_widths": widths,
                    "n_generators": int(eff.size)}
                pvals[key] = 1.0
                continue
            t, p = stats.ttest_1samp(eff, 0.0)
            res = {
                "status": "EVALUATED",
                "eligible_widths": widths,
                "n_generators": int(eff.size),
                "effect_mean": float(eff.mean()),
                "effect_sd": float(eff.std(ddof=1)),
                "t": float(t), "df": int(eff.size - 1),
                "p_raw": float(p)}
            if len(widths) == 1:
                res["single_width_used"] = widths[0]
                res["single_width_named"] = (
                    f"exactly one width (w{widths[0]}) is "
                    "frozen eligible; it is used alone")
            results[key] = res
            pvals[key] = float(p)
        elif key == "checkpoint_effect::primary_pair":
            eff = np.array(
                [checkpoint_pair_effects[g] for g in
                 sorted(checkpoint_pair_effects)])
            if eff.size < 2:
                results[key] = {"status": "NOT_EVALUABLE",
                                "n_generators": int(eff.size)}
                pvals[key] = 1.0
            else:
                t, p = stats.ttest_1samp(eff, 0.0)
                results[key] = {
                    "status": "EVALUATED",
                    "n_generators": int(eff.size),
                    "effect_mean": float(eff.mean()),
                    "t": float(t), "df": int(eff.size - 1),
                    "p_raw": float(p)}
                pvals[key] = float(p)
        elif key.startswith("incremental_prediction::"):
            results[key] = {
                "status":
                    "NON_REJECTING_PLACEHOLDER_M2_FAILED_"
                    "CALIBRATION",
                "calibration_gain":
                    successor["m2_status"]["calibration_gain"],
                "note": "M2 is never fitted or scored on "
                        "CONFIRMATION"}
            pvals[key] = 1.0
        else:
            raise ConfirmationProtocolRefusal(
                f"unknown contrast slot {key!r} — the frozen "
                "family is closed")
    if len(pvals) != 16:
        raise ConfirmationProtocolRefusal(
            f"{len(pvals)} slots reached Holm — the frozen "
            "procedure runs over ALL 16 including placeholders")
    hh = holm(pvals, alpha)
    for k in results:
        results[k]["p_holm"] = hh[k]["p_holm"]
        results[k]["reject_at_alpha"] = hh[k]["reject_at_alpha"]
    return {"alpha": alpha, "n_slots": 16,
            "multiplicity": "holm_over_all_16_slots",
            "contrasts": results}


def width_heterogeneity(successor,
                        per_generator_effects) -> dict:
    """Secondary width-specific effects — labeled, never
    primary."""
    out = {}
    for s in successor["eligible_slots"]:
        cell = s["cell"]
        prefix, w = cell.rsplit("::w", 1)
        eff = per_generator_effects.get(prefix, {}).get(
            int(w), {})
        vals = np.array([eff[g] for g in sorted(eff)])
        out[cell] = {
            "classification": "SECONDARY_HETEROGENEITY",
            "n_generators": int(vals.size),
            "effect_mean":
                float(vals.mean()) if vals.size else None,
            "attrition_observed":
                1.0 - vals.size / CONFIRMATION_PER_SLOT}
    return out


# ---- C36: external authority boundary ----
STATE_ROOT = Path.home() / ".local/share/agent-multi"
MUSASHI_REVIEW_RECORD_PATH = (
    STATE_ROOT / "m4_confirmation_authority"
    / "MUSASHI_M4_CONFIRMATION_DESIGN_REVIEW_RECORD.json")
OWNER_EXECUTION_RECORD_PATH = (
    STATE_ROOT / "m4_confirmation_authority"
    / "OWNER_M4_CONFIRMATION_EXECUTION_RECORD.json")

_REVIEW_KEYS = {
    "schema", "author", "role", "date",
    "reviewed_successor_sha256", "reviewed_tip",
    "reviewed_analysis_statement", "decision",
    "record_sha256"}
_EXEC_KEYS = {
    "schema", "author", "role", "date",
    "authorized_successor_sha256",
    "authorized_review_record_sha256",
    "authorized_population_statement",
    "cpu_limits_statement", "decision", "record_sha256"}


def _read_external(path, expected_keys, schema, author_role,
                   decision_expected, what):
    if not Path(path).is_file():
        raise ConfirmationProtocolRefusal(
            f"{what} record is ABSENT — planning may report "
            "counts, execution refuses before any CONFIRMATION "
            "array or ledger")
    fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        if st.st_uid != os.getuid():
            raise ConfirmationProtocolRefusal(
                f"{what} record is not owned by the operator")
        raw = os.read(fd, 4 * 1024 * 1024)
    finally:
        os.close(fd)
    try:
        doc = json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ConfirmationProtocolRefusal(
            f"{what} record is not strict JSON")
    if set(doc) != expected_keys:
        raise ConfirmationProtocolRefusal(
            f"{what} record keys are not the exact schema "
            f"(diff: {sorted(set(doc) ^ expected_keys)})")
    if doc["schema"] != schema:
        raise ConfirmationProtocolRefusal(
            f"{what} record schema is {doc['schema']!r}")
    if doc["role"] != author_role:
        raise ConfirmationProtocolRefusal(
            f"{what} record role is {doc['role']!r} — only "
            f"{author_role} may author it")
    if "TEMPLATE" in json.dumps(doc).upper() or any(
            isinstance(v, str) and v.strip().startswith("<")
            for v in doc.values()):
        raise ConfirmationProtocolRefusal(
            f"{what} record carries template placeholders — a "
            "template grants nothing")
    if _selfsha(doc, "record_sha256") != doc["record_sha256"]:
        raise ConfirmationProtocolRefusal(
            f"{what} record self-identity does not re-derive")
    if doc["decision"] != decision_expected:
        raise ConfirmationProtocolRefusal(
            f"{what} record decision is {doc['decision']!r}, "
            f"not {decision_expected!r}")
    return doc


def read_musashi_review_record(successor_sha) -> dict:
    doc = _read_external(
        MUSASHI_REVIEW_RECORD_PATH, _REVIEW_KEYS,
        "musashi_m4_confirmation_design_review.v1",
        "EXTERNAL_AUDITOR",
        "M4_CONFIRMATION_DESIGN_APPROVED_FOR_EXECUTION",
        "Musashi design-review")
    if doc["reviewed_successor_sha256"] != successor_sha:
        raise ConfirmationProtocolRefusal(
            "Musashi review record pins a DIFFERENT successor — "
            "a review of other bytes authorizes nothing")
    if doc["reviewed_tip"] != REVIEWED_TIP:
        raise ConfirmationProtocolRefusal(
            "Musashi review record pins a different reviewed "
            "tip")
    return doc


def read_owner_execution_record(successor_sha,
                                review_sha) -> dict:
    doc = _read_external(
        OWNER_EXECUTION_RECORD_PATH, _EXEC_KEYS,
        "owner_m4_confirmation_execution.v1",
        "OWNER",
        "M4_CONFIRMATION_EXECUTION_AUTHORIZED_CPU_ONLY",
        "owner execution")
    if doc["authorized_successor_sha256"] != successor_sha:
        raise ConfirmationProtocolRefusal(
            "owner execution record authorizes a DIFFERENT "
            "successor")
    if doc["authorized_review_record_sha256"] != review_sha:
        raise ConfirmationProtocolRefusal(
            "owner execution record does not chain to the "
            "Musashi review record — the two-record gate is "
            "a chain, not a pair of islands")
    return doc


def require_both_records(successor_doc) -> dict:
    """The ONE gate: both external records, chained, or typed
    refusal before any CONFIRMATION array/ledger."""
    ssha = successor_doc["successor_sha256"]
    rev = read_musashi_review_record(ssha)
    own = read_owner_execution_record(ssha,
                                      rev["record_sha256"])
    return {"review": rev, "execution": own}
