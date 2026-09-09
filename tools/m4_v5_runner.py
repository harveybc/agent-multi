"""M4-C32: the v5 runner — one config-driven CPU entry for the
corrected DEVELOPMENT screen, the four development mechanics
units and the sealed 16-generator CALIBRATION population, with
reserved (never constructed) CONFIRMATION slots.

Custody: 0700 root, every evidence object 0600, intervention
logs created once with O_EXCL and held open for append+fsync, a
per-batch durable state so partial work resumes from the
verified predecessor (absent/mismatched state is UNCERTAIN and
refuses), one canonical terminal report with an exact recursive
inventory (RUN_REPORT lookalikes refuse), second invocation is
read-only idempotent verification, and --execute returns zero
ONLY after the fresh verifier reproduces population, inventory
and accounting exactly.

CONFIRMATION is structurally closed: the bank refuses to build
its arrays and the runner enumerates its slots as RESERVED ids
only. CPU only. Zero scientific conclusion in this module.
"""
import hashlib
import json
import os
import resource
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402
import m4_v5_protocol as pv  # noqa: E402


class RunnerV5Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


CONF_WIDTHS = (16, 64)
CAL_SEEDS = 3


def _excl_json(path: Path, doc: dict):
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(doc, indent=1,
                                sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)


def _self(doc: dict) -> dict:
    doc = dict(doc)
    doc.pop("record_sha256", None)
    doc["record_sha256"] = m4._self_sha(doc, "record_sha256")
    return doc


def _safe(s):
    return s.replace("::", "__")


# ---------------------- population (C29) ----------------------

def _cells(design):
    out = []
    for fam in design["candidate_population"][
            "structured_boolean_families"]:
        out.append((fam, "clean"))
    for fam in design["candidate_population"]["temporal_families"]:
        for nz in design["candidate_population"][
                "noise_regimes_temporal"]:
            out.append((fam, nz))
    return out


def _conf_cells(design):
    out = []
    for fam in design["candidate_population"][
            "structured_boolean_families"]:
        out.append((fam, "clean"))
    for fam in design["candidate_population"]["temporal_families"]:
        for nz in ("clean", "white"):
            if (fam, nz) in _cells(design) or True:
                out.append((fam, nz))
    return out


def screen_units_v5(design, role):
    n_gen = design["populations_v5"][
        "DEVELOPMENT_per_cell" if role == "DEVELOPMENT"
        else "CALIBRATION_per_cell"]
    units = []
    fams = (_cells(design)
            + [("random_label", "clean"),
               ("easy_constant", "clean")])
    for fam, nz in fams:
        for width in design["candidate_population"][
                "hidden_widths"]:
            for gi in range(n_gen):
                units.append({
                    "unit_id": (f"screen::{role}::{fam}::{nz}"
                                f"::w{width}::g{gi}"),
                    "kind": "screen", "role": role,
                    "family": fam, "noise_coord": nz,
                    "width": width, "generator_index": gi,
                    "generator_id": gb.generator_id(
                        role, fam, nz, gi),
                    "model_seed": 0,
                    "unit_role": (
                        "negative_control"
                        if fam == "random_label" else
                        "positive_control"
                        if fam == "easy_constant" else
                        "treatment")})
    return sorted(units, key=lambda u: u["unit_id"])


def intervention_units_v5(design, role):
    units = []
    if role == "DEVELOPMENT":
        specs = [dict(s, model_seeds=[s["model_seed"]])
                 for s in design["four_unit_rule"]["units"]]
        for s in specs:
            for ms in s["model_seeds"]:
                units.append(_iv_unit(role, s["family"],
                                      s["noise"], s["width"],
                                      s["generator_index"], ms))
    else:
        n_gen = design["populations_v5"]["CALIBRATION_per_cell"]
        for fam, nz in _conf_cells(design):
            for width in CONF_WIDTHS:
                for gi in range(n_gen):
                    for ms in range(CAL_SEEDS):
                        units.append(_iv_unit(role, fam, nz,
                                              width, gi, ms))
    return sorted(units, key=lambda u: u["unit_id"])


def _iv_unit(role, fam, nz, width, gi, ms):
    return {"unit_id": (f"intervention::{role}::{fam}::{nz}"
                        f"::w{width}::g{gi}::s{ms}"),
            "kind": "intervention", "role": role,
            "family": fam, "noise_coord": nz, "width": width,
            "generator_index": gi,
            "generator_id": gb.generator_id(role, fam, nz, gi),
            "model_seed": ms,
            "unit_role": "paired_checkpoint_intervention"}


def reserved_confirmation_slots(design):
    """ids ONLY — never constructed (C33 kill 17)."""
    slots = []
    n = design["populations_v5"][
        "CONFIRMATION_reserved_per_cell"]
    for fam, nz in _conf_cells(design):
        for width in CONF_WIDTHS:
            for gi in range(n):
                slots.append(f"RESERVED::CONFIRMATION::{fam}::"
                             f"{nz}::w{width}::g{gi}")
    return slots


# ------------------------ execution ------------------------

def _limits(design, out, t0, acct):
    lim = design["resources"]
    if time.monotonic() - t0 > lim["max_wall_seconds"]:
        return "WALL_STOP"
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if rss > lim["max_rss_bytes"]:
        return "RSS_STOP"
    if (out / "M4_RUN_STOP").exists():
        return "STOP_REQUESTED"
    tmp = out / f".hb_{os.urandom(6).hex()}"
    tmp.write_text(json.dumps({
        "monotonic": time.monotonic(),
        "updates_done": acct["optimization_updates"]}))
    os.chmod(tmp, 0o600)
    os.replace(tmp, out / "M4_RUN_HEARTBEAT.json")
    return None


_DESC_INVALID = {"numerically_invalid_descriptor": True,
                 "compressed_len_zlib9": None,
                 "spectral_rank_W1_1e3": None,
                 "prune_fraction_1e3": None}


def _descriptors(p, acct):
    """C31B: the descriptor numeric domain is the DECLARED
    float32 serialization boundary — finite float64 values
    outside that range are a typed NUMERICALLY_INVALID_DESCRIPTOR
    (infinity bytes are never compressed); SVD failure and
    nonfinite singular values are the same typed invalidity,
    never a zero rank or an apparently valid measurement."""
    t0 = time.monotonic()
    acct["descriptor_evals"] += 1
    w = np.concatenate([np.ascontiguousarray(p[k]).ravel()
                        for k in sorted(p)])
    if not np.isfinite(w).all():
        return {**_DESC_INVALID, "descriptor_seconds": 0.0}
    wr = np.round(w, 6)
    f32max = float(np.finfo(np.float32).max)
    if (np.abs(wr) > f32max).any():
        return {**_DESC_INVALID, "descriptor_seconds": 0.0}
    comp = len(zlib.compress(
        wr.astype(np.float32).tobytes(), 9))
    try:
        sv = np.linalg.svd(p["W1"], compute_uv=False)
    except np.linalg.LinAlgError:
        return {**_DESC_INVALID, "descriptor_seconds": 0.0}
    if sv.size and not np.isfinite(sv).all():
        return {**_DESC_INVALID, "descriptor_seconds": 0.0}
    rank = int((sv > sv.max() * 1e-3).sum()) if sv.size else 0
    prune = float((np.abs(w) < 1e-3).mean())
    dt = time.monotonic() - t0
    acct["descriptor_seconds"] += dt
    # the invalid marker exists ONLY on invalid results, so the
    # committed finite DEVELOPMENT records stay bit-identical
    return {"compressed_len_zlib9": comp,
            "spectral_rank_W1_1e3": rank,
            "prune_fraction_1e3": round(prune, 8),
            "descriptor_seconds": round(dt, 6)}


def _run_screen_unit_v5(design, u, acct):
    g = gb.generate(u["role"], u["family"], u["noise_coord"],
                    u["generator_index"])
    gb.consumer_verify(g)
    kind = pv.task_kind(u["family"])
    ck = pv.build_checkpoints(g, u["width"], u["model_seed"])
    if ck["numerically_invalid"]:
        # C31: numerical failure is a TYPED incomplete unit —
        # it stays in the denominator and never crashes.
        rec = {**{k: v for k, v in u.items()},
               "manifest_sha256":
                   g["manifest"]["manifest_sha256"],
               "task_kind": kind,
               "selected_stop_update": None,
               "stop_trajectory_digest": None,
               "metric_heldout": None,
               "baseline_value": None,
               "improvement": None,
               "numerically_invalid": True}
        return _self(rec)
    acct["optimization_updates"] += \
        ck["checkpoints"]["post_stop_bounded"]["updates"]
    p = ck["checkpoints"]["calibration_stop"]["params"]
    metric, base, invalid = pv.heldout_metric(kind, g, p)
    acct["evaluations"] += 1
    rec = {**{k: v for k, v in u.items()},
           "manifest_sha256": g["manifest"]["manifest_sha256"],
           "task_kind": kind,
           "selected_stop_update": ck["selected_stop_update"],
           "stop_trajectory_digest":
               ck["stop_trajectory_digest"],
           "metric_heldout":
               None if invalid else round(metric, 8),
           "baseline_value":
               None if invalid else round(base, 8),
           "improvement": None if invalid else round(
               pv.improvement(kind, metric, base), 8),
           "numerically_invalid": bool(invalid)}
    return _self(rec)


def derive_learnability_v5(design, recs) -> dict:
    rl = [r["improvement"] for r in recs
          if r["family"] == "random_label"
          and r["improvement"] is not None]
    if not rl:
        raise RunnerV5Refusal("no random_label control records")
    margin = max(0.05, float(np.percentile(rl, 95)))
    cells = {}
    for r in recs:
        if r["family"] in ("random_label", "easy_constant"):
            continue
        if r["numerically_invalid"]:
            oc = "NUMERICALLY_INVALID"
        elif r["improvement"] is not None and \
                r["improvement"] > margin:
            oc = "LEARNABLE_UNDER_FROZEN_BUDGET"
        else:
            oc = "OPTIMIZATION_LIMITED"
        cells[r["unit_id"]] = {"outcome": oc,
                               "improvement": r["improvement"],
                               "margin": round(margin, 8)}
    easy = [r for r in recs if r["family"] == "easy_constant"]
    easy_pass = bool(easy) and all(
        r["improvement"] is not None and r["improvement"] > 0
        for r in easy)
    doc = {"schema": "agent_multi.m4_learnability_table.v2",
           "margin": round(margin, 8),
           "easy_positive_control_passes": easy_pass,
           "cells": cells}
    return _self(doc)


def _state_path(out, uid, arm):
    return out / "intervention" / f"{_safe(uid)}__{arm}.state.npz"


def _write_state(path, params, st, margin, endpoint, next_b,
                 last_rec_sha):
    import io
    buf = io.BytesIO()
    np.savez(buf, **params)
    tmp = Path(str(path) + f".tmp{os.urandom(4).hex()}")
    fd = os.open(str(tmp), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, buf.getvalue())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp, path)
    meta = {"streak": st["streak"], "updates": st["updates"],
            "margin": margin, "endpoint": endpoint,
            "next_batch": next_b, "last_record_sha": last_rec_sha}
    meta["meta_sha256"] = m4._self_sha(meta, "meta_sha256")
    tmp2 = Path(str(path) + f".meta.tmp{os.urandom(4).hex()}")
    fd = os.open(str(tmp2), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(meta, sort_keys=True).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(tmp2, Path(str(path) + ".meta.json"))


def _load_state(path):
    with np.load(str(path)) as z:
        params = {k: z[k].copy() for k in z.files}
    meta = m4._strict_json_file(Path(str(path) + ".meta.json"),
                                "resume state meta")
    if m4._self_sha(meta, "meta_sha256") != meta["meta_sha256"]:
        raise RunnerV5Refusal("resume state meta self-digest "
                              "does not re-derive — UNCERTAIN")
    return params, meta


def _run_arm_durable(design, u, g, tape, ckpt, kind, out, acct):
    """One arm with per-batch durability: the JSONL is created
    once with O_EXCL and appended+fsynced per batch; a durable
    state (atomic replace) follows every batch. Resume continues
    from the VERIFIED predecessor state; an existing log without
    a matching state is UNCERTAIN and refuses. The state files
    are removed when the arm completes."""
    arm_name = ckpt["_name"]
    lp = (out / "intervention"
          / f"{_safe(u['unit_id'])}__{arm_name}.jsonl")
    sp = _state_path(out, u["unit_id"], arm_name)
    start = None
    prev_records = []
    if lp.exists():
        lines = lp.read_text().splitlines()
        if not sp.exists():
            raise RunnerV5Refusal(
                f"{lp.name}: partial intervention log without "
                "its durable predecessor state — UNCERTAIN; "
                "explicit disposition required, silent resume "
                "refused")
        params, meta = _load_state(sp)
        if not lines:
            raise RunnerV5Refusal(
                f"{lp.name}: empty log with a state — UNCERTAIN")
        last = m4._strict_json_text(lines[-1], "last record")
        if last["record_sha256"] != meta["last_record_sha"]:
            raise RunnerV5Refusal(
                f"{lp.name}: durable state does not bind the "
                "last appended record — UNCERTAIN")
        for i, line in enumerate(lines):
            prev_records.append(m4._strict_json_text(
                line, f"{lp.name} {i}"))
        start = {"params": params, "streak": meta["streak"],
                 "updates": meta["updates"],
                 "margin": meta["margin"],
                 "endpoint": meta["endpoint"],
                 "next_batch": meta["next_batch"]}
        fd = os.open(str(lp), os.O_WRONLY | os.O_APPEND
                     | getattr(os, "O_NOFOLLOW", 0))
    else:
        fd = os.open(str(lp), os.O_CREAT | os.O_EXCL
                     | os.O_WRONLY | os.O_APPEND, 0o600)
    holder = {"fd": fd}

    def on_batch(rec, params, st, margin, endpoint):
        rec = _self({**rec, "tape_digest": tape["digest"]})
        os.write(holder["fd"], (json.dumps(
            rec, sort_keys=True) + "\n").encode())
        os.fsync(holder["fd"])
        _write_state(sp, params, st, margin, endpoint,
                     rec["batch"] + 1, rec["record_sha256"])

    try:
        res = pv.run_intervention(g, tape, ckpt, kind,
                                  on_batch=on_batch,
                                  start=start)
    finally:
        os.close(holder["fd"])
    # completed: durable state no longer needed
    for q in (sp, Path(str(sp) + ".meta.json")):
        if q.exists():
            os.unlink(q)
    acct["optimization_updates"] += res["updates_done"] - (
        start["updates"] if start else 0)
    acct["evaluations"] += len(res["records"])
    all_records = prev_records + [
        _self({**r, "tape_digest": tape["digest"]})
        for r in res["records"]]
    # facts over the COMPLETE history
    endpoint = 0
    for r in all_records:
        if r["outcome"] == "ACCEPTED":
            endpoint = r["cumulative_associations"]
    cap = res["cap_reached"]
    cause = res["stopping_cause"]
    fail_b = None if cap else len(all_records) - 1         if all_records[-1]["outcome"] != "ACCEPTED" else None
    return {"restricted_endpoint": min(
                endpoint, pv.MAX_BATCHES * pv.ASSOC_BATCH),
            "cap_reached": cap,
            "stopping_cause": cause,
            "updates_done": res["updates_done"],
            "retention_margin": res["retention_margin"],
            "fail_batch": fail_b,
            "final_params_digest": res["final_params_digest"]}


def _run_intervention_unit_v5(design, u, out, acct):
    g = gb.generate(u["role"], u["family"], u["noise_coord"],
                    u["generator_index"])
    gb.consumer_verify(g)
    kind = pv.task_kind(u["family"])
    tape = pv.association_tape(design["design_sha256"], g,
                               u["width"], u["model_seed"])
    ck = pv.build_checkpoints(g, u["width"], u["model_seed"])
    if ck["numerically_invalid"]:
        rec = {**u, "manifest_sha256":
               g["manifest"]["manifest_sha256"],
               "task_kind": kind,
               "tape_id": tape["tape_id"],
               "tape_digest": tape["digest"],
               "unit_status":
                   "NUMERICALLY_INVALID_TASK_TRAINING",
               "invalid_at_update": ck["invalid_at_update"]}
        return _self(rec)
    acct["optimization_updates"] += \
        ck["checkpoints"]["post_stop_bounded"]["updates"]
    arms = {}
    lineage = {name: {"params_digest": c["params_digest"],
                      "parent": c["parent"],
                      "updates": c["updates"]}
               for name, c in ck["checkpoints"].items()}
    for arm in pv.CHECKPOINTS:
        ckpt = dict(ck["checkpoints"][arm])
        ckpt["_name"] = arm
        a = _run_arm_durable(design, u, g, tape, ckpt, kind,
                             out, acct)
        a["descriptors"] = _descriptors(
            ck["checkpoints"][arm]["params"], acct)
        a["checkpoint_loss_stop"] = round(pv.loss_task(
            kind, ck["checkpoints"][arm]["params"],
            g["X_stop"], g["y_stop"]), 8)
        arms[arm] = a
    traj = ck["stop_trajectory"]
    slope = (traj[-1] - traj[0]) / max(len(traj) - 1, 1) \
        if len(traj) > 1 else 0.0
    rec = {**u, "manifest_sha256":
           g["manifest"]["manifest_sha256"],
           "task_kind": kind,
           "tape_id": tape["tape_id"],
           "tape_digest": tape["digest"],
           "tape_tol": tape["tol"],
           "genesis_digest":
               lineage["initialization"]["params_digest"],
           "checkpoint_lineage": lineage,
           "stop_trajectory_digest":
               ck["stop_trajectory_digest"],
           "stop_trajectory_slope": round(float(slope), 8),
           "selected_stop_update": ck["selected_stop_update"],
           "arms": arms,
           "paired_primary_difference":
               arms["calibration_stop"]["restricted_endpoint"]
               - arms["initialization"]["restricted_endpoint"]}
    return _self(rec)


# --------------------------- run ---------------------------

def plan_v5(design, roles) -> dict:
    out = {}
    for role in roles:
        su = screen_units_v5(design, role)
        iu = intervention_units_v5(design, role)
        out[role] = {"screen_units": len(su),
                     "intervention_units": len(iu)}
    out["confirmation_reserved_slots"] = len(
        reserved_confirmation_slots(design))
    return out


def _assert_pre_outcome_boundary(design):
    """C28/C33 kill 16: the SEALED v5 may be scored only after
    its design bytes are committed AND pushed. Reduced test
    fixtures (different self identity) are engineering surfaces,
    not v5 scores, and pass through."""
    import subprocess
    dp = pv.DESIGN_PATH_V5
    if not dp.exists():
        return
    sealed = m4._strict_json_file(dp, "sealed v5")
    if design["design_sha256"] != sealed["design_sha256"]:
        return                      # fixture design — not v5
    try:
        rel = str(dp.relative_to(REPO))
    except ValueError:
        raise RunnerV5Refusal(
            "PRE_OUTCOME_BOUNDARY: the sealed v5 design is not "
            "COMMITTED — no v5 score may be computed before the "
            "commit/push boundary")
    ls = subprocess.run(["git", "-C", str(REPO), "ls-files",
                         "--error-unmatch", rel],
                        capture_output=True, text=True)
    if ls.returncode != 0:
        raise RunnerV5Refusal(
            "PRE_OUTCOME_BOUNDARY: the sealed v5 design is not "
            "COMMITTED — no v5 score may be computed before the "
            "commit/push boundary")
    blob = subprocess.run(["git", "-C", str(REPO), "show",
                           f"HEAD:{rel}"], capture_output=True,
                          text=True)
    if blob.stdout != dp.read_text():
        raise RunnerV5Refusal(
            "PRE_OUTCOME_BOUNDARY: the committed v5 bytes do "
            "not equal the sealed file")
    br = subprocess.run(["git", "-C", str(REPO), "rev-parse",
                         "--abbrev-ref", "HEAD"],
                        capture_output=True, text=True
                        ).stdout.strip()
    anc = subprocess.run(
        ["git", "-C", str(REPO), "merge-base",
         "--is-ancestor",
         subprocess.run(["git", "-C", str(REPO), "log", "-n1",
                         "--format=%H", "--", rel],
                        capture_output=True,
                        text=True).stdout.strip(),
         f"origin/{br}"], capture_output=True, text=True)
    if anc.returncode != 0:
        raise RunnerV5Refusal(
            "PRE_OUTCOME_BOUNDARY: the v5 seal commit is not "
            "PUSHED to origin — push the boundary before any "
            "v5 score")


def execute_v5(design, out_root: Path, roles) -> dict:
    for role in roles:
        if role not in ("DEVELOPMENT", "CALIBRATION"):
            raise RunnerV5Refusal(
                "only DEVELOPMENT and CALIBRATION may execute "
                "in this order — CONFIRMATION is reserved")
    _assert_pre_outcome_boundary(design)
    out = Path(out_root)
    all_su = []
    all_iu = []
    for role in roles:
        all_su += screen_units_v5(design, role)
        all_iu += intervention_units_v5(design, role)
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic()}
    resumed_verified = 0
    if out.exists() and any(out.iterdir()):
        if (out / "RUN_REPORT.json").is_file():
            v = verify_run_v5(design, out, roles)
            return {"idempotent_verification": v["verified"],
                    "units_new_this_session": 0}
        # C32 durable resume: verify the ledger, REPLAY every
        # completed unit, continue partial arms from their
        # durable predecessor states.
        led = m4._strict_json_file(out / "RUN_LEDGER.json",
                                   "resume ledger")
        if led["design_sha256"] != design["design_sha256"]:
            raise RunnerV5Refusal(
                "resume ledger binds a different design")
        if led["units"] != all_su + all_iu:
            raise RunnerV5Refusal(
                "resume ledger population mismatch")
        vac = {"optimization_updates": 0, "evaluations": 0,
               "descriptor_seconds": 0.0, "descriptor_evals": 0,
               "t0": time.monotonic()}
        for u in all_su:
            q = out / "screen" / f"{_safe(u['unit_id'])}.json"
            if q.is_file():
                r = m4._strict_json_file(q, q.name)
                if _run_screen_unit_v5(design, u, vac) != r:
                    raise RunnerV5Refusal(
                        f"resume: {q.name} does not REPLAY")
                resumed_verified += 1
    else:
        out.mkdir(parents=True, exist_ok=True)
        os.chmod(out, 0o700)
        (out / "screen").mkdir(mode=0o700)
        (out / "intervention").mkdir(mode=0o700)
        ledger = {"schema": "agent_multi.m4_v5_run_ledger.v1",
                  "design_sha256": design["design_sha256"],
                  "roles": list(roles),
                  "units": all_su + all_iu,
                  "confirmation_reserved":
                      reserved_confirmation_slots(design),
                  "scheduling": "lexicographic unit_id per "
                                "kind, outcome-independent"}
        _excl_json(out / "RUN_LEDGER.json", _self(ledger))
    done = 0
    for u in all_su:
        rp = out / "screen" / f"{_safe(u['unit_id'])}.json"
        if rp.exists():
            continue
        stop = _limits(design, out, acct["t0"], acct)
        if stop:
            raise RunnerV5Refusal(f"typed resource stop: {stop}")
        _excl_json(rp, _run_screen_unit_v5(design, u, acct))
        done += 1
    recs = [m4._strict_json_file(
        out / "screen" / f"{_safe(u['unit_id'])}.json", "rec")
        for u in all_su]
    for role in roles:
        tp = out / f"LEARNABILITY_TABLE_{role}.json"
        if tp.exists():
            continue
        rrecs = [r for r in recs if r["role"] == role]
        _excl_json(tp, derive_learnability_v5(design, rrecs))
    for u in all_iu:
        rp = out / "intervention" / \
            f"{_safe(u['unit_id'])}_summary.json"
        if rp.exists():
            continue
        stop = _limits(design, out, acct["t0"], acct)
        if stop:
            raise RunnerV5Refusal(f"typed resource stop: {stop}")
        _excl_json(rp, _run_intervention_unit_v5(design, u, out,
                                                 acct))
        done += 1
    report = {"schema": "agent_multi.m4_v5_run_report.v1",
              "design_sha256": design["design_sha256"],
              "authority": "DEVELOPMENT_AND_CALIBRATION_"
                           "MECHANICS_NO_CONFIRMATION",
              "roles": list(roles),
              "units_total": len(all_su) + len(all_iu),
              "session_complete": True,
              "units_resumed_verified": resumed_verified,
              "accounting": {
                  "optimization_updates":
                      acct["optimization_updates"],
                  "evaluations": acct["evaluations"],
                  "descriptor_evals": acct["descriptor_evals"],
                  "descriptor_seconds":
                      round(acct["descriptor_seconds"], 3)},
              "wall_seconds": round(
                  time.monotonic() - acct["t0"], 2),
              "telemetry_mutable": ["M4_RUN_HEARTBEAT.json"],
              "artifacts_sha256": _inventory_v5(out)}
    _excl_json(out / "RUN_REPORT.json", _self(report))
    v = verify_run_v5(design, out, roles)
    if v["verified"] is not True:
        raise RunnerV5Refusal("post-execution fresh verification "
                              "did not verify")
    return {**{k: report[k] for k in
               ("units_total", "accounting", "wall_seconds")},
            "verified": v["verified"],
            "units_new_this_session": done}


def _inventory_v5(out):
    inv = {}
    skip = {"M4_RUN_HEARTBEAT.json", "M4_RUN_STOP",
            "RUN_REPORT.json"}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in skip:
            if p.name.startswith("RUN_REPORT"):
                raise RunnerV5Refusal(
                    f"RUN_REPORT lookalike {p.name!r} in the "
                    "root — one canonical terminal report only")
            inv[str(p.relative_to(out))] = m4._sha_file(p)
    return inv


def verify_run_v5(design, out_root: Path, roles) -> dict:
    """Fresh reconstruction: ledger population equality (full
    rows), exact inventory (lookalikes refuse), screen and
    intervention replay from the sealed design, tape/genesis/
    lineage digic equality, table re-derivation, accounting
    re-derivation, artifact modes 0600."""
    import stat as _stat
    out = Path(out_root)
    led = m4._strict_json_file(out / "RUN_LEDGER.json", "ledger")
    if m4._self_sha(led, "record_sha256") != led["record_sha256"]:
        raise RunnerV5Refusal("ledger self-digest does not "
                              "re-derive")
    if led["design_sha256"] != design["design_sha256"]:
        raise RunnerV5Refusal("ledger does not bind the sealed "
                              "v5")
    all_su = []
    all_iu = []
    for role in roles:
        all_su += screen_units_v5(design, role)
        all_iu += intervention_units_v5(design, role)
    if led["units"] != all_su + all_iu:
        raise RunnerV5Refusal(
            "pre-result ledger does not enumerate the sealed "
            "population exactly (full rows)")
    if led["confirmation_reserved"] != \
            reserved_confirmation_slots(design):
        raise RunnerV5Refusal(
            "reserved CONFIRMATION slots do not match the "
            "sealed reservation")
    report = m4._strict_json_file(out / "RUN_REPORT.json",
                                  "report")
    _RK = {"schema", "design_sha256", "authority", "roles",
           "units_total", "session_complete",
           "units_resumed_verified", "accounting",
           "wall_seconds", "telemetry_mutable",
           "artifacts_sha256", "record_sha256"}
    if set(report) != _RK:
        raise RunnerV5Refusal("report keys are not the exact "
                              "schema")
    if m4._self_sha(report, "record_sha256") != \
            report["record_sha256"]:
        raise RunnerV5Refusal("report self-digest does not "
                              "re-derive")
    inv = _inventory_v5(out)
    if inv != report["artifacts_sha256"]:
        raise RunnerV5Refusal("artifact inventory does not "
                              "equal the report exactly")
    for rel in inv:
        mode = _stat.S_IMODE((out / rel).stat().st_mode)
        if mode != 0o600:
            raise RunnerV5Refusal(
                f"artifact {rel!r} mode {oct(mode)} is not the "
                "private 0600 contract")
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic()}
    valid_iv_units = 0
    recs = []
    for u in all_su:
        p = out / "screen" / f"{_safe(u['unit_id'])}.json"
        if not p.is_file():
            raise RunnerV5Refusal(
                f"screen record {p.name} is MISSING — the "
                "denominator never shrinks")
        r = m4._strict_json_file(p, p.name)
        fresh = _run_screen_unit_v5(design, u, acct)
        if fresh != r:
            bad = sorted(k for k in fresh
                         if fresh[k] != r.get(k))
            raise RunnerV5Refusal(
                f"screen unit {u['unit_id']} does not replay "
                f"(fields: {bad[:3]})")
        recs.append(r)
    for role in roles:
        tab = m4._strict_json_file(
            out / f"LEARNABILITY_TABLE_{role}.json", "table")
        rrecs = [r for r in recs if r["role"] == role]
        if derive_learnability_v5(design, rrecs) != tab:
            raise RunnerV5Refusal(
                f"{role} learnability table does not re-derive")
    derived = {}
    for u in all_iu:
        p = out / "intervention" / \
            f"{_safe(u['unit_id'])}_summary.json"
        if not p.is_file():
            raise RunnerV5Refusal(
                f"summary {p.name} is MISSING — the denominator "
                "never shrinks")
        r = m4._strict_json_file(p, p.name)
        # replay WITHOUT rewriting logs: reconstruct in memory
        g = gb.generate(u["role"], u["family"],
                        u["noise_coord"], u["generator_index"])
        gb.consumer_verify(g)
        kind = pv.task_kind(u["family"])
        tape = pv.association_tape(design["design_sha256"], g,
                                   u["width"], u["model_seed"])
        if tape["digest"] != r["tape_digest"]:
            raise RunnerV5Refusal(
                f"{u['unit_id']}: tape digest does not "
                "re-derive — arms cannot share an unverified "
                "tape")
        ck = pv.build_checkpoints(g, u["width"], u["model_seed"])
        if ck["numerically_invalid"]:
            if r.get("unit_status") != \
                    "NUMERICALLY_INVALID_TASK_TRAINING" or \
                    r.get("invalid_at_update") != \
                    ck["invalid_at_update"] or \
                    r.get("tape_digest") != tape["digest"]:
                raise RunnerV5Refusal(
                    f"{u['unit_id']}: numerically-invalid unit "
                    "does not replay to the same typed state")
            continue
        if r.get("unit_status") is not None:
            raise RunnerV5Refusal(
                f"{u['unit_id']}: claims an invalid status the "
                "replay does not derive")
        valid_iv_units += 1
        acct["optimization_updates"] += \
            ck["checkpoints"]["post_stop_bounded"]["updates"]
        if r["genesis_digest"] != \
                ck["checkpoints"]["initialization"][
                    "params_digest"]:
            raise RunnerV5Refusal(
                f"{u['unit_id']}: genesis does not re-derive")
        if set(r["checkpoint_lineage"]) != set(pv.CHECKPOINTS):
            raise RunnerV5Refusal(
                f"{u['unit_id']}: the four declared checkpoints "
                "are not all present in the lineage")
        for name, c in ck["checkpoints"].items():
            li = r["checkpoint_lineage"][name]
            if li["params_digest"] != c["params_digest"] or \
                    li["parent"] != c["parent"] or \
                    li["updates"] != c["updates"]:
                raise RunnerV5Refusal(
                    f"{u['unit_id']}: checkpoint {name} lineage "
                    "does not replay")
        if set(r["arms"]) != set(pv.CHECKPOINTS):
            raise RunnerV5Refusal(
                f"{u['unit_id']}: the four declared checkpoint "
                "arms are not all present")
        for arm in pv.CHECKPOINTS:
            res = pv.run_intervention(
                g, tape, ck["checkpoints"][arm], kind)
            acct["optimization_updates"] += res["updates_done"]
            acct["evaluations"] += len(res["records"])
            a = r["arms"][arm]
            if (a["restricted_endpoint"]
                    != res["restricted_endpoint"]
                    or a["cap_reached"] is not
                    res["cap_reached"]
                    or a["stopping_cause"]
                    != res["stopping_cause"]
                    or a["updates_done"] != res["updates_done"]
                    or a["final_params_digest"]
                    != res["final_params_digest"]):
                raise RunnerV5Refusal(
                    f"{u['unit_id']}/{arm}: endpoint facts do "
                    "not equal the replayed derivation")
            lp = out / "intervention" / \
                f"{_safe(u['unit_id'])}__{arm}.jsonl"
            lines = lp.read_text().splitlines()
            if len(lines) != len(res["records"]):
                raise RunnerV5Refusal(
                    f"{lp.name}: ledger length does not equal "
                    "the replay")
            for i, line in enumerate(lines):
                rec = m4._strict_json_text(line,
                                           f"{lp.name} {i}")
                claim = {k: rec[k] for k in rec
                         if k not in ("record_sha256",
                                      "tape_digest")}
                if claim != res["records"][i]:
                    raise RunnerV5Refusal(
                        f"{lp.name} record {i} does not replay")
                if rec["tape_digest"] != tape["digest"]:
                    raise RunnerV5Refusal(
                        f"{lp.name} record {i} binds a foreign "
                        "tape")
            fresh_desc = _descriptors(
                ck["checkpoints"][arm]["params"],
                {"descriptor_seconds": 0.0,
                 "descriptor_evals": 0})
            rec_desc = dict(r["arms"][arm]["descriptors"])
            for tkey in ("descriptor_seconds",):
                fresh_desc.pop(tkey, None)
                rec_desc.pop(tkey, None)
            if fresh_desc != rec_desc:
                raise RunnerV5Refusal(
                    f"{u['unit_id']}/{arm}: descriptor "
                    "validity/values do not re-derive from the "
                    "original float64 parameters")
            derived[f"{u['unit_id']}::{arm}"] = \
                res["restricted_endpoint"]
        want = (derived[f"{u['unit_id']}::calibration_stop"]
                - derived[f"{u['unit_id']}::initialization"])
        if r["paired_primary_difference"] != want:
            raise RunnerV5Refusal(
                f"{u['unit_id']}: paired primary difference "
                "does not re-derive")
    ra = report["accounting"]
    if report.get("session_complete") is True and \
            report.get("units_resumed_verified", 0) == 0:
        if ra["optimization_updates"] != \
                acct["optimization_updates"]:
            raise RunnerV5Refusal(
                "reported optimization accounting does not "
                "equal the replayed derivation")
        if ra["descriptor_evals"] != \
                len(pv.CHECKPOINTS) * valid_iv_units:
            raise RunnerV5Refusal(
                "descriptor accounting does not equal the "
                "replayed derivation (typed-invalid units "
                "compute no descriptors)")
    return {"verified": True,
            "screen_units": len(all_su),
            "intervention_units": len(all_iu)}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--roles", type=str,
                    default="DEVELOPMENT")
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    design = pv.load_design_v5(a.design)
    roles = tuple(a.roles.split(","))
    if a.plan:
        print(json.dumps(plan_v5(design, roles), indent=1))
        return 0
    if a.execute:
        r = execute_v5(design, a.out, roles)
        print(json.dumps(r, indent=1))
        return 0
    if a.verify:
        print(json.dumps(verify_run_v5(design, a.out, roles),
                         indent=1))
        return 0
    raise RunnerV5Refusal("choose --plan, --execute or --verify")


if __name__ == "__main__":
    raise SystemExit(main())
