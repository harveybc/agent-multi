"""M4-C20/C22: the ONE config-driven CPU runner.

Every scientific and resource value comes from the sealed v4
design; the CLI selects only design path, output root and
plan/execute mode. It executes the DEVELOPMENT-only learnability
screen (C20) and the four sealed development intervention units
(C23), with: a pre-result ledger of every unit and role,
deterministic outcome-independent scheduling, private write-once
artifacts and typed terminals, durable resume that re-verifies
every completed unit BY REPLAY, executable wall/RSS/stop/
heartbeat limits, exact accounting of optimization, evaluation
and descriptor costs, and a fresh verifier that reconstructs
generators, fits, batches, transitions, endpoint, censoring and
outcomes from raw persisted evidence — no producer `passed`,
`learnable`, `endpoint` or `verified` field controls a verdict
without re-derivation.

DEVELOPMENT only. CALIBRATION/CONFIRMATION are never generated,
scored or inspected here. CPU only. Zero scientific conclusion.
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
import m4_intervention_design as dz  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402


class RunnerRefusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


LEARN_OUTCOMES = ("LEARNABLE_UNDER_FROZEN_BUDGET",
                  "OPTIMIZATION_LIMITED", "NUMERICALLY_INVALID")


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


# ------------------- deterministic fitting -------------------

def _fit(design, g, width, model_seed):
    """The frozen learnability fit: identical optimizer family
    and budget as the intervention pretrain."""
    lg = design["learnability_gate"]["optimizer"]
    p = m4._mlp_init(gb.N_IN, width,
                     gb._seed("fit_init", g["generator_id"],
                              width, model_seed))
    X, y = g["X_train"], g["y_train"]
    for u in range(lg["updates"]):
        rng = np.random.default_rng(
            gb._seed("fit_mb", g["generator_id"], width,
                     model_seed, u))
        i = rng.integers(0, len(y), size=lg["minibatch"])
        m4._sgd_step(p, X[i], y[i], lg["learning_rate"])
    return p


def _metric(design, g, p):
    """Held-out criterion + frozen baseline, per family class."""
    out = m4._forward(p, g["X_held"])[1]
    if not np.isfinite(out).all():
        return None, None, "NUMERICALLY_INVALID"
    fam = g["family"]
    if fam in gb.BOOL_FAMILIES + ("random_label",):
        acc = float(((out > 0.5) == (g["y_held"] > 0.5)).mean())
        maj = float((g["y_train"] > 0.5).mean())
        base = max(maj, 1.0 - maj)     # train majority class
        return acc, base, None
    # temporal + easy_constant: MSE skill vs persistence
    mse = float(np.mean((out - g["y_held"]) ** 2))
    persist = g["X_held"][:, -1]
    base_mse = float(np.mean((persist - g["y_held"]) ** 2))
    return mse, base_mse, None


def _improvement(g, metric, base):
    if g["family"] in gb.BOOL_FAMILIES + ("random_label",):
        return metric - base                  # accuracy gain
    return (base - metric) / max(base, 1e-12)  # MSE skill


# ---------------------- unit enumeration ----------------------

def screen_units(design):
    units = []
    pop = design["candidate_population"]
    for fam, nz in (list(map(tuple,
                             _sc(design)))
                    + [("random_label", "clean"),
                       ("easy_constant", "clean")]):
        for width in pop["hidden_widths"]:
            for gidx in range(
                    design["population_census"]
                    ["development_generators_per_cell"]):
                units.append({
                    "unit_id": f"screen::{fam}::{nz}::w{width}"
                               f"::g{gidx}",
                    "kind": "screen", "family": fam,
                    "noise_or_NOT_APPLICABLE":
                        nz if fam in gb.TEMPORAL_FAMILIES
                        else "NOT_APPLICABLE",
                    "noise_coord": nz,
                    "width": width, "generator_role":
                        "DEVELOPMENT",
                    "generator_id": gb.generator_id(
                        "DEVELOPMENT", fam, nz, gidx),
                    "model_seed": 0,
                    "optimizer_budget":
                        design["learnability_gate"]["optimizer"],
                    "unit_role": (
                        "negative_control"
                        if fam == "random_label" else
                        "positive_control"
                        if fam == "easy_constant" else
                        "treatment")})
    return sorted(units, key=lambda u: u["unit_id"])


def _sc(design):
    cells = []
    for fam in design["candidate_population"][
            "structured_boolean_families"]:
        cells.append((fam, "clean"))
    for fam in design["candidate_population"]["temporal_families"]:
        for nz in design["candidate_population"][
                "noise_regimes_temporal"]:
            cells.append((fam, nz))
    return cells


def intervention_units(design):
    units = []
    for spec in design["four_unit_rule"]["units"]:
        units.append({
            "unit_id": (f"intervention::{spec['family']}::"
                        f"{spec['noise']}::w{spec['width']}::"
                        f"g{spec['generator_index']}::"
                        f"s{spec['model_seed']}"),
            "kind": "intervention", **spec,
            "generator_role": "DEVELOPMENT",
            "generator_id": gb.generator_id(
                "DEVELOPMENT", spec["family"], spec["noise"],
                spec["generator_index"]),
            "optimizer_budget":
                design["learnability_gate"]["optimizer"],
            "unit_role": "development_mechanics"})
    return sorted(units, key=lambda u: u["unit_id"])


# ------------------------- execution -------------------------

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
    os.replace(tmp, out / "M4_RUN_HEARTBEAT.json")
    return None


def _run_screen_unit(design, u, acct):
    nz = u["noise_coord"]
    g = gb.generate("DEVELOPMENT", u["family"], nz,
                    int(u["generator_id"].rsplit("g", 1)[1]))
    gb.consumer_verify(g)
    p = _fit(design, g, u["width"], u["model_seed"])
    acct["optimization_updates"] += \
        design["learnability_gate"]["optimizer"]["updates"]
    metric, base, invalid = _metric(design, g, p)
    acct["evaluations"] += 1
    # no-learning + random-init band, evaluated without updates
    p0 = m4._mlp_init(gb.N_IN, u["width"],
                      gb._seed("fit_init", g["generator_id"],
                               u["width"], u["model_seed"]))
    m0, b0, inv0 = _metric(design, g, p0)
    band = []
    for k in range(3):
        pk = m4._mlp_init(gb.N_IN, u["width"],
                          gb._seed("rand_band",
                                   g["generator_id"], k))
        mk, _, invk = _metric(design, g, pk)
        band.append(None if invk else round(
            _improvement(g, mk, base), 8))
        acct["evaluations"] += 1
    acct["evaluations"] += 1
    rec = {**u, "manifest_sha256":
           g["manifest"]["manifest_sha256"],
           "metric_heldout":
               None if invalid else round(metric, 8),
           "baseline_value":
               None if invalid else round(base, 8),
           "improvement": None if invalid else round(
               _improvement(g, metric, base), 8),
           "no_learning_improvement":
               None if inv0 else round(
                   _improvement(g, m0, b0), 8),
           "random_init_band": band,
           "numerically_invalid": bool(invalid)}
    return _self(rec)


def derive_learnability_table(design, recs):
    """The frozen threshold rule, applied to RAW records — pure
    and re-derivable: margin = max(0.05, p95 of random_label
    improvements); a structured cell is LEARNABLE iff finite and
    improvement > margin."""
    rl = [r["improvement"] for r in recs
          if r["family"] == "random_label"
          and r["improvement"] is not None]
    if not rl:
        raise RunnerRefusal("no random_label control records — "
                            "the margin cannot be derived")
    margin = max(0.05, float(np.percentile(rl, 95)))
    table = {}
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
        table[r["unit_id"]] = {
            "outcome": oc,
            "improvement": r["improvement"],
            "margin": round(margin, 8)}
    fam_summary = {}
    for r in recs:
        fam = r["family"]
        if fam in ("random_label", "easy_constant"):
            continue
        key = f"{fam}::{r['noise_coord']}"
        fam_summary.setdefault(key, {"LEARNABLE_UNDER_FROZEN_"
                                     "BUDGET": 0,
                                     "OPTIMIZATION_LIMITED": 0,
                                     "NUMERICALLY_INVALID": 0,
                                     "total": 0})
        fam_summary[key][table[r["unit_id"]]["outcome"]] += 1
        fam_summary[key]["total"] += 1
    doc = {"schema": "agent_multi.m4_learnability_table.v1",
           "margin": round(margin, 8),
           "margin_rule": design["learnability_gate"]
           ["threshold_rule"],
           "denominator_note": (
               "every enumerated cell appears; missing or "
               "failed cells stay in the denominator"),
           "cells": table, "family_noise_summary": fam_summary}
    return _self(doc)


def _descriptors(p, acct):
    t0 = time.monotonic()
    w = np.concatenate([np.ascontiguousarray(p[k]).ravel()
                        for k in sorted(p)])
    comp_len = len(zlib.compress(
        np.round(w, 6).astype(np.float32).tobytes(), 9))
    s = np.linalg.svd(p["W1"], compute_uv=False)
    rank = int((s > s.max() * 1e-3).sum()) if s.size else 0
    prune = float((np.abs(w) < 1e-3).mean())
    dt = time.monotonic() - t0
    acct["descriptor_seconds"] += dt
    acct["descriptor_evals"] += 1
    return {"compressed_len_zlib9": comp_len,
            "spectral_rank_W1_1e3": rank,
            "prune_fraction_1e3": round(prune, 8),
            "descriptor_seconds": round(dt, 6)}


def _run_intervention_unit(design, u, out, acct):
    g = gb.generate("DEVELOPMENT", u["family"], u["noise"],
                    u["generator_index"])
    gb.consumer_verify(g)
    ledger = out / "intervention" / f"{_safe(u['unit_id'])}.jsonl"
    results = {}
    for arm, seed_tag in (("treatment", "fit_init"),
                          ("control", "ctrl_init")):
        t_arm = time.monotonic()
        if arm == "treatment":
            p = _fit(design, g, u["width"], u["model_seed"])
            acct["optimization_updates"] += \
                design["learnability_gate"]["optimizer"]["updates"]
        else:
            p = m4._mlp_init(gb.N_IN, u["width"],
                             gb._seed(seed_tag,
                                      g["generator_id"],
                                      u["width"],
                                      u["model_seed"]))
        base_loss = m4._loss(p, g["X_held"], g["y_held"])
        st = {"params": p,
              "assoc_X": np.zeros((0, gb.N_IN)),
              "assoc_y": np.zeros((0,)),
              "family": f"{u['unit_id']}::{arm}",
              "batch_index": 0, "retention_streak": 0,
              "updates_done": 0, "accepted_batches": 0,
              "retention_margin": base_loss * 1.10}
        led = open(ledger.parent /
                   f"{_safe(u['unit_id'])}_{arm}.jsonl", "a")
        endpoint = 0
        censored = False
        cause = None
        for b in range(64):
            stop = _limits(design, out, acct["t0"], acct)
            if stop:
                cause = stop
                censored = True
                break
            rec = m4.apply_batch(st, g["X_train"], g["y_train"],
                                 g["X_held"], g["y_held"])
            acct["optimization_updates"] += m4.UPDATES_PER_BATCH
            acct["evaluations"] += 1
            rec = _self(rec)
            led.write(json.dumps(rec, sort_keys=True) + "\n")
            led.flush()
            os.fsync(led.fileno())
            if rec["outcome"] == "ACCEPTED":
                endpoint = rec["cumulative_associations"]
            else:
                cause = rec["outcome"]
                break
        else:
            cause = "MAX_BATCHES"
            censored = True
        led.close()
        results[arm] = {
            "endpoint_last_passing": endpoint,
            "censored": censored,
            "stopping_cause": cause,
            "updates_done": st["updates_done"],
            "descriptors": _descriptors(st["params"], acct),
            "arm_wall_seconds": round(
                time.monotonic() - t_arm, 3)}
    # matched diagnostic: one no-rehearsal batch from a fresh
    # treatment state (mechanics only)
    p = _fit(design, g, u["width"], u["model_seed"])
    acct["optimization_updates"] += \
        design["learnability_gate"]["optimizer"]["updates"]
    base_loss = m4._loss(p, g["X_held"], g["y_held"])
    st_d = {"params": p, "assoc_X": np.zeros((0, gb.N_IN)),
            "assoc_y": np.zeros((0,)),
            "family": f"{u['unit_id']}::diag",
            "batch_index": 0, "retention_streak": 0,
            "updates_done": 0, "accepted_batches": 0,
            "retention_margin": base_loss * 1.10}
    drec = m4.apply_batch(st_d, g["X_train"], g["y_train"],
                          g["X_held"], g["y_held"],
                          rehearsal=False)
    acct["optimization_updates"] += m4.UPDATES_PER_BATCH
    acct["evaluations"] += 1
    rec = {**u, "manifest_sha256":
           g["manifest"]["manifest_sha256"],
           "arms": results,
           "diagnostic_outcome": drec["outcome"],
           "diagnostic_examples_per_update": m4.MINIBATCH,
           "paired_difference_endpoint":
               results["treatment"]["endpoint_last_passing"]
               - results["control"]["endpoint_last_passing"]}
    return _self(rec)


def _safe(s):
    return s.replace("::", "__")


# --------------------------- run ----------------------------

def plan(design) -> dict:
    su = screen_units(design)
    iu = intervention_units(design)
    return {"screen_units": len(su),
            "intervention_units": len(iu),
            "total_units": len(su) + len(iu),
            "first_unit": su[0]["unit_id"],
            "scheduling": "lexicographic unit_id, "
                          "outcome-independent"}


def execute(design, out_root: Path) -> dict:
    out = Path(out_root)
    su = screen_units(design)
    iu = intervention_units(design)
    all_units = su + iu
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic()}
    resumed_verified = 0
    if out.exists() and any(out.iterdir()):
        resumed_verified = _resume_verify(design, out)
    else:
        out.mkdir(parents=True, exist_ok=True)
        os.chmod(out, 0o700)
        (out / "screen").mkdir(mode=0o700)
        (out / "intervention").mkdir(mode=0o700)
        ledger = {"schema": "agent_multi.m4_run_ledger.v1",
                  "design_sha256": design["design_sha256"],
                  "units": [{k: v for k, v in u.items()
                             if k != "optimizer_budget"}
                            for u in all_units],
                  "scheduling": "lexicographic unit_id, "
                                "outcome-independent"}
        _excl_json(out / "RUN_LEDGER.json", _self(ledger))
    done_new = 0
    for u in su:
        rp = out / "screen" / f"{_safe(u['unit_id'])}.json"
        if rp.exists():
            continue
        stop = _limits(design, out, acct["t0"], acct)
        if stop:
            raise RunnerRefusal(f"typed resource stop: {stop}")
        _excl_json(rp, _run_screen_unit(design, u, acct))
        done_new += 1
    recs = []
    for u in su:
        rp = out / "screen" / f"{_safe(u['unit_id'])}.json"
        r = m4._strict_json_file(rp, f"screen record {rp.name}")
        recs.append(r)
    tab_p = out / "LEARNABILITY_TABLE.json"
    if not tab_p.exists():
        _excl_json(tab_p, derive_learnability_table(design, recs))
    for u in iu:
        rp = out / "intervention" / \
            f"{_safe(u['unit_id'])}_summary.json"
        if rp.exists():
            continue
        stop = _limits(design, out, acct["t0"], acct)
        if stop:
            raise RunnerRefusal(f"typed resource stop: {stop}")
        _excl_json(rp, _run_intervention_unit(design, u, out,
                                              acct))
        done_new += 1
    report = {"schema": "agent_multi.m4_run_report.v1",
              "design_sha256": design["design_sha256"],
              "authority": "DEVELOPMENT_MECHANICS_ONLY_NO_"
                           "SCIENTIFIC_CONCLUSION",
              "units_total": len(all_units),
              "session_complete": done_new == len(all_units),
              "units_new_this_session": done_new,
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
              "artifacts_sha256": _inventory(out)}
    rp = out / "RUN_REPORT.json"
    if rp.exists():
        rp2 = out / f"RUN_REPORT_resume_{os.urandom(4).hex()}.json"
        _excl_json(rp2, _self(report))
    else:
        _excl_json(rp, _self(report))
    return report


def _inventory(out):
    inv = {}
    skip = {"M4_RUN_HEARTBEAT.json", "M4_RUN_STOP",
            "RUN_REPORT.json"}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in skip and \
                not p.name.startswith("RUN_REPORT"):
            inv[str(p.relative_to(out))] = m4._sha_file(p)
    return inv


def _resume_verify(design, out) -> int:
    """C22 durable resume: EVERY completed screen unit is
    re-verified by full replay before new work; a corrupt record
    refuses."""
    n = 0
    led = m4._strict_json_file(out / "RUN_LEDGER.json",
                               "run ledger")
    if led["design_sha256"] != design["design_sha256"]:
        raise RunnerRefusal("resume ledger binds a different "
                            "design")
    for p in sorted((out / "screen").glob("*.json")):
        r = m4._strict_json_file(p, f"resume record {p.name}")
        if m4._self_sha(r, "record_sha256") != r["record_sha256"]:
            raise RunnerRefusal(
                f"resume: {p.name} self-digest does not "
                "re-derive")
        fresh = _run_screen_unit(design, _unit_from(r, design),
                                 {"optimization_updates": 0,
                                  "evaluations": 0,
                                  "descriptor_seconds": 0.0,
                                  "descriptor_evals": 0,
                                  "t0": time.monotonic()})
        if fresh != r:
            raise RunnerRefusal(
                f"resume: {p.name} does not REPLAY — completed "
                "units must reproduce before continuation")
        n += 1
    return n


def _unit_from(r, design):
    return {k: r[k] for k in
            ("unit_id", "kind", "family",
             "noise_or_NOT_APPLICABLE", "noise_coord", "width",
             "generator_role", "generator_id", "model_seed",
             "unit_role")} | \
        {"optimizer_budget":
            design["learnability_gate"]["optimizer"]}


# ------------------------ fresh verifier ------------------------

def verify_run(design, out_root: Path) -> dict:
    """C22: full reconstruction from raw persisted evidence."""
    out = Path(out_root)
    led = m4._strict_json_file(out / "RUN_LEDGER.json",
                               "run ledger")
    if m4._self_sha(led, "record_sha256") != led["record_sha256"]:
        raise RunnerRefusal("ledger self-digest does not "
                            "re-derive")
    if led["design_sha256"] != design["design_sha256"]:
        raise RunnerRefusal("ledger does not bind the sealed v4")
    su = screen_units(design)
    iu = intervention_units(design)
    want_rows = [{k: v for k, v in u.items()
                  if k != "optimizer_budget"}
                 for u in su + iu]
    if led["units"] != want_rows:
        raise RunnerRefusal("pre-result ledger does not "
                            "enumerate the sealed population "
                            "exactly — every unit row (role, "
                            "generator identity, coordinates) "
                            "must match the sealed design")
    report = m4._strict_json_file(out / "RUN_REPORT.json",
                                  "run report")
    if m4._self_sha(report, "record_sha256") != \
            report["record_sha256"]:
        raise RunnerRefusal("report self-digest does not "
                            "re-derive")
    inv = _inventory(out)
    if inv != report["artifacts_sha256"]:
        raise RunnerRefusal("artifact inventory does not equal "
                            "the report exactly")
    acct = {"optimization_updates": 0, "evaluations": 0,
            "descriptor_seconds": 0.0, "descriptor_evals": 0,
            "t0": time.monotonic()}
    recs = []
    for u in su:
        p = out / "screen" / f"{_safe(u['unit_id'])}.json"
        if not p.is_file():
            raise RunnerRefusal(
                f"screen record {p.name} is MISSING — a "
                "missing or failed unit stays in the "
                "denominator and its absence refuses")
        r = m4._strict_json_file(p, f"screen record {p.name}")
        fresh = _run_screen_unit(design, u, acct)
        cmp_r = {k: v for k, v in r.items()
                 if k != "optimizer_budget"}
        cmp_f = {k: v for k, v in fresh.items()
                 if k != "optimizer_budget"}
        if cmp_f != cmp_r:
            bad = sorted(k for k in cmp_f
                         if cmp_f[k] != cmp_r.get(k))
            raise RunnerRefusal(
                f"screen unit {u['unit_id']} does not replay "
                f"(fields: {bad[:3]})")
        recs.append(r)
    tab = m4._strict_json_file(out / "LEARNABILITY_TABLE.json",
                               "learnability table")
    fresh_tab = derive_learnability_table(design, recs)
    if fresh_tab != tab:
        raise RunnerRefusal(
            "learnability table does not re-derive from the raw "
            "records under the frozen rule")
    derived_iv = {}
    for u in iu:
        p = out / "intervention" / \
            f"{_safe(u['unit_id'])}_summary.json"
        if not p.is_file():
            raise RunnerRefusal(
                f"summary {p.name} is MISSING — a missing or "
                "failed unit stays in the denominator and its "
                "absence refuses")
        r = m4._strict_json_file(p, f"summary {p.name}")
        if m4._self_sha(r, "record_sha256") != r["record_sha256"]:
            raise RunnerRefusal(f"{p.name} self-digest does not "
                                "re-derive")
        g = gb.generate("DEVELOPMENT", u["family"], u["noise"],
                        u["generator_index"])
        gb.consumer_verify(g)
        for arm, seed_tag in (("treatment", "fit_init"),
                              ("control", "ctrl_init")):
            lp = out / "intervention" / \
                f"{_safe(u['unit_id'])}_{arm}.jsonl"
            lines = lp.read_text().splitlines()
            if arm == "treatment":
                pmod = _fit(design, g, u["width"],
                            u["model_seed"])
            else:
                pmod = m4._mlp_init(
                    gb.N_IN, u["width"],
                    gb._seed(seed_tag, g["generator_id"],
                             u["width"], u["model_seed"]))
            base_loss = m4._loss(pmod, g["X_held"], g["y_held"])
            st = {"params": pmod,
                  "assoc_X": np.zeros((0, gb.N_IN)),
                  "assoc_y": np.zeros((0,)),
                  "family": f"{u['unit_id']}::{arm}",
                  "batch_index": 0, "retention_streak": 0,
                  "updates_done": 0, "accepted_batches": 0,
                  "retention_margin": base_loss * 1.10}
            endpoint = 0
            censored = False
            cause = None
            for i, line in enumerate(lines):
                rec = m4._strict_json_text(
                    line, f"{lp.name} record {i}")
                if m4._self_sha(rec, "record_sha256") != \
                        rec["record_sha256"]:
                    raise RunnerRefusal(
                        f"{lp.name} record {i} self-digest does "
                        "not re-derive")
                replay = m4.apply_batch(
                    st, g["X_train"], g["y_train"],
                    g["X_held"], g["y_held"])
                claimed = {k: rec[k] for k in rec
                           if k != "record_sha256"}
                if replay != claimed:
                    bad = sorted(k for k in replay
                                 if replay[k] != claimed.get(k))
                    raise RunnerRefusal(
                        f"{lp.name} batch {i} does not replay "
                        f"(fields: {bad[:3]})")
                if replay["outcome"] == "ACCEPTED":
                    endpoint = replay["cumulative_associations"]
                else:
                    cause = replay["outcome"]
            if cause is None:
                censored = True
                cause = ("MAX_BATCHES" if len(lines) == 64
                         else r["arms"][arm]["stopping_cause"])
                if cause not in ("MAX_BATCHES", "WALL_STOP",
                                 "RSS_STOP", "STOP_REQUESTED"):
                    raise RunnerRefusal(
                        f"{lp.name}: censoring cause does not "
                        "reconstruct")
            a = r["arms"][arm]
            if a["endpoint_last_passing"] != endpoint or \
                    a["censored"] is not censored or \
                    a["stopping_cause"] != cause or \
                    a["updates_done"] != st["updates_done"]:
                raise RunnerRefusal(
                    f"{u['unit_id']}/{arm}: endpoint/censoring "
                    "facts do not equal the replayed derivation")
            derived_iv[f"{u['unit_id']}::{arm}"] = endpoint
        want_diff = (derived_iv[f"{u['unit_id']}::treatment"]
                     - derived_iv[f"{u['unit_id']}::control"])
        if r["paired_difference_endpoint"] != want_diff:
            raise RunnerRefusal(
                f"{u['unit_id']}: paired difference does not "
                "re-derive")
        if r["diagnostic_examples_per_update"] != m4.MINIBATCH:
            raise RunnerRefusal(
                "diagnostic does not match the primary example "
                "count")
    # C22/C23 kill 9: when the report covers ONE complete
    # session, its optimization/evaluation accounting must equal
    # the amount the verifier itself re-derived by replaying the
    # sealed population — a forged (or omitted) cost refuses.
    if report.get("session_complete") is True:
        fitu = design["learnability_gate"]["optimizer"]["updates"]
        expected_opt = len(su) * fitu
        expected_evals = acct["evaluations"]
        for u in iu:
            p_ = out / "intervention" / \
                f"{_safe(u['unit_id'])}_summary.json"
            r_ = m4._strict_json_file(p_, "summary")
            expected_opt += (2 * fitu
                             + r_["arms"]["treatment"]
                             ["updates_done"]
                             + r_["arms"]["control"]
                             ["updates_done"]
                             + m4.UPDATES_PER_BATCH)
            expected_evals += (r_["arms"]["treatment"]
                               ["updates_done"]
                               // m4.UPDATES_PER_BATCH
                               + r_["arms"]["control"]
                               ["updates_done"]
                               // m4.UPDATES_PER_BATCH + 1)
        ra = report["accounting"]
        if ra["optimization_updates"] != expected_opt:
            raise RunnerRefusal(
                "reported optimization accounting does not "
                "equal the replayed derivation")
        if ra["descriptor_evals"] != 2 * len(iu) or \
                type(ra["descriptor_seconds"]) is bool or \
                ra["descriptor_seconds"] < 0:
            raise RunnerRefusal(
                "descriptor cost accounting is missing or "
                "impossible — descriptor costs are part of the "
                "predictive comparison")
    return {"verified": True,
            "screen_units": len(su),
            "intervention_units": len(iu),
            "derived_intervention_endpoints": derived_iv,
            "learnability_margin": tab["margin"]}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    design = dz.load_design_v4(a.design)
    if a.plan:
        print(json.dumps(plan(design), indent=1))
        return 0
    if a.execute:
        r = execute(design, a.out)
        print(json.dumps({k: r[k] for k in
                          ("units_total",
                           "units_new_this_session",
                           "units_resumed_verified",
                           "accounting", "wall_seconds")},
                         indent=1))
        return 0
    if a.verify:
        v = verify_run(design, a.out)
        print(json.dumps(v, indent=1))
        return 0
    raise RunnerRefusal("choose --plan, --execute or --verify")


if __name__ == "__main__":
    raise SystemExit(main())
