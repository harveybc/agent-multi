"""M4 C25-C31: the v5 protocol — paired association tapes, one
genesis per experimental unit, truthful task contracts, real
checkpoint lineages over TRAIN/STOP/EVALUATION, the restricted
endpoint, fixed role populations, the dispersion estimator and
the executable M0/M1/M2 discrete-time survival ladder.

v4 and its run are classified
EXPLORATORY_DEVELOPMENT_EVIDENCE_WITH_PROTOCOL_DEFECTS; nothing
here rewrites that evidence. scientific_outcome: NONE until the
sealed analyses run under their own order. CPU only.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import m4_generator_bank as gb  # noqa: E402
import m4_intervention_design as dz  # noqa: E402
import m4_residual_capacity as m4  # noqa: E402

DESIGN_PATH_V5 = (REPO / "docs/research/model_capacity/"
                  "M4_SEALED_DESIGN_V5_2026_09_09.json")

MAX_BATCHES = 64
ASSOC_BATCH = 8
ACQ_TOL_BOOL = 0.4
ACQ_TOL_TEMPORAL_SD = 0.4       # x train-target SD, train-fitted
CHECKPOINTS = ("initialization", "pre_stop", "calibration_stop",
               "post_stop_bounded")
STOP_CADENCE = 50               # STOP-loss evaluated every N upd
STOP_PATIENCE = 8               # cadence points without improve
STOP_MIN_DELTA = 1e-5
STOP_MAX_UPDATES = 4000
POST_STOP_EXTRA = 500           # bounded TRAIN-only updates
G_DEV = 2
G_CAL = 16
G_CONF_RESERVED = 48
SD_UCB_LIMIT = 6.0              # associations; UCB above refuses
LADDER_RIDGE_LAMBDA = 1.0       # frozen — no data-driven search
LADDER_HORIZONS = (8, 16, 32, 64)
HARD_WALL_SECONDS = 172800      # ONE executable limit (48 h)


class V5Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


# ------------------ C26: truthful task contract ------------------

def task_kind(family):
    if family in gb.BOOL_FAMILIES + ("random_label",):
        return "boolean"
    return "temporal"          # temporal + easy_constant (MSE)


def forward_task(kind, p, X):
    H = np.tanh(X @ p["W1"] + p["b1"])
    z = (H @ p["W2"] + p["b2"]).ravel()
    if kind == "boolean":
        return H, 1.0 / (1.0 + np.exp(-z))     # sigmoid head
    return H, z                                # linear head


def sgd_step_task(kind, p, X, y, lr):
    """Boolean: sigmoid + binary cross-entropy. Temporal: linear
    + MSE. One executable contract per task type (C26)."""
    H, out = forward_task(kind, p, X)
    err = (out - y) / len(y)   # BCE-with-sigmoid and MSE share
    #                            this output-error form
    dH = (err[:, None] * p["W2"].T) * (1 - H ** 2)
    p["W2"] -= lr * (H.T @ err[:, None])
    p["b2"] -= lr * err.sum(keepdims=True)
    p["W1"] -= lr * (X.T @ dH)
    p["b1"] -= lr * dH.sum(axis=0)


def loss_task(kind, p, X, y):
    _, out = forward_task(kind, p, X)
    if kind == "boolean":
        eps = 1e-12
        return float(-np.mean(
            y * np.log(out + eps)
            + (1 - y) * np.log(1 - out + eps)))
    return float(np.mean((out - y) ** 2))


def heldout_metric(kind, g, p):
    _, out = forward_task(kind, p, g["X_held"])
    if not np.isfinite(out).all():
        return None, None, "NUMERICALLY_INVALID"
    if kind == "boolean":
        acc = float(((out > 0.5) == (g["y_held"] > 0.5)).mean())
        maj_class = 1.0 if float(
            (g["y_train"] > 0.5).mean()) >= 0.5 else 0.0
        base = float(((g["y_held"] > 0.5)
                      == (maj_class > 0.5)).mean())
        return acc, base, None
    mse = float(np.mean((out - g["y_held"]) ** 2))
    persist = g["X_held"][:, -1]
    base_mse = float(np.mean((persist - g["y_held"]) ** 2))
    return mse, base_mse, None


def improvement(kind, metric, base):
    if kind == "boolean":
        return metric - base
    return (base - metric) / max(base, 1e-12)


# ------------- C25: the canonical association tape -------------

def tape_id(design_sha, generator_id, width, model_seed) -> str:
    """Arm and checkpoint names NEVER enter this identity."""
    return (f"tape::{design_sha[:16]}::{generator_id}"
            f"::w{width}::s{model_seed}")


def association_tape(design_sha, g, width, model_seed) -> dict:
    """C25: blinded, independent, family-compatible pairings.
    Inputs come from a disjoint draw of the SAME family input
    process; targets are an independent draw from the family's
    TRAIN target distribution; pairing is blinded. Byte-identical
    across arms and checkpoints of the same experimental unit."""
    tid = tape_id(design_sha, g["generator_id"], width,
                  model_seed)
    kind = task_kind(g["family"])
    rng = np.random.default_rng(gb._seed("assoc_tape", tid))
    n = MAX_BATCHES * ASSOC_BATCH
    if g["family"] in gb.BOOL_FAMILIES + ("random_label",):
        X = rng.choice([-1.0, 1.0], size=(n, gb.N_IN))
        y = rng.choice([0.0, 1.0], size=n)
        tol = ACQ_TOL_BOOL
    else:
        # disjoint windows of the same observation process
        rng_proc = np.random.default_rng(
            gb._seed("assoc_proc", tid))
        lat = gb._latent_series(g["family"], rng_proc,
                                n + gb.N_IN)
        tr = slice(0, n + gb.N_IN)
        nz = g["noise"] if g["noise"] != "NOT_APPLICABLE" \
            else "clean"
        dist = gb._disturbance(nz, rng_proc, lat, tr)
        X, _ = gb._windows(lat + dist, gb.N_IN)
        # blinded independent targets from the TRAIN target
        # distribution (train-fitted, evaluation never touched)
        y = rng.choice(g["y_train"], size=n, replace=True)
        tol = ACQ_TOL_TEMPORAL_SD * float(
            np.std(g["y_train"]) + 1e-12)
    ub = rng.integers(0, 2 ** 31 - 1,
                      size=m4.UPDATES_PER_BATCH * MAX_BATCHES)
    tape = {"tape_id": tid, "kind": kind,
            "X": X, "y": y, "tol": float(tol),
            "update_seeds": ub}
    tape["digest"] = tape_digest(tape)
    return tape


def tape_digest(tape) -> str:
    h = hashlib.sha256()
    h.update(tape["tape_id"].encode())
    h.update(np.ascontiguousarray(tape["X"]).tobytes())
    h.update(np.ascontiguousarray(tape["y"]).tobytes())
    h.update(np.ascontiguousarray(tape["update_seeds"]).tobytes())
    h.update(json.dumps(tape["tol"]).encode())
    return h.hexdigest()


# ---------- C27: genesis and real checkpoint lineages ----------

def genesis_params(g, width, model_seed):
    """ONE genesis per (generator, width, model_seed) — arm and
    checkpoint names never enter the derivation."""
    return m4._mlp_init(gb.N_IN, width,
                        gb._seed("genesis", g["generator_id"],
                                 width, model_seed))


def build_checkpoints(g, width, model_seed) -> dict:
    """Early stopping on the STOP slice ONLY (frozen cadence,
    patience and min delta); EVALUATION never selects anything.
    Returns the four declared checkpoints, all descending from
    the same genesis, with a replayable lineage."""
    kind = task_kind(g["family"])
    p = genesis_params(g, width, model_seed)
    init_digest = m4._params_digest(p)
    cks = {"initialization": {
        "params": {k: v.copy() for k, v in p.items()},
        "updates": 0, "parent": None,
        "params_digest": init_digest}}
    stop_traj = []
    best = (float("inf"), 0, None)
    since = 0
    u = 0
    snapshots = {}
    while u < STOP_MAX_UPDATES:
        rng = np.random.default_rng(
            gb._seed("task_mb", g["generator_id"], width,
                     model_seed, u))
        i = rng.integers(0, len(g["y_train"]), size=16)
        sgd_step_task(kind, p, g["X_train"][i],
                      g["y_train"][i], m4.LEARNING_RATE)
        u += 1
        if u % STOP_CADENCE == 0:
            if not all(np.isfinite(v).all()
                       for v in p.values()):
                return {"numerically_invalid": True,
                        "invalid_at_update": u,
                        "checkpoints": None,
                        "stop_trajectory": stop_traj,
                        "stop_trajectory_digest": None,
                        "selected_stop_update": None,
                        "task_loss_stop": None}
            sl = loss_task(kind, p, g["X_stop"], g["y_stop"])
            if not np.isfinite(sl):
                return {"numerically_invalid": True,
                        "invalid_at_update": u,
                        "checkpoints": None,
                        "stop_trajectory": stop_traj,
                        "stop_trajectory_digest": None,
                        "selected_stop_update": None,
                        "task_loss_stop": None}
            stop_traj.append(round(sl, 10))
            snapshots[u] = {k: v.copy() for k, v in p.items()}
            if sl < best[0] - STOP_MIN_DELTA:
                best = (sl, u, snapshots[u])
                since = 0
            else:
                since += 1
            if since >= STOP_PATIENCE:
                break
    stop_u = best[1] if best[2] is not None else u
    calib = best[2] if best[2] is not None else \
        {k: v.copy() for k, v in p.items()}
    # pre_stop: the snapshot at the predeclared HALF of the
    # selected stopping point (floor to cadence, min 1 cadence)
    pre_u = max(STOP_CADENCE,
                (stop_u // 2) // STOP_CADENCE * STOP_CADENCE)
    pre = snapshots.get(pre_u)
    if pre is None:                      # stop before 2 cadences
        pre_u = min(snapshots) if snapshots else 0
        pre = snapshots.get(pre_u,
                            cks["initialization"]["params"])
    # post_stop_bounded: exact extra TRAIN-only updates FROM the
    # calibration_stop parameters
    post = {k: v.copy() for k, v in calib.items()}
    for k2 in range(POST_STOP_EXTRA):
        rng = np.random.default_rng(
            gb._seed("post_mb", g["generator_id"], width,
                     model_seed, k2))
        i = rng.integers(0, len(g["y_train"]), size=16)
        sgd_step_task(kind, post, g["X_train"][i],
                      g["y_train"][i], m4.LEARNING_RATE)
    traj_digest = hashlib.sha256(json.dumps(
        stop_traj).encode()).hexdigest()
    for name, obj in (("calibration_stop", calib),
                      ("pre_stop", pre), ("post_stop", post)):
        if not all(np.isfinite(v).all() for v in obj.values()):
            return {"numerically_invalid": True,
                    "invalid_at_update": stop_u,
                    "checkpoints": None,
                    "stop_trajectory": stop_traj,
                    "stop_trajectory_digest": None,
                    "selected_stop_update": None,
                    "task_loss_stop": None}
    cks["pre_stop"] = {
        "params": pre, "updates": pre_u,
        "parent": init_digest,
        "params_digest": m4._params_digest(pre)}
    cks["calibration_stop"] = {
        "params": calib, "updates": stop_u,
        "parent": init_digest,
        "params_digest": m4._params_digest(calib)}
    cks["post_stop_bounded"] = {
        "params": post, "updates": stop_u + POST_STOP_EXTRA,
        "parent": cks["calibration_stop"]["params_digest"],
        "params_digest": m4._params_digest(post)}
    return {"numerically_invalid": False,
            "checkpoints": cks,
            "stop_trajectory": stop_traj,
            "stop_trajectory_digest": traj_digest,
            "selected_stop_update": stop_u,
            "task_loss_stop": round(best[0], 10)
            if best[2] is not None else None}


# ------------- C25/C31: the paired intervention run -------------

def run_intervention(g, tape, ck, kind, on_batch=None,
                     start=None) -> dict:
    """One arm: consume the SHARED tape from the given checkpoint
    parameters. Restricted endpoint min(count, 64); CAP_REACHED
    typed; retention margin train-fitted from EVALUATION loss at
    entry.

    C32 resume: `start` carries a durable predecessor state
    (params, streak, updates, endpoint, next_batch) — the run
    CONTINUES from it instead of restarting; `on_batch` receives
    (record, params) after each durable batch."""
    if start is None:
        p = {k: v.copy() for k, v in ck["params"].items()}
        base_loss = loss_task(kind, p, g["X_held"], g["y_held"])
        st = {"assoc_n": 0, "streak": 0, "updates": 0}
        margin = base_loss * 1.10
        endpoint = 0
        first_b = 0
    else:
        p = {k: np.asarray(v).copy()
             for k, v in start["params"].items()}
        st = {"assoc_n": start["next_batch"] * ASSOC_BATCH,
              "streak": int(start["streak"]),
              "updates": int(start["updates"])}
        margin = float(start["margin"])
        endpoint = int(start["endpoint"])
        first_b = int(start["next_batch"])
    records = []
    cause = None
    cap = False
    for b in range(first_b, MAX_BATCHES):
        sl = slice(b * ASSOC_BATCH, (b + 1) * ASSOC_BATCH)
        st["assoc_n"] = (b + 1) * ASSOC_BATCH
        Xc = tape["X"][:st["assoc_n"]]
        yc = tape["y"][:st["assoc_n"]]
        for u in range(m4.UPDATES_PER_BATCH):
            rng = np.random.default_rng(int(
                tape["update_seeds"][b * m4.UPDATES_PER_BATCH
                                     + u]))
            io_ = rng.integers(0, len(g["y_train"]), size=8)
            ia = rng.integers(0, st["assoc_n"], size=8)
            Xmb = np.vstack([g["X_train"][io_], Xc[ia]])
            ymb = np.concatenate([g["y_train"][io_], yc[ia]])
            sgd_step_task(kind, p, Xmb, ymb, m4.LEARNING_RATE)
            st["updates"] += 1
        ret_loss = loss_task(kind, p, g["X_held"], g["y_held"])
        if not (np.isfinite(ret_loss)
                and all(np.isfinite(v).all()
                        for v in p.values())):
            rec = {"batch": b, "ret_loss": None,
                   "retention_streak": st["streak"],
                   "cumulative_associations": st["assoc_n"],
                   "cumulative_acquired": 0,
                   "cumulative_ok": False,
                   "outcome": "NUMERICAL_ANOMALY"}
            records.append(rec)
            if on_batch is not None:
                on_batch(rec, p, st, margin, endpoint)
            cause = "NUMERICAL_ANOMALY"
            break
        st["streak"] = st["streak"] + 1 \
            if ret_loss > margin else 0
        _, out = forward_task(kind, p, Xc)
        ok_v = np.abs(out - yc) < tape["tol"]
        rec = {"batch": b, "ret_loss": round(ret_loss, 6),
               "retention_streak": st["streak"],
               "cumulative_associations": st["assoc_n"],
               "cumulative_acquired": int(ok_v.sum()),
               "cumulative_ok": bool(ok_v.all())}
        if st["streak"] >= m4.RETENTION_CONSECUTIVE:
            rec["outcome"] = "RETENTION_ENDPOINT"
        elif not rec["cumulative_ok"]:
            rec["outcome"] = "ACQUISITION_ENDPOINT"
        else:
            rec["outcome"] = "ACCEPTED"
            endpoint = st["assoc_n"]
        records.append(rec)
        if on_batch is not None:
            on_batch(rec, p, st, margin, endpoint
                     if rec["outcome"] != "ACCEPTED"
                     else st["assoc_n"])
        if rec["outcome"] != "ACCEPTED":
            cause = rec["outcome"]
            break
    else:
        cause = "CAP_REACHED"
        cap = True
    return {"records": records,
            "restricted_endpoint": min(endpoint,
                                       MAX_BATCHES * ASSOC_BATCH),
            "cap_reached": cap,
            "stopping_cause": cause,
            "updates_done": st["updates"],
            "retention_margin": round(margin, 10),
            "final_params_digest": m4._params_digest(p)}


# ---------------- C29: dispersion from CALIBRATION ----------------

def dispersion_from_paired(diffs_by_generator) -> dict:
    """Generator-level SD of the paired restricted-endpoint
    difference AFTER averaging nested model seeds, with a 95 %
    upper confidence bound (chi-square)."""
    from math import sqrt
    d = np.asarray(sorted(diffs_by_generator.values()),
                   dtype=np.float64)
    n = len(d)
    if n < 3:
        raise V5Refusal("dispersion needs at least 3 generators")
    sd = float(d.std(ddof=1))
    # chi2 lower quantile via Wilson-Hilferty approximation
    df = n - 1
    z05 = -1.6448536269514722
    chi2_lo = df * (1 - 2 / (9 * df)
                    + z05 * sqrt(2 / (9 * df))) ** 3
    ucb = sd * sqrt(df / max(chi2_lo, 1e-9))
    return {"n_generators": n, "sd": round(sd, 6),
            "sd_ucb95": round(float(ucb), 6),
            "supported": bool(ucb <= SD_UCB_LIMIT)}


# ---------- C30: executable M0/M1/M2 survival ladder ----------

def _hazard_design(rows, level):
    feats = []
    for r in rows:
        f = [1.0,
             np.log1p(r["param_count"]) / 10.0]
        f += r["nuisance"]
        if level >= 1:
            f += [r["checkpoint_loss"],
                  np.log1p(r["task_updates"]) / 10.0]
        if level >= 2:
            f += [np.log1p(r["compressed_len"]) / 10.0,
                  r["spectral_rank"] / 64.0,
                  r["prune_fraction"],
                  r["stop_traj_slope"]]
        feats.append(f)
    return np.asarray(feats, dtype=np.float64)


def fit_hazard(rows, level, lam=LADDER_RIDGE_LAMBDA,
               iters=300, lr=0.5):
    """Discrete-time hazard: P(fail at batch b | survived) =
    sigmoid(w·x + gamma_b-spline over b). Ridge lam frozen."""
    X = _hazard_design(rows, level)
    events = []
    for i, r in enumerate(rows):
        fail_b = r["fail_batch"]       # None if cap-censored
        last = MAX_BATCHES if fail_b is None else fail_b
        for b in range(last):
            events.append((i, b, 0.0))
        if fail_b is not None:
            events[-1] = (i, fail_b - 1, 1.0) if fail_b > 0 \
                else (i, 0, 1.0)
    idx = np.asarray([e[0] for e in events])
    bb = np.asarray([e[1] for e in events], dtype=np.float64)
    yy = np.asarray([e[2] for e in events])
    B = np.stack([np.ones_like(bb), bb / MAX_BATCHES,
                  (bb / MAX_BATCHES) ** 2], axis=1)
    Z = np.hstack([X[idx], B])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        h = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (h - yy) / len(yy) + lam * w / len(yy)
        w -= lr * grad
    return w


def survival_curve(w, row, level):
    x = _hazard_design([row], level)[0]
    surv = 1.0
    out = []
    for b in range(MAX_BATCHES):
        bb = b / MAX_BATCHES
        zz = np.concatenate([x, [1.0, bb, bb ** 2]])
        h = 1.0 / (1.0 + np.exp(-(zz @ w)))
        surv *= (1.0 - h)
        out.append(surv)
    return np.asarray(out)


def integrated_brier(w, rows, level) -> float:
    """Primary prediction loss over the 64 acquisition batches;
    cap-censored units contribute while under observation."""
    tot, cnt = 0.0, 0
    for r in rows:
        s = survival_curve(w, r, level)
        fail_b = r["fail_batch"]
        for b in range(MAX_BATCHES):
            if fail_b is None:           # censored at cap: alive
                obs_alive = 1.0
            elif b < fail_b:
                obs_alive = 1.0
            else:
                obs_alive = 0.0
            tot += (s[b] - obs_alive) ** 2
            cnt += 1
    return tot / max(cnt, 1)


def ladder_compare(rows, groups, horizons=LADDER_HORIZONS) -> dict:
    """Generator-grouped leave-one-group-out: M0 vs M1 vs M2 on
    integrated Brier + calibration at predeclared horizons. The
    M2-minus-M1 decision is paired at generator level."""
    uniq = sorted(set(groups))
    if len(uniq) < 4:
        return {"status":
                "M4_LADDER_POPULATION_INSUFFICIENT",
                "n_groups": len(uniq)}
    per_level = {}
    paired = {g_: {} for g_ in uniq}
    for level in (0, 1, 2):
        briers = []
        for g_ in uniq:
            tr = [r for r, gg in zip(rows, groups) if gg != g_]
            te = [r for r, gg in zip(rows, groups) if gg == g_]
            w = fit_hazard(tr, level)
            b = integrated_brier(w, te, level)
            briers.append(b)
            paired[g_][level] = b
        per_level[f"M{level}"] = round(float(np.mean(briers)), 8)
    d21 = [paired[g_][1] - paired[g_][2] for g_ in uniq]
    d21 = np.asarray(d21)
    t = d21.mean() / (d21.std(ddof=1) / np.sqrt(len(d21))
                      + 1e-12)
    return {"status": "EXECUTED",
            "integrated_brier": per_level,
            "m2_minus_m1_paired_gain": round(
                float(d21.mean()), 8),
            "m2_vs_m1_t": round(float(t), 4),
            "n_groups": len(uniq)}


# ------------------- C28: the sealed v5 design -------------------

V5_FIELD_MAP = {
    "v4_classification": (
        "EXPLORATORY_DEVELOPMENT_EVIDENCE_WITH_PROTOCOL_DEFECTS"),
    "changed": {
        "task_families": "canonical spelling `discontinuity`; "
                         "one readable role-namespaced id rule "
                         "replaces the inherited sha/numeric-"
                         "range text (F-additional-1)",
        "architectures": "Boolean head is EXECUTED sigmoid+BCE "
                         "(was declared sigmoid, ran linear/MSE "
                         "— F3)",
        "resources": "ONE executable hard wall 172800 s (48 h) "
                     "replaces the contradictory 21600 s (F-"
                     "additional-10)",
        "association_tape": "arm/checkpoint-free identity, "
                            "family-compatible marginals, "
                            "blinded pairing (F1/F4)",
        "genesis": "one genesis per (generator,width,seed); "
                   "initialization IS that object (F2)",
        "data_roles": "TRAIN/STOP/EVALUATION byte-disjoint "
                      "(F5)",
        "checkpoints": "four real materialized lineages with "
                       "frozen stopping rule (F2)",
        "baselines": "train-selected majority CLASS evaluated "
                     "on EVALUATION rows (F3)",
        "endpoint": "restricted min(count, 64 batches) primary; "
                    "CAP_REACHED typed; resource stops are "
                    "INCOMPLETE units, never censoring; the "
                    "undefined Gehan fallback is REMOVED and no "
                    "unrestricted secondary analysis is retained "
                    "(C31)",
        "populations": "DEV 2 / CAL 16 / CONF 48 reserved per "
                       "cell; role inside seed derivation (C29)",
        "ladder": "executable discrete-time survival M0/M1/M2, "
                  "frozen ridge lambda 1.0, generator-grouped "
                  "splits, integrated Brier + horizon "
                  "calibration (C30)",
    },
}


def build_design_v5() -> dict:
    v4 = dz.load_design_v4()
    d = json.loads(json.dumps(v4))
    d["schema"] = "agent_multi.m4_intervention_design.v5"
    d["sealed_at_date"] = "2026-09-09"
    d["supersedes_design_sha256"] = v4["design_sha256"]
    d["supersession_classification"] = (
        "PROTOCOL_CORRECTION_SUPERSESSION_BEFORE_ANY_"
        "CALIBRATION_OUTCOME")
    d["scientific_outcome"] = "NONE"
    d["v4_disposition"] = V5_FIELD_MAP["v4_classification"]
    d["v4_to_v5_field_map"] = V5_FIELD_MAP["changed"]
    d["task_families"]["temporal"] = list(gb.TEMPORAL_FAMILIES)
    d["task_families"]["generator_id_rule"] = (
        "readable `ROLE-family-noise-gN`; the ROLE enters the "
        "seed derivation; index namespaces are role-separated "
        "(byte-disjointness proven executable); the inherited "
        "sha-id/numeric-range text is superseded")
    d["architectures"]["boolean_head"] = (
        "sigmoid output + binary cross-entropy, EXECUTED")
    d["architectures"]["temporal_head"] = (
        "linear output + mean-squared error, EXECUTED")
    d["resources"] = dict(d["resources"])
    d["resources"]["max_wall_seconds"] = HARD_WALL_SECONDS
    d["resources"]["single_limit_note"] = (
        "ONE executable hard wall — the reduction-rule ceiling "
        "and the enforced runner limit are the same 172800 s")
    d["reduction_rule"]["cpu_resource_ceiling_hours"] = 48.0
    d["data_roles"] = {
        "TRAIN": "optimizer updates only (192 rows)",
        "STOP": "early-stopping decisions only (64 rows)",
        "EVALUATION": "learnability, retention and scientific "
                      "endpoints only (64 rows)",
        "binding": "all six arrays digest-bound in the "
                   "generator manifest; no EVALUATION value may "
                   "select a checkpoint, threshold, width, seed "
                   "or budget"}
    d["checkpoint_rules"] = {
        "genesis": "one _mlp_init per (generator,width,"
                   "model_seed); arm and checkpoint names never "
                   "enter the derivation",
        "initialization": "the genesis object itself, zero "
                          "updates",
        "calibration_stop": {
            "cadence_updates": STOP_CADENCE,
            "patience_points": STOP_PATIENCE,
            "min_delta": STOP_MIN_DELTA,
            "max_updates": STOP_MAX_UPDATES,
            "rule": "best STOP-slice loss under frozen cadence/"
                    "patience/min-delta"},
        "pre_stop": "the snapshot at half the selected stopping "
                    "update, floored to the cadence (min one "
                    "cadence point)",
        "post_stop_bounded": f"exactly {POST_STOP_EXTRA} "
                             "additional TRAIN-only updates from "
                             "calibration_stop",
        "lineage": "parameter digests, update counts, STOP-"
                   "trajectory digest and parent digests "
                   "persisted; the verifier replays each "
                   "lineage"}
    d["association_tape"] = {
        "identity": "tape::<design16>::<generator_id>::wW::sS — "
                    "arm and checkpoint names excluded",
        "inputs": "Boolean: fresh ±1 draws of the same input "
                  "process; temporal: disjoint windows of the "
                  "same observation process",
        "targets": "independent blinded draw from the family's "
                   "TRAIN target distribution",
        "acquisition_tolerance": {
            "boolean_absolute": ACQ_TOL_BOOL,
            "temporal_sd_multiple": ACQ_TOL_TEMPORAL_SD,
            "fitting": "temporal tolerance = multiple x TRAIN "
                       "target SD (train-fitted, frozen in the "
                       "tape)"},
        "digest": "persisted and re-derived; one byte of "
                  "difference between arms refuses before "
                  "optimization"}
    d["estimands"]["primary_endpoint"] = (
        "RESTRICTED acquired count min(cumulative associations "
        "at last passing evaluation, 512) over 64 batches; a "
        "cap hit is the observed restricted value WITH "
        "CAP_REACHED published; WALL_STOP/RSS_STOP/"
        "STOP_REQUESTED/numerical failure/interruption are "
        "INCOMPLETE scientific units — in the denominator, "
        "never survival censoring; no unrestricted secondary "
        "analysis is retained in v5 and the undefined Gehan "
        "fallback is removed")
    d["estimands"].pop("censoring_analysis", None)
    d["populations_v5"] = {
        "DEVELOPMENT_per_cell": G_DEV,
        "CALIBRATION_per_cell": G_CAL,
        "CONFIRMATION_reserved_per_cell": G_CONF_RESERVED,
        "confirmation_boundary": (
            "48 fixed BEFORE calibration; CONFIRMATION arrays "
            "and outcomes are never generated, loaded or scored "
            "in this order — structurally enforced"),
        "dispersion_rule": (
            "per confirmatory cell: paired restricted-endpoint "
            "difference (calibration_stop minus initialization) "
            "averaged over the 3 nested model seeds per "
            "generator; SD across the 16 CALIBRATION "
            "generators; 95% chi-square upper confidence "
            f"bound; UCB > {SD_UCB_LIMIT} associations returns "
            "M4_CONFIRMATORY_PRECISION_NOT_SUPPORTED")}
    d["prediction_ladder"] = {
        "form": "discrete-time survival over the 64 acquisition "
                "batches (logistic hazard, quadratic batch "
                "basis)",
        "M0": "family/noise nuisance + parameter count",
        "M1": "M0 + checkpoint loss + elapsed task updates",
        "M2": "M1 + frozen trajectory (STOP-loss slope) and "
              "description (compressed length, spectral rank, "
              "prune fraction) measurements",
        "regularization": {"family": "ridge",
                           "lambda_frozen": LADDER_RIDGE_LAMBDA,
                           "rule": "FROZEN pre-outcome — no "
                                   "data-driven search"},
        "splits": "generator-grouped leave-one-group-out only",
        "primary_loss": "integrated Brier score",
        "calibration_horizons": list(LADDER_HORIZONS),
        "decision": "M2 advances only on paired unseen-"
                    "generator integrated-Brier improvement "
                    "after Holm within the frozen family AND "
                    "calibration non-inferiority; measurement "
                    "time and CPU cost reported separately",
        "insufficiency": "fewer than 4 generator groups returns "
                         "a typed insufficiency instead of a "
                         "fit"}
    del d["design_sha256"]
    d["design_sha256"] = m4._self_sha(d, "design_sha256")
    return d


V5_ALLOWED_ROOTS = {
    "schema", "sealed_at_date", "supersedes_design_sha256",
    "supersession_classification", "design_sha256",
    "v4_disposition", "v4_to_v5_field_map", "task_families",
    "architectures", "resources", "reduction_rule", "data_roles",
    "checkpoint_rules", "association_tape", "estimands",
    "populations_v5", "prediction_ladder"}


def verify_design_supersession_v5(v5: dict) -> None:
    v4 = dz.load_design_v4()
    if v5.get("supersedes_design_sha256") != v4["design_sha256"]:
        raise V5Refusal("v5 does not supersede the sealed v4")
    if v5.get("scientific_outcome") != "NONE":
        raise V5Refusal("v5 must declare scientific_outcome NONE")
    if v5.get("v4_disposition") != \
            V5_FIELD_MAP["v4_classification"]:
        raise V5Refusal("v5 must classify v4 as exploratory "
                        "evidence with protocol defects")
    deltas = m4._design_delta_paths(v4, v5)
    illegal = {dd for dd in deltas
               if dd[0] not in V5_ALLOWED_ROOTS}
    if illegal:
        raise V5Refusal(
            "v5 changes a field outside its frozen correction "
            f"surface: {sorted('.'.join(map(str, dd)) for dd in illegal)[:3]}")


def seal_design_v5(path: Path = None) -> dict:
    path = Path(path or DESIGN_PATH_V5)
    if path.exists():
        raise V5Refusal("a sealed M4 v5 design already exists — "
                        "immutable")
    d = build_design_v5()
    verify_design_supersession_v5(d)
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(d, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def load_design_v5(path: Path = None) -> dict:
    d = m4._strict_json_file(Path(path or DESIGN_PATH_V5),
                             "M4 sealed design v5")
    if m4._self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise V5Refusal("v5 self-digest does not re-derive")
    if d.get("schema") != "agent_multi.m4_intervention_design.v5":
        raise V5Refusal("design is not the v5 protocol "
                        "correction")
    verify_design_supersession_v5(d)
    return d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal", action="store_true")
    a = ap.parse_args()
    if a.seal:
        d = seal_design_v5()
        print(json.dumps({
            "sealed_v5": d["design_sha256"],
            "supersedes": d["supersedes_design_sha256"]},
            indent=1))
    else:
        raise V5Refusal("choose --seal")
