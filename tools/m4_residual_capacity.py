#!/usr/bin/env python3
"""M4: residual-capacity intervention — superseding PRE-RESULT
design v2 (order M4 C1-C8) + corrected MECHANICS_ONLY CPU
preflight.

M4-C1 (cumulative endpoint): after adding batch b, the UNION of
batches 0..b is evaluated; a batch advances the count only when
EVERY cumulative association meets the frozen per-association
acquisition criterion AND the original-task retention criterion
holds. Once an earlier association is forgotten the endpoint can
never increase. The endpoint is a conditional empirical
intervention result — never "unused bits", "remaining
intelligence" or exact Kolmogorov complexity.

M4-C2 (frozen rehearsal): every update uses balanced minibatches
— 50% original-task examples and 50% uniformly sampled (with
replacement) cumulative associations; batch size, update count,
learning rate and evaluation cadence are frozen pre-result. A
no-rehearsal arm exists as a frozen DIAGNOSTIC only.

M4-C3: retention violation = TWO CONSECUTIVE failing
evaluations at the frozen once-per-batch cadence; the streak
resets after a passing evaluation; raw losses and the streak are
both published.

M4-C4: the restart proof saves the COMPLETE training state
(parameters, cumulative-association inventory, retention streak,
counters; the update sampler is stateless by seed derivation),
reloads it in a FRESH process, applies the next sealed batch,
and requires exact equality with an uninterrupted branch under
the same state and batch.

M4-C5: wall/RSS/stop-file/heartbeat are EXECUTED per batch with
typed outcomes; the matched-compute control RUNS and its update
difference is derived, not declared.

M4-C6: the output root must be empty before anything is written;
artifacts are exclusive 0600 durable files; the report carries
FULL SHA-256 identities for every artifact and an exact
inventory; `verify_preflight()` re-reads everything with a fresh
reader, re-derives digests and reconstructs the stated facts
from the batch ledger. A replaced artifact invalidates the
report; a second invocation makes zero changes before refusing.

MECHANICS_ONLY: no M4 scientific conclusion, no confirmation
generator, no DOIN gene, no production gate."""
import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESIGN_PATH_V1 = (REPO / "docs/research/model_capacity/"
                  "M4_SEALED_DESIGN_2026_09_08.json")
DESIGN_PATH = (REPO / "docs/research/model_capacity/"
               "M4_SEALED_DESIGN_V2_2026_09_08.json")
PREFLIGHT_DIR = (REPO / "docs/audits/evidence/"
                 "m4_mechanics_preflight_v2")
DESIGN_PATH_V3 = (REPO / "docs/research/model_capacity/"
                  "M4_SEALED_DESIGN_V3_2026_09_09.json")
PREFLIGHT_DIR_V3 = (REPO / "docs/audits/evidence/"
                    "m4_mechanics_preflight_v3")
# C10: the EXTERNALLY REVIEWED v2 identity — the only v2 that may
# execute, and the only predecessor a v3 may supersede.
REVIEWED_V2_SHA = ("51ec6cfa8e80188e57f1e7a9f1163b1144e0fe304fa0"
                   "8f03ca4923c441859ed0")
MASTER_SEED_PHRASE = "m4_residual_capacity_2026_09_08"
ACQ_TOL = 0.4
REHEARSAL_FRACTION = 0.5
MINIBATCH = 16
UPDATES_PER_BATCH = 400
LEARNING_RATE = 0.05
RETENTION_CONSECUTIVE = 2


class M4Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


def _sha_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _seed(*parts) -> int:
    h = hashlib.sha256(("|".join(
        [MASTER_SEED_PHRASE, *map(str, parts)])).encode())
    return int.from_bytes(h.digest()[:8], "big")


def _strict_json_file(path: Path, what: str) -> dict:
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise M4Refusal(f"duplicate JSON key in {what}")
        return dict(pairs)
    try:
        return json.loads(
            Path(path).read_text(), object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(
                M4Refusal(f"non-finite constant in {what}")))
    except json.JSONDecodeError as exc:
        raise M4Refusal(f"{what} is not well-formed JSON "
                        f"({exc.msg})")


def _strict_json_text(text: str, what: str) -> dict:
    """C10: strict parse for embedded JSON (ledger lines):
    duplicate keys and non-finite constants refuse."""
    def _no_dupes(pairs):
        keys = [k for k, _ in pairs]
        if len(keys) != len(set(keys)):
            raise M4Refusal(f"duplicate JSON key in {what}")
        return dict(pairs)
    try:
        return json.loads(
            text, object_pairs_hook=_no_dupes,
            parse_constant=lambda c: (_ for _ in ()).throw(
                M4Refusal(f"non-finite constant in {what}")))
    except json.JSONDecodeError as exc:
        raise M4Refusal(f"{what} is not well-formed JSON "
                        f"({exc.msg})")


def _canon_sha(v, what: str) -> str:
    if type(v) is not str or len(v) != 64 or             any(c not in "0123456789abcdef" for c in v):
        raise M4Refusal(
            f"{what} is not a canonical 64-lowercase-hex digest")
    return v


def _req_int(v, what: str, lo: int = 0, hi: int = None) -> int:
    if type(v) is not int:      # bools-as-numbers refuse
        raise M4Refusal(f"{what} must be an exact integer")
    if v < lo or (hi is not None and v > hi):
        raise M4Refusal(f"{what} violates its domain")
    return v


def _req_num(v, what: str):
    if type(v) is bool or not isinstance(v, (int, float)):
        raise M4Refusal(f"{what} must be a number")
    import math as _m
    if not _m.isfinite(float(v)):
        raise M4Refusal(f"{what} is not finite")
    return float(v)


def _req_bool(v, what: str) -> bool:
    if type(v) is not bool:
        raise M4Refusal(f"{what} must be an exact boolean")
    return v


def _flatten_design(d, pre=()):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten_design(v, pre + (k,)))
    elif isinstance(d, list):
        out[pre] = json.dumps(d, sort_keys=True)
    else:
        out[pre] = json.dumps(d)
    return out


def _design_delta_paths(a: dict, b: dict) -> set:
    fa, fb = _flatten_design(a), _flatten_design(b)
    return {p for p in set(fa) | set(fb)
            if fa.get(p, "\x00absent") != fb.get(p, "\x00absent")}


# C10: the exact field-by-field supersession surfaces, frozen.
V2_SUPERSESSION_ALLOWED = {
    ("association_batches", "acquisition_criterion"),
    ("association_batches", "ambiguous_states"),
    ("association_batches", "max_updates_per_batch"),
    ("association_batches", "stop_rule"),
    ("association_batches", "typed_outcomes"),
    ("controls", "matched_compute"),
    ("cumulative_endpoint",
     "acquisition_criterion_per_association"),
    ("cumulative_endpoint", "evaluation"),
    ("design_sha256",),
    ("endpoint_semantics",),
    ("rehearsal_rule", "evaluation_cadence"),
    ("rehearsal_rule", "learning_rate"),
    ("rehearsal_rule", "minibatch_size"),
    ("rehearsal_rule", "no_rehearsal_arm", "multiplicity"),
    ("rehearsal_rule", "no_rehearsal_arm", "role"),
    ("rehearsal_rule", "primary"),
    ("rehearsal_rule", "rehearsal_fraction"),
    ("rehearsal_rule", "updates_per_batch"),
    ("retention", "violation_rule"),
    ("schema",),
    ("supersedes_design_sha256",),
    ("supersession_classification",),
}
V3_SUPERSESSION_ALLOWED = {
    ("schema",),
    ("sealed_at_date",),
    ("supersedes_design_sha256",),
    ("supersession_classification",),
    ("scientific_outcome",),
    ("rehearsal_rule", "no_rehearsal_arm", "matched_minibatch"),
    ("design_sha256",),
}
V3_MATCHED_MINIBATCH_RULE = (
    "the no-rehearsal diagnostic processes the SAME frozen "
    "minibatch size AND the SAME number of examples per update "
    "as the primary arm (16 associations per update, no "
    "original-task examples); frozen before any intervention "
    "outcome")


def verify_design_supersession_v2(v2: dict) -> None:
    """C10: the v2 supersession is validated FIELD BY FIELD
    against the immutable v1 — every delta must lie on the frozen
    supersession surface and the predecessor binding must hold."""
    v1 = _strict_json_file(DESIGN_PATH_V1, "M4 design v1")
    if _self_sha(v1, "design_sha256") != v1.get("design_sha256"):
        raise M4Refusal("v1 self-digest does not re-derive")
    if v2.get("supersedes_design_sha256") != v1["design_sha256"]:
        raise M4Refusal(
            "v2 does not supersede the immutable v1 identity")
    deltas = _design_delta_paths(v1, v2)
    illegal = {d for d in deltas
               if d not in V2_SUPERSESSION_ALLOWED}
    if illegal:
        raise M4Refusal(
            "v2 supersession carries a delta outside the frozen "
            f"surface: {sorted('.'.join(map(str, d)) for d in illegal)[:3]}")


def build_design_v3() -> dict:
    """C14: the matched no-rehearsal diagnostic — a superseding
    v3 whose ONLY change is the matched-minibatch rule; declared
    scientific_outcome NONE (no intervention outcome existed)."""
    v2 = _strict_json_file(DESIGN_PATH, "M4 sealed design v2")
    if _self_sha(v2, "design_sha256") != v2.get("design_sha256"):
        raise M4Refusal("v2 self-digest does not re-derive")
    if v2["design_sha256"] != REVIEWED_V2_SHA:
        raise M4Refusal("v2 on disk is not the reviewed identity")
    verify_design_supersession_v2(v2)
    d = json.loads(json.dumps(v2))
    d["schema"] = "agent_multi.m4_residual_capacity_design.v3"
    d["sealed_at_date"] = "2026-09-09"
    d["supersedes_design_sha256"] = REVIEWED_V2_SHA
    d["supersession_classification"] = (
        "MATCHED_COMPUTE_DIAGNOSTIC_CORRECTION_BEFORE_ANY_"
        "INTERVENTION_OUTCOME")
    d["scientific_outcome"] = "NONE"
    d["rehearsal_rule"]["no_rehearsal_arm"][
        "matched_minibatch"] = V3_MATCHED_MINIBATCH_RULE
    del d["design_sha256"]
    d["design_sha256"] = _self_sha(d, "design_sha256")
    return d


def seal_design_v3(path: Path = None) -> dict:
    path = Path(path or DESIGN_PATH_V3)
    if path.exists():
        raise M4Refusal("a sealed M4 v3 design already exists — "
                        "immutable")
    d = build_design_v3()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(d, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def verify_design_supersession_v3(v3: dict) -> None:
    """C10/C14: v3 must supersede EXACTLY the reviewed v2, carry
    scientific_outcome NONE, state the frozen matched-minibatch
    rule, and differ from v2 ONLY on the frozen v3 surface."""
    v2 = _strict_json_file(DESIGN_PATH, "M4 sealed design v2")
    if _self_sha(v2, "design_sha256") != v2.get("design_sha256"):
        raise M4Refusal("v2 self-digest does not re-derive")
    if v2["design_sha256"] != REVIEWED_V2_SHA:
        raise M4Refusal("v2 on disk is not the reviewed identity")
    verify_design_supersession_v2(v2)
    if v3.get("supersedes_design_sha256") != REVIEWED_V2_SHA:
        raise M4Refusal(
            "v3 does not supersede the REVIEWED v2 identity")
    if v3.get("scientific_outcome") != "NONE":
        raise M4Refusal(
            "v3 must declare scientific_outcome NONE — no "
            "intervention outcome may inform a supersession")
    if v3.get("rehearsal_rule", {}).get(
            "no_rehearsal_arm", {}).get("matched_minibatch") !=             V3_MATCHED_MINIBATCH_RULE:
        raise M4Refusal(
            "v3 does not state the exact frozen "
            "matched-minibatch rule")
    deltas = _design_delta_paths(v2, v3)
    illegal = {d for d in deltas
               if d not in V3_SUPERSESSION_ALLOWED}
    if illegal:
        raise M4Refusal(
            "v3 supersession carries a delta outside the frozen "
            f"surface: {sorted('.'.join(map(str, d)) for d in illegal)[:3]}")


def build_design_v2() -> dict:
    v1 = _strict_json_file(DESIGN_PATH_V1, "M4 design v1")
    d = {
        "schema": "agent_multi.m4_residual_capacity_design.v2",
        "sealed_at_date": "2026-09-08",
        "supersedes_design_sha256": v1["design_sha256"],
        "supersession_classification": (
            "CUMULATIVE_ENDPOINT_AND_MECHANICS_SUPERSESSION_"
            "BEFORE_ANY_INTERVENTION_OUTCOME"),
        "stage": "C2_C3_residual_capacity_intervention",
        "question": v1["question"],
        "endpoint_semantics": (
            "the count of random associations that remain "
            "JOINTLY acquired under the frozen retention margin "
            "— a conditional empirical intervention result under "
            "this exact protocol; NEVER 'unused bits', "
            "'remaining intelligence' or exact Kolmogorov "
            "complexity; sequential acquisition throughput is "
            "NOT the endpoint"),
        "cumulative_endpoint": {
            "evaluation": (
                "after adding batch b, evaluate the UNION of "
                "batches 0..b; the endpoint advances only when "
                "EVERY cumulative association meets the frozen "
                "per-association acquisition criterion AND the "
                "original-task retention criterion holds; once "
                "an earlier association is forgotten the "
                "endpoint cannot increase"),
            "acquisition_criterion_per_association":
                f"|model_output - label| < {ACQ_TOL}",
        },
        "rehearsal_rule": {
            "primary": (
                "balanced minibatches: 50% original-task "
                "examples and 50% uniformly sampled (with "
                "replacement) cumulative associations"),
            "rehearsal_fraction": REHEARSAL_FRACTION,
            "minibatch_size": MINIBATCH,
            "updates_per_batch": UPDATES_PER_BATCH,
            "learning_rate": LEARNING_RATE,
            "evaluation_cadence": (
                "once per batch: retention AND full cumulative "
                "acquisition"),
            "no_rehearsal_arm": {
                "role": "frozen DIAGNOSTIC only",
                "multiplicity": (
                    "descriptive; never part of the "
                    "confirmatory family; frozen now — the "
                    "better policy is never chosen after seeing "
                    "intervention outcomes")}},
        "retention": {
            "metric": v1["retention"]["metric"],
            "margin_rule": v1["retention"]["margin_rule"],
            "violation_rule": (
                f"{RETENTION_CONSECUTIVE} CONSECUTIVE failing "
                "evaluations at the frozen cadence; the streak "
                "resets to zero after a passing evaluation; raw "
                "losses and the streak are both published"),
            "frozen_before_outcomes": True},
        "task_families": v1["task_families"],
        "architectures": v1["architectures"],
        "optimizer_capability_controls":
            v1["optimizer_capability_controls"],
        "checkpoints": v1["checkpoints"],
        "association_batches": {
            "blinding": v1["association_batches"]["blinding"],
            "batch_size": 8,
            "max_batches": 64,
            "typed_outcomes": [
                "RETENTION_ENDPOINT", "ACQUISITION_ENDPOINT",
                "MAX_BATCHES", "WALL_STOP", "RSS_STOP",
                "STOP_REQUESTED", "NUMERICAL_ANOMALY",
                "OPTIMIZATION_LIMITED_ACQUISITION"]},
        "controls": {
            "random_initialization": v1["controls"][
                "random_initialization"],
            "matched_compute": (
                "the control EXECUTES the identical batch/update "
                "sequence; its update-count difference is "
                "derived from both ledgers and must be within "
                "1%")},
        "statistics": v1["statistics"],
        "measurements": v1["measurements"],
        "resources": v1["resources"],
        "seeds": v1["seeds"],
        "grants_nothing": v1["grants_nothing"],
    }
    d["design_sha256"] = _self_sha(d, "design_sha256")
    return d


def seal_design_v2(path: Path = DESIGN_PATH) -> dict:
    if Path(path).exists():
        raise M4Refusal("a sealed M4 v2 design already exists — "
                        "immutable")
    d = build_design_v2()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(d, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def load_design(path: Path = None) -> dict:
    """C10/C14: the ACTIVE design — v3 when sealed, else the
    reviewed v2; self identity, exact schema, the reviewed-v2 pin
    and the full field-by-field supersession chain all verify on
    every load."""
    if path is None:
        path = DESIGN_PATH_V3 if DESIGN_PATH_V3.exists() \
            else DESIGN_PATH
    d = _strict_json_file(path, "M4 sealed design")
    if _self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise M4Refusal("sealed M4 design self-digest does not "
                        "re-derive")
    _canon_sha(d.get("design_sha256"), "design self digest")
    schema = d.get("schema")
    if schema == "agent_multi.m4_residual_capacity_design.v3":
        verify_design_supersession_v3(d)
    elif schema == "agent_multi.m4_residual_capacity_design.v2":
        if d["design_sha256"] != REVIEWED_V2_SHA:
            raise M4Refusal(
                "M4 v2 design is not the REVIEWED identity — a "
                "foreign v2 never executes")
        verify_design_supersession_v2(d)
    else:
        raise M4Refusal("M4 design schema is neither the "
                        "reviewed v2 nor its v3 supersession")
    return d


# ---------------- tiny deterministic MLP ----------------

def _mlp_init(n_in, n_hidden, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((n_in, n_hidden)) * 0.3,
            "b1": np.zeros(n_hidden),
            "W2": rng.standard_normal((n_hidden, 1)) * 0.3,
            "b2": np.zeros(1)}


def _forward(p, X):
    import numpy as np
    H = np.tanh(X @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def _sgd_step(p, X, y, lr):
    H, out = _forward(p, X)
    err = (out - y) / len(y)
    dH = (err[:, None] * p["W2"].T) * (1 - H ** 2)
    p["W2"] -= lr * (H.T @ err[:, None])
    p["b2"] -= lr * err.sum(keepdims=True)
    p["W1"] -= lr * (X.T @ dH)
    p["b1"] -= lr * dH.sum(axis=0)


def _loss(p, X, y):
    import numpy as np
    _, out = _forward(p, X)
    return float(np.mean((out - y) ** 2))


def _params_digest(p):
    import numpy as np
    h = hashlib.sha256()
    for k in sorted(p):
        h.update(k.encode())
        h.update(np.ascontiguousarray(p[k]).tobytes())
    return h.hexdigest()


# ------------- complete serializable state -------------

def _state_digest(st) -> str:
    import numpy as np
    h = hashlib.sha256()
    h.update(_params_digest(st["params"]).encode())
    h.update(np.ascontiguousarray(st["assoc_X"]).tobytes()
             if len(st["assoc_X"]) else b"EMPTY")
    h.update(np.ascontiguousarray(st["assoc_y"]).tobytes()
             if len(st["assoc_y"]) else b"EMPTY")
    h.update(json.dumps(
        {k: st[k] for k in ("family", "batch_index",
                            "retention_streak", "updates_done",
                            "accepted_batches",
                            "retention_margin")},
        sort_keys=True).encode())
    return h.hexdigest()


def _save_state(path: Path, st) -> str:
    import numpy as np
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        import io
        buf = io.BytesIO()
        np.savez(buf, **st["params"],
                 assoc_X=st["assoc_X"], assoc_y=st["assoc_y"])
        os.write(fd, buf.getvalue())
        os.fsync(fd)
    finally:
        os.close(fd)
    meta = {"family": st["family"],
            "batch_index": st["batch_index"],
            "retention_streak": st["retention_streak"],
            "updates_done": st["updates_done"],
            "accepted_batches": st["accepted_batches"],
            "retention_margin": st["retention_margin"],
            "state_digest": _state_digest(st)}
    mp = Path(str(path) + ".meta.json")
    fd = os.open(str(mp), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, json.dumps(meta, sort_keys=True,
                                indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return meta["state_digest"]


def _load_state(path: Path):
    import numpy as np
    with np.load(str(path)) as z:
        params = {k: z[k].copy()
                  for k in ("W1", "b1", "W2", "b2")}
        assoc_X = z["assoc_X"].copy()
        assoc_y = z["assoc_y"].copy()
    meta = _strict_json_file(Path(str(path) + ".meta.json"),
                             "M4 state meta")
    st = {"params": params, "assoc_X": assoc_X,
          "assoc_y": assoc_y, **{k: meta[k] for k in
                                 ("family", "batch_index",
                                  "retention_streak",
                                  "updates_done",
                                  "accepted_batches",
                                  "retention_margin")}}
    if _state_digest(st) != meta["state_digest"]:
        raise M4Refusal("state identity does not re-derive")
    return st


# ------------- the cumulative intervention step -------------

def _gen_unit(family):
    import numpy as np
    rng = np.random.default_rng(_seed("gen", family))
    n, n_in = 256, 8
    X = rng.standard_normal((n, n_in))
    if family == "sine":
        y = np.sin(X[:, 0] * 2.0) + 0.1 * rng.standard_normal(n)
    else:
        y = (X[:, :5].sum(axis=1) > 0).astype(float)
    return (X[:192], y[:192], X[192:], y[192:])


def _batch_assoc(family, b):
    import numpy as np
    rng = np.random.default_rng(_seed("assoc", family, b))
    return (rng.standard_normal((8, 8)),
            rng.choice([0.0, 1.0], size=8))


def apply_batch(st, Xtr, ytr, Xev, yev, rehearsal=True):
    """M4-C1/C2/C3: append the sealed batch to the CUMULATIVE
    inventory, train with the frozen 50/50 rehearsal rule
    (stateless per-(family,batch,update) sampling), then evaluate
    retention (two-consecutive streak) and EVERY cumulative
    association."""
    import numpy as np
    b = st["batch_index"]
    Xa, ya = _batch_assoc(st["family"], b)
    st["assoc_X"] = np.vstack([st["assoc_X"], Xa]) \
        if len(st["assoc_X"]) else Xa
    st["assoc_y"] = np.concatenate([st["assoc_y"], ya]) \
        if len(st["assoc_y"]) else ya
    half = int(MINIBATCH * REHEARSAL_FRACTION)
    for u in range(UPDATES_PER_BATCH):
        rng = np.random.default_rng(
            _seed("mb", st["family"], b, u))
        io_ = rng.integers(0, len(ytr), size=half)
        ia = rng.integers(0, len(st["assoc_y"]),
                          size=MINIBATCH - half)
        if rehearsal:
            Xmb = np.vstack([Xtr[io_], st["assoc_X"][ia]])
            ymb = np.concatenate([ytr[io_], st["assoc_y"][ia]])
        else:
            # C14: the MATCHED diagnostic — the SAME minibatch
            # size and the SAME number of examples per update as
            # the primary arm (16), drawn wholly from the
            # cumulative associations; its own deterministic draw
            # leaves the primary stream untouched.
            ia2 = rng.integers(0, len(st["assoc_y"]),
                               size=MINIBATCH)
            Xmb = st["assoc_X"][ia2]
            ymb = st["assoc_y"][ia2]
        _sgd_step(st["params"], Xmb, ymb, LEARNING_RATE)
        st["updates_done"] += 1
    ret_loss = _loss(st["params"], Xev, yev)
    if ret_loss > st["retention_margin"]:
        st["retention_streak"] += 1
    else:
        st["retention_streak"] = 0
    _, out = _forward(st["params"], st["assoc_X"])
    per_assoc_ok = np.abs(out - st["assoc_y"]) < ACQ_TOL
    cumulative_ok = bool(per_assoc_ok.all())
    facts = {"batch": b, "ret_loss": round(ret_loss, 6),
             "retention_streak": st["retention_streak"],
             "cumulative_associations": int(len(st["assoc_y"])),
             "cumulative_acquired": int(per_assoc_ok.sum()),
             "cumulative_ok": cumulative_ok}
    st["batch_index"] += 1
    if st["retention_streak"] >= RETENTION_CONSECUTIVE:
        facts["outcome"] = "RETENTION_ENDPOINT"
    elif not cumulative_ok:
        facts["outcome"] = "ACQUISITION_ENDPOINT"
    else:
        st["accepted_batches"] += 1
        facts["outcome"] = "ACCEPTED"
    return facts


# ---------------- the corrected preflight ----------------

def _excl_write(path: Path, payload: bytes):
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def mechanics_preflight(out_dir: Path = None) -> dict:
    """TWO units, MECHANICS_ONLY, under the active (v3) design:
    cumulative acquisition with frozen rehearsal, two-consecutive
    retention, REAL fresh-process restart continuation, executed
    limits, matched-compute control, ONE matched no-rehearsal
    diagnostic batch (C14), and write-once digest-bound artifacts
    with mutable telemetry classified apart (C13)."""
    import subprocess
    import sys as _sys
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    design = load_design()
    out = Path(out_dir or PREFLIGHT_DIR_V3)
    # M4-C6/C16: refuse a nonempty root BEFORE writing anything
    if out.exists() and any(out.iterdir()):
        raise M4Refusal(
            "the preflight output root is not empty — outputs "
            "are write-once; nothing was modified")
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    stop = out / "M4_STOP"
    lim = design["resources"]
    t_all = time.monotonic()
    artifacts = {}
    units = []
    for uidx, family in enumerate(("sine", "majority")):
        ledger_path = out / f"u{uidx}_batches.jsonl"
        led = open(ledger_path, "a")
        facts = {"family": family, "typed_outcome": None}
        Xtr, ytr, Xev, yev = _gen_unit(family)
        p = _mlp_init(8, 16, _seed("init", family))
        for _ in range(600):
            rng = np.random.default_rng(
                _seed("pretrain", family, _))
            i = rng.integers(0, len(ytr), size=16)
            _sgd_step(p, Xtr[i], ytr[i], LEARNING_RATE)
        base_loss = _loss(p, Xev, yev)
        st = {"params": p, "assoc_X": np.zeros((0, 8)),
              "assoc_y": np.zeros((0,)), "family": family,
              "batch_index": 0, "retention_streak": 0,
              "updates_done": 0, "accepted_batches": 0,
              "retention_margin": base_loss * 1.10}
        ck_stop = out / f"u{uidx}_stop.npz"
        _save_state(ck_stop, st)
        # treatment: FOUR bounded cumulative batches
        for b in range(4):
            if time.monotonic() - t_all > \
                    lim["max_wall_seconds"]:
                facts["typed_outcome"] = "WALL_STOP"
                break
            rss = resource.getrusage(
                resource.RUSAGE_SELF).ru_maxrss * 1024
            if rss > lim["max_rss_bytes"]:
                facts["typed_outcome"] = "RSS_STOP"
                break
            if stop.exists():
                facts["typed_outcome"] = "STOP_REQUESTED"
                break
            tmp = out / f".hb_{os.urandom(6).hex()}"
            tmp.write_text(json.dumps(
                {"unit": family, "batch": b,
                 "monotonic": time.monotonic()}))
            os.replace(tmp, out / "M4_HEARTBEAT.json")
            rec = apply_batch(st, Xtr, ytr, Xev, yev)
            rec["record_sha256"] = _self_sha(rec,
                                            "record_sha256")
            led.write(json.dumps(rec, sort_keys=True) + "\n")
            led.flush()
            os.fsync(led.fileno())
            if rec["outcome"] != "ACCEPTED":
                facts["typed_outcome"] = rec["outcome"]
                break
        led.close()
        if facts["typed_outcome"] is None:
            facts["typed_outcome"] = "MAX_BATCHES_MECHANICS"
        facts["accepted_batches_mechanics_only"] = \
            st["accepted_batches"]
        facts["cumulative_associations_final"] = \
            int(len(st["assoc_y"]))
        # M4-C4: REAL restart — reload batch-1 state in a FRESH
        # process, apply batch 2 there; the uninterrupted branch
        # replays the same two batches in-process; identical
        st1 = _load_state(ck_stop)
        for b in range(2):
            apply_batch(st1, Xtr, ytr, Xev, yev)
        ck_two = out / f"u{uidx}_after2.npz"
        _save_state(ck_two, st1)
        st_a = _load_state(ck_stop)
        apply_batch(st_a, Xtr, ytr, Xev, yev)
        ck_one = out / f"u{uidx}_after1.npz"
        _save_state(ck_one, st_a)
        code = (
            "import sys; sys.path.insert(0, 'tools');"
            "import m4_residual_capacity as m4;"
            "from pathlib import Path;"
            f"st = m4._load_state(Path({str(ck_one)!r}));"
            f"X = m4._gen_unit({family!r});"
            "m4.apply_batch(st, X[0], X[1], X[2], X[3]);"
            "print(m4._state_digest(st))")
        rc = subprocess.run([_sys.executable, "-c", code],
                            capture_output=True, text=True,
                            cwd=str(REPO))
        if rc.returncode != 0:
            raise M4Refusal(
                f"fresh-process restart failed: "
                f"{rc.stderr[-160:]}")
        facts["restart_fresh_process_digest"] = \
            rc.stdout.strip()
        facts["uninterrupted_digest"] = _state_digest(st1)
        facts["restart_continuation_identical"] = (
            rc.stdout.strip() == _state_digest(st1))
        # M4-C5: matched-compute control EXECUTES the identical
        # sequence from a random init; derive the update diff
        ctrl = {"params": _mlp_init(8, 16,
                                    _seed("ctrl", family)),
                "assoc_X": np.zeros((0, 8)),
                "assoc_y": np.zeros((0,)), "family": family,
                "batch_index": 0, "retention_streak": 0,
                "updates_done": 0, "accepted_batches": 0,
                "retention_margin": st["retention_margin"]}
        treat_updates = st["updates_done"]
        while ctrl["batch_index"] < st["batch_index"]:
            apply_batch(ctrl, Xtr, ytr, Xev, yev)
        diff = abs(ctrl["updates_done"] - treat_updates) / \
            max(1, treat_updates)
        facts["matched_compute_executed"] = True
        facts["matched_compute_update_diff"] = round(diff, 6)
        assert diff <= 0.01
        # C14: ONE matched no-rehearsal diagnostic batch from the
        # persisted pre-treatment state — 16 examples per update,
        # mechanics only, its own ledger and checkpoint
        st_d = _load_state(ck_stop)
        rec_d = apply_batch(st_d, Xtr, ytr, Xev, yev,
                            rehearsal=False)
        rec_d["examples_per_update"] = MINIBATCH
        rec_d["record_sha256"] = _self_sha(rec_d,
                                           "record_sha256")
        diag_path = out / f"u{uidx}_diag.jsonl"
        _excl_write(diag_path, (json.dumps(
            rec_d, sort_keys=True) + "\n").encode())
        ck_diag = out / f"u{uidx}_diag.npz"
        _save_state(ck_diag, st_d)
        facts["diagnostic_outcome"] = rec_d["outcome"]
        units.append(facts)
        for pth in (ck_stop, ck_one, ck_two, ck_diag,
                    ledger_path, diag_path,
                    Path(str(ck_stop) + ".meta.json"),
                    Path(str(ck_one) + ".meta.json"),
                    Path(str(ck_two) + ".meta.json"),
                    Path(str(ck_diag) + ".meta.json")):
            artifacts[pth.name] = _sha_file(pth)
    # C13: the heartbeat is MUTABLE telemetry — classified apart,
    # never digest-bound in the immutable map.
    report = {"schema": "agent_multi.m4_mechanics_preflight.v3",
              "design_sha256": design["design_sha256"],
              "authority": "MECHANICS_ONLY_NO_SCIENTIFIC_"
                           "CONCLUSION",
              "units": units,
              "artifacts_sha256": artifacts,
              "telemetry_mutable": ["M4_HEARTBEAT.json"],
              "wall_seconds": round(time.monotonic() - t_all, 2)}
    report["report_sha256"] = _self_sha(report, "report_sha256")
    _excl_write(out / "M4_PREFLIGHT_REPORT.json",
                json.dumps(report, indent=1).encode())
    verify_preflight(out)
    return report


_REPORT_KEYS = {"schema", "design_sha256", "authority", "units",
                "artifacts_sha256", "telemetry_mutable",
                "wall_seconds", "report_sha256"}
_UNIT_KEYS = {"family", "typed_outcome",
              "accepted_batches_mechanics_only",
              "cumulative_associations_final",
              "restart_fresh_process_digest",
              "uninterrupted_digest",
              "restart_continuation_identical",
              "matched_compute_executed",
              "matched_compute_update_diff",
              "diagnostic_outcome"}
_RECORD_KEYS = {"batch", "ret_loss", "retention_streak",
                "cumulative_associations",
                "cumulative_acquired", "cumulative_ok",
                "outcome", "record_sha256"}
_BATCH_OUTCOMES = {"ACCEPTED", "ACQUISITION_ENDPOINT",
                   "RETENTION_ENDPOINT"}
_META_KEYS = {"family", "batch_index", "retention_streak",
              "updates_done", "accepted_batches",
              "retention_margin", "state_digest"}


def _load_state_checked(path: Path):
    """C11: arbitrary bytes under a repaired digest refuse TYPED
    before any verdict — unreadable state is never verified."""
    try:
        return _load_state(path)
    except M4Refusal:
        raise
    except Exception as exc:
        raise M4Refusal(
            f"state artifact {Path(path).name!r} is not a "
            f"loadable NPZ state ({type(exc).__name__}) — "
            "arbitrary bytes never verify")


def _check_record_schema(rec: dict, i: int, what: str,
                         extra_keys: set = frozenset()):
    """C10: exact keys, exact primitive types, physical domains
    and canonical digests for one ledger record."""
    want = _RECORD_KEYS | set(extra_keys)
    if set(rec) != want:
        raise M4Refusal(f"{what} record {i} keys are not the "
                        "exact schema")
    _req_int(rec["batch"], f"{what} record {i} batch")
    if rec["batch"] != i:
        raise M4Refusal(f"{what} record {i} batch counter is "
                        "impossible")
    _req_num(rec["ret_loss"], f"{what} record {i} ret_loss")
    if rec["ret_loss"] < 0:
        raise M4Refusal(f"{what} record {i} ret_loss violates "
                        "its domain")
    _req_int(rec["retention_streak"],
             f"{what} record {i} retention_streak", 0,
             RETENTION_CONSECUTIVE)
    n_assoc = _req_int(rec["cumulative_associations"],
                       f"{what} record {i} associations", 0)
    n_acq = _req_int(rec["cumulative_acquired"],
                     f"{what} record {i} acquired", 0)
    if n_acq > n_assoc:
        raise M4Refusal(
            f"{what} record {i} counters are impossible — "
            "acquired exceeds the cumulative inventory")
    _req_bool(rec["cumulative_ok"],
              f"{what} record {i} cumulative_ok")
    if rec["outcome"] not in _BATCH_OUTCOMES:
        raise M4Refusal(f"{what} record {i} outcome is foreign")
    _canon_sha(rec["record_sha256"],
               f"{what} record {i} self digest")
    if _self_sha(rec, "record_sha256") != rec["record_sha256"]:
        raise M4Refusal(f"{what} record {i} self-digest does "
                        "not re-derive")


def verify_preflight(out_dir: Path = None) -> dict:
    """C11/C12/C13: the verifier is an independent RECONSTRUCTION
    — it re-derives every task and batch from the sealed design,
    loads and validates every state artifact, replays every
    apply_batch transition from its predecessor, re-derives every
    published number and verdict, compares exact replayed states
    with every stored checkpoint, and re-executes the restart in
    its OWN fresh process. Producer booleans and checksums are
    claims to be checked, never evidence."""
    import subprocess
    import sys as _sys
    import numpy as np
    out = Path(out_dir or PREFLIGHT_DIR_V3)
    report = _strict_json_file(out / "M4_PREFLIGHT_REPORT.json",
                               "M4 preflight report")
    # ---- C10: cheap exact-schema pass BEFORE any replay ----
    if set(report) != _REPORT_KEYS:
        raise M4Refusal("report keys are not the exact schema")
    if report.get("schema") != \
            "agent_multi.m4_mechanics_preflight.v3":
        raise M4Refusal("report schema is foreign")
    _canon_sha(report["report_sha256"], "report self digest")
    if _self_sha(report, "report_sha256") != \
            report["report_sha256"]:
        raise M4Refusal("report self-digest does not re-derive")
    design = load_design()
    _canon_sha(report["design_sha256"], "report design digest")
    if report["design_sha256"] != design["design_sha256"]:
        raise M4Refusal("report does not bind the sealed active "
                        "design")
    if report["authority"] != \
            "MECHANICS_ONLY_NO_SCIENTIFIC_CONCLUSION":
        raise M4Refusal("report authority label is foreign")
    _req_num(report["wall_seconds"], "report wall_seconds")
    if report["wall_seconds"] < 0:
        raise M4Refusal("report wall_seconds violates its domain")
    if report["telemetry_mutable"] != ["M4_HEARTBEAT.json"]:
        raise M4Refusal(
            "telemetry classification is not the exact mutable "
            "set — the heartbeat is mutable, never digest-bound")
    if type(report["artifacts_sha256"]) is not dict or \
            not report["artifacts_sha256"]:
        raise M4Refusal("artifacts_sha256 must be a nonempty map")
    for name, sha in report["artifacts_sha256"].items():
        _canon_sha(sha, f"artifact digest for {name!r}")
        if name in report["telemetry_mutable"]:
            raise M4Refusal(
                "mutable telemetry appears in the immutable "
                "artifact map — it cannot be both digest-bound "
                "and unchecked")
    if type(report["units"]) is not list or \
            len(report["units"]) != 2:
        raise M4Refusal("report must carry exactly two units")
    for u in report["units"]:
        if set(u) != _UNIT_KEYS:
            raise M4Refusal("unit facts are not the exact "
                            "schema")
    # ---- exact inventory (heartbeat classified, stop excluded) --
    expected = set(report["artifacts_sha256"]) | \
        {"M4_PREFLIGHT_REPORT.json"} | \
        set(report["telemetry_mutable"])
    actual = {q.name for q in out.iterdir()
              if q.name != "M4_STOP"}
    if actual != expected:
        raise M4Refusal(
            "artifact inventory does not equal the report "
            f"exactly (diff: {sorted(actual ^ expected)[:5]})")
    for name, sha in report["artifacts_sha256"].items():
        if _sha_file(out / name) != sha:
            raise M4Refusal(
                f"artifact {name!r} does not match its FULL "
                "recorded digest — replaced evidence "
                "invalidates the report")
    hb = _strict_json_file(out / "M4_HEARTBEAT.json",
                           "M4 heartbeat telemetry")
    if set(hb) != {"unit", "batch", "monotonic"}:
        raise M4Refusal("heartbeat telemetry keys are not the "
                        "exact schema")
    # ---- C11: full independent reconstruction per unit ----
    derived_units = []
    for uidx, family in enumerate(("sine", "majority")):
        unit = report["units"][uidx]
        if unit["family"] != family:
            raise M4Refusal("unit family order is foreign")
        # ledger schema first (cheap), then replay
        lines = (out / f"u{uidx}_batches.jsonl"
                 ).read_text().splitlines()
        recs = []
        for i, line in enumerate(lines):
            rec = _strict_json_text(
                line, f"u{uidx} batch record {i}")
            _check_record_schema(rec, i, f"u{uidx}")
            recs.append(rec)
        if not recs:
            raise M4Refusal(f"u{uidx} ledger is empty")
        dlines = (out / f"u{uidx}_diag.jsonl"
                  ).read_text().splitlines()
        if len(dlines) != 1:
            raise M4Refusal(f"u{uidx} diagnostic ledger must "
                            "hold exactly one record")
        drec = _strict_json_text(dlines[0],
                                 f"u{uidx} diagnostic record")
        _check_record_schema(drec, 0, f"u{uidx} diagnostic",
                             {"examples_per_update"})
        if drec["examples_per_update"] != MINIBATCH:
            raise M4Refusal(
                "diagnostic arm does not process the primary "
                "arm's example count per update")
        # rebuild the task and the initial state from the design
        Xtr, ytr, Xev, yev = _gen_unit(family)
        par = _mlp_init(8, 16, _seed("init", family))
        for k in range(600):
            rng = np.random.default_rng(
                _seed("pretrain", family, k))
            i = rng.integers(0, len(ytr), size=16)
            _sgd_step(par, Xtr[i], ytr[i], LEARNING_RATE)
        base_loss = _loss(par, Xev, yev)
        st0 = {"params": par, "assoc_X": np.zeros((0, 8)),
               "assoc_y": np.zeros((0,)), "family": family,
               "batch_index": 0, "retention_streak": 0,
               "updates_done": 0, "accepted_batches": 0,
               "retention_margin": base_loss * 1.10}
        st0_digest = _state_digest(st0)
        stored_stop = _load_state_checked(
            out / f"u{uidx}_stop.npz")
        if _state_digest(stored_stop) != st0_digest:
            raise M4Refusal(
                f"u{uidx} pre-treatment checkpoint does not "
                "equal the state reconstructed from the sealed "
                "design")
        # replay EVERY treatment transition from its predecessor
        st = _load_state_checked(out / f"u{uidx}_stop.npz")
        derived_accepted = 0
        for i, rec in enumerate(recs):
            replay = apply_batch(st, Xtr, ytr, Xev, yev)
            claimed = {k: rec[k] for k in rec
                       if k != "record_sha256"}
            if replay != claimed:
                bad = sorted(k for k in replay
                             if replay[k] != claimed.get(k))
                raise M4Refusal(
                    f"u{uidx} batch {i} does not replay from "
                    f"its predecessor (fields: {bad[:3]}) — "
                    "stored outcomes are not reproductions")
            if replay["outcome"] == "ACCEPTED":
                derived_accepted += 1
            elif i != len(recs) - 1:
                raise M4Refusal(
                    f"u{uidx} ledger continues past a terminal "
                    "outcome")
        if unit["accepted_batches_mechanics_only"] != \
                derived_accepted:
            raise M4Refusal(
                "accepted-batch fact does not equal the "
                "REPLAYED count")
        if unit["cumulative_associations_final"] != \
                int(len(st["assoc_y"])):
            raise M4Refusal(
                "cumulative-association fact does not equal "
                "the replayed inventory")
        # typed outcome re-derived from the replayed ledger
        last = recs[-1]["outcome"]
        if last != "ACCEPTED":
            derived_typed = last
        elif len(recs) == 4:
            derived_typed = "MAX_BATCHES_MECHANICS"
        else:
            raise M4Refusal(
                f"u{uidx} ledger ends ACCEPTED before the batch "
                "bound — no typed outcome reconstructs in "
                "MECHANICS verification")
        if unit["typed_outcome"] != derived_typed:
            raise M4Refusal(
                "typed outcome does not equal the replayed "
                "derivation")
        # checkpoints: replay one and two batches from st0
        st1 = _load_state_checked(out / f"u{uidx}_stop.npz")
        apply_batch(st1, Xtr, ytr, Xev, yev)
        d_after1 = _state_digest(st1)
        apply_batch(st1, Xtr, ytr, Xev, yev)
        d_after2 = _state_digest(st1)
        for name, want in ((f"u{uidx}_after1.npz", d_after1),
                           (f"u{uidx}_after2.npz", d_after2)):
            stored = _load_state_checked(out / name)
            if _state_digest(stored) != want:
                raise M4Refusal(
                    f"checkpoint {name!r} does not equal the "
                    "REPLAYED state — stored states are not "
                    "reproductions")
        # C12: the VERIFIER's OWN causal restart — fresh process
        # from the persisted after1, one declared batch, compared
        # with the replayed uninterrupted after2
        code = (
            "import sys; sys.path.insert(0, 'tools');"
            "import m4_residual_capacity as m4;"
            "from pathlib import Path;"
            "st = m4._load_state(Path("
            f"{str(out / f'u{uidx}_after1.npz')!r}));"
            f"X = m4._gen_unit({family!r});"
            "m4.apply_batch(st, X[0], X[1], X[2], X[3]);"
            "print(m4._state_digest(st))")
        rc = subprocess.run([_sys.executable, "-c", code],
                            capture_output=True, text=True,
                            cwd=str(REPO))
        if rc.returncode != 0:
            raise M4Refusal(
                "verifier fresh-process restart failed: "
                f"{rc.stderr[-160:]}")
        child_digest = rc.stdout.strip()
        derived_restart = (child_digest == d_after2)
        _canon_sha(unit["restart_fresh_process_digest"],
                   "restart fresh-process digest fact")
        _canon_sha(unit["uninterrupted_digest"],
                   "uninterrupted digest fact")
        _req_bool(unit["restart_continuation_identical"],
                  "restart fact")
        if not derived_restart or \
                unit["restart_continuation_identical"] is not \
                True or \
                unit["restart_fresh_process_digest"] != \
                child_digest or \
                unit["uninterrupted_digest"] != d_after2:
            raise M4Refusal(
                "restart continuation does not verify CAUSALLY "
                "— the verifier's own fresh-process replay "
                "disagrees with the stored facts")
        # matched compute RE-EXECUTED, never trusted
        ctrl = {"params": _mlp_init(8, 16,
                                    _seed("ctrl", family)),
                "assoc_X": np.zeros((0, 8)),
                "assoc_y": np.zeros((0,)), "family": family,
                "batch_index": 0, "retention_streak": 0,
                "updates_done": 0, "accepted_batches": 0,
                "retention_margin": st["retention_margin"]}
        while ctrl["batch_index"] < st["batch_index"]:
            apply_batch(ctrl, Xtr, ytr, Xev, yev)
        diff = abs(ctrl["updates_done"] - st["updates_done"]) / \
            max(1, st["updates_done"])
        _req_bool(unit["matched_compute_executed"],
                  "matched-compute fact")
        _req_num(unit["matched_compute_update_diff"],
                 "matched-compute diff fact")
        if unit["matched_compute_executed"] is not True or \
                diff > 0.01 or \
                unit["matched_compute_update_diff"] != \
                round(diff, 6):
            raise M4Refusal(
                "matched-compute control does not re-derive "
                "within tolerance")
        # C14: replay the matched diagnostic batch
        st_d = _load_state_checked(out / f"u{uidx}_stop.npz")
        replay_d = apply_batch(st_d, Xtr, ytr, Xev, yev,
                               rehearsal=False)
        replay_d["examples_per_update"] = MINIBATCH
        claimed_d = {k: drec[k] for k in drec
                     if k != "record_sha256"}
        if replay_d != claimed_d:
            raise M4Refusal(
                "diagnostic record does not replay from the "
                "persisted pre-treatment state under the "
                "matched-minibatch rule")
        stored_diag = _load_state_checked(
            out / f"u{uidx}_diag.npz")
        if _state_digest(stored_diag) != _state_digest(st_d):
            raise M4Refusal(
                "diagnostic checkpoint does not equal the "
                "REPLAYED diagnostic state")
        if unit["diagnostic_outcome"] != replay_d["outcome"]:
            raise M4Refusal(
                "diagnostic outcome fact does not equal the "
                "replayed outcome")
        for ck in ("stop", "after1", "after2", "diag"):
            m = _strict_json_file(
                Path(str(out / f"u{uidx}_{ck}.npz")
                     + ".meta.json"),
                f"u{uidx} {ck} checkpoint metadata")
            if set(m) != _META_KEYS:
                raise M4Refusal(
                    "checkpoint metadata keys are not the exact "
                    "schema")
            _canon_sha(m["state_digest"],
                       f"u{uidx} {ck} state digest")
            _req_num(m["retention_margin"],
                     f"u{uidx} {ck} retention margin")
            for f_ in ("batch_index", "retention_streak",
                       "updates_done", "accepted_batches"):
                _req_int(m[f_], f"u{uidx} {ck} {f_}")
        derived_units.append({
            "family": family,
            "derived_accepted": derived_accepted,
            "derived_typed": derived_typed,
            "derived_restart_causal": derived_restart,
            "derived_matched_diff": round(diff, 6)})
    return {"verified": True,
            "units": len(report["units"]),
            "artifacts": len(report["artifacts_sha256"]),
            "derived": derived_units}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-design-v2", action="store_true")
    ap.add_argument("--seal-design-v3", action="store_true")
    ap.add_argument("--mechanics-preflight", action="store_true")
    ap.add_argument("--verify-preflight", action="store_true")
    args = ap.parse_args(argv)
    if args.seal_design_v2:
        d = seal_design_v2()
        print(json.dumps({"sealed_v2": d["design_sha256"],
                          "supersedes":
                              d["supersedes_design_sha256"]},
                         indent=1))
        return 0
    if args.seal_design_v3:
        d = seal_design_v3()
        print(json.dumps({"sealed_v3": d["design_sha256"],
                          "supersedes":
                              d["supersedes_design_sha256"],
                          "scientific_outcome":
                              d["scientific_outcome"]},
                         indent=1))
        return 0
    if args.mechanics_preflight:
        r = mechanics_preflight()
        print(json.dumps({
            "preflight": "MECHANICS_ONLY_COMPLETE_V3",
            "units": [
                {"family": u["family"],
                 "typed_outcome": u["typed_outcome"],
                 "accepted_batches":
                     u["accepted_batches_mechanics_only"],
                 "restart_identical":
                     u["restart_continuation_identical"],
                 "matched_compute_diff":
                     u["matched_compute_update_diff"]}
                for u in r["units"]],
            "wall_seconds": r["wall_seconds"]}, indent=1))
        return 0
    if args.verify_preflight:
        print(json.dumps(verify_preflight(), indent=1))
        return 0
    raise M4Refusal("choose --seal-design-v2, "
                    "--mechanics-preflight or --verify-preflight")


if __name__ == "__main__":
    raise SystemExit(main())
