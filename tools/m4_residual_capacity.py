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


def load_design(path: Path = DESIGN_PATH) -> dict:
    d = _strict_json_file(path, "M4 sealed design v2")
    if _self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise M4Refusal("sealed M4 v2 self-digest does not "
                        "re-derive")
    if d.get("schema") != \
            "agent_multi.m4_residual_capacity_design.v2":
        raise M4Refusal("M4 design is not the v2 supersession")
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
        else:                       # frozen DIAGNOSTIC arm only
            Xmb = st["assoc_X"][ia]
            ymb = st["assoc_y"][ia]
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


def mechanics_preflight(out_dir: Path = PREFLIGHT_DIR) -> dict:
    """TWO units, MECHANICS_ONLY, under the v2 design: cumulative
    acquisition with frozen rehearsal, two-consecutive retention,
    REAL fresh-process restart continuation, executed limits and
    matched-compute, write-once fully digest-bound artifacts."""
    import subprocess
    import sys as _sys
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    design = load_design()
    out = Path(out_dir)
    # M4-C6: refuse a nonempty root BEFORE writing anything
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
        units.append(facts)
        for pth in (ck_stop, ck_one, ck_two, ledger_path,
                    Path(str(ck_stop) + ".meta.json"),
                    Path(str(ck_one) + ".meta.json"),
                    Path(str(ck_two) + ".meta.json")):
            artifacts[pth.name] = _sha_file(pth)
    artifacts["M4_HEARTBEAT.json"] = _sha_file(
        out / "M4_HEARTBEAT.json")
    report = {"schema": "agent_multi.m4_mechanics_preflight.v2",
              "design_sha256": design["design_sha256"],
              "authority": "MECHANICS_ONLY_NO_SCIENTIFIC_"
                           "CONCLUSION",
              "units": units,
              "artifacts_sha256": artifacts,
              "wall_seconds": round(time.monotonic() - t_all, 2)}
    report["report_sha256"] = _self_sha(report, "report_sha256")
    _excl_write(out / "M4_PREFLIGHT_REPORT.json",
                json.dumps(report, indent=1).encode())
    verify_preflight(out)
    return report


def verify_preflight(out_dir: Path = PREFLIGHT_DIR) -> dict:
    """M4-C6: a FRESH reader — exact schema, exact directory
    inventory, FULL file digests re-derived, and the stated facts
    reconstructed from the batch ledgers. A replaced artifact
    invalidates the report."""
    out = Path(out_dir)
    report = _strict_json_file(out / "M4_PREFLIGHT_REPORT.json",
                               "M4 preflight report")
    _KEYS = {"schema", "design_sha256", "authority", "units",
             "artifacts_sha256", "wall_seconds",
             "report_sha256"}
    if set(report) != _KEYS:
        raise M4Refusal("report keys are not the exact schema")
    if _self_sha(report, "report_sha256") != \
            report["report_sha256"]:
        raise M4Refusal("report self-digest does not re-derive")
    design = load_design()
    if report["design_sha256"] != design["design_sha256"]:
        raise M4Refusal("report does not bind the sealed v2 "
                        "design")
    expected = set(report["artifacts_sha256"]) | \
        {"M4_PREFLIGHT_REPORT.json"}
    actual = {p.name for p in out.iterdir()
              if p.name != "M4_STOP"}
    if actual != expected:
        raise M4Refusal(
            "artifact inventory does not equal the report "
            f"exactly (diff: {sorted(actual ^ expected)[:5]})")
    for name, sha in report["artifacts_sha256"].items():
        if name == "M4_HEARTBEAT.json":
            continue            # mutable telemetry, listed only
        if _sha_file(out / name) != sha:
            raise M4Refusal(
                f"artifact {name!r} does not match its FULL "
                "recorded digest — replaced evidence "
                "invalidates the report")
    for uidx, unit in enumerate(report["units"]):
        recs = []
        for line in (out / f"u{uidx}_batches.jsonl"
                     ).read_text().splitlines():
            r = _strict_json_file.__wrapped__(line, "x") \
                if False else json.loads(line)
            if _self_sha(r, "record_sha256") != \
                    r["record_sha256"]:
                raise M4Refusal("batch record self-digest does "
                                "not re-derive")
            recs.append(r)
        accepted = sum(1 for r in recs
                       if r["outcome"] == "ACCEPTED")
        if accepted != unit["accepted_batches_mechanics_only"]:
            raise M4Refusal(
                "accepted-batch count does not reconstruct from "
                "the ledger")
        streak = 0
        for r in recs:
            if r["retention_streak"] not in (0, streak + 1):
                raise M4Refusal("retention streak does not "
                                "follow the sealed "
                                "two-consecutive rule")
            streak = r["retention_streak"]
        if not unit["restart_continuation_identical"]:
            raise M4Refusal("restart continuation is not "
                            "identical")
        if not unit["matched_compute_executed"] or \
                unit["matched_compute_update_diff"] > 0.01:
            raise M4Refusal("matched-compute control missing or "
                            "out of tolerance")
    return {"verified": True,
            "units": len(report["units"]),
            "artifacts": len(report["artifacts_sha256"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-design-v2", action="store_true")
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
    if args.mechanics_preflight:
        r = mechanics_preflight()
        print(json.dumps({
            "preflight": "MECHANICS_ONLY_COMPLETE_V2",
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
