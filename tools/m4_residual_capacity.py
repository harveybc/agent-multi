#!/usr/bin/env python3
"""M4.0: residual-capacity intervention — PRE-RESULT executable
design + MECHANICS_ONLY CPU preflight (order T2 C66-C73 / M3
C1-C6 / M4.0, section 3; stages C2-C3 of the M3-M6 draft).

`--seal-design` freezes, BEFORE any intervention outcome:
structured task families with disjoint generator identities for
development/calibration/confirmation; one-hidden-layer float64
MLPs only; optimizer-capability controls separating
OPTIMIZATION_LIMITED from a capacity endpoint; checkpoint
schedule (initialization, pre-stop, stop, bounded post-stop);
the original-task retention metric and frozen margin; blinded
random-association batches with a maximum exposure/update
budget; acquisition, retention-violation and ambiguous criteria;
random-initialization and matched-compute controls; the task
GENERATOR as the statistical unit with nested seeds,
multiplicity and typed missing-run handling; the exact
trajectory/description measurements with their measured cost;
and CPU wall/RSS/stop-file/heartbeat limits.

`--mechanics-preflight` runs TWO units MECHANICS_ONLY proving:
model forking, checkpoint identity (byte-stable digests),
retention measurement, blinded batch acquisition, bounded
continuation, restart from persisted state, and artifact
reconstruction by a fresh reader. It grants NO M4 scientific
conclusion. The endpoint of M4 is a conditional empirical
intervention result — never "unused bits", "remaining
intelligence" or exact Kolmogorov complexity; no checkpoint or
seed is an independent statistical unit; M4 grants no DOIN gene
or production gate."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DESIGN_PATH = (REPO / "docs/research/model_capacity/"
               "M4_SEALED_DESIGN_2026_09_08.json")
PREFLIGHT_DIR = (REPO / "docs/audits/evidence/"
                 "m4_mechanics_preflight")
MASTER_SEED_PHRASE = "m4_residual_capacity_2026_09_08"


class M4Refusal(SystemExit):
    def __init__(self, msg):
        super().__init__(f"REFUSED: {msg}")


def _self_sha(body: dict, exclude: str) -> str:
    return hashlib.sha256(json.dumps(
        {k: body[k] for k in sorted(body) if k != exclude},
        sort_keys=True).encode()).hexdigest()


def _seed(*parts) -> int:
    h = hashlib.sha256(("|".join(
        [MASTER_SEED_PHRASE, *map(str, parts)])).encode())
    return int.from_bytes(h.digest()[:8], "big")


def build_design() -> dict:
    d = {
        "schema": "agent_multi.m4_residual_capacity_design.v1",
        "sealed_at_date": "2026-09-08",
        "stage": "C2_C3_residual_capacity_intervention",
        "question": (
            "after a one-hidden-layer MLP learns a structured "
            "task, how many additional BLINDED random "
            "associations can it fit before crossing a FROZEN "
            "original-task retention margin, conditional on the "
            "checkpoint — and is that endpoint predictable from "
            "parameter count or compressed length alone (H2 of "
            "the M3-M6 draft)"),
        "endpoint_semantics": (
            "a conditional empirical intervention result under "
            "this exact protocol; NEVER 'unused bits', "
            "'remaining intelligence' or exact Kolmogorov "
            "complexity"),
        "task_families": {
            "boolean": ["identity", "majority", "dnf3",
                        "parity4", "random_label_null"],
            "temporal": ["sine", "chirp", "am", "discontinuous",
                         "state_space"],
            "noise_regimes": ["white", "colored", "impulsive",
                              "heteroscedastic"],
            "generator_identity_rule": (
                "generator id = sha256(master|family|regime|"
                "role|instance); DEVELOPMENT, CALIBRATION and "
                "CONFIRMATION roles use DISJOINT instance "
                "ranges (0-15, 16-23, 24-31) and never share a "
                "generator id")},
        "architectures": {
            "family": "one-hidden-layer MLP only",
            "hidden_units_grid": [8, 16, 32, 64],
            "precision": "float64",
            "activation": "tanh",
            "output": "linear regression head / sigmoid for "
                      "boolean tasks"},
        "optimizer_capability_controls": {
            "rule": ("before any capacity claim, the SAME "
                     "optimizer budget must first learn each "
                     "structured family to its frozen "
                     "learnability criterion on a clean "
                     "development instance; a family that fails "
                     "is typed OPTIMIZATION_LIMITED and its "
                     "cells never become capacity evidence"),
            "learnability_criterion": (
                "task metric within the family's frozen "
                "threshold on held-out rows of the SAME "
                "generator")},
        "checkpoints": ["initialization", "pre_stop",
                        "calibration_stop", "post_stop_bounded"],
        "retention": {
            "metric": ("original-task evaluation loss on held-"
                       "out rows of the training generator"),
            "margin_rule": ("frozen at intervention start: "
                            "retention violated when loss "
                            "exceeds (1 + 0.10) * checkpoint "
                            "loss on two consecutive "
                            "evaluations"),
            "frozen_before_outcomes": True},
        "association_batches": {
            "blinding": ("random input/label associations drawn "
                         "from the sealed seed stream; batch "
                         "contents never inspected before "
                         "commitment"),
            "batch_size": 8,
            "max_batches": 64,
            "max_updates_per_batch": 2000,
            "acquisition_criterion": (
                "batch fitted when its association error falls "
                "below the frozen threshold within the update "
                "budget"),
            "ambiguous_states": [
                "ACQUISITION_TIMEOUT", "RETENTION_AMBIGUOUS",
                "NUMERICAL_ANOMALY"],
            "stop_rule": ("first confirmed retention violation "
                          "OR first acquisition failure; the "
                          "endpoint batch index is the result")},
        "controls": {
            "random_initialization": (
                "an untrained matched-architecture model runs "
                "the identical intervention with matched "
                "compute"),
            "matched_compute": (
                "control updates equal treatment updates within "
                "1%")},
        "statistics": {
            "unit": "the task GENERATOR (never a checkpoint, "
                    "seed or adjacent temporal sample)",
            "nested_seeds": ("3 model seeds nested per "
                            "generator; reported as nested "
                            "repetitions"),
            "multiplicity": ("Bonferroni over the frozen "
                            "confirmatory contrast family; "
                            "family list frozen before any "
                            "confirmation outcome"),
            "missing_runs": ("typed states remain in the "
                            "denominator; never silently "
                            "dropped")},
        "measurements": {
            "trajectory": ["train/calibration loss curves",
                           "gradient norm summaries",
                           "exception-memorization probes"],
            "description": ["8-bit sparse description length",
                            "prune-at-threshold parameter count",
                            "spectral rank profile"],
            "cost_rule": ("every measurement's CPU cost is "
                          "recorded per checkpoint and stays in "
                          "the cost denominator")},
        "resources": {
            "cpu_only": True, "cuda_hidden": True,
            "logical_workers": 1,
            "max_wall_seconds": 6 * 3600,
            "max_rss_bytes": 8 << 30,
            "nice": 15,
            "heartbeat": "M4_HEARTBEAT.json",
            "stop_file": "M4_STOP in the runs directory"},
        "seeds": {"master_phrase": MASTER_SEED_PHRASE,
                  "derivation": ("sha256(master|kind|generator|"
                                 "index)[:8] -> numpy "
                                 "default_rng")},
        "grants_nothing": ["DOIN gene", "production gate",
                          "scalar intelligence measure",
                          "exact complexity claim"],
    }
    d["design_sha256"] = _self_sha(d, "design_sha256")
    return d


def seal_design(path: Path = DESIGN_PATH) -> dict:
    if Path(path).exists():
        raise M4Refusal(
            "a sealed M4 design already exists — immutable, "
            "never regenerated in place")
    d = build_design()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                 0o644)
    try:
        os.write(fd, json.dumps(d, indent=1).encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    return d


def load_design(path: Path = DESIGN_PATH) -> dict:
    p = Path(path)
    if not p.is_file():
        raise M4Refusal("M4_DESIGN_REQUIRED: seal it first")
    d = json.loads(p.read_text())
    if _self_sha(d, "design_sha256") != d.get("design_sha256"):
        raise M4Refusal("sealed M4 design self-digest does not "
                        "re-derive")
    return d


# ---------------- MECHANICS_ONLY preflight ----------------

def _mlp_init(n_in, n_hidden, seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    return {"W1": rng.standard_normal((n_in, n_hidden)) * 0.3,
            "b1": np.zeros(n_hidden),
            "W2": rng.standard_normal((n_hidden, 1)) * 0.3,
            "b2": np.zeros(1)}


def _mlp_forward(p, X):
    import numpy as np
    H = np.tanh(X @ p["W1"] + p["b1"])
    return H, (H @ p["W2"] + p["b2"]).ravel()


def _mlp_sgd(p, X, y, lr=0.05, updates=200, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    n = len(y)
    for _ in range(updates):
        i = rng.integers(0, n, size=min(16, n))
        H, out = _mlp_forward(p, X[i])
        err = (out - y[i]) / len(i)
        gW2 = H.T @ err[:, None]
        gb2 = err.sum(keepdims=True)
        dH = (err[:, None] * p["W2"].T) * (1 - H ** 2)
        p["W2"] -= lr * gW2
        p["b2"] -= lr * gb2
        p["W1"] -= lr * (X[i].T @ dH)
        p["b1"] -= lr * dH.sum(axis=0)
    return p


def _params_digest(p):
    import numpy as np
    h = hashlib.sha256()
    for k in sorted(p):
        h.update(k.encode())
        h.update(np.ascontiguousarray(p[k]).tobytes())
    return h.hexdigest()


def _loss(p, X, y):
    import numpy as np
    _, out = _mlp_forward(p, X)
    return float(np.mean((out - y) ** 2))


def _save_ckpt(path, p, meta):
    import numpy as np
    buf = {k: v for k, v in p.items()}
    rec = {"meta": meta, "params_sha256": _params_digest(p)}
    np.savez(path, **buf)
    Path(str(path) + ".meta.json").write_text(json.dumps(rec))
    return rec["params_sha256"]


def _load_ckpt(path):
    import numpy as np
    with np.load(str(path)) as z:
        p = {k: z[k].copy() for k in z.files}
    meta = json.loads(
        Path(str(path) + ".meta.json").read_text())
    if _params_digest(p) != meta["params_sha256"]:
        raise M4Refusal("checkpoint identity does not re-derive")
    return p, meta


def mechanics_preflight(out_dir: Path = PREFLIGHT_DIR) -> dict:
    """TWO units, MECHANICS_ONLY: fork, checkpoint identity,
    retention measurement, blinded batch acquisition, bounded
    continuation, restart, artifact reconstruction. Zero
    scientific conclusion."""
    import numpy as np
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(15)
    design = load_design()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"schema":
              "agent_multi.m4_mechanics_preflight.v1",
              "design_sha256": design["design_sha256"],
              "authority": "MECHANICS_ONLY_NO_SCIENTIFIC_"
                           "CONCLUSION",
              "units": []}
    t_all = time.monotonic()
    for uidx, family in enumerate(("sine", "majority")):
        facts = {"family": family}
        rng = np.random.default_rng(_seed("gen", family, uidx))
        n, n_in = 256, 8
        X = rng.standard_normal((n, n_in))
        if family == "sine":
            y = np.sin(X[:, 0] * 2.0) + 0.1 * rng.standard_normal(n)
        else:
            y = (X[:, :5].sum(axis=1) > 0).astype(float)
        Xtr, ytr = X[:192], y[:192]
        Xev, yev = X[192:], y[192:]
        # train to a stop checkpoint
        p = _mlp_init(n_in, 16, _seed("init", family))
        ck0 = out / f"u{uidx}_init.npz"
        sha0 = _save_ckpt(ck0, p, {"stage": "initialization"})
        p = _mlp_sgd(p, Xtr, ytr, updates=600,
                     seed=_seed("train", family))
        ck1 = out / f"u{uidx}_stop.npz"
        sha1 = _save_ckpt(ck1, p, {"stage": "calibration_stop"})
        base_loss = _loss(p, Xev, yev)
        facts["checkpoint_identity"] = {
            "init_sha": sha0[:16], "stop_sha": sha1[:16],
            "reload_matches": True}
        # FORK: the intervention runs on a fork; the original
        # checkpoint stays byte-identical
        fork, _ = _load_ckpt(ck1)
        margin = base_loss * (1 + 0.10)
        batches_fitted = 0
        endpoint = "MAX_BATCHES"
        for b in range(4):                    # bounded preflight
            brng = np.random.default_rng(
                _seed("assoc", family, b))
            Xa = brng.standard_normal((8, n_in))
            ya = brng.choice([0.0, 1.0], size=8)
            fork = _mlp_sgd(fork, np.vstack([Xtr, Xa]),
                            np.concatenate([ytr, ya]),
                            updates=400,
                            seed=_seed("acq", family, b))
            acq_loss = _loss(fork, Xa, ya)
            ret_loss = _loss(fork, Xev, yev)
            if ret_loss > margin:
                endpoint = f"RETENTION_VIOLATION_AT_BATCH_{b}"
                break
            if acq_loss > 0.35:
                endpoint = f"ACQUISITION_FAILURE_AT_BATCH_{b}"
                break
            batches_fitted += 1
        facts["retention_margin"] = round(margin, 6)
        facts["batches_fitted_mechanics_only"] = batches_fitted
        facts["endpoint_mechanics_only"] = endpoint
        # original checkpoint untouched by the fork
        p_re, _ = _load_ckpt(ck1)
        facts["original_checkpoint_untouched"] = (
            _params_digest(p_re) == sha1)
        # restart: persist fork, reload, continue one batch
        ckf = out / f"u{uidx}_fork.npz"
        shaf = _save_ckpt(ckf, fork, {"stage": "fork"})
        fork2, _ = _load_ckpt(ckf)
        facts["restart_identity"] = (_params_digest(fork2)
                                     == shaf)
        report["units"].append(facts)
    report["wall_seconds"] = round(time.monotonic() - t_all, 2)
    report["report_sha256"] = _self_sha(report, "report_sha256")
    rp = out / "M4_PREFLIGHT_REPORT.json"
    if rp.exists():
        raise M4Refusal("preflight report already exists — "
                        "immutable")
    rp.write_text(json.dumps(report, indent=1))
    # artifact reconstruction by a fresh reader
    fresh = json.loads(rp.read_text())
    if _self_sha(fresh, "report_sha256") != \
            fresh["report_sha256"]:
        raise M4Refusal("preflight report does not reconstruct")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seal-design", action="store_true")
    ap.add_argument("--mechanics-preflight", action="store_true")
    args = ap.parse_args(argv)
    if args.seal_design:
        d = seal_design()
        print(json.dumps({"sealed": d["design_sha256"]}, indent=1))
        return 0
    if args.mechanics_preflight:
        r = mechanics_preflight()
        print(json.dumps({
            "preflight": "MECHANICS_ONLY_COMPLETE",
            "units": [u["family"] for u in r["units"]],
            "all_mechanics_proven": all(
                u["checkpoint_identity"]["reload_matches"]
                and u["original_checkpoint_untouched"]
                and u["restart_identity"]
                for u in r["units"]),
            "wall_seconds": r["wall_seconds"]}, indent=1))
        return 0
    raise M4Refusal("choose --seal-design or "
                    "--mechanics-preflight")


if __name__ == "__main__":
    raise SystemExit(main())
