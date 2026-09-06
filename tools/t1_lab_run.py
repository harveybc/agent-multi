#!/usr/bin/env python3
"""T1.3 measurement lab v2 (orders T0-T1 + C1-C12).

C11: before measuring, the runner verifies the SEALED design bytes
and digest, the exact code identities it names, and the bank
inventory with per-unit digests. The preprocessor root is an
EXPLICIT parameter or env var — no worktree default, no home
topology in evidence.

C6: evidence is observation-level — every unit x operator produces
a content-addressed NPZ (denoised + residual arrays per role
support) whose digest enters the record; gate metrics are derivable
from those arrays by an independent verifier.

C8: every gate metric is computed PER ROLE (train / validation /
score) and published separately; the primary verdict inputs are the
SCORE-role facts only.

C19 (audit 2026-09-06): the dimensionality control is
TRAIN-FROZEN INDEPENDENT nuisance channels — deterministic rng
seeded by (unit, operator, channel), scaled by the TRAIN-role
standard deviation of the observed series only. No channel value
depends on any observed row, so no temporal role can leak through
the control; width matches [X,D,R] exactly.

C7: JSON is emitted with allow_nan=False; non-finite gate values
become typed nulls with a reason and can never authorize
calibration."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

HORIZONS = (1, 5)
RIDGE_LAGS = 8
RIDGE_LAMBDA = 1.0
NUISANCE_CHANNELS = 2
CANDIDATES = ("identity", "trailing_mean", "trailing_median",
              "ewma", "local_level_kalman")
ORACLE = "centered_mean_oracle"
PARAMS = {"trailing_mean": {"window": 5},
          "trailing_median": {"window": 5},
          "ewma": {"alpha": 0.3},
          "identity": {}, "local_level_kalman": {},
          "centered_mean_oracle": {"window": 5}}


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()
                          ).hexdigest()


def _rss() -> int:
    with open("/proc/self/statm") as fh:
        return int(fh.read().split()[1]) * 4096


def _finite_or_null(x):
    """C7: a gate-bearing number is finite or a typed null."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return {"null": True, "reason": "non-numeric"}
    if not np.isfinite(v):
        return {"null": True, "reason": "non-finite"}
    return v


def resolve_preprocessor_root() -> Path:
    root = os.environ.get("B4_T1_PREPROCESSOR_ROOT")
    if not root:
        raise SystemExit(
            "REFUSED: set B4_T1_PREPROCESSOR_ROOT to the reviewed "
            "preprocessor checkout — no default worktree topology")
    p = Path(root)
    if not (p / "app/causal_operators.py").is_file():
        raise SystemExit(
            "REFUSED: preprocessor root does not carry the operator "
            "module")
    return p


def load_co():
    # spec-load by file: agent-multi carries its OWN `app` package,
    # so a name import can collide when other modules loaded first.
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "t1_causal_operators",
        resolve_preprocessor_root() / "app/causal_operators.py")
    co = ilu.module_from_spec(spec)
    spec.loader.exec_module(co)
    return co


def _spec(co, kind, columns):
    oid = (f"{kind}_t1" if kind != ORACLE
           else "NON_CAUSAL_ORACLE_ONLY_centered_mean")
    return {"schema": co.SCHEMA_VERSION, "operator_id": oid,
            "kind": kind, "version": "1",
            "params": dict(PARAMS[kind]), "columns": list(columns),
            "fit_role": "train",
            "lookback": co.derived_lookback(kind, PARAMS[kind]),
            "availability_rule": "bar_close"}


def ridge_r2(X, y, Xs, ys, lam=RIDGE_LAMBDA):
    XtX = X.T @ X + lam * np.eye(X.shape[1])
    w = np.linalg.solve(XtX, X.T @ y)
    pred = Xs @ w
    ss = float(np.sum((ys - pred) ** 2))
    tot = float(np.sum((ys - ys.mean()) ** 2))
    return 1.0 - ss / max(tot, 1e-12)


def _lag_matrix(series_list, lo, hi, h):
    rows = []
    for t in range(lo + RIDGE_LAGS, hi - h):
        feats = []
        for s in series_list:
            feats.extend(s[t - RIDGE_LAGS + 1: t + 1])
        rows.append(feats)
    return np.array(rows)


def _targets(clean_var, lo, hi, h):
    return np.array([clean_var[t + h]
                     for t in range(lo + RIDGE_LAGS, hi - h)])


def nuisance_channels(obs_j, roles, ident: str) -> list:
    """C19: width-control channels generated INDEPENDENTLY of the
    observed series — deterministic rng seeded by the declared
    identity, amplitude frozen on the TRAIN role's std only. No
    value depends on any observed row, so future-row mutations and
    role boundaries cannot reach the control."""
    lo_t, hi_t = roles["train"]
    scale = float(np.std(obs_j[lo_t:hi_t]))
    out = []
    for k in range(NUISANCE_CHANNELS):
        seed = int(hashlib.sha256(
            f"t1_nuisance|{ident}|{k}".encode()
        ).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        out.append(rng.normal(0.0, scale if scale > 0 else 1.0,
                              obs_j.shape[0]))
    return out


def assay_arms(clean_j, obs_j, d_j, r_j, roles, h,
               nuisance) -> dict:
    """Frozen assays: fit on train, report on score. The width
    control matches [X,D,R]'s input width with train-frozen
    INDEPENDENT channels (see nuisance_channels); every arm uses
    the same eligible support."""
    lo_t, hi_t = roles["train"]
    lo_s, hi_s = roles["score"]
    n1, n2 = nuisance
    arms = {"X": [obs_j], "D": [d_j], "XDR": [obs_j, d_j, r_j],
            "width_control_X_nuisance": [obs_j, n1, n2]}
    out = {}
    yt = _targets(clean_j, lo_t, hi_t, h)
    ys = _targets(clean_j, lo_s, hi_s, h)
    for name, series in arms.items():
        Xt = _lag_matrix(series, lo_t, hi_t, h)
        Xs = _lag_matrix(series, lo_s, hi_s, h)
        out[name] = _finite_or_null(ridge_r2(Xt, yt, Xs, ys))
    if isinstance(out["XDR"], float) and \
            isinstance(out["width_control_X_nuisance"], float):
        out["residual_incremental_r2"] = (
            out["XDR"] - out["width_control_X_nuisance"])
    else:
        out["residual_incremental_r2"] = {
            "null": True, "reason": "component non-finite"}
    return out


def _role_metrics(clean_j, obs_j, d_j, sup_j, roles) -> dict:
    """C8: reconstruction/SNR/extremes/tails PER ROLE."""
    facts = {}
    for role, (lo, hi) in roles.items():
        sl = slice(lo, hi)
        sup = sup_j[sl]
        c, o, d = clean_j[sl][sup], obs_j[sl][sup], d_j[sl][sup]
        if len(c) < 20:
            facts[role] = {"null": True, "reason": "support<20"}
            continue
        mse_o = float(np.mean((o - c) ** 2))
        mse_d = float(np.mean((d - c) ** 2))
        gain = (10 * np.log10(mse_o / mse_d)
                if mse_o > 0 and mse_d > 0 else 0.0)
        peaks = np.argsort(np.abs(c))[-5:]
        denom = np.where(np.abs(c[peaks]) < 1e-9, 1.0, c[peaks])
        retention = float(np.mean(np.clip(d[peaks] / denom,
                                          -2, 2)))
        noise_like = o - c
        tail = (float(np.max(np.abs(d - c))
                      / max(np.max(np.abs(noise_like)), 1e-12)))
        facts[role] = {
            "mse_observed": _finite_or_null(mse_o),
            "mse_denoised": _finite_or_null(mse_d),
            "snr_gain_db": _finite_or_null(gain),
            "extreme_retention": _finite_or_null(retention),
            "tail_ratio": _finite_or_null(tail)}
    return facts


def measure_unit_operator(co, unit_dir: Path, kind: str,
                          npz_dir: Path) -> dict:
    rec = json.loads((unit_dir / "UNIT.json").read_text())
    clean = np.load(unit_dir / "clean_signal.npy")
    additive = np.load(unit_dir / "additive_noise.npy")
    obs = np.load(unit_dir / "observed_signal.npy")
    support = np.load(unit_dir / "metric_support.npy")
    for name, arr in (("clean_signal", clean),
                      ("additive_noise", additive),
                      ("observed_signal", obs),
                      ("metric_support", support)):
        if _sha_arr(arr) != rec["digests"][name]:
            raise SystemExit(f"REFUSED: {name} digest broken in "
                             f"{rec['unit_id']}")
    v = clean.shape[0]
    columns = [f"v{j}" for j in range(v)]
    roles = {k: tuple(x) for k, x in rec["temporal_roles"].items()}
    lo_t, hi_t = roles["train"]
    t0 = time.perf_counter()
    result = {"schema": "agent_multi.t1_measurement.v2",
              "unit_id": rec["unit_id"], "operator": kind,
              "family": rec["family"],
              "perturbation": rec["perturbation"],
              "declared_snr_db": rec["declared_snr_db"],
              "heterogeneous": rec["heterogeneous"],
              "seed": rec["seed"],
              "causal": kind != ORACLE,
              "oracle_only": kind == ORACLE,
              "source_digests": rec["digests"]}
    spec = _spec(co, kind, columns)
    try:
        if rec["missing_mask"]:
            raise co.CausalOperatorError(
                "unit contains licensed missingness; current "
                "candidates declare NO missingness policy — typed "
                "refusal recorded as the missingness behavior")
        stream = f"t1_unit:{rec['unit_id']}"
        train_m = obs[:, lo_t:hi_t].T
        art = co.fit(spec, train_m, columns, "train",
                     co.make_train_contract(
                         train_m, train_m.shape[0],
                         float(lo_t), stream))
        n = obs.shape[1]
        tc = co.make_bar_close_contract(n, 0.0, stream)
        d = co.transform_batch(art, obs.T, columns, tc).T
    except co.CausalOperatorError as exc:
        result.update({"status": "TYPED_REFUSAL",
                       "refusal": str(exc),
                       "cpu_wall_seconds":
                           round(time.perf_counter() - t0, 3)})
        return result
    r = obs - d
    npz_dir.mkdir(parents=True, exist_ok=True)
    payload = {"denoised": d, "residual": r}
    blob_sha = hashlib.sha256(
        d.tobytes() + r.tobytes()).hexdigest()
    npz_path = npz_dir / f"{blob_sha}.npz"
    if not npz_path.exists():
        np.savez_compressed(npz_path, **payload)
    per_var = []
    for j in range(v):
        role_facts = _role_metrics(clean[j], obs[j], d[j],
                                   support[j], roles)
        est_std = float(np.std(np.diff(obs[j, lo_t:hi_t]))
                        / np.sqrt(2))
        true_std = rec["noise_std_per_var"][j]
        lags = range(-10, 11)
        xc = [float(np.corrcoef(
            clean[j, 10:-10],
            d[j, 10 + k:len(d[j]) - 10 + k])[0, 1])
            for k in lags]
        delay = int(list(lags)[int(np.argmax(xc))])
        h_facts = {}
        nuis = nuisance_channels(
            obs[j], roles, f"{rec['unit_id']}|{kind}|v{j}")
        for h in HORIZONS:
            h_facts[f"h{h}"] = assay_arms(clean[j], obs[j], d[j],
                                          r[j], roles, h, nuis)
        per_var.append({
            "variable": columns[j],
            "true_additive_snr_db":
                rec["true_additive_snr_db"][j],
            "true_total_error_snr_db":
                rec["true_total_observation_error_snr_db"][j],
            "snr_estimator_std_error": _finite_or_null(
                abs(est_std - true_std) / true_std
                if true_std > 0 else 0.0),
            "delay_bars": delay,
            "by_role": role_facts,
            "assays_score_fit_train": h_facts})
    result.update({
        "status": "MEASURED",
        "per_variable": per_var,
        "denoised_residual_npz_sha256": blob_sha,
        "output_column_expansion": {"X": 1, "D": 1, "XDR": 3},
        "cpu_wall_seconds": round(time.perf_counter() - t0, 3),
        "peak_rss_bytes": _rss(),
        "artifact_sha256": art["artifact_sha256"],
        "spec_sha256": art["spec_sha256"]})
    return result


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", type=Path, required=True)
    ap.add_argument("--design-sha", required=True)
    ap.add_argument("--bank-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--npz-dir", type=Path, required=True)
    args = ap.parse_args()
    import torch
    if torch.cuda.is_available() and os.environ.get(
            "CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("REFUSED: CUDA visible — T1 is CPU-only")
    # C11: sealed-design binding BEFORE anything else
    raw = args.design.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.design_sha:
        raise SystemExit(
            "REFUSED: sealed design bytes differ from the reviewed "
            "digest")
    design = json.loads(raw)
    co = load_co()
    want_co = design["operator_protocol"][
        "causal_operators_sha256"]
    if co.code_identity() != want_co:
        raise SystemExit(
            "REFUSED: operator code identity differs from the "
            "sealed design")
    inv_path = args.bank_dir / "BANK_INVENTORY.json"
    inv = json.loads(inv_path.read_text())
    if design.get("bank_inventory_sha256") not in (
            None, _sha_file(inv_path)):
        raise SystemExit(
            "REFUSED: bank inventory differs from the sealed design")
    if inv.get("schema") != "agent_multi.t1_bank_inventory.v2":
        raise SystemExit(
            "REFUSED: the bank inventory does not bind the "
            "physical population (v2 schema required)")
    # C17: recompute EVERY unit digest from bytes BEFORE measuring
    for uid in inv["unit_ids"]:
        bound = inv["units"][uid]
        ud = args.bank_dir / uid
        if _sha_file(ud / "UNIT.json") != \
                bound["unit_json_sha256"]:
            raise SystemExit(
                f"REFUSED: unit {uid} metadata differs from the "
                "sealed inventory")
        for name, want in bound["arrays"].items():
            if _sha_file(ud / f"{name}.npy") != want:
                raise SystemExit(
                    f"REFUSED: unit {uid} array {name} differs "
                    "from the sealed inventory")
        u = json.loads((ud / "UNIT.json").read_text())
        if u.get("schema") != "agent_multi.t1_unit.v2":
            raise SystemExit(f"REFUSED: unit {uid} is not a v2 "
                             "unit")
    records = []
    for uid in inv["unit_ids"]:
        for kind in CANDIDATES + (ORACLE,):
            records.append(measure_unit_operator(
                co, args.bank_dir / uid, kind, args.npz_dir))
    payload = {"schema": "agent_multi.t1_measurements.v2",
               "design_sha256": args.design_sha,
               "bank_inventory_sha256": _sha_file(inv_path),
               "code_identity": {
                   "causal_operators_sha256": co.code_identity(),
                   "lab_sha256": _sha_file(Path(__file__))},
               "records": records}
    args.output.write_text(json.dumps(payload, indent=1,
                                      allow_nan=False))
    # C18: immutable measurement-population manifest — the exact
    # identities a reviewer promotes; the candidate cannot rewrite
    # them coherently without changing this digest.
    manifest = {
        "schema": "agent_multi.t1_measurement_manifest.v1",
        "design_sha256": args.design_sha,
        "bank_inventory_sha256": _sha_file(inv_path),
        "measurements_sha256": _sha_file(args.output),
        "records_total": len(records),
        "records_measured": sum(1 for r in records
                                if r["status"] == "MEASURED"),
        "records_refused": sum(1 for r in records
                               if r["status"] == "TYPED_REFUSAL")}
    manifest["manifest_sha256"] = hashlib.sha256(json.dumps(
        {k: manifest[k] for k in sorted(manifest)},
        sort_keys=True).encode()).hexdigest()
    mp = args.output.with_name(args.output.stem +
                               "_MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=1))
    print(json.dumps({"units": len(inv["unit_ids"]),
                      "records": len(records),
                      "refusals": sum(
                          1 for r in records
                          if r["status"] == "TYPED_REFUSAL")},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
