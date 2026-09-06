#!/usr/bin/env python3
"""T1.1 known-truth CPU bank (work plan 43 §5.1, order T0-T1).

Self-generated, seeded, no downloads. Every unit carries the clean
signal, the realized perturbation, the observation, exact provenance
(family, perturbation, per-variable SNR, seed, params) and temporal
roles materialized BEFORE any operator is fit. The predeclared
matrix identifies each main effect and the interactions needed for
the per-variable decision — never a blind cartesian sweep."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
N = 2048
ROLES = {"train": (0, 1228), "validation": (1228, 1638),
         "score": (1638, 2048)}          # 60/20/20, materialized
SNR_GRID_DB = ("inf", 20, 10, 5, 0, -5)
FAMILIES = ("blocks", "bumps", "heavisine", "doppler",
            "sine", "chirp", "am", "state_space_mv")
PERTURBATIONS = ("white", "colored", "cross_correlated",
                 "heteroscedastic", "impulsive", "missing",
                 "delayed")
SEEDS = (11, 12, 13)


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()
                          ).hexdigest()


# ------------------------- signal families ------------------------
def clean_signal(family: str, rng: np.random.Generator
                 ) -> np.ndarray:
    t = np.linspace(0, 1, N, endpoint=False)
    if family == "blocks":
        pos = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65,
               0.76, 0.78, 0.81]
        hts = [4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 2.1, -4.2]
        x = np.zeros(N)
        for p, h in zip(pos, hts):
            x += h * (1 + np.sign(t - p)) / 2
        return x[None, :]
    if family == "bumps":
        pos = [0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65,
               0.76, 0.78, 0.81]
        hts = [4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2]
        wid = [0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01,
               0.005, 0.008, 0.005]
        x = np.zeros(N)
        for p, h, w in zip(pos, hts, wid):
            x += h / (1 + np.abs((t - p) / w)) ** 4
        return x[None, :]
    if family == "heavisine":
        return (4 * np.sin(4 * np.pi * t)
                - np.sign(t - 0.3)
                - np.sign(0.72 - t))[None, :]
    if family == "doppler":
        eps = 0.05
        return (np.sqrt(t * (1 - t))
                * np.sin(2 * np.pi * (1 + eps) / (t + eps)))[None, :]
    if family == "sine":
        return np.sin(2 * np.pi * 8 * t)[None, :]
    if family == "chirp":
        return np.sin(2 * np.pi * (4 * t + 12 * t ** 2))[None, :]
    if family == "am":
        return ((1 + 0.6 * np.sin(2 * np.pi * 2 * t))
                * np.sin(2 * np.pi * 16 * t))[None, :]
    if family == "state_space_mv":
        # 3 variables: shared AR(1) common component + private AR(1)
        common = np.zeros(N)
        priv = np.zeros((3, N))
        for i in range(1, N):
            common[i] = 0.97 * common[i - 1] + rng.normal(0, 0.3)
            for j in range(3):
                priv[j, i] = (0.9 * priv[j, i - 1]
                              + rng.normal(0, 0.2))
        loads = np.array([1.0, 0.7, -0.5])
        return loads[:, None] * common[None, :] + priv
    raise SystemExit(f"REFUSED: unknown family {family!r}")


# -------------------------- perturbations -------------------------
def realized_noise(kind: str, clean: np.ndarray, snr_db,
                   rng: np.random.Generator) -> dict:
    v, n = clean.shape
    sig_pow = np.mean(clean ** 2, axis=1)
    if snr_db == "inf":
        return {"noise": np.zeros_like(clean), "mask": None,
                "delay": None,
                "noise_std_per_var": [0.0] * v}
    target = sig_pow / (10 ** (float(snr_db) / 10.0))
    std = np.sqrt(np.maximum(target, 1e-24))
    if kind == "white":
        noise = rng.normal(0, 1, (v, n)) * std[:, None]
    elif kind == "colored":
        e = rng.normal(0, 1, (v, n))
        noise = np.zeros_like(e)
        for i in range(1, n):
            noise[:, i] = 0.8 * noise[:, i - 1] + e[:, i]
        noise *= (std / np.std(noise, axis=1))[:, None]
    elif kind == "cross_correlated":
        shared = rng.normal(0, 1, n)
        own = rng.normal(0, 1, (v, n))
        noise = 0.8 * shared[None, :] + 0.6 * own
        noise *= (std / np.std(noise, axis=1))[:, None]
    elif kind == "heteroscedastic":
        scale = 1.0 + 1.5 * (np.arange(n) / n)
        noise = rng.normal(0, 1, (v, n)) * scale[None, :]
        noise *= (std / np.std(noise, axis=1))[:, None]
    elif kind == "impulsive":
        base = rng.normal(0, 0.3, (v, n))
        spikes = (rng.random((v, n)) < 0.01).astype(float)
        noise = base + spikes * rng.normal(0, 5, (v, n))
        noise *= (std / np.std(noise, axis=1))[:, None]
    elif kind == "missing":
        noise = rng.normal(0, 1, (v, n)) * std[:, None]
        mask = rng.random((v, n)) < 0.05
        return {"noise": noise, "mask": mask, "delay": None,
                "noise_std_per_var": [float(s) for s in std]}
    elif kind == "delayed":
        noise = rng.normal(0, 1, (v, n)) * std[:, None]
        return {"noise": noise, "mask": None, "delay": 2,
                "noise_std_per_var": [float(s) for s in std]}
    else:
        raise SystemExit(f"REFUSED: unknown perturbation {kind!r}")
    return {"noise": noise, "mask": None, "delay": None,
            "noise_std_per_var": [float(s) for s in std]}


def predeclared_matrix() -> list:
    """Main effects + needed interactions, per §5.1 — NOT the full
    cartesian product:
    - white noise x FULL SNR grid x every family (SNR main effect);
    - every non-white perturbation at 10 dB on two contrasting
      families (perturbation main effect);
    - impulsive x {0, -5} dB on bumps (extreme-preservation
      interaction);
    - state_space_mv heterogeneous per-variable SNR (10/0/-5 dB)
      under white and cross-correlated (the per-variable decision);
    all crossed with three seeds."""
    cells = []
    for fam in FAMILIES:
        for snr in SNR_GRID_DB:
            cells.append({"family": fam, "perturbation": "white",
                          "snr_db": snr, "heterogeneous": False})
    for pert in ("colored", "cross_correlated", "heteroscedastic",
                 "impulsive", "missing", "delayed"):
        for fam in ("heavisine", "bumps"):
            cells.append({"family": fam, "perturbation": pert,
                          "snr_db": 10, "heterogeneous": False})
    for snr in (0, -5):
        cells.append({"family": "bumps", "perturbation": "impulsive",
                      "snr_db": snr, "heterogeneous": False})
    for pert in ("white", "cross_correlated"):
        cells.append({"family": "state_space_mv",
                      "perturbation": pert,
                      "snr_db": [10, 0, -5],
                      "heterogeneous": True})
    return cells


def materialize_unit(cell: dict, seed: int, out_dir: Path) -> dict:
    rng = np.random.default_rng(
        int(hashlib.sha256(json.dumps(
            {"cell": cell, "seed": seed},
            sort_keys=True).encode()).hexdigest()[:8], 16))
    clean = clean_signal(cell["family"], rng)
    v = clean.shape[0]
    if cell["heterogeneous"]:
        assert v == len(cell["snr_db"])
        parts = [realized_noise(cell["perturbation"],
                                clean[j:j + 1], cell["snr_db"][j],
                                rng) for j in range(v)]
        noise = np.vstack([p["noise"] for p in parts])
        mask = None
        delay = None
        stds = [p["noise_std_per_var"][0] for p in parts]
    else:
        p = realized_noise(cell["perturbation"], clean,
                           cell["snr_db"], rng)
        noise, mask, delay = p["noise"], p["mask"], p["delay"]
        stds = p["noise_std_per_var"]
    observed = clean + noise
    if mask is not None:
        observed = observed.copy()
        observed[mask] = np.nan
    if delay:
        observed = np.roll(observed, delay, axis=1)
        observed[:, :delay] = observed[:, delay][..., None] * 0 + \
            observed[:, delay:delay + 1]
    unit_id = (f"{cell['family']}__{cell['perturbation']}__"
               f"snr{cell['snr_db']}__"
               f"{'het' if cell['heterogeneous'] else 'hom'}"
               f"__seed{seed}").replace(" ", "").replace(",", "_")
    unit_dir = out_dir / unit_id
    unit_dir.mkdir(parents=True, exist_ok=True)
    np.save(unit_dir / "clean_signal.npy", clean)
    np.save(unit_dir / "realized_noise.npy", noise)
    np.save(unit_dir / "observed_signal.npy", observed)
    true_snr = []
    for j in range(v):
        npow = float(np.mean(noise[j] ** 2))
        true_snr.append("inf" if npow == 0 else
                        float(10 * np.log10(
                            np.mean(clean[j] ** 2) / npow)))
    record = {
        "schema": "agent_multi.t1_unit.v1",
        "unit_id": unit_id,
        "family": cell["family"],
        "perturbation": cell["perturbation"],
        "declared_snr_db": cell["snr_db"],
        "true_realized_snr_db": true_snr,
        "heterogeneous": cell["heterogeneous"],
        "seed": seed,
        "n_variables": v, "n_samples": N,
        "noise_std_per_var": stds,
        "missing_mask": bool(mask is not None),
        "observation_delay_bars": delay or 0,
        "temporal_roles": {k: list(vr) for k, vr in ROLES.items()},
        "roles_materialized_before_any_fit": True,
        "digests": {"clean_signal": _sha(clean),
                    "realized_noise": _sha(noise),
                    "observed_signal": _sha(observed)},
    }
    (unit_dir / "UNIT.json").write_text(json.dumps(record, indent=1))
    return record


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    cells = predeclared_matrix()
    ledger = []
    for cell in cells:
        for seed in SEEDS:
            ledger.append(materialize_unit(cell, seed,
                                           args.output_dir))
    inventory = {
        "schema": "agent_multi.t1_bank_inventory.v1",
        "predeclared_cells": len(cells),
        "seeds": list(SEEDS),
        "units_total": len(ledger),
        "unit_ids": [r["unit_id"] for r in ledger],
        "design_rule": predeclared_matrix.__doc__,
    }
    (args.output_dir / "BANK_INVENTORY.json").write_text(
        json.dumps(inventory, indent=1))
    print(json.dumps({"cells": len(cells),
                      "units": len(ledger)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
