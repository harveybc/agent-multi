"""PRE freeze for order T0-T1 C1-C12 (@audit 2026-09-06): the twelve
public counterexamples reproduced against preprocessor@9359ccb4 and
custody@a8d2bcd2."""
import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "tools"))
PREP = Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"
sys.path.insert(0, str(PREP))
from app import causal_operators as co  # noqa: E402
import t1_adjudicator as adj  # noqa: E402
import t1_known_truth_bank as bank  # noqa: E402

COLS = ["a"]
SPEC = {"schema": co.SCHEMA_VERSION, "operator_id": "e",
        "kind": "ewma", "version": "1", "params": {"alpha": 0.3},
        "columns": COLS, "fit_role": "train", "lookback": 1,
        "availability_rule": "bar_close"}
TRAIN = np.random.default_rng(0).normal(0, 1, (100, 1))
ART = co.fit(SPEC, TRAIN, COLS, "train")

print("== 1: bool and numeric-string matrices coerce and execute ==")
out_b = co.transform_batch(ART, np.array([[True], [False]]), COLS)
out_s = co.transform_batch(ART, np.array([["1.25"], ["2.5"]],
                                         dtype=object), COLS)
print("bool matrix executed ->", out_b.ravel().tolist())
print("string matrix executed ->", out_s.ravel().tolist())

print("\n== 2: arbitrary future availability rule executes ==")
s2 = dict(SPEC, availability_rule="tomorrow_after_decision")
art2 = co.fit(s2, TRAIN, COLS, "train")
out = co.transform_batch(art2, TRAIN[:5], COLS)
print("rule 'tomorrow_after_decision' executed", out.shape,
      "— the API consumes no timestamps at all")

print("\n== 3: state from alpha 0.3 accepted by alpha 0.9 ==")
state = co.init_state(ART)
_, state = co.transform_incremental(ART, state, TRAIN[:10], COLS)
art9 = co.fit(dict(SPEC, params={"alpha": 0.9}), TRAIN, COLS,
              "train")
out, _ = co.transform_incremental(art9, state, TRAIN[10:12], COLS)
print("foreign state ACCEPTED by a different artifact ->",
      out.ravel().tolist())

print("\n== 4: DAG edge a->z validates ==")
child = dict(SPEC, operator_id="child", columns=["z"])
order = co.validate_dag({"root": dict(SPEC), "kid": child},
                        [("root", "kid")], COLS)
print("incompatible child ACCEPTED; order:", order)

print("\n== 5: same-name artifact save overwrites ==")
import tempfile
with tempfile.TemporaryDirectory() as td:
    p1 = co.save_artifact(ART, Path(td), "x")
    first = p1.read_text()
    co.save_artifact(art9, Path(td), "x")
    print("overwritten:", p1.read_text() != first)
    assert p1.read_text() != first

print("\n== 6: delayed observed != clean + realized_noise ==")
with tempfile.TemporaryDirectory() as td:
    rec = bank.materialize_unit(
        {"family": "heavisine", "perturbation": "delayed",
         "snr_db": 10, "heterogeneous": False}, 11, Path(td))
    u = Path(td) / rec["unit_id"]
    clean = np.load(u / "clean_signal.npy")
    noise = np.load(u / "realized_noise.npy")
    obs = np.load(u / "observed_signal.npy")
    disc = float(np.max(np.abs(obs - (clean + noise))))
    print(f"max |observed - (clean + recorded_noise)| = {disc:.2f}")
    assert disc > 1.0

print("\n== 7: committed measurement JSON has non-finite "
      "literals ==")
MEAS = (REPO / "docs/audits/evidence/t1_lab_20260906/"
        "T1_MEASUREMENTS.json")
raw = MEAS.read_text()
print("literal NaN occurrences:", raw.count("NaN"))
assert raw.count("NaN") > 0

print("\n== 8: NaN in every gate metric -> LAB_CALIBRATED ==")
recs = []
for seed in (11, 12, 13):
    recs.append({"unit_id": f"u{seed}", "operator": "ewma",
                 "family": "bumps", "perturbation": "white",
                 "declared_snr_db": 10, "heterogeneous": False,
                 "seed": seed, "causal": True,
                 "oracle_only": False, "status": "MEASURED",
                 "per_variable": [{
                     "variable": "v0", "true_snr_db": 10,
                     "snr_estimator_std_error": float("nan"),
                     "mse_observed": float("nan"),
                     "mse_denoised": float("nan"),
                     "snr_gain_db": float("nan"),
                     "delay_bars": 0,
                     "extreme_retention": float("nan"),
                     "tail_ratio": float("nan"),
                     "ljung_box_p_residual_train": float("nan"),
                     "assays": {h: {"X": float("nan"),
                                    "D": float("nan"),
                                    "XDR": float("nan"),
                                    "capacity_control_XXX":
                                        float("nan"),
                                    "residual_incremental_r2":
                                        float("nan"),
                                    "persistence_obs_mse":
                                        float("nan")}
                                for h in ("h1", "h5")}}]})
out = adj.adjudicate({"records": recs, "expected_units": 3,
                      "expected_operators": 1})
v = list(out["verdicts"].values())[0]
print("all-NaN gates verdict:", v["verdict"])
assert v["verdict"] == "LAB_CALIBRATED"

print("\n== 9: self-declared 3-record population adjudicates ==")
print("expected_units/operators come from the PRODUCER dict:",
      "3 records adjudicated above with self-declared counts")

print("\n== 10: gate metrics computed across train+val+score ==")
lab = (REPO / "tools/t1_lab_run.py").read_text()
seg = lab[lab.index("mse_obs = "):lab.index("est_std")]
print("reconstruction uses obs[j] (FULL 2048 samples):",
      "obs[j] - clean[j]" in seg)
assert "obs[j] - clean[j]" in seg

print("\n== 11: [X,X,X] changes ridge's effective "
      "regularization ==")
rng = np.random.default_rng(5)
X = rng.normal(0, 1, (200, 8))
y = X @ rng.normal(0, 1, 8) + rng.normal(0, 0.1, 200)
lam = 1.0
w1 = np.linalg.solve(X.T @ X + lam * np.eye(8), X.T @ y)
X3 = np.hstack([X, X, X])
w3 = np.linalg.solve(X3.T @ X3 + lam * np.eye(24), X3.T @ y)
p1 = X @ w1
p3 = X3 @ w3
print(f"max |pred_X - pred_XXX| = "
      f"{float(np.max(np.abs(p1 - p3))):.4f} (nonzero: "
      "duplication under a fixed penalty is NOT capacity-neutral)")
assert float(np.max(np.abs(p1 - p3))) > 1e-3

print("\n== 12: execution without the sealed design ==")
print("lab verifies sealed design digest:",
      "T1_LAB_DESIGN" in lab)
adj_src = (REPO / "tools/t1_adjudicator.py").read_text()
print("adjudicator consumes design/inventory:",
      "T1_LAB_DESIGN" in adj_src or "BANK_INVENTORY" in adj_src)
print("lab default embeds a worktree path:",
      "prep-t0t1" in lab)
assert "T1_LAB_DESIGN" not in lab and "prep-t0t1" in lab

print("\nPRE CONFIRMED: all twelve reproduce")
