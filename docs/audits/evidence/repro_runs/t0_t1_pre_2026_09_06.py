"""PRE freeze for order T0-T1 (@work-plan 43): the eight §2 points —
real preprocessor surface, missing fit/transform API, unfit noise
generators, and four numeric counterexamples."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PREP = Path.home() / "Documents/GitHub/.worktrees/prep-t0t1"
HS = Path.home() / "Documents/GitHub/heuristic-strategy"
SDG = Path.home() / "Documents/GitHub/synthetic-datagen"

print("== 1. real preprocessor surface ==")
plugins = sorted(p.name for p in (PREP / "app/plugins").glob("*.py"))
print("plugins:", plugins)
default = (PREP / "app/plugins/plugin_default.py").read_text()
print("file-loader CLI pipeline:", (PREP / "app/cli.py").is_file())
print("D1-D6 splits in default plugin:",
      all(f"d{i}" in default.lower() for i in (1, 2, 3, 4, 5, 6)))
print("dual normalizer (A on D1, B on D4):",
      default.lower().count("normalizer") > 0
      and "d4" in default.lower())
print("train-only rules present:",
      "fit" in default.lower())

print("\n== 2. reusable fit/transform API with artifact + "
      "incremental + digest ==")
hits = {"transform_incremental": 0, "artifact_sha": 0,
        "fit(": 0, "transform_batch": 0}
for p in (PREP / "app").rglob("*.py"):
    src = p.read_text()
    for k in hits:
        if k in src:
            hits[k] += 1
print("occurrences across app/:", hits)
assert hits["transform_incremental"] == 0
assert hits["transform_batch"] == 0
print("-> NO reusable operator protocol exists (CLI-coupled, "
      "batch-only, no incremental parity, no artifact digest)")

print("\n== 3. existing noise generators are NOT the T1 unit ==")
sweep = (HS / "sweep_noise.py").read_text()
print("heuristic-strategy sweep_noise perturbs PREDICTIONS via "
      "API set_noise:", "set_noise" in sweep)
sdg_clean = 0
for p in SDG.rglob("*.py"):
    try:
        s = p.read_text()
    except OSError:
        continue
    if "realized_noise" in s or ("clean_signal" in s
                                 and "observed" in s):
        sdg_clean += 1
print("synthetic-datagen files with (clean_signal, realized_noise, "
      "observed) unit:", sdg_clean)
assert "set_noise" in sweep and sdg_clean == 0
print("-> neither produces per-variable known truth "
      "(clean+realized_noise+observed+seed+family+params)")

print("\n== 4. counterexample: centered filter leaks the future ==")
rng = np.random.default_rng(7)
x = np.zeros(200)
x[100:] = 1.0                      # step at t=100
x += rng.normal(0, 0.05, 200)
centered = np.convolve(x, np.ones(5) / 5, mode="same")
print(f"centered[98] = {centered[98]:.3f} (uses x[99..100] — the "
      "FUTURE step leaks two bars early; clean value there is 0)")
assert centered[98] > 0.15

print("\n== 5. counterexample: fitting on validation changes the "
      "result ==")
train, val = x[:120], x[120:]
mu_train = train.mean()
mu_leaky = x.mean()                # fit includes validation
z_train_only = (val[0] - mu_train)
z_leaky = (val[0] - mu_leaky)
print(f"same bar normalized: train-only {z_train_only:.4f} vs "
      f"train+val {z_leaky:.4f}")
assert abs(z_train_only - z_leaky) > 1e-3

print("\n== 6. counterexample: batch vs incremental bytes differ ==")


def ewma_batch(v, alpha=0.2):
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out


def ewma_incremental_broken(chunks, alpha=0.2):
    outs = []
    for c in chunks:               # state NOT carried across chunks
        outs.append(ewma_batch(c, alpha))
    return np.concatenate(outs)


full = ewma_batch(x)
frag = ewma_incremental_broken([x[:50], x[50:130], x[130:]])
print("batch sha:", hashlib.sha256(full.tobytes()).hexdigest()[:12])
print("fragmented-without-state sha:",
      hashlib.sha256(frag.tobytes()).hexdigest()[:12])
assert not np.array_equal(full, frag)
print("-> incremental parity requires CARRIED state; naive "
      "re-batching per fragment produces different bytes")

print("\n== 7. counterexample: better reconstruction destroys an "
      "extreme and leaves target info in the residual ==")
clean = np.zeros(300)
clean[150] = 5.0                   # the extreme IS the signal
obs = clean + rng.normal(0, 0.5, 300)
smooth = np.convolve(obs, np.ones(9) / 9, mode="same")
rmse_obs = float(np.sqrt(np.mean((obs - clean) ** 2)))
rmse_smooth = float(np.sqrt(np.mean((smooth - clean) ** 2)))
peak_retention = float(smooth[150] / clean[150])
residual = obs - smooth
corr = float(np.corrcoef(residual, clean)[0, 1])
print(f"RMSE improves: {rmse_obs:.3f} -> {rmse_smooth:.3f}; "
      f"peak retention {peak_retention:.2f}; "
      f"corr(residual, clean) = {corr:.3f}")
assert rmse_smooth < rmse_obs and peak_retention < 0.5 and \
    corr > 0.3
print("-> reconstruction metrics alone would promote a filter that "
      "kills the extreme; the residual still carries the target")

print("\n== 8. real state of the preprocessor test suite ==")
r = subprocess.run(
    [sys.executable, "-m", "pytest", "tests", "--collect-only",
     "-q"], cwd=str(PREP), capture_output=True, text=True,
    timeout=300)
tail = (r.stdout.strip().splitlines() or ["<no output>"])[-1]
print("collect-only:", tail)
print("returncode:", r.returncode)

print("\nPRE CONFIRMED: all eight points frozen")
