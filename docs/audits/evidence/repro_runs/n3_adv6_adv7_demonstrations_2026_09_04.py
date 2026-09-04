"""Adversaries 6 and 7 (order @a13671ab §7), demonstrated against
the REAL regenerated table and acquisition:

adv6 — a feature generated from future data: shifting one feature
column by -1 bar (injecting the future) must blow the sealed parity
envelope on the frozen overlap.

adv7 — an internal gap hidden by forward fill: deleting one 2026
bar from a COPY of the acquisition and regenerating must refuse
with the executable 1458-bar grid check; ffill never bridges it.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
STAGING = (Path.home()
           / ".local/share/agent-multi/n3_fresh_staging_20260904")
PY = str(Path.home()
         / "anaconda3/envs/trading-stack/bin/python")

# ---- adv6: future-shifted feature vs the sealed parity envelope
frozen = pd.read_csv(
    Path.home() / "Documents/GitHub/predictor/examples/data/"
    "project3/ethusdt_4h_tech_stat_full_model_ready.csv",
    usecols=["rsi_14"])
regen = pd.read_parquet(STAGING / "model_ready_extended.parquet",
                        columns=["rsi_14"])
n = len(frozen)
honest = regen["rsi_14"].to_numpy()[:n]
leaky = np.roll(regen["rsi_14"].to_numpy(), -1)[:n]  # future bar
b = frozen["rsi_14"].to_numpy()


def envelope_violations(a, b):
    absdev = np.abs(a - b)
    with np.errstate(divide="ignore", invalid="ignore"):
        reldev = absdev / np.maximum(np.abs(b), 1e-30)
    both_nan = np.isnan(a) & np.isnan(b)
    inside = (absdev <= 1e-6) | (reldev <= 1e-5)
    return int((~inside & ~both_nan).sum())


v_honest = envelope_violations(honest, b)
v_leaky = envelope_violations(leaky, b)
print(f"adv6: honest rsi_14 cells outside envelope = {v_honest}")
print(f"adv6: future-shifted rsi_14 cells outside envelope = "
      f"{v_leaky} of {n}")
assert v_honest == 0
assert v_leaky > n * 0.9, "parity failed to catch the future shift"
print("adv6 BITES: any future-dependent computation destroys the "
      "overlap parity -> SOURCE_OR_PIPELINE_DRIFT")

# ---- adv7: one 2026 bar deleted from a staging COPY
tmp = Path.home() / ".cache" / "n3_adv7_staging"
if tmp.exists():
    shutil.rmtree(tmp)
tmp.mkdir(parents=True)
acq = pd.read_parquet(STAGING / "acquired.parquet")
open_ms = pd.to_numeric(acq["open_time"])
victim = int(pd.Timestamp("2026-04-15 08:00:00+00:00")
             .timestamp() * 1000)
trimmed = acq[open_ms != victim]
assert len(trimmed) == len(acq) - 1
trimmed.to_parquet(tmp / "acquired.parquet")
proc = subprocess.run(
    [PY, str(REPO / "tools/n3_fresh_confirmation.py"),
     "regenerate", "--staging", str(tmp)],
    capture_output=True, text=True, cwd=REPO,
    env={"PATH": "/usr/bin:/bin", "CUDA_VISIBLE_DEVICES": "",
         "HOME": str(Path.home())})
tail = proc.stdout.strip().splitlines()[-1] \
    if proc.stdout.strip() else proc.stderr[-200:]
print(f"adv7: regenerate on gapped copy -> exit {proc.returncode}, "
      f"{tail}")
assert proc.returncode == 1
assert "grid incomplete" in proc.stdout
shutil.rmtree(tmp)
print("adv7 BITES: a deleted 2026 bar refuses at the executable "
      "1458-bar grid check; forward fill never bridges it")
