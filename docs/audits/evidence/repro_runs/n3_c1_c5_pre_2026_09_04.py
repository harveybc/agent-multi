"""PRE freeze for order @17f6e574 (C1, C2, C4, C5): reproduce every
finding EXACTLY as Musashi reported it, before any edit."""
import copy
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
spec = importlib.util.spec_from_file_location(
    "n3f", REPO / "tools" / "n3_fresh_confirmation.py")
n3f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(n3f)
from agent_plugins.experiment_runtime import sha_obj  # noqa: E402

STAGING = (Path.home()
           / ".local/share/agent-multi/n3_fresh_staging_20260904")

print("== C1: timestamp-unit bug ==")
lake = pd.read_parquet(n3f.LAKE_PARQUET)
print("frozen parquet open_time dtype:", lake["open_time"].dtype)
buggy = (pd.to_datetime(lake["open_time"], utc=True)
         .astype("int64") // 10 ** 6)
true_ms = (pd.to_datetime(lake["open_time"], utc=True)
           - pd.Timestamp(0, tz="UTC")) \
    // pd.Timedelta(milliseconds=1)
print(f"final frozen bar: buggy={int(buggy.iloc[-1])} "
      f"true={int(true_ms.iloc[-1])}")
start_ms = 1735689600000  # 2025-01-01T00:00Z
lake_2025_buggy = lake[buggy >= start_ms]
print(f"buggy lake_2025 rows: {len(lake_2025_buggy)} "
      "(overlap check was 0 == 0 -> vacuous pass)")
receipt = json.loads((STAGING / "acquisition_receipt.json")
                     .read_text())
print("committed v1 receipt: rows_total=%s "
      "rows_overlap_verified_exact=%s rows_2026=%s verdict=%s"
      % (receipt["rows_total"],
         receipt["rows_overlap_verified_exact"],
         receipt["rows_2026"], receipt["verdict"]))
assert int(buggy.iloc[-1]) == 1767211
assert receipt["rows_overlap_verified_exact"] == 0
assert receipt["rows_2026"] == 3648

print("\n== C2: four forgeries vs the committed verifier ==")
BUNDLE = (REPO / "docs/audits/evidence/"
          "N3_FRESH_CONFIRMATION_BUNDLE_2026_09_04.json")
orig = json.loads(BUNDLE.read_text())


def redigest(unit):
    unit["payload_sha256"] = sha_obj(
        {k: v for k, v in unit.items() if k != "payload_sha256"})


def try_verify(b, tag):
    p = Path.home() / ".cache" / f"c2_pre_{tag}.json"
    p.write_text(json.dumps(b, default=float))
    try:
        out = n3f.verify(p)
        print(f"forgery {tag}: ACCEPTED -> {out['verdict']} "
              f"(decision {out['rederived_decision']})")
        return True
    except n3f.FreshRefusal as r:
        print(f"forgery {tag}: refused ({r})")
        return False
    finally:
        p.unlink(missing_ok=True)


accepted = []
b = copy.deepcopy(orig)
b["contract_sha256"] = "0" * 64
accepted.append(try_verify(b, "F1_zero_contract_sha"))

b = copy.deepcopy(orig)
b["blocks_complete"] = False
b["verdict"] = "FRESH_CONFIRMATION_INSUFFICIENT"
accepted.append(try_verify(b, "F2_blocks_complete_false"))

b = copy.deepcopy(orig)
u = b["units"][0]
u["n_score"] = 1
u["class_support_score"] = {"0": 999, "1": 999, "2": 999}
u["arms"]["arm1"]["multiclass_logloss_mean"] = -123.0
redigest(u)
accepted.append(try_verify(b, "F3_absurd_unit_redigested"))

b = copy.deepcopy(orig)
for u in b["units"]:
    if u["horizon"] == 6:
        u["arms"]["arm3"]["per_obs_logloss"] = [
            round(0.98 * v, 8)
            for v in u["arms"]["arm2"]["per_obs_logloss"]]
        redigest(u)
contrasts, stats, complete = n3f._rederive(b["units"])
b["contrasts"] = contrasts
b["verdict"] = n3f.decide(stats, b["blocks_complete"],
                          b["licenses_ok"])
print("  (forged verdict now:", b["verdict"] + ")")
accepted.append(try_verify(b, "F4_coherent_fake_neural_passer"))

print("\n== C4: staging custody modes ==")
st = os.stat(STAGING)
print(f"root mode: {stat.filemode(st.st_mode)} "
      f"({oct(st.st_mode & 0o777)})")
page = STAGING / "page_000.json"
sp = os.stat(page)
print(f"page_000 mode: {stat.filemode(sp.st_mode)} "
      f"({oct(sp.st_mode & 0o777)})")
assert (st.st_mode & 0o777) == 0o775
assert (sp.st_mode & 0o777) == 0o664

print("\n== C5: parser boundaries ==")
print("json.loads NaN accepted:",
      json.loads('{"a": NaN}')["a"] != json.loads('{"a": NaN}')["a"])
print("json.loads duplicate keys (keeps last):",
      json.loads('{"a": 1, "a": 2}'))
print("float(True) coerces:", float(True))
print("ffill 2026 effect: NEVER MEASURED in v1 (no per-role count "
      "exists in any artifact)")

print("\nPRE CONFIRMED:", sum(accepted), "of 4 forgeries accepted; "
      "unit bug, vacuous continuity, 0775/0664 custody and parser "
      "holes all reproduce")
assert all(accepted)
