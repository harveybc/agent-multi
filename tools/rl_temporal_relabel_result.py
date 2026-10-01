#!/usr/bin/env python
"""Relabel an existing RESULT.json to DEVELOPMENT_NOT_CONFIRMATORY and add the equal-bars
(prefix-free) view recomputed from trace.csv. Idempotent."""
import csv, json, sys
from pathlib import Path
import numpy as np

d = Path(sys.argv[1]); res = json.loads((d / "RESULT.json").read_text())
rows = list(csv.DictReader((d / "trace.csv").open()))
sc = [r for r in rows if r["is_context_prefix"] != "True"]
eq = np.array([float(r["equity"]) for r in rows]); e_sc = np.array([float(r["equity"]) for r in sc])
prev = np.concatenate([[10000.0], eq[:-1]]); ret_all = eq / prev - 1.0
first_scored = rows.index(sc[0]); ret_sc = ret_all[first_scored:]
sd = lambda a: float(a.mean() / a.std(ddof=1)) if a.std(ddof=1) > 0 else None
peak = np.maximum.accumulate(e_sc); dd = float(((peak - e_sc) / peak).max())
res["status"] = "DEVELOPMENT_NOT_CONFIRMATORY"
res["status_history"] = ["RESULT (relabelled: FROZEN_DEVELOPMENT manifest is not a confirmatory freeze)"]
res["caveats"] = ["checkpoint selected on the same validation episode it is reported on (selection-on-validation); not a held-out number",
                  "DEVELOPMENT availability class; one seed is not a comparison"]
res["equal_bars_view"] = {"bars_total_with_prefix": len(rows), "bars_prefix_excluded": len(rows) - len(sc), "bars_scored": len(sc),
    "sharpe_with_prefix_bars": sd(ret_all), "sharpe_scored_bars_only": sd(ret_sc),
    "max_drawdown_fraction_scored": dd, "net_return_scored": float(e_sc[-1] / 10000.0 - 1.0),
    "exposure_fraction_scored": float(np.mean([r["position"] != "0" for r in sc])),
    "convention": "per-bar equity returns, ddof=1, not annualized; scored bars = rows [13699,15895) only"}
res["metrics_as_originally_written"] = res["metrics"]
res["metrics"]["sharpe"] = {"value": res["equal_bars_view"]["sharpe_scored_bars_only"], "convention": res["equal_bars_view"]["convention"], "undefined_reason": None}
res["metrics"]["max_drawdown_fraction"] = dd; res["metrics"]["exposure_fraction"] = res["equal_bars_view"]["exposure_fraction_scored"]
(d / "RESULT.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
print(json.dumps(res["equal_bars_view"]))
