#!/usr/bin/env python3
"""C23 evidence: coverage simulation under intracluster
correlation. Proves the naive between-series normal interval
fabricates precision when series share a panel, and that the
predeclared design-effect rule (n_eff = n/(1+(n-1)*ICC_hat), with
INCONCLUSIVE when n_eff < 3) does not."""
import json
import sys
from pathlib import Path

import numpy as np


def icc_hat(x: np.ndarray) -> float:
    n = len(x)
    c = x - x.mean()
    denom = float(np.sum(c ** 2))
    if denom <= 0 or n < 3:
        return 1.0
    s, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            s += c[i] * c[j]
            cnt += 1
    return float(np.clip((s / cnt) / (denom / (n - 1)), 0.0, 1.0))


def simulate(n_series=36, icc=0.5, n_sim=4000, z=1.96,
             k_panels=4, seed=20260906):
    """Three rules under a shared panel effect:
    (a) naive between-series CI in ONE panel — under-covers;
    (b) within-panel ICC design-effect — CANNOT even see the
        shared effect (the mean removes it), so it under-covers
        identically: dependence is NOT identifiable from inside
        one panel — this is WHY the chosen rule refuses;
    (c) panel-level CI over K independent panels — covers."""
    rng = np.random.default_rng(seed)
    naive_cover = 0
    icc_cover = 0
    panel_cover = 0
    for _ in range(n_sim):
        shared = rng.normal(0, np.sqrt(icc))
        x = shared + rng.normal(0, np.sqrt(1 - icc), n_series)
        m = x.mean()
        sd = x.std(ddof=1)
        se_naive = sd / np.sqrt(n_series)
        if m - z * se_naive <= 0 <= m + z * se_naive:
            naive_cover += 1
        ih = icc_hat(x)
        n_eff = n_series / (1 + (n_series - 1) * ih)
        se_icc = sd / np.sqrt(max(n_eff, 1e-9))
        if m - z * se_icc <= 0 <= m + z * se_icc:
            icc_cover += 1
        # (c) K independent panels, panel means as the unit
        pm = []
        for _k in range(k_panels):
            sh = rng.normal(0, np.sqrt(icc))
            xs = sh + rng.normal(0, np.sqrt(1 - icc),
                                 n_series // k_panels)
            pm.append(xs.mean())
        pm = np.array(pm)
        from scipy import stats as _st
        tq = float(_st.t.ppf(0.975, k_panels - 1))
        se_p = pm.std(ddof=1) / np.sqrt(k_panels)
        if pm.mean() - tq * se_p <= 0 <= pm.mean() + tq * se_p:
            panel_cover += 1
    return {"icc_true": icc, "n_series": n_series,
            "k_panels": k_panels, "n_sim": n_sim,
            "naive_single_panel_coverage":
                round(naive_cover / n_sim, 4),
            "within_panel_icc_coverage":
                round(icc_cover / n_sim, 4),
            "panel_level_coverage":
                round(panel_cover / n_sim, 4),
            "single_panel_rule": "INCONCLUSIVE by design — no "
                                 "interval is licensed"}


def main() -> int:
    rows = [simulate(icc=v) for v in (0.0, 0.2, 0.5, 0.8)]
    out = {"schema": "agent_multi.t2_coverage_simulation.v2",
           "nominal": 0.95,
           "rule": "panel_replication_or_descriptive: K>=3 "
                   "panels -> panel-level CI; one panel -> "
                   "DESCRIPTIVE + INCONCLUSIVE (within-panel "
                   "dependence is not identifiable, proven by "
                   "the within_panel_icc row equalling naive)",
           "rows": rows}
    p = (Path.home() / ".local/share/agent-multi/"
         "t2_coverage_sim_20260906.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
