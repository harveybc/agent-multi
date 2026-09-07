#!/usr/bin/env python3
"""C30 evidence: operating characteristics of the COMPOSITE
six-panel screen rule (t lower bound + exact sign sensitivity +
leave-one-panel-out + non-inferiority), simulated at the panel
level. No confirmatory outcome is consumed — panel effects are
synthetic draws.

Questions answered, all predeclared:
1. Boundary type-I: with the true grand effect AT the practical
   margin (no real superiority beyond it), how often does the
   composite rule ADVANCE? Must be well under alpha=0.05.
2. Power: with true effects 2x and 3x the margin across
   between-panel sd tau, how often does it ADVANCE?
3. Damaged-panel sensitivity: five helpful panels + one harmed
   panel — the sign and LOPO rules must kill ADVANCE even when
   the grand mean clears the margin.
4. Dominant-panel sensitivity: one huge panel effect + five nulls
   — LOPO must kill ADVANCE.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

MARGIN = 0.02
NI_MARGIN = 0.02
K = 6
TQ = float(stats.t.ppf(0.975, K - 1))


def composite_advance(effects: np.ndarray) -> bool:
    """The EXACT frozen rule from adjudicate_screen (harm gates
    beyond the effect sign are exercised by the battery, not
    here)."""
    grand = float(effects.mean())
    se = float(effects.std(ddof=1) / math.sqrt(K))
    ci_low = grand - TQ * se
    if ci_low <= MARGIN:
        return False
    if int((effects > 0).sum()) != K:
        return False
    for i in range(K):
        if float(np.delete(effects, i).mean()) <= MARGIN:
            return False
    if float(effects.min()) < -NI_MARGIN:
        return False
    return True


def rate(gen, n_sim, rng):
    hits = 0
    for _ in range(n_sim):
        if composite_advance(gen(rng)):
            hits += 1
    return hits / n_sim


def main() -> int:
    rng = np.random.default_rng(20260906)
    n_sim = 20000
    rows = []
    # 1-2: type-I at the boundary and power, across tau
    for mu_x, label in ((MARGIN, "boundary_type_I"),
                        (2 * MARGIN, "power_2x_margin"),
                        (3 * MARGIN, "power_3x_margin")):
        for tau in (0.005, 0.01, 0.02, 0.04):
            r = rate(lambda g: g.normal(mu_x, tau, K),
                     n_sim, rng)
            rows.append({"scenario": label, "true_mean": mu_x,
                         "tau_between_panel": tau,
                         "advance_rate": round(r, 4)})
    # 3: five helpful panels + one materially harmed panel;
    # grand mean still clears the margin
    for harm in (-0.03, -0.06):
        def gen(g, h=harm):
            eff = g.normal(0.06, 0.005, K)
            eff[0] = h
            return eff
        grand = 0.06 * 5 / 6 + harm / 6
        r = rate(gen, n_sim, rng)
        rows.append({"scenario": "one_damaged_panel",
                     "harmed_effect": harm,
                     "grand_mean_approx": round(grand, 4),
                     "advance_rate": round(r, 4)})
    # 4: one dominant panel + five nulls
    def gen_dom(g):
        eff = g.normal(0.0, 0.005, K)
        eff[0] = 0.50
        return eff
    r = rate(gen_dom, n_sim, rng)
    rows.append({"scenario": "one_dominant_panel",
                 "dominant_effect": 0.50,
                 "grand_mean_approx": round(0.50 / 6, 4),
                 "advance_rate": round(r, 4)})
    out = {"schema": "agent_multi.t2_screen_sim.v1",
           "rule": "t_ci_low(df=5) > margin AND signs 6/6 AND "
                   "LOPO mean > margin in all 6 omissions AND "
                   "min panel effect >= -non_inferiority",
           "margin": MARGIN, "non_inferiority": NI_MARGIN,
           "n_sim": n_sim, "rows": rows}
    p = (Path.home() / ".local/share/agent-multi/"
         "t2_screen_sim_20260906.json")
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
