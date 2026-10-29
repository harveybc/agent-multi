# Prospective stress / ablation protocol (declared before any run of these arms)

Declared 2026-10-29 by the auditor, before executing any arm below. Inputs: agent-multi 3644ece9
(code identical to 935ee95c), engine heuristic-strategy e0a1c2b, risk CONFIG sha 99d7d4dc...,
panels panel_specs(7) (calibration) and panel_specs(43) (confirmation). Policies: P1 (08fca639649c)
and Only3h (control only_3: BASE genes, enabled {(3,+1),(3,-1)}). No tracked file is modified; arms are
implemented by subclassing World / patching objects in the auditor's own script. No retuning, no selection.

Notation: d = side (+1 buy, -1 sell); D = d*(entry - SL) > 0 = the stop distance of the position or
candidate (SL from the frozen median profile geometry). Native market bars are NEVER changed; only the
reports consumed by the arbiter and the close/hold controller change.

## G1 (requested): adverse report bias k = 0.5 x stop distance, entries and occupied positions
- Occupied position (close/hold): every sample path, every bar and all four OHLC fields shifted by
  -d * 0.5 * D (towards the stop).
- Flat (entry bids): for each candidate (h, d) separately, bars 1..H-1 of every sample path shifted by
  -d * 0.5 * D(candidate); bar 0 (the entry open) unchanged, so the entry estimate and geometry are unchanged.
- Valuation then follows the unchanged code (first passage per sample, common actions, same genes).
## G1b: same as G1 but only occupied-position reports are biased (entry bids unbiased).
## G2: favourable (optimistic) bias, +d * 0.5 * D, same placement as G1 (entries and occupied).
## G3 rotation ablations (request item 5), unbiased reports:
- G3a never-early-close: the controller always returns HOLD (native SL/TP/forced flat unchanged).
- G3b cooldown 1: after an AGENT_CLOSE exit at bar j, no entry fill at bar j+1 (earliest j+2).
- G3c cooldown 6: after an AGENT_CLOSE exit at bar j, no entry fill before bar j+7.
## G4 duplicate-group dispersion (diagnostic): confirmation of every frontier policy sharing the exact
calibration vector of P1, P2 and P3 (64, 44, 10 policies).

Reported for every arm and panel: mean weekly return %, worst MTM DD %, orders, orders/week, activity
eligibility (mean >= 3/week), TP / SL / forced flat / early counts, wins, early closes whose passive
counterfactual (true bars, native rules) is SL / TP / forced flat, destructive early closes
(early net < passive net). All outputs are reported whatever they show.
