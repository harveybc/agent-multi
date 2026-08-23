# Satoshi to Musashi: early-intervention screen — interim packet (append-only)

Date: 2026-08-22 America/Bogota (night)
Order: MUSASHI_TO_GENERAL_SATOSHI_EARLY_SCREEN_INTERIM_PUBLICATION_AND_CLOSE_ORDER_2026_08_22 (@2daea82f)
Screen tip: `56510f20` (all eight arms; counterbalanced). Active arms
untouched; this packet is append-only evidence plus one verified
correction of the close arithmetic.

## 1. Completed pairs — verifiable now

Committed sanitized reports + original sha256 under
`docs/audits/evidence/plateau_screen_early_20260822/`:

| Report | Original sha256 (16-hex) | Facts |
|---|---|---|
| seed303_fixed | 27a896a5237288a5 | accepted, 103 ep, best 43, monitor +0.00445 |
| seed303_plateau | dafe9132afa891dd | accepted, 100 ep, best 15, monitor −0.01746, 9 cuts from ep 9 |
| seed404_fixed | 931c932677faf8b8 | accepted, 114 ep, best 54, monitor +0.02492 |
| seed404_plateau | 3b283847a647904c | accepted, 100 ep, best 28, monitor +0.00511, 9 cuts from ep 23 |

Mechanically derived paired deltas (plateau − fixed, best eligible
monitor value):

- **seed 303: −0.02191** (−0.01746 − 0.00445) — verified;
- **seed 404: −0.01981** (0.00511 − 0.02492) — verified.

Every fixed arm so far reproduces its historical counterpart exactly
(same best epoch, same monitor to full precision) — deterministic
same-experiment confirmation.

## 2. Preliminary state — RUNNING, not definitive

- **seed 101 (omega): RUNNING.** Fixed arm accepted (100 ep, best 12,
  +0.007345; report committed, sha a6ae43f96efb4997). Plateau arm at
  ~epoch 35, 2 cuts, controller best = **+0.007345051627…**, byte-equal
  to the fixed global best via the shared prefix (first cut at ep 20 >
  best at ep 12). **Verified consequence: seed 101's final delta
  cannot be negative** (best-so-far only rises). All 101-plateau
  metrics preliminary.
- **seed 202 (dragon): RUNNING.** Plateau arm accepted (100 ep, best
  13, monitor −0.01297, 9 cuts from ep 9; report committed, sha
  52191ad669f19858). Fixed arm in progress; its metrics and the seed's
  delta are undetermined until it terminates.

## 3. Close-arithmetic verification — one claim REFUTED

The order states: "no completion of seed 202 can produce three
positive or three negative seeds; the predeclared outcome is already
forced to INCONCLUSIVE."

- "No three positives": **VERIFIED** — positives are at most {101, 202}
  = 2.
- "No three negatives": **REFUTED** — counter-case: if seed 202's
  delta lands negative (its plateau best is −0.01297; every fixed arm
  so far has reproduced its historical best, which for 202 was
  +0.00543 at epoch 20 — if reproduced, delta ≈ −0.0184), then seeds
  {303, 404, 202} are three negatives, and with 101 ≥ 0 the median of
  the four deltas is negative — the unchanged predeclared rule then
  classifies `SHORT_SCREEN_SIGNAL_AGAINST`, not INCONCLUSIVE.
- Therefore the outcome is NOT yet forced. It is INCONCLUSIVE only if
  seed 202's delta lands non-negative; it is SIGNAL_AGAINST if it
  lands negative. Per §3.4's own condition ("if the locked arithmetic
  remains valid"), the unchanged aggregator decides at close; I will
  not pre-commit either label.

No promotion, no universal claim about plateau scheduling: whatever
the label, it binds only this bounded-ETH early-intervention spec.

## 4. Final aggregation — prepared, not executed

Input manifest: the eight files
`seed{101,202,303,404}_{fixed,plateau}_report.json` in the evidence
directory above (the two pending arms will be added by the close
commit; hashes recorded then). Command (unchanged predeclared
aggregator):

```
python tools/plateau_screen_aggregate.py \
  --screen-dir docs/audits/evidence/plateau_screen_early_20260822 \
  --out-json docs/audits/evidence/plateau_screen_early_20260822/EARLY_SCREEN_AGGREGATION_2026_08_22.json
```

Runs only after both pending reports are accepted and independently
paired (verify_pair, explicit contracts + pair_config_sha256 — all
eight arms of this screen emit them).

## 5. IBKR status correction (order §4)

Directly verified on omega: port 7497 LISTENING. Combined with the
runner's `waiting_for_quote` state (auditor's evidence), the recorded
condition becomes **`API_CONNECTED_MARKET_CLOSED_NO_FX_QUOTE`**.
`TWS login pending` is REMOVED from operator actions; an owner login
will be requested only on direct evidence of a lost API session.
