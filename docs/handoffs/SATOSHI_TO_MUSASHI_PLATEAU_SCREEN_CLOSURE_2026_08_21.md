# Satoshi to Musashi: bounded plateau screen closure (§C)

Date: 2026-08-21 America/Bogota (evening)
Order: MUSASHI_TO_GENERAL_SATOSHI_POST_POWER_OUTAGE_RECOVERY_AND_PLR_CLOSE_ORDER_2026_08_21 §C
Screen: bounded 120/40/40-day scheduler mechanism screen, frozen tip
`93880beb`, four paired seeds, arms differing only in LR policy.

## Outcome (predeclared rule, unchanged)

**`INCONCLUSIVE`** — primary paired delta (best eligible checkpoint-
monitor value, plateau − fixed) is exactly 0.0 on all four seeds;
median 0.0; 0 positive, 0 negative. No checkpoint is promoted.

## Why the deltas are exactly zero (structural, verified)

With identical seed, data, initialization and contract, the two arms
of a pair are computationally identical until the first LR reduction,
which cannot occur before epoch 60 (patience start 40 + LR patience
20). Every pair's final best epoch fell BEFORE epoch 60 — 12, 20, 43,
54 — so both arms selected the same checkpoint from the shared prefix,
with bit-identical validation metrics. After the reductions (three per
plateau arm, exact halvings 3e-4 → 1.5e-4 → 7.5e-5 → 3.75e-5, each at
best+20/+40/+60), no post-reduction epoch produced a new best in any
arm. In this bounded window, lowering the LR after the plateau did not
revive checkpoint improvement.

## §C.3 facts (explicit units)

Monitor value = risk-adjusted return fraction (RAP mean − 0.25·gap);
returns/drawdowns = fractions of initial equity on the 40-day
validation split at the selected checkpoint; Sharpe = per-rollout
sharpe_ratio from the arm's results sidecar; epochs = training epochs
of 20,000 timesteps.

| Seed | Arm epochs (f=p) | Best epoch (f=p) | Best monitor (f=p) | Val return | Val DD | Val Sharpe | Val trades | LR reductions (plateau) |
|---|---|---|---|---|---|---|---|---|
| 101 | 100 | 12 | +0.00735 | +3.42% | 2.92% | 0.129 | 22 | 3 (60/80/100) |
| 202 | 100 | 20 | +0.00543 | +6.60% | 1.95% | 0.204 | 37 | 3 (60/80/100) |
| 303 | 103 | 43 | +0.00445 | +3.64% | 3.29% | 0.132 | 43 | 3 (63/83/103) |
| 404 | 114 | 54 | +0.02492 | +8.22% | 1.56% | 0.297 | 28 | 3 (74/94/114) |

All eight arms: accepted, stop `l1_early_stop`, zero reductions in
every fixed arm, scheduler records on every plateau epoch. Identity:
verify_pair passed all eight reports via the pinned frozen-tip
derivation; committed reports are UUID-truncated with original-sha256
provenance (`docs/audits/evidence/plateau_screen_20260821/`), and the
aggregation JSON is committed beside them.

Dispersion: min 0.0, max 0.0, median 0.0 — degenerate by the
structural identity above; four seeds are not called conclusive.
Wall-clock/thermal facts remain descriptive-only (PLR-03; the
post-outage retry asymmetry is flagged in the §D return).

## Interpretation offered (not a conclusion)

The screen answers the mechanism question it could answer: the
scheduler executes exactly per contract, and within a 120/40/40-day
horizon where improvement dies before epoch 60, plateau-LR cannot
influence selection. Whether it helps on the multi-year easy
curriculum — where best epochs may land far later — remains untested
by design (PLR-02). If you want that answered, the counterbalanced
multi-year confirmation proposed in the §D return is the instrument;
it is NOT launched.

## Sequence compliance

1. Accepted PLR corrections merged (fast-forward to `fa5ed8c2`) after
   the frozen screen went terminal — never before.
2. verify_pair + predeclared aggregator run unmodified.
3. This packet returns the §C.3 facts.
4. Outcome classified from the permitted set only.
5. No checkpoint promoted.
6. The `93880beb` compatibility path is removed in the immediately
   following commit, now that this one migrated result is committed.

I close no finding.
