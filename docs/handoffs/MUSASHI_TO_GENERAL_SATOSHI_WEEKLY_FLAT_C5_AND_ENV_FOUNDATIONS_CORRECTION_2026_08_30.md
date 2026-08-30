# Musashi to General Satoshi: C5 and environment foundations correction

Date: 2026-08-30

Source audit:
`docs/audits/AUDIT_SATOSHI_WEEKLY_FLAT_P1_P3_AND_C5_RETURN_2026_08_30.md`

Priority: correct the core environment before WP3, WP4 or any training.

## G1: one explicit action contract

Carry the original continuous model output, mapped discrete command and current
signed exposure as separate values. Classify risk from the mapped command using
a discrete-command adapter, or convert the raw output to the exact signed
target contract before classification. Never feed command ids to a target-value
classifier. A masked entry/enlargement/reversal must submit HOLD or an explicit
safe reduction, never the original command.

Test long/short entry, hold, reduction, enlargement, reversal and close through
the real env for continuous and discrete modes. Assert actual plugin command and
economic effect, not only diagnostic labels.

## G2: causal reopen evidence

Materialize `ReopenEvidence` from causal observations: elapsed time from bound
calendar, fully closed bars, spread relative to a past-only baseline, opening
gap sigma, realized-volatility ratio, quote continuity and consecutive stable
checks. Bind units and baseline windows in policy. Prove exact blackout exit,
reset determinism, missing inputs fail-closed and no future bars influence it.

## G3: real missing-session semantics

Replace synthetic closure bars with historical timestamp gaps. The simulator
must perform no step during a closure. Test pre-close forced flatten, direct
jump to reopen context, blackout observations and first actionable post-blackout
bar. Do not zero a reward to conceal an economic change; assert no actionable
transition/reward exists.

## G4: order inventory and fresh reconciliation

Derive pending entry versus protective reduce-only orders from actual order
identities, sides, sizes, parent/child relation and reduce-only semantics. No
`or 0` coercion. Forced flatten is accepted only after the shared typed direct
evidence proves fresh zero positions and zero pending orders; otherwise emit a
typed incident.

## G5: non-vacuous lifecycle tests

Force termination with known open exposure and assert exact preservation before
migration. Add the flat counterpart separately. Assert every branch executes.

## G6: repair foundational findings

1. Fix F-A and prove all eleven OANDA fields vary correctly when date is index
   or column, with bit-parity for equivalent inputs.
2. Fix F-B and require `observation_space.contains(obs)` across representative
   feature/price/return/session configurations.
3. Add an executable authority block for Nautilus economic comparisons while
   F-C remains.
4. Add explicit H1/H4 refusal in Nautilus while F-D remains; propose its proper
   timeframe implementation separately.

Return PRE/POST reproducers, full suites and a claim map. Do not deploy, touch
the current MT5 position, launch training or begin WP3/WP4 until independent
acceptance.
