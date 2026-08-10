# L1 Matched Factorial — Typed Result (decision identity 2de49ea9225e2baf)

Date: 2026-08-10. Sealed chain: COLLECTION_SEALED (zero refusals at
collection), sealed tree digest
`f3bb41516f8f3bb9…`, dragon replica whole-tree digest EQUAL, 16/16
terminal artifacts loaded on the replica, envelope aggregation from
the sealed root only, `sealed_digest_unchanged: true` after
publication. Full artifacts live in the operator-local collection root
(private-by-default policy); this summary carries no machine paths.

## Typed outcome

**INCONCLUSIVE** — with a crisp, uniform, two-layer cause:

1. **Total activity collapse in ALL 16 cells.** Every cell — 4 seeds ×
   phase-1 {normal, easy} × phase-2 LR multiplier {1.0, 0.3} — ended
   `activity_stop_no_eligible_checkpoint` at exactly 80 phase-2 epochs
   (40 floor + 40 inactivity streak), with ZERO trades on train,
   train-tail and validation from the first epoch. Every cell is VALID
   per §7.1 (terminal loads, tensor chain consistent, phase-2 updates
   occurred) and INACTIVE. Paired deltas are 0 at both LR levels; no
   §7.2 rule matches.
2. **Metric-completeness refusals (contract working as specified):**
   with zero trades and zero P&L (final equity exactly initial cash),
   `sharpe_ratio` is null on every cell's evaluation; the WP3 rule
   (every raw metric finite or refusal) correctly refuses 16 times
   rather than averaging a void.

## Scientific reading (diagnostic, for the conditional plan)

- Neither the easy phase-1 curriculum nor the reduced normal-phase LR
  (0.3×) rescued trading activity under the exact decision protocol.
- The M0 mechanism screen's "reduced-LR-from-anchor retains activity"
  result did NOT replicate under the L1 protocol. Differences to
  inspect first (per the LR-pair conditional plan: inspect the typed
  cause before any new compute):
  1. the paired activity gate (train-tail AND validation must both
     trade) versus M0's gate;
  2. the v3 exact normal contract (full spread 1e-4 + enforced
     protected entries) — noting the PRE-correction diagnostic run
     under the older cost profile also collapsed, so costs alone do
     not explain it;
  3. phase-1→phase-2 boundary specifics (4-epoch phase-1 then reload)
     versus M0's direct anchor fine-tuning;
  4. budget/epoch-length interactions (M0 measured survival at ~10-14
     epochs; the L1 stopper demands sustained eligibility past epoch
     40).
- Per the conditional plan, INCONCLUSIVE ⇒ no LR-pair experiment, no
  broad sweep: one bounded mechanistic inspection of the unresolved
  contrast is the only next compute.

The 16-record table, per-cell §7.1 facts and raw metrics (with typed
unavailables) are in the sealed aggregation artifact; Musashi
reproduces from the sealed root and the collection manifest.
