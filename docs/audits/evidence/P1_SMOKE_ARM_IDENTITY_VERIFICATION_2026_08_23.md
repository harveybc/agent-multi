# P1 smoke: N/EN-W outer equality — mechanism VERIFIED (not hypothesized)

Practical order item 3 (2026-08-23).

Observation: N, EN-W and EN-F scored identically on outer-2024
(231 trades, +25.0014%, dd 7.95%) from three different artifact
sha256s.

First hypothesis (sign-coincidence of near-init policies) — REFUTED
by the executing traces: the raw actions of N and EN-W are EXACTLY
EQUAL on all 2,195 compared outer bars (0 values differ), not merely
sign-aligned.

Verified mechanism (state-map comparison, per-tensor sha256): ALL 148
named tensors of N's selected state (normal phase, epoch 2) equal
EN-W's selected state (easy phase, epoch 2, exposed as bundle epoch 0
of its normal phase). At this smoke scale the easy solvency
relaxation NEVER BINDS — 512-step episodes produce no insolvency
events — so `easy_chronological_continuation` and `normal_realistic`
generate byte-identical trajectories, gradients and models under one
seed. The artifact zips differ only in serialization metadata, which
is why artifact hashes diverged while state maps did not.

Consequences, stated for the record:

1. The smoke is MECHANICS evidence only (bundles, exact-state
   verification, endpoint plumbing). It is not a performance
   comparison and is never to be read as one.
2. For the LONG experiment, treatment activation must be PROVEN, not
   assumed: the aggregation step will compare each seed's easy-phase
   selected state map against arm N's same-seed state trajectory —
   `easy_treatment_diverged: false` marks an inert treatment and the
   seed's EN arms as uninformative rather than silently equal.
3. The per-tensor state maps introduced by finding 309 are what made
   this verification a one-line comparison instead of a speculation.
