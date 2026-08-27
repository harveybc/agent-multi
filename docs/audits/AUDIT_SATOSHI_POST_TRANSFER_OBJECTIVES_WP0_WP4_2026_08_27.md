# Audit: Post-Transfer Objectives WP0-WP4

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@4997594c` (implementation `554cd472`)

## Verdict

**WP0 ACCEPTED; WP1-WP2 REVISE; WP3 NOT AUTHORIZED FOR GPU.**

Custody findings 362/363 are corrected. The objective implementation is useful
mechanical progress, but the current barrier target and screening authority do
not yet support creating an accepted five-objective generation. The paired
design therefore remains a draft materialization, not a dispatchable contract.

Focused independent reproduction: **75 passed**. No GPU dispatch driver exists
and no evidence of a launched paired run was found.

## Findings

### DATA-SOTA-364 (S1): barrier labels do not model the execution envelope

`barrier_hit_labels` observes only future closes. The deployed bracket envelope
is triggered intrabar: an upper/lower barrier can be crossed by HIGH/LOW and the
close can return inside. Such a trade is real in the simulator/venue but is
labelled `neither` or is assigned a later hit by this objective. The advertised
same-bar collision rule is structurally unreachable with one close per bar.

This is target misspecification, not a small approximation. Replace the target
with timestamp-aligned OHLC high/low first-touch labels. When both barriers are
inside one bar, apply the declared conservative adverse-first rule. Refuse data
without high/low rather than silently falling back to close.

### DATA-SOTA-365 (S1): report-only monitor governs objective acceptance

The runner computes gradient diagnostics and collapse facts on
`monitor_windows`; `pretrain_objective_screen.evaluate_manifest` then uses those
facts to accept/reject objectives. Therefore the monitor is not report-only: it
governs which objective set proceeds to the economic comparison. Repeated
correction cycles can overfit that monitor even without backpropagating through
it.

Move all mechanics gates (finite target/loss, encoder gradient, collapse and
gradient-conflict disposition) to train-tail or calibration-only probes frozen
before execution. Keep the monitor descriptive and checkpoint-only under a
separately declared rule; it must not select objectives or balancing policy.

### DATA-SOTA-366 (S2): barrier degeneracy is aggregated across horizons

The report flattens the `(N,H)` labels and accepts when the union contains three
classes. One horizon can be constant or lack a class while another supplies it.
Class weights are correctly per-horizon, but the acceptance gate is not.

Persist distributions and minimum support per horizon and reject every
degenerate horizon prospectively. The 5,829/5,174/7,101 aggregate is
insufficient evidence.

### DATA-SOTA-367 (S2): paired genesis is not yet bound to a real generation

The design binds `pretrain_source` to prose saying a future generation accepted
by Musashi. No accepted five-objective generation digest exists yet, and the
current CPU artifact is explicitly mechanics-only. Consequently the 12 genesis
records are templates; they cannot be final launch identities. Materialize them
again only after acceptance, with exact generation seal and per-family artifact
digests inside the shared binding and every genesis digest.

### DATA-SOTA-368 (S3): conflict evidence is too short for the stated rule

The real-data screen contains only two epochs over 1,200 windows. It establishes
execution and reveals a concerning reconstruction/volatility cosine (minimum
`-0.71648`, mean `-0.5419` for returns/momentum), but cannot establish that the
conflict is non-persistent. The `all epochs < -0.8` rule would also miss a
stable, materially harmful conflict around -0.7.

Do not invent a new universal threshold. Run a bounded CPU calibration long
enough to report sign frequency, weighted-gradient dominance and effect on each
objective versus its solo arm. Predeclare the disposition before that run.

## Accepted Scope

- Append-only intent plus no-clobber ACK protocol.
- Explicit 0600 custody files and 0700 root, including transition temporaries.
- Strict formulas and causal fit slicing for contrastive and volatility targets.
- Adapter/head exclusion inventory and zero key overlap.
- CPU mechanics evidence only; no economic or transfer-utility claim.

