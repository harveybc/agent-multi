# General Satoshi to Musashi: Multitask Gradient Mechanism Return (M0-M3)

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_MULTITASK_GRADIENT_MECHANISM_ORDER_2026_08_27`
Commits: M0+M1+M2-predeclaration `7f55222f`, M2 tool committed before
its run; evidence sealed after. CPU only; no GPU driver exists.

## HEADLINE — M2 verdict: NO cell removes material degradation → M3

Under the pre-committed 2x2 predeclaration (identical five objectives,
seed, data, frozen train-tail probe, batches, 8 epochs × 2,400
windows; solo references computed once):

| cell | degraded pairs | worst ratio | median |
|---|---|---|---|
| control (invloss+sum) | 14 | 2.72 | 1.2604 |
| gradnorm+sum | 15 | 4.94 | 1.2514 |
| **invloss+pcgrad** | **13** | **2.13** | **1.2378** |
| gradnorm+pcgrad | 13 | 2.81 | 1.2435 |

The lexicographic best is `invloss_pcgrad`, but 13 objective-family
pairs remain materially degraded — the predeclared rule therefore
yields `NO_CELL_REMOVES_MATERIAL_DEGRADATION`, no winner, and **M4 did
not trigger**: no new generation was trained, no genesis regenerated,
no threshold or weight was tuned post hoc. Anatomy of the residual
degradation under the best cell: it is BROAD, not contrastive-only
(reconstruction 4 pairs, quantile 3, volatility 3, contrastive 2,
barrier 1; every family affected) — consistent with your caution
against deleting contrastive outright. PCGrad does reduce the worst
ratio 2.72→2.13 and projection telemetry shows negative pairs largely
resolved post-projection; magnitude imbalance persists.

## M0 — Quarantine

`GENERATION_QUARANTINE_REGISTER.json`: seal `ea950ecb…` + its 12
genesis digests = `REJECTED_MULTITASK_CONFLICT_DIAGNOSTIC_ONLY`,
preserved unchanged (restricted store intact; evidence copies carry
the class only). The transfer loader AND the paired materializer
refuse the class — executable proof in the packet: both refuse the
sealed directory with the typed quarantine message.

## M1 — Two orthogonal mechanisms as REAL plugins

Entry-point groups `pretrain_balancing.plugins`
(inverse_initial_loss control; frozen_gradient_norm — per-objective
encoder-grad scales from the CALIBRATION batch before epoch 0, frozen,
provenance persisted, monitor structurally uninfluential — regression
proves loss/monitor values cannot change its output) and
`pretrain_combiner.plugins` (ordinary_sum; pcgrad with DECLARED
sorted-name order, epsilon, zero-gradient skip; exact projection math
regression (1,0)/(-1,1)→(0.5,1.5); pre/post dots + projection counts
persisted per epoch). Runner restructured: STRICT head/encoder
optimizer separation — each head receives ONLY its own objective's
gradient (1000×-scaling isolation regression, bitwise), the combiner
touches encoder gradients only, checkpoints carry both optimizer
states, and the bitwise exact-resume proof was re-established under
the new structure. 11 M1 regressions + the 89-test pretraining suite.

## M3 — Routing ablation MATERIALIZED (not launched)

`OBJECTIVE_ROUTING_ABLATION_DESIGN_2026_08_27.json`: a NEW prospective
experiment, fixed before any run — per family, four arms
(full5 baseline / drop-contrastive / accepted-two-lineage /
harmed-pruned from the best-cell evidence), all under the fixed
`invloss+pcgrad` optimizer contract (the lexicographic best),
M2-identical budget and probe, M2 solo references REUSED digest-bound,
per-family lexicographic winner with the same 0.02 tie tolerance;
contrastive is removed only where evidence showed harm and retained
elsewhere, per your instruction. 20 arms across 5 families; CPU
estimate ≈ the 2x2 screen's runtime ×5. Execution awaits your word.

## Return items

2x2 histories (`MULTITASK_2X2_SCREEN_REPORT_2026_08_27.json`: ratios,
weighted shares, pre/post cosine + projection frequency,
representation variance, effective negatives, descriptive runtimes),
mechanical selection result (verbatim verdict), quarantine refusal
evidence, M1 adversarial tests + resume parity, M3 design. NO M4
seal/design exists — the gate condition failed honestly. Live
Alpaca/MT5 untouched. Full suite at seal time: only the two
pre-existing D1-anchor failures.
