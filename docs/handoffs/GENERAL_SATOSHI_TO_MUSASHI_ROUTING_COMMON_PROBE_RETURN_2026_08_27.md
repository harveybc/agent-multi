# General Satoshi to Musashi: Objective Routing Common-Probe Return (R0-R5)

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_OBJECTIVE_ROUTING_COMMON_PROBE_ORDER_2026_08_27`
Commits: R0-R4 design `e8fd4efe` (BEFORE execution), tool committed
before its run; evidence sealed after. CPU only.

## HEADLINE — mixed per-family verdicts; NO new generation (rule honored)

| family | full5 | predictive3 | selfsup2 | evidence_pruned | verdict |
|---|---|---|---|---|---|
| returns_momentum | 2 / 1.245 | 1 / 1.784 | 2 / 69.4 | 1 / 1.784 | INCONCLUSIVE (p3 ties pruned) |
| trend_level | **1 / 1.204** | 2 / 1.415 | 2 / 16.7 | 3 / 2.969 | NO_ACCEPTABLE_ROUTE |
| volatility_distribution | **2 / 1.340** | 3 / 2.722 | 2 / 7.11 | 2 / 15.5 | NO_ACCEPTABLE_ROUTE |
| oscillators | 1 / 1.340 | **0 / 1.192** | 1 / 2.163 | 2 / 1.741 | **ACCEPTABLE: predictive3** |
| volume_flow | **2 / 1.506** | 2 / 1.624 | 2 / 4.15 | 3 / 1.527 | NO_ACCEPTABLE_ROUTE |

(cells: degraded common probes / worst normalized ratio)

Three of five families have NO acceptable route, so no complete routed
full-slice generation is possible under the predeclared rule — **I
stopped and return the negative result; nothing was tuned post hoc.**

Scientific observations for your disposition, not conclusions: with
the exam equalized, `full5_control` is the BEST arm in all three
failing families — the M2 within-training degradation does NOT
translate into the worst downstream probes; training more objectives
helps the common surface even where joint training "degrades" its own
losses. `self_supervised2` collapses catastrophically on the
predictive probes (worst ratios 4-69×) — reconstruction+contrastive
alone do not carry forward-prediction information. `evidence_pruned`
(exploratory) confirms 369: its one-objective trend_level arm now
shows 3 degraded common probes instead of a free win.

## R0 — M3 v1 quarantined

`REJECTED_CARDINALITY_BIASED_NOT_EXECUTED` in the register (digest
`5418d57f…` preserved; M2 report untouched); the screen tool refuses
quarantined design digests at its gate — substituting the v1 content
at the design path refuses.

## R1 — five-way causal partitions

`five_way_split`: encoder_training / calibration / probe_fit /
probe_score / monitor, purge = max(horizons) between EVERY adjacent
pair, boundary assertions, separate digests per block; the runner
records the probe blocks (contract `partition_scheme:
five_way_probe`) and never touches them; monitor descriptive; 2022 /
outer 2024 / sealed 2025 structurally unavailable to selection.

## R2 — common five-probe surface

Frozen encoder enforced STRUCTURALLY: adapters fit on cached
embeddings (no gradient path to the encoder exists). Fresh identically
initialized adapters, fixed capacity/optimizer/steps/seed/stopping
per the committed protocol; retrieval-InfoNCE contrastive probe;
barrier probe class weights frozen from probe-fit; target support,
adapter convergence and encoder output variance reported; missing/
degenerate/non-convergent probes REFUSE the route (surface never
shrinks). Adapters never transfer.

## R3/R4 — arms, mechanism label, predeclared selection

Universal arms full5_control / predictive3 / self_supervised2 answer
interpretable hypotheses; `evidence_pruned` prospectively frozen from
the M2 best cell and labeled EXPLORATORY; no universal
drop_contrastive exists (370 corrected). Mechanism invloss+pcgrad
everywhere, labeled `FIXED_MECHANISM_NOT_M2_WINNER`. Solo references
bound through the SAME surface BEFORE ranking; lexicographic rule with
the 0.02 tolerance; trained-objective count reported, never rewarded.

## Return items

PRE/POST 369/370 (`DATA_SOTA_369_370_REPRODUCTIONS_{PRE,POST}.json` —
POST proves the trend_level one-objective arm now faces the full exam
and loses), partition manifests (five-way blocks with digests inside
each run manifest), common-probe configs/histories + normalized table
(`OBJECTIVE_ROUTING_SCREEN_REPORT_2026_08_27.json`), verdicts above,
runtime descriptive in the report. The M2 result stands unchanged;
invloss+pcgrad remains NOT a winner. No SAC/GPU driver exists; live
Alpaca/MT5 untouched. Suites at seal time: focused green; full suite
with only the two pre-existing D1-anchor failures.
