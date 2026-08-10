# Musashi to General Satoshi III: L1 Accepted and Conditional LR-Pair Plan

Date: 2026-08-10 America/Bogota  
Authority: owner-approved document 38 and independent round-3 acceptance  
Runtime state: decision identity `2de49ea9225e2baf` active on four GPUs

## 1. Acceptance

Findings 196-200 are independently verified corrected. Preserve the smoke-v3
seal and replica. Do not alter the running decision identity. Monitor all four
workers, collect only after four clean terminal seed outcomes, replicate the
sealed decision tree, load every terminal artifact on the replica and aggregate
exclusively from the collection envelope.

Report raw and consistently scaled metrics for every split and seed, including
trades, activity, mean weekly return, annualized return, maximum drawdown,
Sharpe/RAP, train-inner and inner-outer gaps, gradient updates, epochs, elapsed
time and artifact hashes. The typed result remains one of
`EASY_CONTRIBUTES`, `LR_ONLY`, `INTERACTION`, `EASY_HARMFUL` or
`INCONCLUSIVE`.

## 2. What This Factorial Answers

Phase-1 LR is fixed at `1e-4`. The current cells cross:

- `N` versus `E` phase-1 difficulty; and
- normal-phase LR `1e-4` versus `3e-5`.

Use paired-seed contrasts to estimate difficulty, normal LR and interaction.
Do not describe this run as optimizing the easy-phase LR.

## 3. Conditional Next Experiment

Prepare, but do not launch or materialize into the current identity, a typed
`LR_easy x LR_normal` response-surface contract. Its launch condition is:

- `EASY_CONTRIBUTES` or `INTERACTION`: execute a bounded paired design around
  the empirically viable LR region, containing the equal-LR diagonal and
  asymmetric pairs;
- `LR_ONLY`: do not spend compute on easy LR; retain no easy phase and tune the
  normal LR in the next appropriate component domain;
- `EASY_HARMFUL`: remove easy from this SAC recipe; preserve it only as a
  rejected mechanism result;
- `INCONCLUSIVE`: inspect the typed refusal/activity cause first; add evidence
  only for the unresolved contrast, never an automatic broad sweep.

The exact LR levels are derived from this run's learning curves and stability
facts. Do not invent a broad range in advance. Hold fixed replay handling,
optimizer reset/carry semantics, phase budgets, data splits, costs, protection,
seeds and model topology. Otherwise an apparent LR interaction would be
confounded with a transition-policy change.

Selection is always under normal-realistic inner/outer validation. Easy-phase
performance is diagnostic, never the release objective. The sealed 2025 split
remains unopened.

## 4. Future DOIN Encoding

Only if the bounded LR-pair experiment finds a stable asymmetric region, add
two separate continuous genes to the later component domain:
`log_lr_easy` and `log_lr_normal`. Bound them from the measured viable region.
Do not encode a single shared LR and do not add transition/reset genes in the
same first evolutionary campaign.

## 5. Immediate Work

1. Monitor heartbeat freshness, exact experiment identity, current cell and
   GPU binding without restarting healthy workers.
2. Alert only on fresh unresolved anomalies; suppress duplicate Telegram
   noise.
3. On terminal completion, perform collection, replica load verification and
   sealed aggregation automatically under standing authority.
4. Return the typed result and paired raw table to Musashi for independent
   reproduction.
5. Implement the non-blocking persisted post-write digest improvement only in
   a future identity, never by mutating this run.

