# Objective-Contract and Curriculum Audit (AT-F1-012)

Audit ID: AUDIT-AT-F1-012-20260731-01
evidence_observed_at: 2026-07-31 14:20–14:45 America/Bogota
report_written_at: 2026-07-31 14:50 America/Bogota
Auditor: Satoshi
Precondition: Invocation 07 completed and persisted
(`AUDIT_AT_F1_001_CORRECTION_2026_07_31.md`); `AT-F1-001` confirmed
`reported (finding open)`; no authoritative objective selected by Harvey yet.
Evidence: chain.db (read-only, quick_check ok, 11 blocks, 18 transactions);
CSV at `evidence/AT_F1_012_OBJECTIVE_RANKING_2026_07_31.csv`;
`rl_pipeline_with_validation.py`; curriculum materializer and template.
Business priority context (owner, 2026-07-31): 1. live testing of best
performers, 2. optimization, 3. academic, 4. social.

## 1. Ranking Sensitivity (§3.1) — `reproduced`

**Coverage: 5 of 18 chain transactions carry complete atomic split evidence**
(weeks, full-period RAP, mean-weekly RAP, trades per split). The 13 exclusions
are enumerated in the CSV pipeline output: 1 genesis-population transaction
and 12 optimae records written in the compact summary form without a `splits`
object. Coverage is a result: **the research profile persists atomic split
evidence only on the full-evidence optimae variant**, so any future ranking
audit inherits this ceiling. Non-champion `candidate_evaluated` results carry
scalar fitness only. No value was imputed.

Results over the 5 fully-evidenced accepted candidates (β = 0.25 both ways):

| Statistic | Value |
| --- | --- |
| Spearman ρ (full-period vs weekly ranks) | **0.80** |
| Kendall τ | **0.60** |
| Top-1 identical | **No — the champion flips** |
| Top-5 overlap | 5/5 (trivially — n = 5) |
| Sign flips (L2 > 0 → L2 < 0) | **5 of 5 (100 %)** |
| Eligibility flips (either floor interpretation) | 0 of 5 |
| Split horizons | train_tail 2–3 weeks vs validation 53 weeks (ratio ≈ 18–26×) |

- Configured-objective champion: block 8, `L2_full = +0.000482231`,
  `L2_week = −0.000044253`, weekly rank **2**.
- Weekly-objective champion: block 6, `L2_week = −0.000043580`, full rank 2.
- **Unequal-horizon bias, direction and magnitude:** full-period RAP scales
  with split length, so the 53-week validation term dominates the composite by
  roughly the horizon ratio; the 2–3-week train-tail contributes almost
  nothing to the mean but fully drives the gap penalty. Under weekly
  normalization every accepted candidate's objective is **negative** — the
  fleet's positive L2 trajectory is, in weekly units, selection among
  weekly-negative candidates whose full-period validation accumulation is
  positive.
- Definitional note (`observed`): `mean_weekly_rap` is not a rescaling of
  full-period RAP — it charges drawdown *within each week* then averages, a
  strictly harsher risk measure. The two objectives differ in definition, not
  only normalization. Both statements above are therefore about the contract
  choice, not about an arithmetic bug.
- Statistical honesty: n = 5, champion-lineage-biased sample (accepted
  optimae are monotone improvements under the configured objective).
  ρ = 0.80 must not be read as reassurance; the decision-relevant facts are
  the champion flip and the universal sign flip, both of which are exact, not
  sampled.

## 2. Objective Semantics (§3.2) — classification, neither called "correct"

| Property | Full-period L2 (configured) | Mean-weekly L2 (task-specified) |
| --- | --- | --- |
| Rewards | accumulated split return net of one worst full-split drawdown, gap-penalized | average weekly return net of weekly drawdowns, gap-penalized |
| Unit | fraction per split-period (horizon-dependent) | fraction per week (horizon-comparable) |
| Cross-split comparability | **No** — 3-week and 53-week terms are added as if commensurable | Yes |
| DEAP-scalar suitability | usable but horizon-confounded; monotone under fixed splits | suitable; comparable across future split changes |
| Failure modes | horizon dominance (validation swamps tail); gap penalty operates on incommensurable units; masks weekly-negative behavior; invites long-split accumulation strategies | harsher risk charge may reward inactivity near zero (mitigated by existing 12-trade floor); weekly noise increases variance of the scalar on short tails |

The configured objective is documented job-0 behavior (a deliberate
full-period proxy per document 18's staged design) — but AT-F1-001's contract
said weekly, and the two disagree materially. That conflict is Harvey's to
resolve, not either agent's.

## 3. Curriculum Inheritance (§3.3) — traced through code and config

`Observed`, exact symbols:

1. **What job 1 inherits:** the archived champion policy
   (`${ARTIFACT_ROOT}/full_genome/usdcad_4h/champion_policy.zip` →
   `warm_start_model` + `warm_start_model_sha256`,
   [materialize_execution_curriculum_followup.py:168-170](../../examples/scripts/materialize_execution_curriculum_followup.py#L168))
   and the decoded champion parameters as `initial_candidate_decoded`
   (`_champion_parameters`, lines 68–86, fail-closed on missing decodes),
   plus `source_champion` lineage (line 134). **It does not inherit the
   fitness scalar or the job-0 ranking.**
2. **Fitness is recalculated**, not carried:
   `training.selection_metric = "robust_weekly_rap_fitness"` (line 168) routes
   through `_selection_value()`'s dedicated branch
   ([rl_pipeline_with_validation.py:228-235](../../pipeline_plugins/rl_pipeline_with_validation.py#L228))
   to the immutable multi-scenario **mean-weekly-RAP** robust fitness of
   document 19 §5 — weekly units by construction.
3. **Repair or inherit:** job 1 therefore **repairs** the selection bias for
   authoritative selection; job-0 bias survives only as *initialization*
   (champion + up to five diverse elites chosen under the full-period
   objective). Mitigating fact from §1: the weekly-preferred candidate
   (block 6) sits at full-period rank 2 and thus enters the elite warm-start
   set anyway.
4. **Preserved contracts** (`observed` in template): mandatory SL/TP brackets,
   activity floors, `evaluate_test_split = false`, artifact hashing and
   lineage all present in the curriculum template; the materializer fails
   closed without champion + decoded parameters.
5. **One verification-required risk (`inferred`, not verified):** the template
   carries **two** selection keys —
   `/objectives/selection_metric = train_validation_l1_score` and
   `/training/selection_metric = robust_weekly_rap_fitness`. If the pipeline
   consumes the `objectives` key rather than the `training` key at any call
   site, the repair silently degrades to the full-period composite. Cheapest
   check (proposed to Musashi, pre-launch): one unit test asserting that the
   materialized job-1 config drives `_selection_value` into the
   `robust_weekly_rap_fitness` branch. Until that assertion exists, item 2
   above is design intent, not verified runtime behavior.
6. **Cross-difficulty comparability:** the robust fitness is computed over the
   same immutable scenario suite at every difficulty phase, in fraction-per-
   week units — comparable across easy/nominal/stress by construction
   (doc 19 §4–5).

## 4. Decision Packet for Harvey (§3.4)

| | A — job 1 authoritative | B — boundary stop + v3 objective | C — retain full-period as authoritative |
| --- | --- | --- | --- |
| Information retained | all job-0 compute as warm start; weekly-champion enters elites | all archived, but search restarts under v3 | all, but champion semantics stay horizon-biased |
| Compute cost | zero additional | ~7 remaining days of job-0 search re-spent under v3 | zero |
| Comparability | job-1 scalar comparable across difficulty and future assets | clean but delayed | future cross-asset comparisons inherit horizon confound |
| Artifact consequence | job-0 champion relabeled "alpha handoff under full-period proxy objective" — initialization evidence, never a performance claim | job-0 becomes baseline-only | job-0 champion presented as selected-best; §1 shows it is weekly-rank 2 with negative weekly RAP |
| Falsifiable selection rule | authoritative selection = robust weekly RAP over the immutable suite; guard: elite set must contain the top-2 weekly-ranked candidates (verified true today) | v3 = weekly L2 from generation 0 | requires "ranking sensitivity immaterial", which §1 falsifies (top-1 flips) |

**Recommendation (not executed; Harvey owns it): Alternative A**, with two
riders — (i) Musashi's one-test verification of §3.3.5 before job-1 launch,
and (ii) the relabeling rider on job-0's champion. Rationale under the
owner's stated priorities: A wastes no compute, does not delay the path to
live testing, and the already-designed curriculum objective is exactly the
weekly-unit contract the correction demands. **C is falsified by measurement**
(champion flips; every sign flips). **B buys statistical cleanliness the
curriculum already provides**, at the price of the fleet's next week —
inconsistent with priority 1 (live testing soonest on best performers).

## 5. Research Disposition (§4)

1. Both: an implementation-level contract conflict here, and a genuinely
   publishable micro-phenomenon — horizon-ratio-driven rank inversion in
   composite train/validation objectives.
2. Falsifiable hypothesis: *rank correlation between full-period and
   per-period-normalized composite objectives decreases with split-horizon
   ratio; at ratios ≥ ~18× top-1 inversion occurs on real campaigns.*
   Minimum experiment: recompute both objectives over the 24 archived E4
   baselines (complete split evidence exists) plus this CSV — zero training,
   one afternoon.
3. Prior-art collision categories (no novelty asserted): backtest-length and
   selection bias (Bailey/López de Prado — already in the verified ledger),
   multi-fidelity noise in hyperparameter optimization (Hyperband — verified),
   objective mis-specification in evolutionary search.
4. Placement: **P2** (fitness-contract section + one future-work line);
   registry checked — no new line registered, none needed.

## 6. State Updates and Confirmations

- CSV delivered at `evidence/AT_F1_012_OBJECTIVE_RANKING_2026_07_31.csv` (one
  row per included candidate; identity, block/tx provenance, split weeks,
  both RAP forms, both L2s, both ranks, all eligibility flags, test-closure
  flag).
- Verified non-finding: all five evidenced candidates carry
  `test: evaluation_skipped / protected_test_disabled_for_optimization` on
  chain — the firewall is auditable from chain data alone.
- AT-F1-012 → `reported`. AT-F1-001 remains `reported (finding open)` pending
  Harvey's objective decision (this packet is that decision's input).
- No swarm, chain, OLAP, fitness code or queued-job configuration was touched;
  all database access was `mode=ro`; no training or backtest was launched; no
  deadline, delegation or new packet was created.
