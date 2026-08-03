# Proposal: Solvency-Relaxation Curriculum (Owner's NEAT Observation, Formalized)

Date: 2026-08-02 21:35 America/Bogota
Author: Satoshi II, novice technical lead
Origin: the owner's direct experimental observation (NEAT-era gym-fx)
For: General Musashi (research-lead collision test + registry disposition)
and the owner (future-domain decision)
Runtime mutation by this document: none. Job 0 untouched. No GPU consumed.

## 1. The Owner's Observation (primary evidence, historical)

Training NEAT agents in the original gym-fx with full fees and margin-call
termination: candidates died at margin call so early that no model ever
learned to trade. With margin call accidentally disabled — negative
balances allowed, fitness still fully punished by the damage — models
began learning, some reaching excellent profit/risk levels. Re-tested
under realistic margin-call constraints, they performed; adding rates
improved them further. This experience is the origin of the project's
difficulty-level requirement.

## 2. Formal Framing

Three named phenomena agree with the observation:

1. **Absorbing-state starvation**: early termination truncates episodes,
   destroying credit assignment over the recovery/consequence horizon —
   the learner never experiences what follows a drawdown.
2. **Survivorship pathology in evolutionary search**: selection over
   short, ruin-truncated episodes rewards timidity or luck, not policy —
   the population dies before the required structure can evolve. (Its
   mirror is our 2026-07-29 activity-gate incident: survive by not
   trading. Both are fitness-landscape pathologies, opposite signs.)
3. **Constraint-relaxation curriculum**: train under relaxed dynamics,
   select under honest fitness, validate/test under full realism —
   exactly the architecture of the doc-19 cost curriculum, of which the
   owner's experience is the ancestor. The proposal adds a SECOND
   curriculum axis, orthogonal to costs: solvency/termination.

The critical property the owner preserved, and we must too: only the
EPISODE DYNAMICS relax. Fitness always sees the losses; validation
scenarios stay immutable and realistic; the protected test is untouched.

## 3. Code Fact: the Phenomenon is Structurally Present Today

`gym-fx/app/env.py:343`:
`terminated = bool(self.bridge.terminated or new_equity <= self.min_equity)`
with `min_equity` defaulting to 1% of initial cash and already
configurable (`env.py:131`). The Nautilus path additionally pre-denies
under-margined entries (softer: the agent lives but cannot add exposure).
Whether the deployed v2 campaign config triggers material early
termination in practice is UNMEASURED — measuring it is step 1 below.

## 4. Proposed Program (evidence before doctrine)

1. **Measure first (CPU-side log analysis, zero interference):** from the
   existing four workers' logs/OLAP, count episodes terminated by
   `min_equity` versus data-end during current training. If ruin
   termination is rare, the effect is small here and the line de-priorities
   itself honestly.
2. **Bounded local A/B (only in a declared GPU window, never beside the
   campaign):** same asset/seed/config, `min_equity` terminating versus
   relaxed (deeply negative, financing accruing), compare learning-curve
   activity, L1 progression and — decisive — validation under the
   REALISTIC constraint. The owner's historical result predicts the
   relaxed arm wins on realistic validation.
3. **If the signal reproduces:** register the solvency axis in the doc-19
   curriculum contract (train phases relax, validation never does) for a
   FUTURE domain — new domain hash at a job boundary only; job 0 finishes
   untouched; the queued job-1 template is amended only with an explicit
   owner decision plus auditor review BEFORE materialization, or the axis
   waits for the next campaign.
4. **Research registry:** propose as a P6+ line (constraint-relaxation
   curriculum for trading RL). Collision test requested from the research
   lead: curriculum RL (Bengio 2009 lineage), safe-RL constraint
   scheduling, early-termination bias literature; the owner's historical
   NEAT evidence is a legitimate motivating observation for P2/P6.

## 5. Hard Boundaries

- Live/demo layers NEVER relax solvency: LTS caps, margin checks and the
  daily-loss hold are reality, not curriculum.
- No mid-chain change of any kind; no GPU experiment while the campaign
  owns the devices, except in a measured window per doc 23 §5 discipline.
- Selection/validation/test contracts are immutable under this proposal;
  only training-episode termination is in scope.
