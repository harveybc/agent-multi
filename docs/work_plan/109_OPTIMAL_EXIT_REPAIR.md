# Optimal-exit learning repair

Date: 2026-10-28. Supersedes the next-action list in plan 108, not its results.

The previous study is closed. Do not interrupt it, alter its artifacts, or call
its six failed policies qualified. The new work has a separate identity.

## Requirements and test design

| Requirement | Regression / evidence |
|---|---|
| Exit optimum begins at the current decision, not an earlier peak | Uniform-start suffix targets; dense and terminal return equivalence |
| No order-state or future labels in network observations | Observation equality under suffix reset and future-label mutation |
| Measure exploration and actual updates | Executed command counts, oracle-opportunity visits, advantages by command, finite gradient norms by component |
| Distinguish capacity from RL learning | Separate supervised synthetic capacity control; never reused as RL weights |
| Preserve deployment semantics | Deterministic argmax qualification from the original first decision, both directions |
| Reject an effectively constant stopping clock | Exact sampled expectation versus TRAIN-frozen constant-hazard control, plus deterministic blind controls |
| Normalize combined objective using calibration only | Frozen control scales; unit-invariance test; confirmation cannot set scales |
| Report scope and termination truthfully | Reference-aware grid labels; terminal STATUS and PROGRESS agree |
| Keep evidence and resume safely | Input hashes, source identity, durable per-unit receipts, checkpoint replay |

## Ordered experiment

1. Add tests before implementation. Inspect training-only observation aliases,
   sampled/executed actions, advantages, gradients and state visitation.
2. Run a separately labelled supervised capacity diagnostic on the identical
   grouped TCN and causal observations. This is not pure RL, not a trading
   result, and never a donor for the RL experiment.
3. Compare fresh pure-RL fits with the original reset and uniform TRAIN-only
   decision-start sampling. The latter exposes already-open synthetic positions
   at different ages; it does not add a label or age to network inputs. Starting
   later requires recomputing regret from that decision, never the old optimum.
4. Fix training budget and seeds before confirmation. Diagnose on TRAIN only;
   every declared arm is reported. Do not silently replace argmax by sampling,
   change thresholds on confirmation, or call a positive random-hazard return
   learned timing. Keep costs, 13 observations / stride 2, architecture and
   financial inputs unchanged (synthetic channels only in this diagnostic).
5. Re-run affected DEAP comparisons with calibration-only objective scales,
   explicit boundary warnings and paired case-level uncertainty. Two close
   biases are an exit-controller subproblem, never whole-strategy optimization.
6. Full-strategy optimization, financial fits and floating selection remain
   downstream of learned-control qualification and the strategy cross-audit.

No inference that financial data lack predictive information follows from a
failed synthetic control. A numerical gradient is not proof of useful learning.
Current faults are a diagnostic/qualification failure and incorrect reporting
or comparison claims, not evidence that native quote accounting is defective.

## TRAIN-only pilot, completed before confirmation

The first-reset 32768-step replication reproduced the original final policy
tensor hash exactly. All sampled commands equalled executed commands. Core,
feature branches and actor had finite nonzero gradients. Frozen risk-head
gradients were zero as expected. The critic's final explained variance was
-0.2419, so gradient existence is not evidence of a useful value function.

Of 32768 sampled decisions, 4464 (13.62%) were at a remaining-menu optimum
where closing was appropriate. Uniform decision starts raised this to 18808
(57.40%), but produced immediate closure rather than learned timing. This
refutes the claim that changing exploration alone already fixes the learner.

| TRAIN-only control, seed 7 | Mean regret / initial risk | Early-close share |
|---|---:|---:|
| Original first reset, 32768 steps | 1.164652 | 0% |
| Uniform decision reset, 32768 steps | 0.415370 | 100% (immediate) |
| Separate supervised capacity diagnostic, 100 epochs | 0.259571 | 88.89% |

The capacity diagnostic beats the blind controls on both sides in TRAIN, but
does not establish generalization or provide RL weights. There are no exact
byte-identical observation/mask collisions with opposing labels among the
4142 unique TRAIN observations. This is only an exact-alias check.

## Frozen follow-up and automatic ordering

Configuration: `configs/validation/exit_learning_repair.json`. Fresh fits,
seeds 7/19/43, first versus uniform decision resets at matched budgets 32768
and 131072. The 131072-step arms run first. One separate 100-epoch supervised
capacity diagnostic precedes them. No early stopping or confirmation-driven
budget choice. Total: 12 pure-RL fits and one diagnostic. No expert checkpoint
is imported into a pure-RL fit.

After all fits, evaluate the prescribed policies once on the retained synthetic
confirmation corpus. It is a reused engineering control, not untouched market
validation. Both TRAIN and confirmation must beat passive, immediate and the
TRAIN-frozen constant-hazard control, separately in each direction, using
deterministic argmax. All arms, including failed ones, remain in the result.

If any pure-RL arm qualifies, run the nine exit-subproblem DEAP searches
automatically: 20 population, 100 generation cap, patience 10, relative threshold
1e-8; three objectives and three seeds. Bias bounds are now [-5,5], inside the
ideal-logit margin of 6. Combined Sharpe/return uses RMS scales from the three
calibration controls. Search-bound hits remain warnings, not proof of an optimum.
Freeze objective finalists before confirmation, and report paired bootstrap
intervals clustered by path parameters (both sides and noise variants together).
No objective is declared the winner merely because it maximizes its own score.

If no arm qualifies, the chain closes with an explicit failed-control result,
not a strategy release. In either outcome, full-strategy optimization and its
cross-audit remain distinct from this two-bias exit subproblem; no financial
floating selection is launched by this diagnostic.
