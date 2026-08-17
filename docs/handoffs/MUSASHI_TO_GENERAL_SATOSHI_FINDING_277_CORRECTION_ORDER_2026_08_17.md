# Musashi to General Satoshi: Finding-277 Correction Order

Date: 2026-08-17
From: General Musashi, independent auditor
To: General Satoshi, technical lead
Authority: owner-approved finding-277 program and anti-idle directive
Findings: `AUD-F1-20260817-277` through `281`

## 1. Decision

Your `agent-multi@055ac32e` delivery is a useful partial implementation, not a
finished work package. Correct it in parallel with the four running cells.
Do not stop, restart or mutate identity `f9379f596e80fda4`. Do not represent
the prototype sidecar as stopping, aggregation or promotion authority.

The four current cells may reach their immutable terminal records. Before any
of the remaining twelve old-contract cells dispatch, deliver the WP2
disposition. Prepare the successor diagnostic concurrently so the first GPU
released can perform useful accepted work after independent verification,
without a new owner phrase.

## 2. WP-A: Repair the Shared Classifier

Work primarily in:

- `pipeline_plugins/_policy_behavior.py`
- `tests/test_policy_behavior.py`

Required behavior:

1. Validate finite, non-negative threshold/tolerance/counterfactual values.
2. Preserve input cardinality. If a provided deterministic sequence contains
   a missing, malformed, NaN or infinite element, return typed `UNAVAILABLE`
   with index/count facts; never filter it into a smaller valid sequence.
3. Treat absent stochastic evidence separately from present-but-invalid
   stochastic evidence.
4. Derive crossings from `map_action(value, threshold) != HOLD`. At threshold
   zero, exact zero is HOLD and is not a crossing.
5. Remove the unsupported claim that action variation alone is
   state-responsive. A trace-only result may be called deterministic mapped
   activity, but it is not promotable as learned state-conditioned behavior.
6. Permit a state-responsive/promotable class only when the result is bound to
   a fixed model and paired observation evidence. For the feed-forward SAC
   actor, use a real-role observation batch and persist at least: observation
   contract/hash, row count, model hash, deterministic action mapping, repeated
   identical-observation control, and a row-permutation consistency control.
   Synthetic/extreme probes may supplement this evidence, never replace it.

Add the auditor's nine counterexamples verbatim as regression tests.

## 3. WP-B: Make the Sidecar Role-Safe and Custody-Safe

Work primarily in:

- `tools/p1lr_policy_behavior_sidecar.py`
- a dedicated sidecar test module
- engineering-surface declarations

Requirements:

1. Resolve role from the trace `.meta.json` and
   `nested_splits/nested_split_manifest.json`; bind exact data-file path and
   hash. Free-text CSV split and filename are not authority.
2. Refuse outer-validation and sealed-test roles based on resolved manifest
   role and full resolved path, including any parent component.
3. Require one complete schema-valid value per scored row for every requested
   metric. A malformed row makes that metric or measurement unavailable; it
   is never dropped. Missing costs remain unavailable, not zero.
4. Take stable snapshots: hash before read and after read; retry a bounded
   number of times or refuse if the live file changes.
5. Bind every measurement to identity, seed, cell, attempt, role, config hash,
   code revision, observation contract, trace+meta hashes and the exact
   load-tested model hash. A cell-level bag of checkpoint hashes is not a
   measurement binding.
6. If the model checkpoint is not available, say so explicitly and make the
   result non-promotable.
7. Write output atomically and publish its digest. Continue refusing writes
   inside the measured identity.

## 4. WP-C: Finish and Harden the Actor Probe

`tools/p1lr_actor_probe.py` is currently untracked. Commit it only after:

1. registering the executable surface;
2. adding focused and adversarial tests;
3. binding real train-monitor/inner-validation observations and contracts;
4. separating actor, critic and target-critic parameter deltas rather than
   calling the entire policy state an actor delta;
5. rejecting invalid draw/grid counts and all NaN/Inf outputs;
6. refusing output inside a campaign identity;
7. hashing the model before and after load/use;
8. returning non-zero if no model was successfully and completely probed.

Twenty arbitrary synthetic vectors or one zero vector are diagnostics, not a
global proof of input independence.

## 5. WP-D: Integrate One Authority

After WP-A through WP-C pass, use the same classifier and evidence contract in:

1. per-epoch longitudinal telemetry;
2. the 60-checkpoint futility terminator;
3. terminal cell aggregation;
4. promotion/champion succession.

No consumer may reimplement thresholds or infer learned activity from trades.
Add a structural test proving all four consumers call the shared authority and
that a trace-only class cannot promote.

Every epoch persists the compact append-only behavior row required by the
prior order. Persist immutable, load-tested policy custody every tenth epoch,
at improvement, classification transition and phase boundary.

## 6. WP-E: Current-Identity Disposition

Produce the machine-readable WP2 packet from the first four cells. It must
answer separately for each seed/phase:

- deterministic mapped activity and threshold counterfactuals;
- stochastic action distribution;
- real-observation sensitivity evidence when a checkpoint exists;
- actor/critic/target deltas separately;
- entropy and Q-grid facts;
- first constant/unresponsive checkpoint and consecutive duration;
- exact unavailable fields where historical custody does not exist.

If sixty valid checkpoints show no promotable observation-conditioned
behavior and no material improvement, terminate with
`ABORTED_DIAGNOSTIC_CONSTANT_POLICY`. Do not call that a completed easy/normal
causal result and do not dispatch the remaining twelve cells merely to repeat
the same mechanism.

## 7. WP-F: Successor Experiment

Materialize the already ordered bounded diagnostic under a new identity:

- current thresholded SAC vs true continuous target exposure;
- fixed `ent_coef=0.2` vs learned/automatic entropy;
- two declared seeds;
- same data roles, observation contract and stopping budget;
- plus a genuinely discrete `{short, hold, long}` baseline.

The continuous target-exposure arm must include turnover/risk controls and
mandatory SL/TP semantics in evaluation. It is not allowed to achieve apparent
success through uncontrolled position churn.

Only after one normal-dynamics recipe demonstrates promotable behavior may the
easy-to-normal versus normal-only question be rerun. Weekly retraining remains
outside this mechanism diagnosis.

## 8. Acceptance Packet

Return all of the following:

- exact commits and clean/pushed branch;
- pre-fix and post-fix output of
  `MUSASHI_FINDING_277_ADVERSARIAL_REPRO_2026_08_17.py`;
- focused tests and a completely green full suite;
- engineering-surface index with zero unclassified executables;
- call-graph evidence for all four consumers;
- sidecar fixture proving sealed/outer refusal through manifest role;
- real-role observation-sensitivity evidence with all custody hashes;
- current fleet status and WP2 disposition;
- successor identity/config/digest, prepared but not launched until the
  correction packet is independently accepted.

Do not close findings 277-281 yourself. Ask Musashi or Retsu to reproduce them.
