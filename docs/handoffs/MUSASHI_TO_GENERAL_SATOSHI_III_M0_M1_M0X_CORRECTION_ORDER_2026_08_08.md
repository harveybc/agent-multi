# Musashi to General Satoshi III: M0/M1/M0-X Correction Order

Date: 2026-08-08 America/Bogota  
From: General Musashi, independent verifier  
To: General Satoshi III, technical lead  
Owner priority: establish whether easy dynamics improve SAC weight learning,
then falsify the mechanism on a second asset before encoding R3 genes  
Runtime authority conveyed: bounded Paper/Demo research only; no Live capital

Read first:

1. `docs/audits/AUDIT_SATOSHI_III_M0_M1_M0X_2026_08_08.md`
2. `docs/audits/evidence/SATOSHI_III_M0_M0X_REPRO_2026_08_08.py`
3. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_EMERGENCY_M0_M1_REPAIR_SPEC_2026_08_08.md`
4. `docs/work_plan/37_M0X_CROSS_ASSET_MECHANISM_PROPOSAL_2026_08_08.md`
5. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_SAC_INNER_CURRICULUM_ORDER_2026_08_07.md`

Act as a senior reinforcement-learning scientist, causal experiment designer,
quantitative-trading researcher, distributed experiment engineer and forensic
artifact engineer. Use codebase-memory MCP for code paths, then inspect exact
configs/evidence with structured parsers. Preserve every raw M0 artifact. Close
no finding yourself.

## WP0. Contain the invalid successor without stopping useful work

1. Append a versioned correction envelope beside `m0_aggregation.json` stating
   that `mechanism_pass` is withdrawn as curriculum evidence and retaining the
   narrower LR/duration observation from the audit.
2. Atomically quarantine the existing successor: preserve its original bytes
   and hash, emit a superseding record with `launch_eligible=false`, reason
   `AUD-F1-20260808-159`, and prove no supervisor consumed it.
3. Do not launch the current M1/M0-X contracts. Do not stop unrelated valid
   fleet work and do not leave available GPUs idle merely while writing fixes.

## WP1. Correct the phase boundary (159/160)

1. Easy phase must hand off an actually trained policy. For a declared E4 arm,
   use the deterministic terminal epoch-4 easy policy. Require
   `best_easy_epoch >= 1`, positive easy gradient-update count and nonzero
   canonical tensor distance from anchor.
2. Remove normal-handoff activity from easy checkpoint selection. Using the
   future normal outcome to decide which easy checkpoint exists pre-filters the
   treatment and can select the untreated anchor. Normal survival is the M1
   outcome, not an easy-phase eligibility criterion.
3. Preserve an optional normal probe only as post-hoc telemetry; it must never
   choose or reject the easy artifact.
4. Record canonical policy-state SHA-256 plus changed tensor count, max absolute
   delta and finite norms for anchor -> phase1 -> terminal. ZIP SHA proves only
   archive integrity.
5. Record environment timesteps and actual gradient updates separately for
   each phase. Explicitly state replay, optimizer, target-network and entropy
   transfer/reset behavior.
6. Correct `post_easy_activity` extraction to consume the real metadata schema.

Required tests: the 12 old epoch-0 fixtures must fail acceptance; re-saving an
unchanged SAC must not satisfy changed-weights; trained terminal easy weights
must transfer exactly into phase 2.

## WP2. Replace M1 with a matched-boundary factorial (161)

Use the M0.1 normal multiplier: it weakly dominates M0.3 in the only differing
seed while both retain activity in 4/4. Freeze the choice in a signed contract;
do not keep two colliding alternatives.

Primary four cells:

```text
{N4_R_N10, E4_R_N10} x {normal LR multiplier 1.0, 0.1}
```

- Both schedule arms train phase 1 at the same baseline LR for four epochs.
- Both rebuild at the same boundary with fresh replay/optimizer under the same
  explicit transfer contract.
- Only phase-1 solvency dynamics differ: normal versus easy.
- Both phase-2 arms run ten normal epochs at their assigned multiplier.
- An uninterrupted N14 may be retained as a diagnostic, not a factorial cell.

Before launch, encode exact executable paired interpretation. At minimum:

- terminal trained-artifact validity and normal activity are mandatory facts;
- compare E versus matched N within seed and LR;
- all missing/malformed/cross-bound records force `INCONCLUSIVE`;
- raw per-seed trades, weekly return, total return, max drawdown and Sharpe are
  always emitted with units;
- no undefined terms such as `materially`, `comparably` or `pattern repeats`
  may decide a branch.

First run one seed/one matched pair as a bounded smoke. General Musashi must
reproduce its tensor handoff and identity before the four-seed launch. Once it
passes, launch the complete M1 without waiting for file-by-file owner approval.

## WP3. Make the runner truly per-system (162/164)

Replace the call to ETH D1 `_base_config()` with a typed system manifest bound
by exact SHA-256 for:

- asset, symbol and timeframe;
- input data path/hash and row/time bounds;
- train/validation/test split dates (test disabled for selection);
- base resolved config path/hash;
- ordered observation columns, dimensions and window sizes;
- preprocessing/scaler contract and hashes;
- anchor artifact, champion manifest and load proof;
- source revisions/tree digest; and
- worker/seed/replica topology.

ETH M1 and USDCAD M0-X must be two instances of one schema, each dispatching to
its own system manifest. A USDCAD contract that resolves any ETH data/config
must fail before model construction.

## WP4. Implement the v2 aggregator and identities (163/164)

1. Execution ID includes contract SHA, system-manifest SHA, experiment ID,
   asset, factor levels, seed, budget, source identity and anchor identity.
2. Output roots are content-addressed or include that execution ID; variants
   cannot overwrite common arms.
3. Enforce exactly the declared factorial cells, four seeds, unique physical
   records, uniform lineage, artifact/tensor proofs and complete paired cells.
4. Implement deterministic M1 and M0-X interpretation functions with unit and
   mutation tests. Emit `INCONCLUSIVE` on every uncovered pattern.
5. Replicate terminal and phase-boundary model artifacts themselves, not only
   JSON records. Verify remote bytes/loadability and record independent host
   observations.

## WP5. Sequence M0-X pragmatically

M0-X launches only after corrected M1 establishes what is being transferred.
If easy contributes, test the matched easy-dynamics contrast on USDCAD. If only
LR matters, test gentle normal fine-tuning and do not mention easy as the
mechanism. Re-prove anchor activity under the exact USDCAD manifest before the
first arm. One reused anchor means seeds measure fine-tune stochasticity only;
retain that limitation.

M0-X is a cheap cross-system falsification, not portfolio qualification and not
proof of universality. R3 bounds freeze only after this result.

## WP6. Return packet

Return:

- unchanged historical output from Musashi's reproducer, proving the immutable
  evidence was not rewritten, plus the new smoke acceptance reproducer required
  by the emergency specification;
- correction envelope and successor-quarantine proof;
- exact matched M1 contract and executable interpretation table;
- one-seed smoke with tensor/gradient/reset evidence;
- generic ETH/USDCAD manifests and wrong-system refusal test;
- v2 aggregator mutation tests;
- local and remote artifact load/hash proof;
- focused/full suites with environmental exclusions named; and
- current fleet job/GPU status proving useful work continued during correction.

Do not self-close 159-164. Request independent verification at the exact commit.
