# Explicit frozen-reference rollout correction

2026-10-30. Owner orders implementation and unattended execution now; no
additional design approval is required. Supersedes the fictitious fixed
two-order continuation as the next candidate, not the archived controls.

## Implemented contract to test

At a flat tick consider NOP and all eight BUY/SELL x3/6/12/24h openings. At an
occupied tick consider HOLD and CLOSE. For each action and frozen reference,
execute that action in the native account, then run the reference each hour
until native weekly flat, with no two-order limit. References are the frozen
simple and predictive-proxy controls from163; passive continuation is also
included so inactivity is a real alternative. Neither reference recursively
calls the rollout selector.

Value each common action/reference pair over every current report sample:
expected final return minus the fixed account-DD weight times worst sample
DD. No additional fitted coefficients and no per-sample oracle selection.
Conservative tied actions prefer NOP/HOLD. Execute the chosen first action
and re-evaluate next tick; no hidden persistence override after selection.
Reference policies retain their own close persistence and memory, observed
once per real tick and independently cloned into each candidate branch.

The quantity is explicitly Q of the frozen reference continuation, not Q of
the resulting adaptive selector. Under deterministic ideal reports, exact
finite-horizon rollout with an available reference action has a policy-
improvement comparison against the reference, not global optimality. Under
noisy changing reports or robust sample criteria there is no such general
guarantee; measure the result. Do not relabel the170 defect as solved merely
because an implementation exists.

## Information boundary

Simulated prices use the declared current report paths. Future reference
decisions receive only suffixes of the SAME whole report ensemble, with the
original observed-through timestamp retained and lineage recorded. They never
receive the path/sample identity, a sample-specific perfect forecast, or a
fresh report fetched from observed real future. The forecast is carried
forward, not magically improved. Simulated account/position state stays in
the controller; no agent-state feedback is added to learned model inputs.

This frozen-information continuation is deliberately conservative and
explicit. It does not simulate a trained model's future forecast updates.
The real next tick may receive a newer report, so report-update sensitivity
remains observable rather than assumed away.

## Tests before implementation

| ID | Contract |
|---|---|
| FR01 | Root action is common across samples; no sample-wise maximization |
| FR02 | Future report suffix preserves ensemble/weights/lineage; future source callbacks forbidden |
| FR03 | Same observed prefix and report yield same action despite altered actual future |
| FR04 | Snapshot and controller memory isolated; exactly one real observation per reference/tick |
| FR05 | Both directions, all horizons, NOP, occupied CLOSE/HOLD and native protection respected |
| FR06 | Ideal fixed-reference rollout reproduces the reference's native continuation |
| FR07 | Small ideal native fixtures do not underperform included frozen references in terminal utility |
| FR08 | Frozen controls unchanged; fresh seeds, matched worlds, no optimization on confirmation |
| FR09 | Resumable per-account receipts reject code/config/source changes and mutated artifacts |

## Automatic experiment sequence

Tests -> measured smoke -> fresh ideal controls and rollout variants ->
actual-policy close attribution -> descriptive noise calibration -> other
four reporter settings -> final tables, receipts and audit packet.
No positive-profit admission gate and no owner approval between stages.
First compare joint reference rollout plus its simple-only/proxy-only
ablations against the two unchanged references and both legacy sequential
controls. Seven fixed policies, no coefficient search. Fresh calibration
seed990000 and confirmation1000000; descriptive calibration never selects
policies. Each bank has36 worlds. Total1,512 native accounts.

After the complete ideal block, automatically measure one-tick CLOSE-to-HOLD
interventions for every executed close of the joint rollout controller on
those same worlds, continuing from cloned native/controller snapshots. This
checks the actual adaptive policy separately from reference-value claims.
Include a cold full-replay equivalence test for the snapshot intervention.

Bounded four-worker CPU execution,6GiB aggregate memory cap, durable per-case
completion and restart. Runtime is measured before stating ETA. No financial
training, feature-selection launch, data alteration or automatic promotion.

## Implementation and Preflight

Implemented in `app/reference_rollout.py`; the unattended entry point is
`tools/run_reference_rollout.py`, with
`configs/reference_rollout_contrast.json`. The policy specification and
reference configurations are pinned per arm in MANIFEST; the underlying
account `config` field retains the common native runner configuration and
must not be mistaken for the rollout policy specification.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python tools/run_reference_rollout.py \
  --source <completed-pragmatic-contrast> \
  --engine-root <pinned-native-engine> \
  --output <new-run-root> --workers 4
```

The same command resumes authenticated units. Add `--smoke` and use a
different output root for the disjoint small bank. Reference policies are
never fitted on either new bank. Every confirmation report receives the same
world and native business/cost contract.

Final preflight identity:
`9e6dc21b10d9d56892a3cbd1398e6ab50062cf0125cd8de3a6832b8898c8fe55`.
285 tests pass with no skips; lint and completed-resume verification pass.
21 accounts and six actual-policy close interventions completed. Five closes
improve account profit and utility; one worsens both. Three CLOSE-dominant,
one HOLD-dominant and two return/DD tradeoffs. This is a small engineering
test, not a final comparison or proof that adaptive continuation is solved.

Preflight receipts and complete tables:
[evidence](../audits/evidence/reference_rollout_preflight_20261030/RESULTS.md).
The copied COMPLETE file indexes the full local archive; not every raw unit
is duplicated in git. No negative close outcome is excluded from the queue.

The measured small-bank work extrapolates to about1.7 CPU-wall hours with
four workers under similar load; initial operational allowance is2-3hours.
This is an estimate, not a completion promise. Phase ETA remains undefined
until a duration for every outstanding policy family has been observed;
fast controls are not used to predict unmeasured rollout costs.

Automatic terminal outputs: REPORT.json, RESULTS.md, per-account and per-close
receipts, COMPLETE.json and AUDIT_REQUEST.md. Review reference valuation,
actual adaptive continuation and learned-interface qualification separately.
