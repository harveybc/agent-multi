# Discrete SAC Exploration Correction

## Observed Failure

Revision2 did not qualify: seeds7 and43 classified168/216 and151/216 synthetic
cases correctly. Incorrect opening choices saturated near probability1, and
some correct actions fell to probability0. No full financial candidate fit
started. Worker and collector services were stopped; original evidence remains
unchanged. Report174 describes a historical launch, not current running state.

The contextual one-step qualification sampled only its current actor. That
starved Q actions following an early collapse. Fixed entropy alpha0.01 could
not recover the saturated actor. This was found before financial experiments,
not inferred from poor validation returns.

## Isolated Engineering Checks

On the small two-feature fixture, uniform legal-action behavior alone with
alpha0.01 still failed120/216. With the same behavior, fixed alpha1 learned
all216 cases, each expected-action probability above0.6, in70.77seconds CPU.
These are debugging controls, not trading or hyperparameter-selection results.

The production correction uses official TorchRL automatic entropy, initialized
at1, with target0.6*mean(log(number of legal actions)). It includes the
temperature loss in optimization and records alpha and target entropy. Forced
single-action NOP contributes zero target entropy. The fraction0.6 is an
explicit initial probe setting, not a measured optimum or success probability.
Uniform legal-action behavior applies to synthetic qualification. Financial
training retains1024 warmup actions followed by SAC sampling and uniform replay.
The corrected automatic-entropy run on that same small fixture passed216/216;
the minimum expected-action probability was0.817954. This does not substitute
for qualification of the full120-column network on each production seed.

## Verification and Scope

29 focused tests pass, including temperature learning, mask-aware entropy,
duration discounting, native accounting, both directions, checkpoint reload
and input invariance. The full216-case learning check and mirrored native
positive controls remain compulsory for every seed before financial fits.
No threshold is lowered to turn a failed qualification into a pass.

The12 authorized cells retain horizons3/6/12/24h, seeds7/19/43,120 selected
features,31 groups,13 observations at stride2, fixed TRAIN scaling and no
learned agent-state input. Synthetic weights and optimizer are discarded.
The real probe is fresh discrete SAC, never PPO or supervised ZigZag learning.

New immutable source, campaign identity and output roots are required for the
corrected run. Completed evidence from earlier revisions cannot satisfy its
receipts. Final allocation remains6/5/1 cells across the three authorized GPUs;
each worker is sequential and the collector consolidates all12 outcomes.
