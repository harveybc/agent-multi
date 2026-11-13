# Critic Coverage And Representation Review

Owner authorizes review after318. No optimizer, source edits, feature selection,
actor refinement, broker action or STL. Source299,71 inputs,49 cumulative absent
features remain preserved. This is a finite CPU diagnostic.

## Prespecified Measurements

Reuse authenticated318 annual native labels, folds and original FIT membership.
Reuse309's complete source-actor TRAIN embeddings only when checkpoint and
support identities match. Never use its candidate embeddings. Capture online
critic twins using the corrected native-parity cache; no retained hooks.

Use the same label-free cohort for raw windows, source actor and both critics:
up to512 evenly spaced valid TRAIN rows per year plus all239 panel rows.
Report exact coverage; this is a systematic geometry sample, not new targets.

For each318 fold and representation, report centered spectrum/effective rank,
99-percent energy dimension, numerical duplicates, marginal variation absent
in FIT, and nearest-FIT distances for STOP/audit/same-year population. Report
both original geometry and diagnostic scaling fitted ONLY on that fold's FIT.
The latter never changes model preprocessing or inputs; constant FIT dimensions
are reported explicitly rather than divided by a tiny arbitrary number.

Label coverage: original FIT counts by year/quarter/position, target-optimal
action sets including ties, overlapping feature supports, and common-value
versus action-difference target energy. These are descriptive, not estimates
of Shannon capacity or the number of required network parameters.

Use fixed1/5-nearest-neighbor label averaging as a representation diagnostic,
not a replacement learner or selector. Compare against the FIT mean and actor
on identical audit labels, with99 FIT-label permutations (seed319) as a
descriptive negative control. Report all combinations, no winner promotion,
no claims of causality/statistical significance or independent confirmation.
Source encoders/scalers already saw these TRAIN years.

## Test Design

| Requirement | Evidence |
| --- | --- |
| No future normalization | FIT-only geometry parameters; changing query cannot change fit scale |
| Geometry correctness | Known rank-one/constant matrices, duplicates and constant-dimension drift |
| Neighbor correctness | Known nearest rows; stable tie order; FIT-only label prototypes |
| No false independent sample count | Window overlap and quarter counts explicitly reported |
| Source parity | Cached source identity, critic reconstruction1e-6, tensor equality and no retained hooks |
| Authentic lineage | Verify318 panel/fold seals and309 source arrays against their manifests |
| No automatic experiments | Publish review and next bounded contrast, without modifying the champion |

After the first geometry pass showed matching actor/critic neighborhoods,
direct checkpoint inspection found all337 encoder tensors identical for each
twin, consistent with290's explicit actor-to-critic encoder initialization.
Add exact tensor comparison, storage-alias counts and activation differences
to the reproducible report; rerun in a new root, retaining the first pass.
Do not propose an actor-encoder transplant as a new contrast when it already
is the same representation.

Also count native online critic-head parameters directly from the source and
diagnostic-head parameters from the sealed318 checkpoint.317 intentionally
matched the routed and averaged variants at three heads per twin;318 reused
that control. This is not the original one-head-per-twin SAC architecture,
and must not be attributed to a limitation of that native architecture.

## Results And Disposition

Final run identity`1aa7518cf5542a9cf423eced9d7a9c30a3c7badb83d9298dca4236ac5d54ef1a`;
6.94seconds on CPU.18 focused tests pass.239 labeled anchors,2266 systematic
geometry rows drawn from24026 valid TRAIN windows. All artifact seals and
source tensors verified. The geometry tables reproduce exactly across the
last two passes; neither pass uses an optimizer or changes any input.

### Encoder Identity

Both online critics match the actor on all337 encoder tensors, including308
floating tensors: maximum weight difference0, no storage aliases. This is the
deliberate initialization documented in290, not a newly discovered sharing bug.
Different inference dispatch gives at most1.91e-6 in the captured activations;
critic Q reconstruction errors stay below7.01e-7. All nearest-neighbor decisions
are identical for actor core and critic cores in the declared comparisons.
An actor-to-critic encoder transplant would repeat the current configuration.

### Capacity And Coverage

| Item | A | B |
| --- | ---: | ---: |
| FIT anchors | 49 | 98 |
| Corresponding valid TRAIN windows | 5938 | 12160 |
| FIT long / short open-position examples | 6 / 6 | 12 / 12 |
| Exact distinct FIT representation rows | 49 | 98 |
| Pairs with overlapping time intervals | 11 | 34 |
| Native core effective rank / width | 7.97 / 208 | 11.20 / 208 |
| FIT-scaled core effective rank / width | 15.69 / 208 | 22.07 / 208 |

There is no collapse to constant or duplicate embeddings. The overlap count
uses time intervals, not exact intersection of the stride2 sampled timestamps;
it is not an effective independent sample-size estimate. Native latent audit
distance exceeds the FIT leave-one-out95th percentile for about6.8%/6.7% of
rows. Sparse sampling is real, but this does not establish widespread geometric
extrapolation, nor a theorem about sufficient training data or capacity.

Direct tensor counts: actor head1699 parameters; native twin critic heads3398;
the diagnostic's six heads10194. The threefold difference was intentional for
317's parameter-matched routing comparison, but318 inherited it. Those runs
must not be described as a direct test of native critic capacity.

### Targets And Local Information

Common-across-actions value accounts for88.1%/84.2% of centered FIT target
energy; action differences account for11.9%/15.8%. This is target variance,
NOT a measured decomposition of current loss gradients or proof that MSE is
the cause. Action-centered objectives already exist in earlier diagnostics;
do not present them as an untested new invention.

Fixed5-neighbor native geometry, lower regret is better:

| Representation/control | A audit regret | B audit regret |
| --- | ---: | ---: |
| Preserved actor actions | 0.168404 | 0.249439 |
| FIT mean | 0.180611 | 0.230557 |
| Raw input windows | 0.247822 | 0.327006 |
| Actor/critic core | 0.232934 | 0.219126 |
| Actor hidden8 | 0.221643 | 0.200831 |

The hidden actor geometry is informative in B relative to its shuffled-label
median0.257634, but fails to beat the actor or FIT-mean control in A. This is
not a validated new trading policy, a causal discovery or an architecture win.
All1/5-neighbor, native/scaled and shuffled-control results are retained.

### Next Controlled Contrast

First compare the native one-head-per-twin critic with318's three-head average
on the SAME annual labels, folds, MSE, baseWD0.01, LR, seed, schedule and
stopping rule. Four new native-capacity fits; reuse the four sealed baseWD
controls. Match each native head to head0 of its corresponding diagnostic twin
at initialization rather than introducing an unrelated random initialization.
Record the function difference at step0; equal common weights do not mean
equal ensemble outputs. Keep the source actor frozen. Do not simultaneously
change input features, label support, representation, loss or trading rules.

Then decide whether a coverage contrast is warranted;288 already tested a
different113-versus437 panel, so a new coverage run must justify its distinct
identity rather than repeat it. No GPU training or further floating pass was
started by this review;71 inputs and49 cumulative return candidates remain.

Evidence: [complete table](generalization/critic_coverage_geometry_20261113/RESULTS.md),
[numeric summary](generalization/critic_coverage_geometry_20261113/SUMMARY.csv),
[contract](generalization/critic_coverage_geometry_20261113/CONFIG.json).
