# General Musashi Return: Model Capacity and Information M0-M2

Date: 2026-09-07

Disposition: `M0_M2_PILOT_ACCEPTED_AS_BOUNDED_RANGE_EVIDENCE`

## 1. What was executed

The complete bounded CPU order was implemented and run in an isolated
`agent-multi` worktree. No financial data, sealed-2025 data, venue, service,
GPU, B4 cell, T2 score, DOIN gene or live path was read or modified.

The final design v3 was committed and pushed before scoring:

- commit: `72e91212`;
- design self-digest: `5df1c6dfde6b06267d44c50355c71211396318c523ce7ebea221650422947c2a`;
- runner digest: `ca632974dc7edfc1c009f81ce30eab8ca5633dade314b1c490fa1147973f1899`;
- measurement-contract digest: `6c5d35c29ea8aa9d121b93260c3990c4a145ec36f513e79fd5cd0cdcfcedeb62`.

The population contained 195 units:

| Family | Units | Contents |
| --- | ---: | --- |
| Threshold neuron | 45 | K={4,8,16}, five N/K ratios, three seeds |
| Boolean MLP | 90 | identity, majority, DNF, XOR and random-label null; clean/20% exceptions; three widths/seeds |
| Temporal MLP | 60 | sine, chirp, heavisine; white, colored and impulsive noise; two widths/seeds |

Every scientific result was reconstructed from design, unit specification and
frozen code. The run took 1,833.014 seconds including in-process full
reconstruction. Peak RSS was 947,482,624 bytes, below the 1 GiB ceiling.

## 2. Defects found before the final run

### V1 evidence-authority defect

The first verifier trusted coherent producer output after checking self-digests.
Three attacks passed: a forged threshold accuracy, a forged checkpoint metric
with all nested digests repaired, and an undeclared result field. They are
frozen in `MODEL_CAPACITY_M0_M2_V1_VERIFIER_PRE_2026_09_07.json`.

V2 killed all three by re-executing each unit and requiring exact result
equality. A fresh process reproduced all 135 v2 units exactly.

### V2 scope audit

V2 still omitted explicit identity/random-label controls, structured temporal
noise and pruning/low-rank curves, and left the theoretical input assumption
ambiguous. These were scope omissions, not altered outcomes. V3 superseded v2
append-only and executed the full 195-unit bank.

One earlier test draft also contained repeated dead assertions. It was replaced
before any design was sealed. An integrated smoke then found that a post-stop
diagnostic epoch could update the selected state; that defect was corrected and
regression-tested before v1 scoring.

## 3. Results

### Single threshold neuron

The majority-fit transition was observed around the expected N/K region:

| K | Last ratio with majority perfect fit | First ratio with majority failure |
| ---: | ---: | ---: |
| 4 | 1.5 | 2.0 |
| 8 | 2.5 | 3.0 |
| 16 | 2.0 | 2.5 |

Independent random-label evaluation stayed near chance. This is range evidence
only: K is small, there are three seeds, and the pocket perceptron mixes
separability with finite-budget optimization. It does not confirm a universal
2K law.

### Memorization and generalization controls

- The first-bit identity control reached evaluation accuracy 1.0 with clean
  labels and with 20% corrupted training labels; selected checkpoints memorized
  none of those exceptions on average.
- The random-label null averaged about 0.50 evaluation accuracy while selected
  train accuracy was about 0.53. Extra width did not create out-of-sample rule
  recovery where no rule existed.
- XOR remained near chance for all widths. This is classified as an optimizer
  or training-protocol limitation, not evidence that the architecture lacks
  representational capacity.

### Temporal known-truth signals

Mean selected evaluation R2 across families and widths was approximately:

| Perturbation | SNR | Mean evaluation R2 |
| --- | ---: | ---: |
| White | clean | 0.941 |
| White | 10 dB | 0.858 |
| White | 0 dB | 0.665 |
| Colored | 10 dB | 0.743 |
| Impulsive | 10 dB | 0.860 |

Equal declared SNR did not imply equal learning difficulty: colored noise was
materially worse than white or impulsive noise in this bank. Noise structure
must remain part of the feature/operator evidence, not be collapsed into one
SNR scalar.

### Stopping and description length

Early stopping fired in 114 of 150 MLP units. Among stopped units, the bounded
post-stop continuation changed evaluation score by -0.0119 on average and was
worse in 53/114 units. Across all MLP units, selected mean evaluation score was
0.7765 versus 0.7674 at the endpoint. The diagnostic branch never selected or
promoted a checkpoint.

Within-unit Pearson associations for raw float32 zlib length were 0.352 with
train score, 0.303 with evaluation score and -0.084 with sample-specific
memorization (45 defined units). These are descriptive associations, not
information content or causality. Raw compressed length is rejected as a
standalone measure of knowledge or residual capacity.

The cost result is equally important. Producer-measured diagnostic CPU totals
were approximately 535 s for model descriptions, 45 s for initial descriptions,
17 s for data descriptions, 1.93 s for gradient/Fisher/Hessian sketches, 0.70 s
for losses/scores, 0.39 s for activation geometry and 0.26 s for exception
memorization. Full compression/pruning/SVD instrumentation consumed most of
the pilot cost and should be sparse in confirmation.

## 4. Accepted interpretation

The pilot establishes that the measurement machinery works and that the
questions are empirically separable. It does not establish exact Kolmogorov
complexity, intelligence in bits, knowledge stored in individual weights,
capacity remaining from repeated weights, or a general law for MLPs.

The strongest scientific conclusions are:

1. the single-neuron reference is recoverable as a bounded calibration;
2. training failure must be separated from representational capacity;
3. stopping, memorization and generalization are not interchangeable;
4. noise structure matters beyond scalar SNR; and
5. checkpoint compression is too weak and too costly to carry as the primary
   estimator, though sparse measurements remain a valid secondary descriptor.

## 5. Verification and next design

Focal pilot tests: 20 passed. Pilot plus engineering-surface index: 37 passed.
Ruff and `git diff --check`: clean. The first full-suite pass reported 3,138
passed, 5 skipped and four failures: the new executable declaration, one B4
claim race inherited from the branch base, and the two known D1 host-evidence
failures. The declaration was corrected; the final full-suite count is recorded
from the final unchanged tree: 3,140 passed, 5 skipped and the same two known
D1 host-evidence failures. The inherited B4 claim-race test passed in this
second full run.

Five production-code mutations were run and reverted before publication:

| Mutation | Tests broken |
| --- | ---: |
| Disable fresh scientific recomputation | 2 |
| Permit repeated weights to imply residual capacity | 1 |
| Call an all-fit endpoint a saturation transition | 1 |
| Permit evaluation-driven stopping in the design | 1 |
| Remove pruning and low-rank curves | 1 |

After reversal, both code-file SHA-256 values again matched the sealed design.
The generated public summary has self-digest
`743ed8dca3d9783d95d9d374e8b4d6ffc67e4ea717b6baf968b722f2a9f3748d`.
The complete summary and both superseded-stage disclosures are committed under
`docs/audits/evidence/`.

The proposed M3-M6 confirmation is fully written in
`M3_M6_CONFIRMATORY_DESIGN_DRAFT_2026_09_07.md`. It uses an
optimization-independent separability solver for the Cover calibration,
introduces the retention-constrained residual-capacity intervention, and makes
incremental decision utility over simple controls the admission gate. It is a
proposed confirmatory design, not an executed result and grants no GPU, DOIN or
live authority.
