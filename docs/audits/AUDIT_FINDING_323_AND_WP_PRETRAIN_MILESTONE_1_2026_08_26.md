# Audit: Finding-323 Cleanup and WP-PRETRAIN Milestone 1

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@73514a3b`
Auditor: General Musashi
Disposition: **323 TIP CLEANUP ACCEPTED; PRETRAIN MECHANICS ONLY**

## Independent reproduction

Focused cleanup and pretraining tests: **136 passed**. The runner executes,
builds all five branch artifacts, records identities and demonstrates a bounded
resume path. These facts are accepted as mechanics.

Finding-323 active-tip cleanup is accepted for the audited branch. The secrets
sweep reports no real credential requiring rotation. History is not rewritten.

The CPU smoke artifacts are **MECHANICS_ONLY_NOT_TRANSFER_ELIGIBLE** pending the
findings below.

## Findings

### DATA-SOTA-341 — S2 — Pretraining consumes reserved validation years

The contract sets `fit_end=2023-12-31` and the evidence reports 13,699 rows.
Under the established nested split, 2022 is monitor and 2023 is inner
validation; neither is ordinary fit data. Pretraining is model fitting, so its
weights have learned from both roles. Excluding only outer-2024/sealed-2025 does
not preserve the declared validation design.

Required: causal per-origin contracts. For score-2022, all pretraining ends
before 2022; later origins may expand only after the earlier decision is frozen.
No artifact trained through 2023 may enter a 2022/2023 comparison.

### DATA-SOTA-342 — S2 — Pretrain/runtime observation distributions differ

Pretraining reads the exported CSV directly and applies per-window channel
z-score. SAC receives observations through the executing GymFxEnv preprocessing
contract. The packet does not prove numerical parity between those tensors.
Transferring an encoder across different normalization distributions can erase
the intended benefit and repeats the observation-contract defect family.

Required: generate pretraining windows through the same pinned observation
pipeline as RL, or implement one shared transform called by both. Prove
same-timestamp tensor equality by feature/family before weights become loadable.

### DATA-SOTA-343 — S3 — Masked targets influence visible normalized inputs

`instance_normalize` runs on the complete window before the temporal mask is
applied. Changing a masked raw value changes mean/std and therefore changes
visible normalized values. The reconstruction objective can exploit statistics
contaminated by what it is supposed to infer.

Required: normalization statistics for masked reconstruction exclude masked
positions, or use training-fitted causal statistics independent of the current
mask. Add an adversary where masked raw values change while all visible raw
values remain fixed; visible model inputs must remain identical.

### DATA-SOTA-344 — S3 — Declared normalization epsilon is ignored

The contract declares `input_normalization.eps`, but the runner calls
`instance_normalize(...)` without passing it. Changing the config changes the
contract digest while leaving execution unchanged.

Required: parse/validate/persist and apply epsilon, with sensitivity and
call-path tests. More broadly, use typed per-family normalization: binary,
bounded, return, level and volume channels must not all inherit an arbitrary
single transform without evidence.

### DATA-SOTA-345 — S3 — Objective weighting is arbitrary and reconstruction-dominated

Both losses receive weight 1 although their observed scales differ by roughly
one to two orders of magnitude. A falling total mainly demonstrates easier
reconstruction, not a representation useful for returns. No monitor split,
gradient norms, gradient conflict or held-out objective metric supports the
weights.

Required: report normalized per-objective losses, gradient norms/cosines and
held-out fit-tail metrics. Select weighting by a predeclared train/monitor rule
(or a bounded balancing method), never by defaults or total-loss monotonicity.
Quantile crossing must be measured and preferably constrained.

### DATA-SOTA-346 — S3 — Resume identity and crash durability are incomplete

`resume_identity` omits runner SHA, code commit, Torch version and normalization
implementation identity even though the manifest records some of them. The
checkpoint and manifest are written non-atomically; an interruption can corrupt
or mismatch them. The exact-resume test compares encoder weights but not heads,
optimizer, losses and final artifact digests.

Required: bind every executing identity into resume, atomically write+fsync
checkpoint and manifest as one generation, reject torn generations, and compare
the complete resumed artifact set bit-for-bit.

## Next disposition

Correct 341-346 before implementing three more objectives. Adding contrastive,
volatility and barrier losses atop contaminated roles/distributions would
multiply uncertainty. Collector implementation may continue independently.
No pretrained encoder from this smoke may load into SAC.
