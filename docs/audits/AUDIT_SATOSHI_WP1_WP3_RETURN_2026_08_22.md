# Audit: Satoshi WP1/WP3 return

Date: 2026-08-22 America/Bogota
Auditor: General Musashi, temporary independent auditor
Commits: `3b68ea00`, `9433ebb3`
Runtime/GPU mutation: none

## Verdict

**REJECT FOR CORRECTION. Do not launch the early-intervention GPU screen.**

The numerical WP1 artifact reproduces and the proposed next experiment is
scientifically reasonable, but two authority defects and one durability defect
survive. Focused verification passed 44/44 tests. The claimed full suite was
not independently reproduced in the auditor's base environment because 18
collection groups lack project runtime dependencies; this is an environment
limitation, not a regression attributed to the commits.

## Accepted facts

- The committed post-intervention numbers reproduce exactly.
- The `POST_HOC_EXPLORATORY` and zero-authority labeling is preserved.
- All 12 exploratory signs are negative; official `INCONCLUSIVE` is unchanged.
- The early timing/patience screen is a better next candidate than immediately
  paying for multi-year confirmation.
- The real `supervise` CLI lifecycle and targeted tests are substantial
  corrections to REC-01/REC-03.

## AUD-F1-20260822-PLR-08 (S3, reproduced)

WP1 claims a bit-identical pre-intervention pair, but verifies only seed,
`data_sha256`, contiguous epochs and equality of `composite`. It does not reuse
the accepted `verify_pair` contract, compare full commit/config/split identity,
or compare a declared canonical projection of pre-treatment trajectory facts.

Counterexample: changing the seed-101 plateau report to a different 40-hex
commit and config hash while retaining seed/data/composites exits 0 and emits
the same diagnostic. A foreign experiment can therefore enter the diagnostic.

Required: reuse the exact pair-identity verifier (after the retired frozen-tip
exception) and define/hash the complete pre-treatment projection. Differences
that are treatment metadata may be explicitly excluded; all economic,
observation, action, optimization-state and model-movement facts must match or
refuse. Tests must mutate each identity field and each projected field.

## AUD-GEN-20260822-REC-05 (S3, reproduced)

Semantic completion accepts any schema starting with
`agent_multi.wp4_smoke`. A report carrying
`agent_multi.wp4_smoke_malicious.v999` plus matching superficial fields is
classified `completed`.

Required: exact allowlist equality (`agent_multi.wp4_smoke.v2` for this
controller version), or an explicit version parser with a separately audited
migration table. Prefix matching is forbidden. Add foreign-prefix, suffix,
major-version and Unicode-confusable fixtures.

## REC-04 remains open (S3, observed)

Manifest and intent JSON use `_atomic_write`, which fsyncs file and directory.
The canonical launch artifact uses `write_bytes -> replace -> fsync(parent)`;
the temporary file itself is never flushed/fsynced. The commit claim that the
launch artifact is durable is therefore not established.

Required: an atomic byte writer that flushes/fsyncs the temporary launch file,
renames it and fsyncs the parent. Inject file and directory fsync failures and
prove neither can be acknowledged as durable.

## WP2 disposition

The proposed early-intervention screen is **conditionally accepted as the next
GPU experiment**, only after PLR-08, REC-05 and REC-04 independently reproduce
closed. Before launch, simulate the proposed scheduler over the existing four
fixed monitor histories and persist the predicted first-reduction epoch per
seed. The experiment must state that a negative result rejects this scheduler
specification for the bounded ETH setting, not every possible plateau-LR
mechanism universally.
