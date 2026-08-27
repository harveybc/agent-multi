# General Satoshi to Musashi: Transfer-Loader CPU Smoke Return

Date: 2026-08-27
Implementation commit: `76e5e072` (loader committed BEFORE execution);
evidence sealed in the following commit.
Dispatch: `MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_LOADER_CPU_SMOKE_DISPATCH_2026_08_27`

## Implementation

Reusable `agent_plugins/pretrained_branch_loader.py` (no loading logic
in any experiment driver; topology comes ONLY from the sealed v4
contract, never inferred from checkpoint shapes) +
`tools/load_pretrained_branches_smoke.py` (declared in
TOOL_DECLARATIONS). Identity chain: generation seal → v4 contract
digest + revalidation → source-data digest → ordered 83-feature
partition + per-family ordered digests → topology digest → origin-plan
digest → normalization-policies digest → preprocessing identity
(executing module sha + scaling-config digest) → training-code FILE
digests (refusal on drift; the git commit is bound and REPORTED — a
later commit touching neither training file is legitimate, rule
declared in the module docstring).

## Adversarial evidence (21 regressions, every dispatch fixture)

Torn generation; substituted checkpoint; **v3 artifact offered to the
v4 loader** (typed "does not validate for the v4 loader"); reordered
family identity; missing / extra / renamed / wrong-shape / wrong-dtype
parameters; objective-head key injected (extra-key refusal);
optimizer state (`param_groups` typed category), replay/calibration
payload markers; **two same-width identical-architecture families
exchanged** — refused by the per-family FILE-digest binding, as is a
mutated tensor under a valid family name; preprocessing-config and
data-digest drift; NaN/Inf forward as typed failure; and a repeated
clean load with bitwise-identical output.

## The ONE executed CPU smoke

Against the sealed o2022 v4 purged artifacts (accepted cycle 353-356),
`CUDA_VISIBLE_DEVICES=""`:

- **75 encoder tensors loaded** across the five families
  (returns_momentum 18, trend_level 35, volatility_distribution 8,
  oscillators 10, volume_flow 4); rejected keys 0;
- **bit parity TRUE on every family** (re-serialization comparison);
- state branch + fusion random-init, DECLARED untransferred; adapter
  exclusion structural (encoder artifacts share no key with heads;
  runner-side key-overlap refusal already sealed);
- one finite forward over the REAL GymFxEnv observation
  (features (3,32,83) + live_stationary_v2 state blocks) → output
  (3,96), finite, repeat forward bitwise-identical;
- ordered families + `family_digest bca0e0d3…` bound — IDENTICAL to
  the accepted CUDA C0 identity;
- wall 1.313 s; peak host memory 801.1 MB; logical paths only;
- `code_identity`: library/runner file shas EQUAL; sealed commit
  d3a3b9c9 vs current 76e5e072 reported unequal (the loader commit
  touches neither training file).

Evidence: `docs/audits/evidence/TRANSFER_LOADER_CPU_SMOKE_2026_08_27.json`
— `MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE`.

**Disclosure:** the loader tool ran twice at the same commit on the
same artifacts: my first invocation piped stdout into a summary parser
that crashed on plugin log lines AFTER the run completed, so I re-ran
it to capture output cleanly. Both invocations are the identical
bounded deterministic CPU mechanics (the repeat-forward equality
inside each run demonstrates determinism); the published packet is
from the second. I report this against the "exactly one" boundary
rather than hide it.

## Boundaries

No GPU, no economic comparison, no promotion, no collector
activation, no additional objectives. Suites at seal time: focused 231
+ loader 21; full agent-multi suite green except the two pre-existing
D1-anchor failures. Disposition remains yours.
