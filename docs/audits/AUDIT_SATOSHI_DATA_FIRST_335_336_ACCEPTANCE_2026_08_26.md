# Audit: DATA-SOTA-335/336 Acceptance

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@4389e115`
Auditor: General Musashi
Disposition: **ACCEPTED; BOUNDED CUDA C0 DISPATCHED**

## Independent evidence

- Tier-A focused set: **188 passed** against the real ETH CSV, CPU forced.
- Independent counterexamples now refuse before Torch construction:
  boolean patch length, string dropout, fractional window, boolean branch width
  and duplicate family identity.
- Fusion consumes named runtime records; equal-width branch swaps refuse by
  family identity. The extractor supplies those records and persists the ordered
  family digest.

DATA-SOTA-335 and DATA-SOTA-336 are independently verified corrected. Findings
329-336 are accepted for the first-wave scope.

## Authorized next action

Run exactly one bounded CUDA C0 mechanics smoke on one available GPU. It must:

1. use the accepted real-env Tier-A fixture and tip `4389e115`;
2. perform no economic comparison, checkpoint promotion or B4 dispatch;
3. publish device/UUID redacted identifier, CUDA/PyTorch versions, wall time,
   peak allocated/reserved memory, parameter count, output finiteness, nonzero
   gradients by every named branch/fusion/actor-facing output, and save/load
   output parity;
4. bind command, config, data digest, code commit and family digest;
5. return the GPU to the scheduler after completion.

CPU implementation of collectors and the pretraining runner continues in
parallel. No long architecture campaign is authorized by this acceptance.
