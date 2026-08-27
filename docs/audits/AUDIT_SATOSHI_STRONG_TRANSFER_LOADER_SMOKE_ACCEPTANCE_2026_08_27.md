# Acceptance Audit: Strong Transfer-Loader Replacement Smoke

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@0ebd32d0`

## Verdict

**ACCEPTED AS END-TO-END MECHANICS EVIDENCE.** The one authorized replacement
CPU smoke completed through the strong effective config, canonical SAC-shared
materializer, strict encoder loader and authenticated single-use custody.

It remains `MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE`. This acceptance does not
claim that pretraining improves trading, authorize promotion, or authorize a GPU
economic experiment.

## Independent Verification

- Focused/Tier-A reproduction: **93 passed**.
- Evidence SHA-256 independently recomputed:
  `1b3cb916cce57950bf31e8694c6f0a750d95ecd314c3bdba1afb34290730d2be`.
- Private ledger observed: state `completed`, attempt 1, transition sequence 4,
  `forward_started=true`, matching evidence digest/schema/run/dispatch.
- Completion-intent marker absent.
- Authenticated `--render <ledger-key>` reproduced the packet without model
  construction.

## Accepted Facts

- Config snapshot digest: `e6c05b51...c90b8b`.
- Effective architecture digest: `fda91f37...8a4da5`.
- Five family encoders: 75 tensors, 331,992 bytes.
- Conservation: offered 75 = loaded 75 + rejected 0.
- Every family preserves bit parity after strict loading.
- State branch and cross-family fusion come from the strong config and remain
  intentionally untransferred/random-initialized.
- Real GymFxEnv observation produced finite `(3, 96)` output; repeated forward
  was bit-identical.
- Runtime 1.357 seconds on CPU; measured host peak 802.4 MB.
- Authorization is consumed; model execution cannot be repeated under this key.

## New Finding

### DATA-SOTA-363 (S3): ledger record mode drifts from 0600 to 0664

The live completed record is mode `0664`, despite the custody contract claiming
records remain `0600`. Initial O_EXCL creation uses 0600, but later atomic
transition writes create a temporary file under the process umask and rename it
over the record, changing permissions.

This does not invalidate the smoke: the ledger contains no broker secret and
its digest/authentication facts remain intact. It blocks claiming private-ledger
permissions and must be corrected with DATA-SOTA-362 before authority beyond
mechanics. Every temporary and final custody file must be explicitly fchmod
0600; ledger directory remains 0700; tests must inspect actual modes after every
transition and completion.

## Next Gate

Proceed with CPU implementation of the remaining pretraining objectives and the
paired comparison harness. No GPU economic run until the companion order is
implemented and independently audited.
