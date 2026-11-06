# Generalization: diagnose before expanding context or synthetic experience

Owner approval2026-11-06. This follows202 in phase2; it does not restart feature
selection, change the trading contract, or authorize an arbitrary feature set.
Keep93 selected inputs, train2020-2023, validation2024, fixed TRAIN scaler,
market-only observations, attention, categorical SAC and current order logic.

G1 executed2026-11-06: see [measured results](204_GENERALIZATION_DIAGNOSIS_RESULTS.md).
The sequential CPU collector completed all three runs. G2/G3 training has NOT
started. BN variants are not promoted: one has diverging critic values, the
other collapses to inactivity. These are optimization-health findings, not
evidence that the market inputs contain no useful information.

## Ordered execution

G1 CPU diagnosis uses authenticated completed R1/R3/R4 artifacts. Inspect all
epoch histories, selected versus final weights, full-account quarterly equity,
replay coverage and frozen TD/Q diagnostics; do not run additional training.
Reconstruct replay dates only if its ring cursor/write count, original decision
sequence and EVERY stored observation match exactly. Actual sampled-batch
history was not logged and cannot be reconstructed as measured evidence.
BN counterfactual train-mode forwards are diagnostics on restored copies only;
they must restore buffers and RNG and must never replace normal evaluation.

G2 multiscale contract: retain13 observations at stride2, add13 at stride6 with
a small branch. The current recipe uses observed admitted H1 rows, NOT fixed
clock-hour lags (closures consume no slot). Report actual elapsed-time spans.
No interpolation, new normalization or hidden removal of baseline decisions.
Audit context availability first; unavailable long context requires an explicit
branch-availability contract, not fabricated values. Budget, seeds, business,
monitor and all non-context settings remain matched. Reuse an existing control
only when its data support and training budget match; no control retraining
merely for convenience. This stage is conditional on G1 implementation health.

G3 synthetic contrast: initially10% synthetic transitions, remaining90% real,
same update budget; all real TRAIN remains available. This is a separate arm
from multiscale, not a combined treatment. Fit generator only within TRAIN;
jointly generate dependencies and recalculate derived features/rewards.
Do not couple synthetic EURUSD with unrelated historical cross-asset/COT/rates.
Bootstrap sampling of unmodified complete episodes is a separate resampling
control, not new synthetic information. Never join positions/windows across
artificial boundaries. Audit producer coverage for all93 columns before launch.
Existing single-asset OHLCV bootstrap is not automatically a joint93-input
generator. No fabricated or silently dropped feature may close this gap.

## Requirements and tests

| ID | Requirement | Acceptance |
|---|---|---|
| G01 | Read-only existing runs | hash identities before/after; write only diagnostic output |
| G02 | Correct replay timeline | synthetic ring wrap tests; exact real observation matching |
| G03 | Honest period metrics | continuous account equity incl. entering equity; no quarter resets; exit-attributed trades labelled |
| G04 | Frozen inference | no updates; BN buffers/RNG restored after counterfactual probes |
| G05 | No synthetic promotion by appearance | coverage, coherence, tails, dependencies, real-validation utility |
| G06 | Same baseline support | context coverage reported without deleting baseline rows |
| G07 | No false generalization | inactivity and degraded TRAIN are not improvements by gap alone |

Compare net return, Sharpe, drawdown, activity, direction balance, early-close
quality, learning curves and stability. Positive candidate differences require
additional paired seeds before combining treatments. The primary accelerator
is reserved for subsequent training; secondary worker prepares synthetic evidence;
coordinator CPU runs diagnostics. No autonomous unbounded search or TEST reads.
