# 391: paired, bounded SAC pilot

Use 390's qualified four-year paths 7 and 19 (eight synthetic market-years
available, separate accounts). This authorization covers a short pilot only,
not a new general release of the banks. Four real training epochs, seed 7,
original selection and fixed scaler, real online account, no new agent state,
no validation-generated data, TEST closed. Each of the two feature widths
has its own real-only control. Mixed arms replace 10% of replay samples
cumulatively at the same batch size and cadence, not 10% extra updates.
Synthetic transitions come from the same width's frozen stochastic policy.

71 starts at the intact 299 checkpoint. 61 physically removes ten input
channels and six whole branches. Since the encoder through fusion is frozen,
preserve each removed group's zero-standardized-input response as a single
fixed channel-by-time offset before the fusion activation. Derive offsets
separately for the actor, online critics and target critics. There are no
invented market observations or additional policy inputs. Keep parameters
and optimizer moments for the trainable attention/core/head. Test masked
parent vs compact predictions (all five networks) and action agreement;
save and reload the model before banking or training.

Admission: host RAM headroom after combined caps at least 2 GiB; workers
limited to 4 or 6 GiB; require free device memory; use UUID selection.
Compare the real epoch-zero accounts against the CPU preparation on each
GPU and reject any trade-count discrepancy. The initial 61 observations are
not an activity-eligible champion (52 validation trades); record the starting
point without selecting it. Seed and evaluation dates are shared within pairs.
Other hardware can introduce numerical differences after the first update.
Preserve all four epoch curves, actual sample and update counts, native
trading metrics and checkpoint eligibility. No promotion on this pilot.

Automation: `tools/pilot391.py prepare` per width, then bounded allocation
and history collection with `tools/fleet391.py`. The fleet comparison is
written only after both contracts in each pair agree and epochs 1..4 exist.
