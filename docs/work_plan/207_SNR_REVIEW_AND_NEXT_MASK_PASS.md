# SNR Report Review and Next Ordered Mask Pass

Execution closed: [208](208_R1_FROZEN_76_TWO_PASS_RESULTS.md) reports the completed
93 ->76 ->76 pass,28 groups and104 evaluations with authenticated collection.

## Review Decision

Reviewed owner-supplied `snr_shannon_musashi_2026-11-06.md` (SHA256
`e824938ac5e8bbbe2f02eb45c0a70530241e9e7ff9bb6f71dcb63ae863f7a7b8`).
The source report is preserved unchanged. Its descriptive tables may motivate
further measurement; its70x capacity deduction does not apply to the current
model and is not a sufficient reason to launch the prepared capacity arm.

### Findings

1. **Wrong model identity.** The report counts a historical expert-guided model
   with15 inputs,9 feedback variables, widths2/2/16 and no attention. The current
   SAC is market-only with93 inputs and30 groups, residual causal TCNs, a causal
   patch-attention adapter with positional encoding and categorical NOP/BUY/SELL.
   The reference has32,781 actor parameters and98,344 total trainable parameters;
   G2 has41,065 and123,196 respectively. The report's1,440 does not describe either.
   Adding actor and critics is an accounting convention, not proof they form one
   shared memory available to the trading policy.
2. **Mean/std is not identified predictive SNR.** The level ratio mostly reflects
   price offset versus dispersion. For returns it measures unconditional drift
   relative to variability, not the predictable component conditional on the
   observations. A near-zero drift does not imply no learnable temporal structure.
3. **Shannon conversion is not justified here.** Channel capacity requires a
   specified channel, signal/noise powers and bandwidth assumptions. Substituting
   squared mean/std of a market series does not establish those quantities or
   measure its useful predictive information. Thus neither206,492 nor0.0073 is
   an established count of noise-free information in our TRAIN. See
   [Shannon's original channel analysis](https://fab.cba.mit.edu/classes/S62.12/docs/Shannon_noise.pdf).
4. **Two bits per connection is not a universal deep-network sizing law.**
   Separating dichotomies under the assumptions of a threshold-function model
   is different from learning a temporal control policy with shared convolution
   weights, attention and critics. Raw-data storage and policy generalization
   are different problems. See [Cover's original result](https://isl.stanford.edu/~cover/papers/paper2.pdf).
5. **Scope and sampling need care.** ZigZag hindsight partitions are acceptable
   for retrospective diagnostics, not causal availability claims. Observed H1
   rows span closures, so elapsed time is not simply row count times one hour.
   The report's stated exclusion of2-bar segments can be a design choice, but
   sample variance with ddof1 already exists for2 observations.

### Executed Counterexample

For256 alternating returns `[-0.01,+0.01,...]`, the proposed
`N*log2(1+(mean/std)^2)` is exactly0. Yet predicting the next return as the
negative of the previous return is correct on100% of the255 transitions.
Adding1 to every value changes the proxy to3400.2459249678027 bits while a
one-to-one translation preserves the deterministic temporal relationship.
This is an algebraic counterexample to that interpretation, not a market result.

Reproduction (CPU, no market data):

```python
import numpy as np
r = np.tile(np.array([-0.01, 0.01]), 128)
proxy = lambda x: len(x) * np.log2(1 + (x.mean()/x.std(ddof=1))**2)
print(proxy(r), np.mean(-r[:-1] == r[1:]), proxy(r + 1))
# 0.0 1.0 3400.2459249678027
```

No synthetic generator was fitted, no input was filtered, and the prepared
7,083,946-parameter arm was not launched. Larger models remain an empirical
option, not ruled out by this review; the70x claim is simply unsupported.

## Continue Existing Work

G2 remains an independent full-data multiscale training contrast on the
larger-RAM worker. The memory-protected primary worker is used for inference
selection, without replay allocation or another training process.

The next ordered removal/reinsertion pass uses the completed R1 seed7 checkpoint
from epoch7, not its overfit final epoch40 and not an unfinished G2 checkpoint.
R1 has critic-core dropout0.1; the actor's inference does not add dropout.

| Parent evidence | Value |
|---|---|
| Parent identity | `be89e2e5b123183dbb751e805ac5b497b15d705b1eaff265b39789beae2e44bf` |
| Checkpoint SHA256 | `bac17a9c517393e171117fffc08f52db38ca6727c4a62b81a9f282ff790ac3af` |
| Pass identity | `9b831ec28a336aa2585bbd5c36a9bc921bec2621c3ad3b61ca5eb21f7c98ffe3` |
| TRAIN return | +39.0376% |
| Validation return | -1.1815% |
| Validation orders / win rate | 266 /50.7519% |
| Validation Sharpe / maxDD | -0.1157 /7.2645% |

These are existing checkpoint metrics, not results of the new selection.

### Frozen Selection Rules

- Start with93 features,30 groups; preserve full TRAIN2020-2023, VAL2024,
  fixed TRAIN scaling,13 observations/stride2, support and business rules.
- Authenticate weights and reproduce the probe and unmasked native accounts
  exactly before any ablation. Repeat the unmasked check after the pass.
- Use the existing `feature_selector`0.2.4 `two_pass_greedy` plugin with its
  package digest recorded. The base worker environment had0.2.2; use the already
  pinned0.2.4 overlay, without altering another worker's environment.
- One ordered deletion pass: compare each candidate to the currently accepted
  subset; accept if validation net return/week improves or ties exactly, provided
  activity remains strictly above2orders/week in BOTH periods.
- Then one reinsertion pass over removed blocks only; accept only strict
  improvement. Keep existing coherent blocks (for example sine/cosine) atomic.
  This is the user's ordered two-pass variant, not exhaustive floating search.
- TRAIN is reported but cannot compensate validation deterioration. No epsilon
  degradation is accepted. Original code uses tolerance0 and remove_on_tie=true.
- Frozen-weight ablation is not retraining or a causal proof. No architectural
  compaction or automatic structural promotion occurs in this pass.
- Persist every candidate result, trace, initial/post-deletion/final counts and
  native final statistics. Cache and systemd restart allow continuation without
  a chat turn. Verify collected result digest against the close receipt.

G3 synthetic work remains separate: the93-column joint producer is not yet
implemented. Do not use the failed information estimate to choose a noise level,
filter columns, or splice synthetic EURUSD beside unrelated real external data.
