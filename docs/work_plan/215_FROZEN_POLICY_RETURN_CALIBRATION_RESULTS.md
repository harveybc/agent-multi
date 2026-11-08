# Frozen Policy Return Calibration Results

Completed 2026-11-08 at 15:55:27 UTC. Design: [214](214_FROZEN_POLICY_RETURN_CALIBRATION.md).
This closes the accepted diagnostic, not a new training or feature-selection run.

## Execution and Verification

- Runtime: `5a429ce1`; identity: `ffa695f1f6032f288ab5c14b46d6d5de26100c18b1af3f02fbc3190a6d2c8aee`.
- RTX5090 executed 24 native accounts in 20.24 seconds after loading the frozen
  checkpoints. Batched market-only inference explains this speed; this is not a
  measurement of sequential training throughput.
- Checkpoints 0, 6 and 40; greedy and categorical sampling seeds 7/19/43;
  full TRAIN 2020-2023 and VALIDATION 2024. No TEST access or weight updates.
- All six greedy accounts reproduced their original metrics. Model tensors
  remained unchanged. Support, costs, strategy and 70 inputs / 28 groups stayed fixed.
- 14 focused tests passed. Independent coordinator verification authenticated all
  153 terminal-manifest files. Explicit discounted sums at 408 trace positions
  reproduced recursive soft returns with maximum error `1.9984014443252818e-15`.
- `ACCOUNTS.json` SHA256: `42819f810b3ffa0ff6b02f8f558697079c8a3b3385fe0db6c46195f4d14f369d`.
- `REPORT.md` SHA256: `b93696528ba2eaf6eaf31f01095e80f0e2e68be0e11f8e16438047f0064f6954`.

## Account Results

Sampling rows are arithmetic means of three action-sampling seeds, not three
independent model fits. Orders in these rows are mean counts, not pooled trades.
DD is the mean of each account's maximum drawdown. Returns include native costs.

| Epoch | Policy | TRAIN return % | VALIDATION return % | VAL Sharpe | VAL orders | VAL wins % | VAL DD % |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | Greedy | 11.51 | 13.81 | 2.10 | 206 | 57.77 | 3.62 |
| 0 | Sampling | -42.34 | -14.04 | -1.76 | 609.00 | 46.17 | 17.54 |
| 6 | Greedy | 5.23 | -3.32 | -1.16 | 35 | 40.00 | 3.85 |
| 6 | Sampling | -20.66 | -10.05 | -1.99 | 172.33 | 42.38 | 12.20 |
| 40 | Greedy | -10.82 | 4.79 | 2.08 | 34 | 61.76 | 1.35 |
| 40 | Sampling | -10.94 | 3.72 | 1.60 | 32.67 | 61.24 | 1.45 |

Initial stochastic VALIDATION returns by seed: -11.6766%, -10.4968%, -19.9598%.
The initial greedy checkpoint remains the selected model: 206 orders, 3.81/week.
Epoch40 has only 34 orders, 0.63/week, and fails the declared activity requirement.
The head-only training ended after 40 epochs / 961000 steps at patience 20/20,
restoring epoch0. It did not produce an eligible improvement.

## Critic Calibration

Compare the minimum online twin Q for the sampled action to its realized soft
discounted return, including future action entropy and native transition durations.
Bias is Q minus return. Units are discounted reward percentage points relative to
initial capital, NOT annual account return percentages or dollars.

| Epoch | TRAIN mean bias | VALIDATION mean bias |
|---|---:|---:|
| 0 | 1.1136 | 1.0823 |
| 6 | -1.1515 | -1.0988 |
| 40 | -25.0222 | -26.3082 |

Greedy ordinary returns are reported separately as policy-mismatch diagnostics;
they are not on-policy calibration targets for a stochastic SAC critic.
Temporal dependence and only three sampling trajectories preclude an IID
significance claim. These measurements establish a substantial aggregate value
calibration problem, not its exact algorithmic cause.

## Interpretation and Next Controlled Work

1. Structural weight preservation succeeded: the profitable initial greedy
   account is reproducible. The later deterioration is not loss of those stored weights.
2. Training samples actions while deployment evaluation takes their argmax.
   The same initial probability distribution performs very differently under
   these two policies. Sampling also produces more trades and higher costs;
   costs alone have not been isolated as the explanation for all losses.
3. Critic values drift far below observed soft returns during refinement. Lower
   TD loss in the preceding short diagnostic did not certify useful calibration.
4. Preserve epoch0 and inspect critic targets, terminal/duration handling,
   online/target network modes and optimizer updates with actor fixed. Verify
   numerical targets before trying an altered learning rule. Quantify the
   stochastic/greedy gap explicitly in subsequent comparisons.

No new architecture, reward, input, business rule or model promotion follows
automatically from this diagnostic. The independent replication is a separate run.
