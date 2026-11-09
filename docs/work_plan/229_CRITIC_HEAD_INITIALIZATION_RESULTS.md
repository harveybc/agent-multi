# Critic Head Initialization Results

Design: [228](228_CRITIC_HEAD_AND_SELECTION_UNIVERSE.md).
Completed2026-11-09 at10:37:46 UTC (05:37:46 America/Bogota).
Pinned runtime3153f902; physically verified RTX5090. Both sequential arms used
4096 updates,batch128,all24025 TRAIN transitions,identical labels/schedule,
empty AdamW memory and the same epoch40 actor,temperature and encoders.
Only online critic head initialization differed. No actor updates or promotion.

## Calibration

| Updates | Retained head RMSE | Fresh head RMSE |
|---:|---:|---:|
|0|25.525045|0.435829|
|256|23.884747|0.305694|
|1024|19.863303|0.293295|
|2048|12.913522|0.292381|
|4096|6.062444|0.290319|

Final bias:retained-4.157739,fresh-0.004713. The fresh head reduces RMSE95.21%
relative to the matched retained-head endpoint. The zero-Q reference is0.295424:
the fresh head improves only about1.73% over that simple reference. Much of the
retained/fresh difference already exists immediately after initialization.
Do not interpret95.21% as a comparable gain in prediction skill or profitability.

Alternate action draws19/43 on the same TRAIN market dates give fresh
RMSE0.274457/0.288893 versus retained6.062777/6.058929. These are not independent
market holdouts or independent initialization replications. One initialization
seed was tested. Metrics describe realized soft discounted returns, not dollars
of trading gain or validation performance.

| Signal | Rows | Retained RMSE | Fresh RMSE | Zero-Q RMSE |
|---|---:|---:|---:|---:|
|NOP|23024|5.573359|0.269138|0.273612|
|BUY|261|18.987084|0.628795|0.657777|
|SELL|740|9.981344|0.585866|0.592980|

NOP comprises95.83% of this degraded-policy replay. Signal counts are decisions,
not executed orders. BUY remains negatively biased(-0.211668) despite the overall
near-zero bias. All three actions improve over the zero-Q RMSE reference here,
but this does not establish correct counterfactual action rankings at every state.

## Interpretation and Next Gate

The same frozen representation permits much better realized-return fitting with
fresh online heads at the same budget. Retaining the degraded head weights is
therefore a material impediment in this diagnostic; resetting optimizer memory
alone was insufficient. This does not yet identify the historical cause of the
degradation or prove that resetting heads under ordinary bootstrapped SAC prevents
recurrence. These diagnostic labels deliberately bypass bootstrap recursion.

Next: qualify a bounded restoration on a COPY of the preserved policy, first
collecting returns from that policy rather than borrowing epoch40 labels. Compare
critic calibration and actor drift/activity after bounded updates, then native
TRAIN/VALIDATION accounts. Verify bootstrap behavior separately before resuming
long joint refinement. Do not substitute this diagnostic critic into a different
policy and claim it is already on-policy calibrated.

Selected69 and selected70 remain unchanged; no new trading result is claimed.
The cumulative phase2 feature readmission design in228 is unchanged and has not
yet executed. This queue is complete; no successor is claimed running.

## Verification

32 focused tests passed before launch. Both arms maintained exact actor and
frozen-parameter equality at recorded updates. Native source calibration passed
before intentional head initialization. Coordinator independently verified all12
terminal-listed files,both DONE receipts and the combined comparison digest.
Arm durations260.72s and266.23s; about9minutes including orchestration.

Queue identity:
`b101d65fa3e4ca65c9eb5136d9a0f6a4b5da56a6e9486eccc76fa60778bf20ab`.
COMPARISON SHA256:
`e917dc6475327a7c2ece0bb2c7ed9d749bfc791ff21215023f6d9ce54ea45b71`.
