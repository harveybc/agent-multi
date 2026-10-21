# Fixed-Stride Window Study

Owner-approved successor to the closed stride contrast. Keep stride 2 and run
7 then 25 observations against the authenticated 13-observation control.
Observed-bar spans are 12, 24 and 48 intervals; elapsed hours may include market
closures. Current completed bar is included; no padding or synthetic weekends.
Inputs, normalizer, business envelope, execution costs, optimizer and fixed
100000-transition budget remain unchanged. Four diagnostic checkpoint pairs;
no early stopping or best-checkpoint restore. TRAIN2020-2023, VALIDATION2024,
TEST closed. Native SAC and DOIN internals are not changed.

Branch2/core4/kernel3/depth1/head8 stay fixed. Flattened head input changes with
window length: parameter counts and synthetic exact-reload witnesses must be
recorded at each actual shape. This is not a parameter-count-matched experiment.
The native environment warms up 1+(window-1)*stride observed bars:13/25/49.
The normalizer uses up to256 past rows but does not impose a256-row reset.
Retain that native behavior; report the initial-support difference explicitly.
Do not claim exactly identical evaluation origins or insert missing history.
Paired comparisons at matching spans use the already closed 13x1 and13x4
contrasts; do not call same-window and same-span comparisons the same thing.

Finite sequence: two seed7 fits, then baseline/finalist on shared seeds17/29
only if validation daily growth improves without worsening gap or validation
drawdown and with activity. Six logical trials maximum; two fresh-root attempts
per trial maximum. A failed or negative trial never becomes an unreported zero.
One heavy process on the preferred GPU, automatic publication/readback and
independent closed collection. Check the closed predecessor's spec, state,
summary and native trial closures before admitting work. Existing closed runs
are never rewritten or retrained merely because publication is delayed.

## Test Design And Traceability

| Requirement | Probe |
| --- | --- |
| Only window changes | v11 strict allow-list and complete common-config comparison |
| Exactly7 then25, fixedstride2 | recipe and finite-queue tests, including conditional replication |
| Preserved origin and causality | real preprocessor probes at all window lengths, future/current perturbation |
| No hidden architecture adjustment | actual-shape native SAC gradient/reload and parameter counts |
| Explicit evaluation support | real environment reset/inference row count at each actual window |
| No overlapping predecessors | checksum-bound completed-sweep gate; corrupt close rejection |
| No human stage boundary | inherited worker/publication/collection finite timers and recovery tests |

CPU follow-up: inspect retained final traces for proposed/executed actions,
recorded commission/financing and gross/net PnL. Mark absent telemetry unknown;
summary inspection is not an independent replay of fills or proof of a cause.
Current stage: unit-test design, before governing implementation.
