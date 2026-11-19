# 378: role routing before any further selector

Freeze before running: market-only shared encoder; flat [0,buy,sell], long
[0,close_long], short [0,close_short]. No reversal on a close. Zero is a
logit reference, not a Q/reward. Twin critics and continuation targets must
route by actual current and next account mode, including native stops.
No order state in the learned observation. Preserve the legacy comparator.

CPU diagnostic, not a training release:

1. Reproduce all 16 native316 forks including their original full suffix
   reward vectors, state and timestamp; require the original three conflicts.
2. From deterministic source trajectory in TRAIN, choose up to64 flat
   origins per calendar year by seeded sampling before scoring (seed378).
   Exclude original witness timestamps and their neighboring decisions.
   Do not replace refused or stopped positions. Years2020-22 are development;
   2023 is a reserved diagnostic panel, not independent of source training.
3. At t+2 compare flat, and long/short opened at t or t+1, using real
   executable positions only. Require equal timestamp and market tensor.
   Score first decision then the same frozen greedy continuation for24
   valid decisions with ending marked equity, not oracle trading or full
   account lifetime profit. All native fees, protection and calendars apply.
4. Regret = best ending equity for this state minus ending equity of chosen
   signal. Minimum equal-state mean regret: before routing, and after routing
   separately by actual position. Within a same-side pair, material means
   minimum shared regret > mean native entry-fill charges of the pair (USD).
   Treat this as a one-fill cost proxy, not a universal economic threshold.
   Publish by side, year, distance to SL vs TP, USD and initial risk units.
5. Any material same-side witness on reserved panel falsifies a universally
   sufficient scalar closer. An empty panel is inconclusive. Report magnitude
   and support before recommending a richer interface, not just existence.
   Distance buckets use direction-adjusted mark-to-barrier distances.

Next gate: ideal controller scenarios including mirrors, no-edge abstention,
ties, early close, beneficial hold, gaps and weekly force-flat. Compare
account regret and pre-stop optimal exit separately. No GPU specialist fit
until the actual winning interface passes this gate. Then paired SAC and
feature sensitivity controls before floating; no parallel replay augmentation
in the first training contrast.

## Results and decision

All16 original forks reproduced (state digest and all outcome dictionaries);
same three conflicts. New panel:256 origins,64/year. Development shared vs
role-routed minimum mean regret:1.744 vs0.639 USD; reserved:1.238 vs0.766 USD.
These are lower bounds under equal reachable-state weights at the same origin,
not achieved returns or population estimates. Trades closed or refused before
comparison were not replaced. State pairs from the same tape are dependent.

| 2023 same-side barrier distances (relative to mark) | long material / pairs | short material / pairs |
|---|---:|---:|
| Both closer to SL |2/14|0/8|
| Both closer to TP |0/15|2/15|
| One closer to each |1/5|3/10|
| Total |3/34|5/33|

Maximum irreducible equal-pair mean regret:17.55 USD long,22.19 USD short.
Development material pairs:11/97 long and14/106 short. Routing removes some
but not all ambiguity. Do not train the scalar-only two-closer candidate as
if it had passed. External position-conditioned evaluation required.

Follow-up (exploratory on same panel): use actual native execution to compare
closing now with every delay until native protection, force-flat or24 decisions
(mark equity if still open at endpoint). Give this isolated-position oracle
perfect foresight, choose HOLD for ties, and score its first decision against
the frozen account continuation. Material misses:11/75 individual long states,
30/74 short; average account regret3.26 and16.00 USD respectively. These are
states, not pairs. This comparison changes both the continuation policy and
re-entry rights. In a separate matched control keep the same source policy but
disallow re-entry once flat:2/75 and3/74 material misses against full account.
Do not attribute all41 misses to opportunity cost. A per-order ideal exit curve
alone is not a sufficient account-value target even with ideal information.

38 focused tests passed,1 skipped: new role router plus existing mirrored
position-aware close/hold, accounting metrics and identifiability regressions.
No full learned-controller or GPU acceptance implied. No regime feedback, no
entry changes, no production or live edits.

Next: retain the tested router. For each direction evaluate a market-produced
surface of **account continuation advantage** at the controller's actual
barrier distances and remaining executable time. Targets must compare CLOSE
and HOLD under the same continuation with re-entry, costs and native deadlines.
Isolated best-before-stop remains a separate diagnostic, not that target.
Require a new outcome-blind panel, mirrored/time/gap tests and paired account
replay before admitting an expensive learned-head comparison. Do not use
the2023 counts above to tune a grid or thresholds and then claim confirmation.

Private evidence: role-conflict378-panel. DESIGN sha256
1ea00fbaf2cefc685bfbcf2969b32afac73235dc8da87a539f3df3bebf670044;
PANEL bd23eb2079f083a030d3e0441ee66c62a3e522778299fe5aeed150ea5d93adc6;
RESULT 6f7c76c5bb0903b19bbbc7d1ac1a4af724295c3a65c48b0355c5714f4eb9ecd2;
EXIT_GATE 790587dbb51924e55636f88d88af3475ebae8e7b9269bdedf0a8792d067405f4.
DESIGN references the pre-results version of this document.
