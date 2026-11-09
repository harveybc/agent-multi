# Anchored Action Attribution Results

## Verified Execution

Completed 2026-11-09T19:07:07.956701Z (14:07 Bogota), 62.16 seconds.
Runtime 8f26cbfd; 29 focused tests passed. All 11 terminal artifacts authenticated
after automatic collection. Identity:
`3482b7b783b891d552bdbd6438a8574c11be850ae7ba54e9971b9b825a18b523`.
Both KL256 actors reproduce every archived decision. Online critics remain
byte-identical to their corrected calibration checkpoints. Source69/70 and
candidate checkpoint hashes unchanged; zero optimizer steps or policy changes.

Private full evidence: `/var/tmp/sac70-anchored-attribution-20261109`.
Compact public evidence:[RESULT](generalization/anchored_attribution_20261109/RESULT.json),
[all 156 events](generalization/anchored_attribution_20261109/EVENTS.csv).
The source terminal receipt is retained as SOURCE_DONE.json; the public subset
is not presented as a complete copy of the full private collection.

## Exact Account Attribution

Each comparison uses the unchanged 70-input parent, identical native execution and
the corresponding KL256 actor. Contributions are chronological full-suffix
effects, including later opportunities and sizing, not isolated trade profits.
The 624 forced-action suffix simulations plus both-account replays reconcile
with error 0.0 USD in all four comparisons.

| Replica | Period | Changed signals | Economically effective | Helpful | Harmful | Net change USD |
|---|---|---:|---:|---:|---:|---:|
|r19|TRAIN|35|14|7|7|+63.619385|
|r19|VALIDATION|11|3|2|1|+24.888990|
|r43|TRAIN|84|24|11|13|+76.056250|
|r43|VALIDATION|26|10|4|6|-96.904290|

## Mechanisms Distinguished

1. **Wrong critic preference in these cases.** Six of the seven harmful
   validation event occurrences have a positive minimum-twin Q advantage for
   the new action, but a negative realized discounted-reward advantage from
   an identical-state continuation under the parent policy. There are six
   unique harmful validation dates; the August13 event occurs in both replicas.
   TRAIN also has17 ordering conflicts among20 harmful occurrences. These
   are diagnostic event counts, not independent samples or a global error rate.
2. **Actor changes despite contrary critic preference.** On November6,r43
   changes NOP to BUY while the minimum-twin critic prefers NOP. The new minus
   old Q gap is-0.02144569; including the parent entropy term gives-0.02102427.
   Row soft utility decreases0.00068692; the chronological cash contribution
   is-49.527395 USD. A shared actor update does not enforce pointwise Q
   improvement on every row. This is observed inconsistency, not by itself
   proof of an optimizer implementation bug or its precise origin.
3. **Small average KL does not preserve discrete actions.** Some changes cross
   very small probability margins; others do not. The harmful December19
   opening crosses a parent probability margin of0.05278114 (5.278114pp), with
   row KL0.00591004 versus r43's full-validation mean0.00014054. Reporting the
   mean alone hides local movement. These probabilities are not calibrated
   estimates of trading success.
4. **Twin consensus is insufficient by itself.** Critics disagree in4/7 harmful
   validation occurrences, but both favor the new action in the remaining3.
   They also disagree in1/6 helpful occurrences. No consensus-based veto is
   adopted from this retrospective diagnostic.
5. **Cash versus discounting is not the observed validation explanation.**
   All13 economically effective validation changes have identical signs for
   isolated cash and discounted-reward differences. There is one such sign
   conflict among the38 effective TRAIN changes. Do not generalize beyond
   this panel or infer that a new fitness function is required by these results.

## Harmful Validation Changes

| Replica | UTC | Change | Chronological USD | Q gap(new-old) | Realized discounted gap |
|---|---|---|---:|---:|---:|
|r19|2024-08-13 13:00|NOP to SELL entry|-57.308575|+0.056838|-0.548556|
|r43|2024-02-16 09:00|HOLD to close short|-10.128905|+0.010300|-0.090120|
|r43|2024-03-21 23:00|SELL entry to NOP|-0.582745|+0.006114|-0.004598|
|r43|2024-06-25 11:00|SELL entry to NOP|-52.267030|+0.018017|-0.497802|
|r43|2024-08-13 13:00|NOP to SELL entry|-57.159345|+0.032366|-0.548556|
|r43|2024-11-06 05:00|NOP to BUY entry|-49.527395|-0.021446|-0.488831|
|r43|2024-12-19 21:00|NOP to BUY entry|-95.473185|+0.031025|-0.880702|

Q and discounted gaps are reward units, not dollars. The second row is one
additional harmful early-close command. The remaining six harmful occurrences
are opening decisions. No broad early-close improvement claim is made.

## Next Decision

Keep source policies and all diagnostic endpoints. The next bounded design
should test TRAIN-only critic action-advantage/ranking calibration against the
current absolute-return calibration, using held-out TRAIN diagnostics and
matched actor refinement. Measure both ranking and actual native accounts;
separately monitor actor changes contrary to critic preference and local KL.
Do not fit a margin threshold to the validation dates above, add observation
feedback, change the strategy, or start another feature-removal pass on the
assumption that this issue is solved. This report launches no new training.
