# Correct consumer activity recovery

This supplements the selected69 handoff dated2026-11-10. Use
[step276](../work_plan/276_STL_CONSUMER_RECOVERY.md), not the retired global
activity export in step275. The current consumer's six-estimator bundle was
recovered with exact probabilities on all31116 rows and exact reload. Its hash
is `fdc95a0cdabf9a5327c991187f62c8be928476f385551c7e8db3972045167d6b`.

For historical replay use the recorded fold selectors; after the fit window
the producer uses the full estimator (index0), never an average of the six.
The receipt's ordered14 lagged inputs remain mandatory. This does not claim
live freshness or authorize changing a broker runner.

Private artifact relative to the operator home:
`.local/share/agent-multi/stl-recovery-20261111/activity/consumer_recovered/CONSUMER_ACTIVITY.joblib`.
The adjacent `RESULT.json` and parent `CONSUMER_SPEC.json` pin the producers,
scope and actual consumer store. This recovers the historical producer; it
does not supply its14 real-time input feeds.

Separate candidate price files repair the November three-hour source defect,
with18 verified H1 observations across six pairs. They are not a released
consumer store. USDCHF May2022 source qualification and pre-TRAIN feature
history still need completion. No synthetic prices, source-gap interpolation,
zero filling, model promotion or service replacement was performed.
