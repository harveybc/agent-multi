# 382: reachable training positions helped account net in both target arms

[Protocol](382_SHORT_FACTORIAL_PROTOCOL.md). 397 examples per fit, equal
counts per quarter of 2020-21. Later-year panel: 247 short decisions on 122
positions in 34 represented weeks in 2022. Same panel for local regret;
separate full-year trajectories for account outcomes. No GPU, new SAC fit,
2023+ replay, threshold adjustment or feature selection.

| Short head / continuation label | Mean regret R (scalar target) | Buy / sell trades | Wins | Net USD | Return % | Max DD % |
|---|---:|---:|---:|---:|---:|---:|
| Constructed / source |0.205|127 / 172|167| -377.58|-3.78|6.10|
| Constructed / scalar |0.198|127 / 164|167| -61.71|-0.62|4.71|
| Reachable / source |0.190|126 / 172|167| +145.20|+1.45|5.40|
| Reachable / scalar |0.176|127 / 171|169| -20.02|-0.20|6.10|
| Scalar short control, early fit |0.223|127 / 173|163| -321.43|-3.21|5.92|
| Unchanged 379 table |0.175|125 / 164|167| +243.69|+2.44|3.47|

Positive numbers in the net column mean ending settled balance exceeds the
initial 10,000 USD after native costs. Replacing constructed states with
actual exposure improved net by 522.78 USD under source labels and 41.69 USD
under scalar labels. Changing the continuation alone was not a consistent
account improvement: +315.87 USD on constructed states, -165.22 USD on actual
exposure. The best local new head did not produce the best new account.

No arm cleared the frozen gate. For reachable/source the week-block interval
for regret difference vs constructed/source was [-0.0675,+0.0312] R; for
reachable/scalar it was [-0.0843,+0.0197] R. Both include zero. The incumbent
is a preservation control fitted partly on 2022, not an unbiased challenger.
The 2022 account is unseen by the four new short fits, conditional on common
inherited components; it is not an independent test of the entire system.

Recommendation: keep the current checkpoint, do not start floating or
choose a different threshold. First repeat only the constructed/source vs
reachable/source comparison on chronological folds with matched fit budgets
and all nuisance components demonstrably trained before their scoring dates
(including the opener/encoder). Check availability of those checkpoints
before committing compute. Treat this one-year population effect as a
hypothesis, not evidence to replace the incumbent.

Evidence: persisted fit support, examples and both targets; four reloaded
heads; per-tick paired decisions; reconciled account results; 15 focused tests
passed. PANEL sha256 20350d561ef04cb10ab65e12303fc31d444bfd680dfba3b3c678bda2a59c95c2;
RESULT sha256 6dca665167521f63f733d2181106b917b0e44a8cda1b943f40010fc5690cd0b0.
