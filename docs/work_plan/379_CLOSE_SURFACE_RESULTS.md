# 379: closer surface does not pass the account gate

[Frozen protocol](379_CLOSE_SURFACE_GATE.md). Two complete CPU runs produced
byte-identical design, fitted heads, row ledgers and results. RESULT SHA256:
82b0113230186d61356d0dd9ecf80e3f64510c22988a036e39aa0b675926a68d.
Fit: 1199 reachable position states. Reserved 2023 (inside TRAIN): 432 states.
This is a diagnostic ridge head, not a new SAC or a validation return.

| Population | States | Scalar regret R | Table regret R |
|---|---:|---:|---:|
| Long |221|0.3913|0.3725|
| Short |211|0.4127|0.3593|
| Material same-market conflicts |26|0.2717|0.2128|
| All |432|0.4018|0.3661|

R is initial stop risk. Mean USD regret fell from 18.15 to 16.50.
Table minus scalar was -0.0357 R, paired 95% weekly bootstrap interval
[-0.0872, +0.0117], 39 represented weeks. Harmful closes fell from
131/247 to 114/227. By direction the table did not beat the best of the
two constant controls: always HOLD for longs (0.3680 R) and always CLOSE
for shorts (0.3582 R). These controls are descriptive, not a selected rule.
The lookup saturated the time axis for 22% of long and 23% of short states.

| Full 2023 account | Trades | Buys | Wins | Net USD | Return | Maximum drawdown |
|---|---:|---:|---:|---:|---:|---:|
| Frozen source |204|34|123|308.64|3.09%|4.02%|
| Scalar heads |277|40|157|600.84|6.01%|3.21%|
| Lookup heads |280|42|158|378.99|3.79%|3.36%|

Decision: do not promote or start another floating round. Local targets use
the source policy after the first action; full replay repeatedly uses the
candidate, changing subsequent exposures and opportunities. This difference
in estimands is known; it is not proof it caused the entire performance gap.
Next diagnostic: replay the first differing close from the same account
snapshot and telescope the return difference using a fixed scalar continuation.
Separate cash saved on the current position from the result of subsequent
re-entries; check that the components reconcile to the full -221.85 USD
difference. Do not tune another grid on the reserved panel.

Reproduction: run generalization/role_surface379.py on CPU with the same
old, parent, bundle and cache inputs as 378, plus --previous pointing to
its panel directory and --out pointing to a new evidence directory.
41 focused tests passed. The 72 legacy synthetic quote integration cases
were skipped without their explicit separate engine/recipe environment;
they are not counted as verified for this run.
