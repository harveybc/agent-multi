# 403: historical opportunity census, not trained specialists

Executed tools/specialist_exposure403.py on three CPU hosts. The 32 receipts
in /var/tmp/specialist-exposure403 on the coordinator cover every combination
of 2016-23 and the native 3/6/12/24-hour risk profiles. All have the same
code, market contract, risk, data and source model identities. No actor was
loaded or used for decisions; no validation or optimization was run.

From an always-flat reference, at each valid legal tick fork isolated native
BUY and SELL accounts, then hold each filled position until native settlement.
Count observation ticks while the position is still open. Clocks restart
annually, and receipts count positions settled on the interval's last bar.
These are reachable position exposures, not executed policy trades, independent
markets, or comparisons of the value of exercising CLOSE. At the next open,
broker protections can still take precedence over a discretionary exit.

| TRAIN | Profile | Eligible flat ticks | BUY position-ticks | SELL position-ticks | BUY age >=24h | SELL age >=24h |
|---|---:|---:|---:|---:|---:|---:|
| 2020-23 | 3 | 19197 | 87108 | 87270 | 454 | 329 |
| 2020-23 | 6 | 19197 | 107612 | 107317 | 1405 | 1212 |
| 2020-23 | 12 | 19197 | 166296 | 174716 | 10049 | 12430 |
| 2020-23 | 24 | 19197 | 215184 | 220914 | 23773 | 25743 |
| 2016-23 | 3 | 38782 | 188756 | 188465 | 2079 | 2047 |
| 2016-23 | 6 | 38782 | 230831 | 232234 | 4525 | 5607 |
| 2016-23 | 12 | 38782 | 354807 | 376019 | 25972 | 33982 |
| 2016-23 | 24 | 38782 | 456929 | 478010 | 57821 | 68351 |

Every eligible fork filled on both sides on this flat baseline. Market
timestamps are common across profiles: do not add eligible counts over
profiles. Four years are nested in eight. Next: native paired alternative
actions and distinct reachable close outcomes by regime, using a common
continuation and reporting broker-priority/censored cases separately. This
census alone does not admit GPU fits or establish a wall time for labeling.
