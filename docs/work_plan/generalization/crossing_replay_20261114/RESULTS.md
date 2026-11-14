# First Harmful Crossing Results

Authenticated replay, not new account trading performance. All rows are FIT.
Each row is its first harmful transition in a case; rows can recur across cases.

|Case|Step|Row|Before -> proposed -> accepted|In batch|Pair margin before / proposed / accepted|
|---|---:|---:|---|---:|---:|
|A_19|10|316|BUY -> NOP -> NOP|0|+0.00003809 / -0.00001716 / -0.00000486|
|A_19|32|317|SELL -> SELL -> BUY|0|+0.00002798 / +0.00032972 / -0.00008792|
|A_19|51|245|BUY -> BUY -> NOP|1|+0.00010685 / +0.00023338 / -0.00018224|
|A_43|238|257|BUY -> SELL -> SELL|1|+0.00087258 / -0.00015508 / -0.00006534|
|B_19|5|257|BUY -> BUY -> NOP|0|+0.00029790 / +0.00021037 / -0.00007286|
|B_19|33|505|BUY -> NOP -> NOP|1|+0.00008000 / -0.00008107 / -0.00004611|
|B_43|67|457|SELL -> NOP -> NOP|0|+0.00002140 / -0.00015941 / -0.00002817|
|B_43|88|434|SELL -> BUY -> BUY|1|+0.00011086 / -0.00004470 / -0.00004684|
|B_43|144|497|SELL -> NOP -> NOP|1|+0.00004040 / -0.00005398 / -0.00008804|
|B_43|152|530|BUY -> BUY -> NOP|0|+0.00005904 / +0.00018891 / -0.00008205|

1024 transactions and 8 original endpoints exactly reproduced; 10 cold witness replays exact.
Independent Torch64 derivative maximum difference: 6.94e-16.

Finite margin contributions sum exactly across stages. Positive protects the old action against the accepted harmful action; negative weakens it.
The raw proposal is a counterfactual branch, not a valid replacement for the protected update: it may violate the original guard.

[Interpretation and next action](../../340_FIRST_HARMFUL_CROSSING_REPLAY.md).
