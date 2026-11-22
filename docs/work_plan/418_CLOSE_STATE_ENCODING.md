# 418: categorical dispatch and 16-level position state

Research branch only. No changes to live runners or native account economics.
BUY=[1,0], SELL=[0,1] dispatch to independent close heads. FLAT has no
valid closer. The signed direction still used to compute executable prices
and distances is arithmetic, not a numeric learning input or target.

For each existing bounded state coordinate u, q=min(15,floor(16*u)).
The bins are uniform in normalized coordinates, not in hours or risk units.
Inputs below are before fixed train standardization; all codes are concatenated
with the same eight frozen market logits, not with a constant side feature.

|State coding|Per coordinate|Total input width|
|---|---:|---:|
|Continuous|1|11|
|Quantized scalar q/15|1|11|
|Reflected Gray q XOR (q >> 1), MSB first|4|20|
|One-hot q|16|56|
|Thresholds q >= k, k=1..15|15|53|

Paired fit 2019-22; managed replay 2023-H2, unchanged 417 tests and seeds.
Continuous direct readout and native account matched prior controls. Fit labels
occupied 12/12/13 of 16 bins for SL/TP/time respectively. Four bits give
16 representable levels; that does not imply equally frequent levels.

|Encoding|Same-entry mean net bp (7/19/43)|Sequential net % (7/19/43)|
|---|---|---|
|Native curve|-4.067 / -3.622 / -3.219|-20.83 / -23.12 / -16.27|
|Continuous direct|-4.215 / -4.131 / -3.200|-19.42 / -20.80 / -19.35|
|Quantized scalar|-4.066 / -3.745 / -2.852|-21.70 / -22.64 / -17.92|
|Gray|-4.163 / -4.030 / -3.222|-22.38 / -21.35 / -18.18|
|One-hot|-4.271 / -3.630 / -3.115|-22.27 / -20.35 / -15.49|
|Thresholds|-4.402 / -3.684 / -3.188|-21.40 / -21.35 / -16.73|

Gray improved the conflicted-tick regret vs the native curve in all three
seeds, but none of its same-entry economic comparisons improved. No new
encoding cleared all existing gates. No evidence here for changing the
business rule, BUY/SELL outputs, or production weights.
