# 381: separate target misalignment from short closer errors

Development only. Frozen 379 heads, original 71 channels, same opener,
native costs, no learning, no 2023 replay or later evaluation. Reset the
account each January 2020, 2021 and 2022. Walk the table trajectory. Per
calendar quarter draw up to 12 reachable open-short decision ticks uniformly
(seed 381), and separately up to 12 ticks where the table wants CLOSE and
the scalar wants HOLD. Draw before counterfactuals, allow cross-panel
overlap, compute it once; never pool the two samples. These are tick samples
not uniformly sampled trades. The heads were fitted within these years.

Force CLOSE and HOLD once on identical snapshots. Measure the signed closing
advantage from ending marked equity (positive favors CLOSE):

* A: 24 valid decisions including the intervention; frozen source resumes.
* B: same 24 decisions, scalar resumes (same opener, re-entry allowed).
* C: same scalar, to end-of-year terminal settlement. Keep original-order
  net advantage separately; it is not the account target. Require terminal
  cash identity and exact identity of original position on both forks.

Freeze keys before scoring. Compare A to the prior 378 label implementation;
check B endpoints match across paired forks and C stops at native terminal.
Cash ties within 1e-8 USD have zero directional information. Report strict
sign flips and ties separately; tabulate regret for both heads in initial
risk units, and actual head mistakes already present versus A or introduced
by comparing against C. Report paired differences by quarter and cluster
bootstrap over quarters (descriptive: only 12 blocks). Neither target is
information supplied to a policy. No retraining, threshold or selector change.
