# Ordered two-pass feature selection

Owner clarification on 2026-11-02 replaces 191's search traversal, not its
native evaluator, checkpoints, metrics, masks, data or business contract.
The earlier SBFS evaluated all deletions at each size and interleaved additions.
That explains the 1173 new evaluations before the service was stopped. Its
results are preserved; it is disabled and will not restart after reboot.

Use feature_selector `two_pass_greedy`, with the original manifest block order:

1. Visit each input block ONCE. Remove it provisionally, measure both frozen
   seeds on full TRAIN/INNER_A, and compare with the currently accepted subset.
   Keep the removal only if eligible and score improves by more than 1e-5.
   Otherwise restore it before the next trial.
2. Only after finishing the entire first pass, revisit the blocks actually
   removed, in removal order. Keep each addition only if it improves that
   current subset by the same rule. Stop after this second pass.

There are 120 columns in 111 atomic coding blocks. Therefore at most 223
evaluation requests including baseline, often fewer. No full-neighborhood
restarts, new training or smaller data/windows. Cached evaluations matching
this exact path can be reused. Differences in selector traversal are allowed;
differences in evaluator code, AST of native compute, data, checkpoint, device,
precision, support or costs invalidate reuse. Authenticate every reused row.

Acceptance tests: ordered call trace; compare against updated current state;
reject worsening/equal candidates; improvements allowed with negative scores;
no interleaved additions; atomic codings; minimum size; counts after each pass;
cached resume; reject cross-search reuse after any scientific-contract change.

Outputs: PASSES.json updates after every decision and at the first-pass boundary;
FEATURE_COUNTS.json and RESULT.json record initial, after-deletion and final
counts, kept/removed columns, per-seed metrics and the full trace.

Lineage precision: the currently pinned store inventory contains 316 feature
columns (317 including timestamp). That inventory is NOT proof that phase 1
evaluated all 316. The admitted phase-1 recipe has 120 columns after removal of
the constant px.er_1 from the prior 121-column recipe. Do not substitute an
inventory count or an older pool revision for the phase-1 evaluated count.
