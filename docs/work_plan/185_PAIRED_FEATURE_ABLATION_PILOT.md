# Paired feature-ablation pilot

Owner accepted 2026-11-01. Before the full floating search, compare selection
by best TRAIN against validation-protected combined scoring on the actual
grouped double-TCN discrete SAC. The diagnostic planner belongs to
feature_selector (`fixed_ablation` installed plugin); native learning and
accounting remain here. See its `docs/PAIRED_ABLATION_PILOT.md` for the frozen
design and requirement-to-test mapping.

Use 524288 decisions per subset/seed, evaluations every 32768; A selects best
eligible mean TRAIN return/week within this budget, B uses the existing
combined score, validation guard and patience 5. B's decision freezes while A
continues. These are paired procedures over one fit, not independent trials.
Control reuses the authenticated identical-budget prefix of the stopping
contrast. Three deletions x seeds 7/19 require six fresh fits. No other
architecture, risk, cost, reward or input preprocessing change.

Inputs: full120/31groups; remove RAW_CLOSE_LOG_RETURN119/31; remove the atomic
hod_sin/hod_cos pair118/31; remove wavelet/logret_1 group111/30. Fixed parent
support is deliberate: pruning must not silently change trading dates.

No phase-2 selection is certified yet. Existing upstream feature construction
has internal-period exposure; all comparisons remain development. Seed43's
prior synthetic failure is retained, not replaced. 2024 and TEST stay closed.

Acceptance before launch: coherent subset tests; native structural projection
and parameter reduction; frozen stopping rules; exact prefix reuse; a short
CPU smoke including replay, selected-weight reload and native full accounts.
Each queue verifies receipts, waits for predecessor/resource admission,
retries a failed unit at most once from fresh state, and records terminal
failure. No unsupported claim of mid-fit restart. Collection automatically
reports full accounts and paired equal-duration monitor deltas and signal
disagreement. Failures do not become zeros or silently disappear.

Choose the next floating evaluator from reproducible sensitivity, activity,
and validation behavior, not from the smallest train/validation gap alone.
Two seeds do not justify a significance or generalization claim. A negative
baseline may still discriminate useful inputs; a high TRAIN score may instead
reward memorization. This pilot tests those alternatives rather than assuming
one is correct.
