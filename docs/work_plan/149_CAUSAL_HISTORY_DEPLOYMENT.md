# Causal history CPU lanes: deployment and collection

Date: 2026-10-29. Implements [148](148_CAUSAL_HISTORY_AUTOMATION.md).

## Verified execution

- Zero-drift lane: code `382d25b2`, identity
  `71d597f34ee659e8bd6ee4ffbe59e34e7a8f3cac783664e79fa51021d556896c`.
  Completed at22:47 Bogota, exit0, 3min31s elapsed, peak memory1.3GiB.
- Median-drift lane: code `b37ef5cd`, identity
  `4acfcdae876bbba833f7af8ccbd748f14c29e53c9aece78f492e4248305eae77`.
  Calibration completed and confirmation running at this deployment snapshot.
- Both have200 runtime tests passed and zero skips. Services are enabled at
  boot with lingering,2 CPU workers each,4GiB per-lane memory ceiling, no GPU.
- The collector has a separate passing tamper-rejection test. It authenticates
  manifests, receipt chains, all output hashes and test receipts before merging
  results. It does not select a policy from confirmation or start another job.

The original median launch failed one test, not an account result: a cached
versus direct diagnostic rounding residual was4.44e-16 on one CPU versus
6.66e-16 on the original CPU. Commit b37ef5cd permits1e-12 absolute difference
ONLY in this diagnostic, bounds its absolute value by1e-10, and still requires
exact equality of every trade, account metric and decision. No strategy code
changed. The failed run was retained; the corrected launch uses a new identity
and output root. The zero-drift lane was not interrupted or rerun.

## Automatic branch taken

Both calibration gates returned false: none of the three fixed history arms
improved return over the no-history arm without increasing worst drawdown while
meeting activity. No full DEAP was launched. This is a completed decision of the
preregistered queue, not a wait for user permission. Both proceed to frozen
confirmation, per-close native counterfactuals, RESULTS.md and AUDIT_REQUEST.md.

The full smoke exercised the alternative search branch with58 candidates,
two generations, and authenticated completed resume. Smoke search is a wiring
test and has no scientific eligibility. A full eligible run would use20 members,
300 generations and seeds7/19/43, without intermediate approvals.

## Operations

Private service locations and remote names live only in operator configuration.
`examples/systemd/causal-history.service.template` defines the lane service.
`tools/collect_causal_history_lanes.py` polls both lanes from a private JSON
config, copies a remote lane only after completion, verifies all artifacts,
and writes a combined report automatically. A disconnected host is retried;
an integrity error is not treated as a successful empty result.

The final packets are diagnostic audits, not final strategy qualification.
No learned model, feature floating selection or financial TEST is run. The next
scientific decision must use these negative controls and the still-required
PC1--PC6 qualification, not reinterpret this penalty as a proven correction.
