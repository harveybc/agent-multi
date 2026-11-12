# Selected71 Boundary Account Replay

Date: 2026-11-12. Full-account check after304, before another refinement or
cumulative phase2 pass. Keep the selected71 source from299. No live model was
replaced and this is not a new feature-selection pass.

| Endpoint | TRAIN return | TRAIN orders | Validation return | Validation orders |
| --- | ---: | ---: | ---: | ---: |
| Preserved299 source |16.05822355% |745 |18.36994575% |207 |
| Fixed304 witness |15.99510755% |747 |18.17232350% |207 |

TRAIN Sharpe changes0.781969 ->0.777856; validation2.690712 ->2.659815.
TRAIN maximum drawdown falls9.358326% ->8.958845%; validation remains near
3.2573%. The lower TRAIN drawdown does not erase the lower return in both periods.

## Exact Attribution

The candidate was fixed from FIT evidence before either full account was read.
Seven TRAIN and two validation signal changes occur; three and one respectively
have nonzero chronological account effects. Reconciliation error is0USD.

| TRAIN decision UTC | Signal | Chronological contribution USD |
| --- | --- | ---: |
|2021-07-15 15:00 |NOP ->SELL |+51.154510 |
|2023-04-25 01:00 |NOP ->SELL |+0.191175 |
|2023-12-26 19:00 |NOP ->SELL |-57.657285 |

Total TRAIN delta:-6.311600USD. The effective validation change is an additional
SELL signal at2024-07-04 12:00, with account delta-19.762225USD. These are
chronological continuation contributions, not independent trade profits.

Protecting the113-row FIT panel does not protect the entire account: the head
shares parameters across observations. This measured failure is not evidence
that the inputs cannot support profitable decisions.

## Verification

- Authenticated304 probe, directions,303 floor,299 bundle and parent data.
- Evaluated only prequalified FIT candidates; no validation-driven search.
- Native chronological engine, full periods, same support, costs and rules.
- Diagnostic export and cold reload reproduce all valid logits and both
  account metrics exactly on the same CPU.
- Restored all source tensors and verified the original model file hash.
- No optimizer step, TEST read, broker operation or promotion.

Runtime27.8 seconds. Six accounts: source, diagnostic and cold diagnostic,
each on TRAIN and validation. `tools/replay_boundary_account.py` persists
progress and terminal hashes.

Identity:`f6c36932a83f350a80c0c48325d579b24b5bfa6a54e561b026fee035d7072663`.
Source weights:`69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536`.
Public evidence:[RESULT.json](generalization/selected71_account_20261112/RESULT.json).
Private replay root:`/var/tmp/sac71-boundary-account-20261112-r1`.

## Next Main-Lane Step

Qualify a finite TRAIN-only counterexample-guided account guard. After a bounded
proposal, inspect every changed TRAIN decision in the native account; add
harmful TRAIN counterexamples to the next constraint set, with rollback and a
fixed maximum iteration count. Validation remains reporting-only. Do not fit
its counterexample, repeat floating on unchanged weights, or promote panel-only
success. STL source preparation proceeds separately.
