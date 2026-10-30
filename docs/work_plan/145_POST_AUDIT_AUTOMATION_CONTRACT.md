# Post-audit automated sequence: executable contract

2026-10-29. Owner authorizes execution of144 as far as implementable without
intermediate approvals. Preserve v2 identities and files. This contract precedes
implementation and fixes requirements/tests for the next increment.

## Acceptance and test design

| ID | Requirement | Witness |
| --- | --- | --- |
| PA01 | Report/bid caches cannot survive execution-config change | Warm old costs, change spread/fees, compare new object and cached result; fixed old config reproduces archive |
| PA02 | Cached values bind report and position identity | Alter same-shaped paths, position id, risk or values; reject before decision; direct/native parity |
| PA03 | Forecast records are immutable and only mature observations score | Freeze submitted arrays; refuse duplicate issue, future issue and early observation; restart produces identical scores |
| PA04 | Calibrated intervals use only prior matured residuals | Separate calibration prefix; query before current observation; no infinite/empty calibrated bands; prefix perturbation invariance |
| PA05 | Every finalist gets fixed-position close counterfactuals | Full trace/native ledger replay; O* includes passive first passage and all executable later exits; classify both sides and passive SL/TP/flat |
| PA06 | Close value is also measured on full accounts | Same genes and world; suppress voluntary close only, not SL/TP/calendar; report changed entries and balance |
| PA07 | Queue is durable, bounded and identity-bound | Atomic sealed unit receipts; restart verifies files and skips completed units; any changed code/config/receipt rejects resume |

Implementation order: PA01/02 -> PA03/04 -> native tests and old-account parity
-> PA05/06 over all five frozen representatives and five report rungs -> sealed
aggregate. CPU tasks are sequential per unit with no financial training.
This closes audit-analysis omissions without choosing new genes from v2
confirmation. New causal reporter calibration/controls and R6 optimization
require their own implemented/qualified identity; this queue must not call
itself that optimization or silently start it after the diagnostic aggregate.

Counterfactual tolerance stays auditor-fixed0.02*sigma*units; sigma0.001 in v2.
No new parameter fitted here. Helpful/destructive labels compare executed exit
with passive hold, while regret uses the best executable exit O*. Full-account
no-close contrasts are separate from summed fixed-position counterfactuals.
