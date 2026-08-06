# Audit: Corrections 135-142, RT0 v2 and RT1-A Materialization

Date: 2026-08-06 America/Bogota  
Auditor: General Musashi, independent verifier  
Technical-lead packet: `docs/handoffs/SATOSHI_III_135_142_ACCEPTANCE_PACKET_2026_08_06.md`  
Audited heads: `agent-multi@ae598521`, `lts@a7db6be`  
Runtime mutation: none; no broker call, venue restart, campaign pause or RT1 run

## 1. Verdict

Corrections 135, 137, 138, 139 and 142 are independently reproduced for the
defects they name and may move to `independently_verified_pending_owner_closure`.
Corrections 136, 140 and 141 are rejected. RT1-A remains
`MATERIALIZED_NOT_EXECUTED`; it is not authorized to run.

The decisive evidence is executable:

`docs/audits/evidence/SATOSHI_III_135_142_ACCEPTANCE_REPRO_2026_08_06.py`

The supplied `AFTER_2026_08_06.json` is not acceptance evidence. Its harness
maps every exception to `still_reproduced=false`; it therefore calls an API
rename (`_score_interval` to `score_interval`), a removed helper
(`_join_manifest`) and an invalid fake ZIP load error successful corrections.
Postconditions, expected exception classes and current APIs must be tested
directly.

## 2. Findings First

### AUD-F1-20260806-143 - S3 - The after-probe converts harness failures into passes

`after_probe.py` catches any exception and marks the case corrected. Three of
eight advertised passes did not exercise the corrected contract at all. A
renamed symbol, removed symbol or malformed fixture is evidence that the probe
is stale, not that the defect is fixed.

Evidence: `after_probe.py:48-70`, supplied `AFTER_2026_08_06.json`; independent
comparison against current public functions.

### AUD-F1-20260806-144 - S2 - Promotion still accepts nonexistent and unloadable models

The runner's validator checks bytes and a self-asserted `load_proven=true`, but
does not load the artifact during validation and does not bind the terminal
evaluation path/hash to the terminal artifact. The aggregator does not call
that validator at all. Four packets whose terminal paths do not exist are
accepted with `promotion_eligible=true`; arbitrary non-ZIP bytes with matching
hashes and `load_proven=true` pass the runner validator with zero problems.

This keeps finding 136 open. A local sibling directory named `replica` is also
not the second-host copy required by document 33.

Evidence: independent probes `136_artifact_load_and_reference` and
`136_aggregator_filesystem`; `tools/aggregate_curriculum_decision.py:78-137`;
the existing `test_aggregator_complete_packet_is_promotion_eligible` fixture
itself uses nonexistent model paths.

### AUD-F1-20260806-145 - S2 - RT warm-up still trades, scores h+1 facts and discards open exposure

The v2 runner removes the first 256 equity samples arithmetically, but the model
actively trades during those 256 bars. In the independent three-bar probe, the
warm-up changed equity from 10,000 to 9,961.67, the scorer still used 10,000 as
its baseline, counted four scored samples, and ended with a long position. The
next origin receives only `initial_cash`; the open position, protection state
and handover cost disappear.

Therefore warm-up is not economically excluded, interval activity is not an
interval delta, and the claimed account/effect continuity is absent. This keeps
finding 140 open.

Evidence: independent `140_warmup_and_state` probe against the real ETH CSV and
`gym-fx`; `rolling_origin_adaptation.py:198-220`, `:403-437`, `:457-464`.

### AUD-F1-20260806-146 - S2 - The SQLite/pointer crash window loses model and account state

The OLAP row is committed before `current_state.json` is replaced. A crash in
that exact interval leaves a durable row and the old pointer. On restart the
code explicitly accepts an empty pointer, skips the committed origin and does
not restore its model path or carried equity. The following origin can retrain
from the wrong model/account state. SQLite plus `os.replace` is not one atomic
transaction.

This keeps finding 141 open.

Evidence: independent `141_commit_pointer_crash`; runner lines 338-352 and
452-464.

### AUD-F1-20260806-147 - S2 - RT1 would optimize fresh SAC initialization, not champion adaptation

The RT runner has no starting champion/anchor artifact in its identity or CLI.
At origin zero it constructs `SAC("MlpPolicy", ...)` from scratch. The RT0
three-origin zero-trade result is consequently not an adaptation measurement of
the current/best ETH policy. Executing the 128 cells would answer a different
question and violate the roadmap order, which places RT0/RT1 after the SAC
topology/learning domain fixes the mature contract.

Evidence: independent `141_identity_and_deadline_guard`; runner lines 239-262
and 354-378; document 33 sections 3.6, R7 and Current Build Order.

### AUD-F1-20260806-148 - S3 - The deadline guard claims a handover condition it never measures

The summary text requires zero unreconciled handovers, but the OLAP schema has
no handover status/count and `satisfied` checks only latency, update count and
deadline misses. A cell can therefore be labeled satisfied without any
reconciliation evidence.

Evidence: independent `141_identity_and_deadline_guard`; runner lines 496-508.

### AUD-F1-20260806-149 - S3 - Execution identity ignores dirty source state

`run_identity()` stores only Git HEAD revisions. An uncommitted tracked change
can execute under the same run id. Decision-bearing runs must require clean
trees and bind a source-tree/content digest (or an explicit dirty diff digest
for diagnostic runs).

Evidence: runner lines 85-95 and 239-262; residual gap confessed in the packet.

### AUD-F1-20260806-150 - S3 - Rejoin proof has no freshness/deadline contract

Tip ancestry and component/domain equality are corrected, but a pending rejoin
has no timeout and `verify_rejoin()` does not require each worker observation to
be newer than `resume_accepted_at`. The correction order explicitly required
fresh evidence from every expected worker and a bounded timeout. A stale cache
must not be accepted, and unavailable evidence must eventually return the
supervisor to a stable paused state rather than remain pending forever.

Evidence: `campaign_supervisor.py:3535-3725`; absence of a rejoin deadline or
per-worker post-resume observation timestamp.

## 3. Accepted Corrections

| Finding | Appended disposition | Independent basis |
| --- | --- | --- |
| 135 | independently verified for component/domain drift and tip ancestry; 150 tracks remaining freshness/timeout | current API proves descendant and rejects a foreign block at the bound index |
| 137 | independently verified pending owner closure | duplicate physical packets, malformed identity and per-arm code drift are rejected |
| 138 | independently verified pending owner closure | typed schema required, forbidden membership checked, repair applied with provenance |
| 139 | independently verified pending owner closure | inactive/stale/missing/mismatched seats cannot become authoritative; exact complete join can |
| 142 | independently verified pending owner closure | dormant year fields absent from both executable RT and N/EN/E configs; explicit dates remain |

Correction 136 remains open through 144. Correction 140 remains open through
145. Correction 141 remains open through 146/147/148/149.

## 4. RT0 and RT1 Disposition

RT0 v2 proves only that three short model-save/load cycles completed inside the
12-hour wall-clock budget. It does not prove correct interval accounting,
handover continuity, restart safety, adaptation utility or an eligible cadence.
Its deadline guard correctly remains false for fewer than 20 observations, but
its handover clause is unevidenced.

The RT1-A grid has the requested 128 semantic cells: 64 adaptive and 64 paired
frozen controls over four cadences, two lookbacks, four blocks and two seeds.
Its status remains correct: `MATERIALIZED_NOT_EXECUTED`. It must not run until:

1. 144-150 pass independent verification;
2. the mature R3 ETH artifact/config/observation contract exists and is bound;
3. exact flat handover with explicit closing cost is implemented, or complete
   protected exposure is carried without disappearing; and
4. a corrected RT0 supplies at least 20 valid handovers and deadline facts.

## 5. Active Campaign Observation

Read-only fleet evidence at approximately 14:33 America/Bogota:

- one plan/domain/genesis/population/tip across omega, dragon and both gamma
  workers; all four claimed distinct candidates; no supervisor alert;
- GPU temperature/utilization: omega 48 C/29%, dragon 50 C/34%, gamma 5070 Ti
  48 C/40%, gamma 5090 56 C/37%; workers are active, not idle;
- only one of 20 generation-zero candidates is evaluated and no champion exists;
- active workers run `agent-multi@5437a31`, which predates the bounded-activity
  correction `5aca0450`;
- current candidates report 1,002-1,353 of 2,000 epochs, 8.0M-10.8M steps and
  zero train/train-tail/validation trades.

This is not a chain failure. It is a zero-activity compute sink under code that
cannot apply the later activity budget. The campaign was left untouched by this
audit. Continuing the current four candidates needs an explicit owner decision;
their current outputs cannot become eligible champions.

## 6. Verification

```text
independent acceptance reproducer: completed; network_used=false
agent-multi focused: 74 passed
lts controller inventory focused: 9 passed, 1 warning
agent-multi full: 609 passed, 2 warnings
lts full: 661 passed, 1 warning
```

Passing suites do not override the counterexamples above.

