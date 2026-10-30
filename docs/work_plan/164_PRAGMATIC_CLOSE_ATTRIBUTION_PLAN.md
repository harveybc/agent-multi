# Frozen pragmatic close attribution

2026-10-30. Owner requests continuation of163, without further babysitting.
This diagnostic precedes learned-adapter qualification and floating selection.

## Question and frozen intervention

For every actually executed AGENT_CLOSE of both frozen sequential finalists,
replay its account with exactly that one decision changed from CLOSE to HOLD.
At the next tick the identical controller resumes normally, including early
close. Preserve controller memory updates from the original decision. Do not
force the alternative to remain open until SL/TP and do not remove all closes.
The original close is evaluated against three distinct references:

1. Native passive exit of the same position (already measured in163).
2. Exit of that position after suppressing just this close under the policy.
3. Final whole-account outcome under that one intervention, including all
   subsequent orders, costs and balance-dependent position sizes.

Positive delta always means **the original CLOSE was better than the HOLD
intervention**. Decompose account delta into same-position and continuation
deltas. Report return and drawdown separately; more return with more drawdown
is a tradeoff, not dominance. Never add overlapping per-intervention gains
into an achievable new policy return.

## Scope and automation

Authenticate the completed163 manifest and all its artifact hashes. Select
only its frozen sequential representatives, not the best observed worlds.
Full scope:36 confirmation worlds xfive reporters xtwo policies =360 exact
baseline replays; every non-preempted close receives one intervention.
Ideal reports first, then noise0.5/1/2 and causal-prefix diagnostic, all using
unchanged native risk geometry and business restrictions. No financial data,
training, strategy retuning, or cross-lane changes.

Disjoint native fixture tests -> archive-subset wiring smoke (no tuning) ->
persistent CPU queue -> per-case receipts ->
per-world summaries -> complete report and audit addendum. At most four
single-threaded workers and6GiB total memory. CPU is appropriate for native
account replay; GPU allocation is unnecessary. Failed integrity checks stop
explicitly; completed units resume only when identities and seals match.

## Tests declared before implementation

| ID | Required evidence |
|---|---|
| CA01 | Observer with no intervention reproduces the normal runner exactly |
| CA02 | Long and short target identity includes tick, entry tick, direction; exactly one override |
| CA03 | Original controller is called and its memory advances; only returned action changes |
| CA04 | Before-target trace is identical; wrong/missing/duplicate target rejects |
| CA05 | Native protection can preempt a request; only actual AGENT_CLOSE is targeted |
| CA06 | Position + continuation = account delta; local and account signs can disagree |
| CA07 | Reuse caches or cold replay give identical account results after intervention |
| CA08 | Altered source, receipts or completion hash reject; resumed completed case does not rerun |
| CA09 | Full baseline equals archived163 metrics, trace and ledger before any intervention |
| CA10 | Output preserves every close and all negative results; no selection or summed-policy claim |

Implementation lives in a new diagnostic controller wrapper and runner; the
production controller and sealed163 runtime are unchanged. Counterfactual
future values are diagnostic outputs only and never controller observations.
