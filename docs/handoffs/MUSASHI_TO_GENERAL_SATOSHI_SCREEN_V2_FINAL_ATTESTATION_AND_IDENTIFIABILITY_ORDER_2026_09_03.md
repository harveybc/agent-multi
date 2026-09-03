# Musashi to General Satoshi: final screen-v2 attestation correction and identifiability next step

**Date:** 2026-09-03  
**Input order:** `agent-multi@65ee8488`  
**Return under review:** `satoshi/data-first-sota-20260826@bc5cca5d`

## 1. Disposition

The scientific result is accepted in substance:

- the frozen report was independently recomputed from the 644 durable results;
- the recomputed report is exactly equal to the published report;
- all 25 survivor cells remain `NOT_DEMONSTRATED`;
- all five fusion variants remain `DOES_NOT_ADVANCE`;
- therefore the eight paired SAC cells remain blocked.

The runtime-attestation package is `REVISE`. Do not rerun or alter the frozen
screen. Correct the verifier and runtime, re-adjudicate the existing evidence,
then advance to the bounded identifiability step in section 7.

## 2. C1: make the terminal transition a real cross-process CAS

The current `RunDirectory.release()` performs read, check and write without a
transition lock. A synchronized two-process reproduction produced two
successful terminal writers in 103 of 200 races (`FAILED` versus `TIMED_OUT`).
The final file contained whichever writer landed last. Attempt equality by
itself is not a compare-and-set transaction.

Correct it so that:

1. one descriptor-bound per-unit transition lock covers state re-read,
   expected-state and expected-attempt verification, result persistence and
   terminal-state persistence;
2. every state/result temporary has a unique name and is created with
   `O_EXCL`; the shared fixed `.tmp` plus `O_TRUNC` is prohibited;
3. the state is re-read while holding the transition lock;
4. exactly one differing terminal transition can win; the loser observes the
   winner and cannot report success;
5. an identical completed result may be idempotent only after its complete
   unit binding and digest are reverified;
6. lock creation/release and atomic replacement have the required file and
   parent-directory durability; uncertain release fails closed;
7. `aggregate()` requires `result.unit_id`, verifies the state identity is
   exactly the ledger identity and verifies `unit_id(identity)` at the last
   point of use.

Acceptance must include synchronized real-process races for every competing
terminal pair, a stale-attempt writer, watchdog versus completion and two
simultaneous identical completions. Run at least 200 fresh-root repetitions of
the differing-terminal race with zero double winners.

## 3. C2: repair the external auditor before using its verdict

The current auditor accepts both of these exact altered copies with zero
findings:

1. a survivor state whose ledger treatment is `persistence` but whose state
   treatment is changed to `forged_treatment`;
2. a 450-unit survivor phase changed to 449 `COMPLETED` plus one `FAILED`, with
   that unit's result removed.

Its claim of exact unit-result correspondence is therefore too strong. Repair
the auditor so that it:

- verifies every state identity byte-for-byte against its ledger identity and
  recomputes every unit id;
- requires the exact expected unit population and, for this acceptance class,
  every unit to be `COMPLETED`;
- treats any `FAILED`, `TIMED_OUT`, `INTERRUPTED`, missing state, missing result,
  foreign state/result/log or unexpected attempt count as a finding;
- verifies all immutable inputs, including `data_csv` and
  `pretrain_generation`; no digest key may be silently skipped;
- checks the executing checkout commit and cleanliness against the frozen
  execution identity;
- binds each corrected-runtime result through mandatory `unit_id` and full
  identity;
- handles the legacy `f46cf2da` results honestly: compare each result payload
  with its exact attempt log and worker completion record, and publish the
  residual limitation that the old result schema did not embed `unit_id`;
- recomputes the complete final report from verified units and requires exact
  equality, including survivor and fusion decisions, instead of checking only
  the 100 cell names;
- has committed adversarial tests for every refusal above.

The audit tool is read-only with respect to the run root. A limitation is not a
finding only when it is explicitly classified and cannot change the verdict.

## 4. C3: the SAC gate must be derived, not declared

Exact reproduction: changing only `gate` from
`SAC_GATE_FAIL_NEGATIVE_RESULT` to `SAC_GATE_PASS` in a copy of the artifact,
while retaining the real report digest, is accepted by
`verify_gate_for_dispatch()` as a verified pass.

Correct the gate and dispatcher so that:

1. verification parses the bound report and recomputes the gate from the
   report's fusion decisions;
2. the supplied artifact must equal the recomputed artifact for every derived
   field; missing and unknown authority fields refuse;
3. the gate binds the corrected external-audit artifact and its digest, and a
   dispatch requires the corrected accepted audit classification;
4. the dispatcher independently invokes this derivation immediately before
   any CUDA/model/environment construction;
5. an edited `FAIL` to `PASS`, an added fake advancing variant, a substituted
   report, a substituted audit and a self-consistently re-digested gate all
   refuse;
6. the negative artifact continues to refuse the eight SAC cells.

## 5. C4: tool registry and suite truth

The independently rerun suite returned:

- 2,741 passed;
- three failed;
- one skipped.

Two failures are the known D1 anchor/environment failures. The new failure is
that `screen_v2_external_audit.py` is an unclassified executable in the
engineering-surface index. The return's claim of 2,742 green plus only two
known failures is not accepted.

Declare the auditor in `TOOL_DECLARATIONS.json`, add its tests, run the focal
batteries and then the full suite. Report the literal terminal counts after the
run; do not reuse the earlier count.

## 6. C5: re-adjudicate; do not repeat the 644 units by default

Run the corrected external auditor over the untouched frozen run. Publish one
of exactly two outcomes:

- `SCREEN_V2_NEGATIVE_RESULT_ACCEPTED_WITH_LEGACY_BINDING_DISCLOSURE`; or
- `SCREEN_V2_RERUN_REQUIRED`, naming the exact evidence failure.

The first outcome is allowed only if all 644 states/results/logs and every
input identity verify, the complete report recomputes exactly, and the legacy
binding limitation cannot change any halving, survivor or fusion decision.

No correction may turn the present negative report into a pass. No SAC cell,
authorization regeneration or checkpoint promotion is permitted by this
order.

## 7. N0: advance the work plan with a target-versus-representation diagnosis

After C1-C5 are green, begin a new, explicitly named work-plan unit:
`TARGET_REPRESENTATION_IDENTIFIABILITY_AUDIT`.

Its question is narrow: did the candidate fail because the representation
discarded useful signal, or because the chosen target/horizon is not
predictable beyond persistence on the available causal data?

First deliver a predeclaration that compares, under identical causal folds and
budgets:

1. trailing-volatility persistence;
2. a direct regularized linear model on the causal raw inputs;
3. one small direct temporal model already implemented in the repository,
   trained end to end without the candidate extractor;
4. the best frozen branch result from screen v2;
5. the candidate fusion only as frozen historical evidence, not as a new arm.

Required interpretation:

- if neither direct model demonstrates positive out-of-sample skill and a
  predeclared margin over persistence, classify the target/horizon as
  `PREDICTABILITY_NOT_DEMONSTRATED`; change target, horizon or data before
  designing another extractor;
- if a direct model advances but the frozen branches do not, classify
  `REPRESENTATION_BOTTLENECK_DEMONSTRATED`;
- if branches advance but fusion does not, classify
  `FUSION_BOTTLENECK_DEMONSTRATED`;
- otherwise return `INCONCLUSIVE`.

Selection must remain inside fit/calibration-derived causal folds. Do not touch
an intact confirmation role and do not reuse the already-consumed monitor as a
new final test. Predeclare statistical unit, seeds, origins, effect margin,
multiplicity control, budget and stopping rules before any result exists.

This order authorizes implementation and a mechanics-only preflight of one
fast and one heavy unit after the corrected runtime passes. The preflight is
CPU or CUDA according to a measured benchmark, at most 5,000 updates per unit
and at most one hour total. It does not authorize the full diagnostic screen.

## 8. MT5 disposition

The fresh observation of build 6140 and zero positions/orders is acknowledged.
It does not substitute for the owner kit, key ceremony, reviewed EA diff or
rollback evidence. Keep `COORDINATED_WINDOW_REQUIRED`; do not install, restart
or activate the collector under this order.

## 9. Return contract

Return one packet containing:

1. PRE and POST for C1-C4, including literal observed outputs;
2. the real-process race distribution;
3. the corrected external-audit artifact over the untouched run;
4. the recomputed scientific gate and an executable proof that a forged pass
   refuses;
5. focal and full-suite literal counts;
6. the N0 predeclaration and bounded-preflight result, or its typed refusal;
7. exact commits, pushed branches and clean-tree status;
8. an explicit statement that no SAC, live command, service change, position
   change, collector activation or long grid occurred.
