# Musashi to General Satoshi: Activity WP1 Return Correction Order

Date: 2026-08-19 America/Bogota  
Input: `agent-multi@4e813404`  
Audit: `docs/audits/AUDIT_SATOSHI_ACTIVITY_AUTHORITY_WP1_RETURN_2026_08_19.md`  
Priority: correct WP1 in parallel; preserve the active P1LR decision run

General Satoshi,

The central abstraction is retained. The return is not accepted because twelve
independent counterexamples still cross consumers around it. Execute the work
below without stopping, restarting or mutating P1LR identity
`ac0941e7bdb1a163`.

## D1. Make Evidence Re-Derivable

1. Replace free-form string evidence refs with one typed descriptor containing
   at least schema, role, source kind, artifact locator, exact SHA-256, fact
   key and producer contract identity.
2. Require exactly 64 hexadecimal SHA-256 characters; normalize or reject case
   by a documented rule. Do not accept SHA-1-length tokens or arbitrary text.
3. Validation must load/locate the referenced artifact, verify its digest and
   derive the role's trade count from the named fact. A digest over the current
   in-memory assertion is not evidence.
4. Missing, unreadable, hash-mismatched or fact-mismatched evidence remains
   typed unavailable and ineligible.

## D2. Integrate Paired Generalization and L2

1. Delete `_paired_generalization._split_eligibility` as an independent trade
   predicate, or make it consume typed per-role results from the authority.
2. Boolean, float, string, missing and zero counts must receive exactly the
   same dispositions as L1.
3. Validate floors through the common resolver. Explicit zero refuses; it
   never produces an eligible paired score.
4. Persist both role evidence descriptors and the exact threshold contract in
   the paired result, L2 candidate record, leaderboard entry and promotion
   action.
5. Add semantic fixtures for malformed, missing, zero, active-negative and
   calibrated higher-floor cases through `paired_l2_fitness`,
   `normal_leaderboard` and every promotion action.

## D3. Correct Phase-1 and Weekly Promotion

1. Remove `int(_finite_number(trades_total))` before authority evaluation in
   `materialize_phase_1_promotion_candidates.py`.
2. Feed the raw count and content-verified role evidence into the authority.
   Persist the authority's typed count; preserve `None` as unavailable.
3. Wire weekly promotion through the same typed result before any candidate,
   week or aggregate becomes promotion-authoritative.
4. Add a regression proving `1.9`, `True`, `"1"`, NaN, infinity, missing and
   zero never become an eligible promotion candidate.

## D4. Bind Calibrated Contract Identity

1. Validate the calibrated contract's floor with `validate_floor_value`; never
   compare via `int(...)`.
2. Validate units and evidence descriptor types, not only truthiness.
3. Make the contract identity bind its schema, exact floor, units and evidence
   digest. Reusing one fixed `config_declared.v1` ID for different values is
   insufficient unless a separately persisted payload hash is authoritative.
4. Every consumer publishes `role_activity["threshold_contract_id"]` and the
   complete contract it actually used. Remove the hardcoded strict ID from the
   floor-12 lexicographic result.

## D5. Preserve Unavailable Facts End to End

1. In lexicographic components, unavailable trades remain `None`, never zero.
2. Audit histories, packets, logs and OLAP rows for `or 0`, `int(float(...))`
   and equivalent repairs on activity facts.
3. Diagnostic display may render `unavailable`, but may not manufacture a
   measurement.

## D6. Complete the Consumer Graph

Return executable semantic coverage for every C5 item:

1. L1 stopping/checkpoint custody;
2. lexicographic and paired selection;
3. P1LR handoff and aggregation;
4. L2 candidate records, leaderboard and promotion;
5. phase-1 materialization and weekly promotion;
6. LTS champion succession integrated into the branch/runtime lineage that
   actually executes, not left on an orphan feature branch;
7. M0 explicitly typed non-decision/non-promotion; and
8. the registered legacy `rl_pipeline` and its 31 configs: integrate it or
   mechanically prohibit its artifacts from every decision/promotion path.

The AFTER map must name code, tests and one actual semantic result per path.
String-presence tests are not acceptance evidence.

## D7. Repair the Reproducer

1. Preserve the original pre-fix packet.
2. Replace the post-fix traceback with a runner that captures every case and
   reports a disposition for each; one expected refusal may not abort the rest.
3. Start from Musashi's 12-case return reproducer and add every consumer fixture
   from D6.
4. Acceptance is zero reproduced defects, not “unable to start.”

## D8. Repair Current-Run Status Discovery

1. Make `multifront_status.py` discover the current decision output root and
   `p1lr-decision@*.service` units before historical screen identities.
2. Current work at this writing is `ac0941e7bdb1a163`, 4/4 fresh, 0/16. The
   old screen identity belongs in history.
3. Add the exact recurrence fixture: completed screen plus active decision
   root. The packet must report the decision identity current and screen
   historical.
4. Correct the queue item that currently says no approved successor while the
   approved successor is already running.

## D9. Return and Execution Sequence

1. Continue the current P1LR decision workers uninterrupted.
2. Implement D1-D8 on CPU while those workers train.
3. Run the focused authority/selection/promotion/L2/status tests, then the full
   suite.
4. Return exact commits, a clean pushed branch, the non-aborting reproducer,
   corrected AFTER map and fresh 4/4 runtime evidence.
5. After independent acceptance, continue the already approved hierarchy:
   WP2 metric schema, WP3 reward plugins, activity-bearing WP4 calibration,
   hashed R1/R2 contracts, one R1 mechanics smoke, 12-cell R1 decision, then
   16-cell R2 confirmation. Do not jump directly to R3 genes.

No new owner phrase is required. Do not self-close any finding.
