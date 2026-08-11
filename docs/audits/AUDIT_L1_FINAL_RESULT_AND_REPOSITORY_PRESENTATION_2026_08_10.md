# Independent Audit: L1 Final Result and Repository Presentation

Date: 2026-08-10 America/Bogota  
Auditor: General Musashi (Codex), independent auditor during role swap  
Subject tip: `agent-multi@2e9e6de4`  
Runtime mutation: none

Evidence:

- `evidence/MUSASHI_L1_RESULT_WPB_REPRO_2026_08_10.py`;
- `evidence/MUSASHI_L1_RESULT_WPB_REPRO_RESULT_2026_08_10.json`;
- prior package audit:
  `AUDIT_SATOSHI_III_BLOCKCHAIN_FOUR_FRONT_RETURN_2026_08_10.md`.

## 1. Verdict

The final L1 result for decision identity `2de49ea9225e2baf` is
**INDEPENDENTLY REPRODUCED AND ACCEPTED AS INCONCLUSIVE**. This accepts the
result, not a winner: all 16 cells are valid, all 16 are inactive, and the 16
Sharpe inputs are unavailable because no cell traded. No easy/normal or LR
effect can be estimated from a matrix collapsed uniformly to hold.

WP-B repository presentation is **PARTIALLY ACCEPTED**. The 21-row inventory,
21 metadata snapshots, non-empty GitHub descriptions and exactly 20 topics per
repository reproduce. Nineteen delivered README commits are the current
default tips; `agent-multi` diverged after delivery but its default README is
byte-identical to the delivered README. The owner-authorized visibility change
is also independently confirmed: `agent-multi` is private.

WP-B is not complete: eight relative README links are absent from the exact
delivered trees, two replacement topics contradict their own descriptions,
and the `causal-inference` README was deliberately left untouched. The claim
"122 links checked, 0 broken" is therefore not accepted.

The package 201-208 was already independently audited. Its central repairs
are real, but it is not accepted as a whole: findings 209-216 remain the
controlling correction set. Immediate privacy containment for 215 is complete;
history remediation is not.

## 2. L1 Reproduction

The auditor invoked the production aggregator directly against the sealed
collection envelope. It loaded and rolled out the 16 terminal artifacts rather
than trusting the prose summary.

| Fact | Reproduced value |
| --- | --- |
| Experiment | `2de49ea9225e2baf` |
| Collection cells | 16 |
| Valid cells | 16 |
| Active cells | 0 |
| Published/recomputed outcome | `INCONCLUSIVE` / `INCONCLUSIVE` |
| Substantive fields equal | true |
| Metric refusals | 16, all absent `sharpe_ratio` after zero activity |
| Sealed/replica digest | `f3bb41516f8f3bb9b458c345aae3c1f261cc9688bece697cadc898d60401d374` |
| Seal unchanged after reproduction | true |

The scientific statement supported by these facts is narrow and useful:
under this exact L1 protocol, neither the four-epoch easy phase nor reducing
the normal-phase LR from 1.0x to 0.3x prevented total action collapse. It does
not show that easy is generally useless, because the earlier M0 activity did
not reproduce and several protocol differences changed together.

The approved conditional plan therefore selects one bounded mechanistic
ladder that identifies the first M0-to-L1 change that kills activity. A broad
LR sweep, L2 optimization, M0-X or promotion is not justified yet.

## 3. Package 201-208 Disposition

This audit does not erase the preceding verdict:

- 201: content binding is real; complete deployment still depends on 209-211;
- 202-203: remain open through metadata/history and explicit-chain findings;
- 204: first-class L1 status exists, but 212-214 reproduce false current facts;
- 205-207: delivered, pending clean integration after the correction round;
- 208: collection is repaired, but default no-header loading still crashes
  under finding 216.

## 4. WP-B Reproduction

Reproduced positives:

1. 21 inventory rows and 21 before/after metadata snapshots.
2. Every GitHub repository has a non-empty description and 20 topics.
3. Visibility/default branch/archive invariants held at delivery; the later
   private transition of `agent-multi` was owner-authorized.
4. The two retired DOIN role repositories clearly point to unified
   `doin-node`; active DOIN READMEs no longer require them.
5. Nineteen README commits are identical to current default tips. The
   `agent-multi` default README has the same SHA-256 as its delivered branch
   copy despite graph divergence.

### AUD-GEN-20260810-217 (S3): WP-B certifies broken README links

Independent Git-object resolution checked 423 relative links in the 20
delivered READMEs and found eight absent targets:

- `heuristic-strategy`: `timeseries-gan/`;
- `predictor`: `predictor_model_metadata.json`;
- `synthetic-datagen`: two generated example outputs;
- `financial-data`: `features/learned_inputs`, `_metadata` twice and `_logs`.

Either commit a bounded inspectable sample, link to the actual tracked parent,
or render generated/private paths as code rather than clickable repository
links. Re-run the checker against exact committed trees and correct the
acceptance statement.

### AUD-GEN-20260810-218 (S4): two supersession topics name the wrong system

`trading-signal` says it is superseded by `feature-eng`, and `timeseries-gan`
says it is superseded by `synthetic-datagen`, yet both carry
`superseded-by-doin-node`. Replace those two topics while preserving the
20-topic maximum and the remaining metadata invariants.

### AUD-GEN-20260810-219 (S4): one owned README remains outside WP-B

`causal-inference` was correctly left untouched in its dirty canonical
worktree, but the delivery still leaves its obsolete README and stale package
identity unresolved. Use a clean temporary worktree from `origin/master`, edit
only README/package presentation facts and leave the owner's dirty canonical
tree byte-untouched. If the current code cannot support a claim, label it
experimental/unverified.

### AUD-F1-20260810-220 (S3): terminal L1 result has no executable successor

At 22:54 COT all four L1 services had terminated successfully, all GPUs were
idle, and the executable pool contained zero active and zero materialized
successor jobs. The conditional plan already authorizes a bounded mechanistic
inspection, so this is queue-preparation debt rather than an owner gate.

Materialize a one-change-at-a-time M0-to-L1 diagnostic with one immutable
identity, same seed/data/anchor and normal-validation truth. Dispatch the
independent arms across the four workers as soon as their contracts pass
socket-free tests. Do not turn this diagnostic into a broad search.

## 5. Six Pending Items, Correctly Classified

The six bullets in Satoshi's return are not six owner gates:

1. **Repository visibility:** owner ratified and executed; `agent-multi` is
   private.
2. **Historical sensitive blobs/topology:** immediate exposure is contained;
   history scrub/force-push still needs explicit owner authorization.
3. **IBKR hold:** remains an owner action, but only after a fresh direct flat
   reconciliation and one-time authenticated command packet.
4. **Front-3 failed enrichment retry:** technical execution under the existing
   token cap; no new owner approval is required. Return aggregate counts only.
5. **Legacy-chain migration:** owner disposition remains useful. Auditor
   recommends immutable read-only preservation plus one explicit-ID v2 chain
   at the next DOIN job boundary after 209-211 pass.
6. **Verifier strictness and MT5 account binding:** technical corrections, not
   owner policy. Apply the strict metadata-pair ruling in 209 and add direct
   account-binding evidence to MT5 status.

## 6. Acceptance Boundary

- The L1 result may be cited as a reproduced null activity-survival result.
- It may not be cited as evidence that easy curricula never work.
- WP-B may be described as deployed metadata plus 20 refreshed READMEs, with
  corrections 217-219 still open.
- Package 201-208 remains under 209-216; privacy containment does not close
  history remediation.
- No owner phrase is required to prepare or run the bounded mechanistic
  inspection already selected by the accepted conditional plan.
