# General Satoshi to Musashi: DATA-SOTA-347..352 Return

Date: 2026-08-27
Correction commits: agent-multi `9fbd181e`, lts `b6bef6c`
(evidence commit follows on the agent-multi branch; clean-tree proof in §2).
Order: `MUSASHI_TO_GENERAL_SATOSHI_DATA_SOTA_347_352_CORRECTION_ORDER_2026_08_26`

## 1. PRE and POST counterexample outputs

`docs/audits/evidence/DATA_SOTA_347_352_REPRODUCTIONS_PRE.json` (at
f59aa495/cb0d604, before edits) and `..._POST.json` (at the corrected
commits). Highlights: PRE — impossible date 2021-13-45 validated, the
string "yes" minted o2023, 2 of 6 features silently dropped, the same
monitor calibrated AND judged, store-crash session said "completed",
same broker_time stored twice, spread −50 stored, failure bound 0
accepted, "garbage-da" stored as a TRM date, no as-of rule. POST —
every one refuses with a typed error or behaves honestly, proven also
against the REAL stores.

## 2. Exact commits and clean trees

agent-multi `satoshi/data-first-sota-20260826`: corrections
`9fbd181e`, evidence sealed in the following commit (tree clean at
push; prepush sensitivity gate green). lts
`satoshi/alpaca-quote-scheduler-20260826@b6bef6c` (pushed, clean).

## 3. Suites

Focused: agent-multi 225 (contract v3, 347-350 regressions, TRM 352
suite, Tier-A env parity, sanitization scan, surface index); lts
scheduler 22. Full suites re-run at seal time; the ONLY failures are
the two pre-existing D1-anchor tests (documented baseline).

## 4. Exact 83-feature coverage manifest

`validate_branch_partition` enforces exactly-once coverage, non-empty
families and canonical within-family order; the v3 contract's five
families cover 83/83, and the smoke manifest persists the global
ordered digest plus per-family ordered digests
(`feature_partition` block of
`WP_PRETRAIN_O2022_V3_CPU_SMOKE_2026_08_26.json`).

## 5. Train/calibration/monitor boundaries and digests

Chronological, wholly before origin score start (o2022): train 2,800
windows (…→2021-06-12), calibration 600 (2021-06-13→2021-09-20),
monitor 600 (2021-09-21→2021-12-29); per-partition step digests in the
smoke manifest. Weights froze from CALIBRATION initial losses
(`initial_calibration_losses` + `calibrated_on`); monitor only
reports/checkpoints. Monitor losses DECREASE for both objectives on all
five branches; crossing 0.0.

## 6. Quote crash/restart idempotency evidence

REAL-sqlite regressions: same (venue,symbol,broker_time) replayed
across two sessions → ONE `quote_canonical` row, TWO
`quote_session_membership` ledgers; store-write crash → session
`failed_unexpected` and the exception propagates; operator interrupt →
`interrupted`; `completed` only after the final requested tick; typed
rejection counters for NaN/Inf/crossed/zero/negative-size/malformed
timestamps. POST JSON carries the executed demonstration.

## 7. TRM future-effective and as-of evidence

Strict calendar parsing, ordered intervals, finite positive COP, unit
check, atomic provenance. `trm_as_of` is the single consumption API:
on the REAL store, the future-effective publication collected on
08-26 (vigencia 08-27) raises typed Unavailable for 08-26 and returns
3118.24 COP only on its validity day. Weekend-span, gap, overlap
(typed Ambiguous) and revised-row fixtures frozen.

## 8. Proposed transfer-loader smoke (NOT launched)

One CPU command, to be implemented and executed only after your
acceptance of this packet:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python \
      tools/load_pretrained_branches_smoke.py \
      --pretrain-dir <accepted o2022 v3 output dir> \
      --arch-config examples/config/project3_ethusdt_4h_sac_grouped_features_v1.json \
      --strict

It will: verify the manifest generation seal and complete identity;
refuse on any digest drift (contract/data/feature/family/topology);
load ONLY encoder weights (adapters excluded — key-overlap refusal
already in the runner) into the grouped extractor's matching branches
by family digest; prove bitwise weight equality post-load; run one
forward on the real env observation and compare against the standalone
encoders. Mechanics only — no SAC training, no economics.

## Boundaries

No GPU, no SAC transfer, no economic comparison; the three remaining
objectives wait for your acceptance. Scheduler and TRM collector
remain non-authoritative and unactivated. Disposition remains yours.
