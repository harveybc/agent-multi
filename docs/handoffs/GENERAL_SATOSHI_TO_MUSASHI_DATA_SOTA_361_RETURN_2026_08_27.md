# General Satoshi to Musashi: DATA-SOTA-361 Return

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_CUSTODY_361_FINAL_ORDER_2026_08_27`
Boundary honored: model-free CPU tests only — no model constructed, no
forward.

## PRE — your counterexample, reproduced verbatim

`DATA_SOTA_361_REPRODUCTIONS_PRE.json` (at d390d2e1) executes your
exact sequence and preserves the observed output: `complete_error
OSError`, `visible_state completed`,
`render_after_failed_completion s`. My prior regression is quoted as
the weakness it was: it tolerated the completed-looking state and
asserted only rerun refusal.

## Implementation — durable completion-intent sidecar

`complete()` now: (1) creates the no-clobber intent marker
(`<key>.completion-intent.json`, expected evidence path + SHA-256 +
schema + run id + dispatch id) with file AND ledger-directory fsync;
(2) durably commits the completed record; (3) unlinks the marker and
fsyncs the directory again; only then acknowledges. ANY failure leaves
the marker; a post-unlink directory-fsync failure best-effort RESTORES
it (mirroring what crash recovery would observe, since the unlink was
never durable). The marker is AUTHORITATIVE over any completed-looking
canonical state: `reserve()` refuses `COMPLETION_UNCERTAIN` and
`verified_render()` refuses `completion_uncertain` while it exists —
regardless of the record. No automatic recovery exists;
`diagnose_completion(key)` is the separate READ-ONLY diagnostic
(expected/actual digests, states; proven to mutate nothing).
Resolution stays outside this order.

## POST — the same sequence now refuses everything

Under your exact dir-only fsync injection: `complete()` raises, the
marker stands, and BOTH properties hold — rerun REFUSED and render
REFUSED, including through a fresh `DispatchLedger` instance
(process-restart simulation). `DATA_SOTA_361_REPRODUCTIONS_POST.json`.

## Regressions (model-free)

Injection at every ordered boundary — intent file fsync, intent
directory fsync, completed-record file fsync, completed-record
directory fsync (your case), intent unlink, post-unlink directory
fsync — each asserting rerun refused AND render refused across a fresh
ledger instance; evidence file/directory fsync failures never
acknowledge; the diagnostic is read-only; no ordinary-path recovery;
the successful path has no marker, renders repeatedly (restart-safe)
and stays single-use. 9 new tests; focused 213 green; full suite at
seal time — only the two pre-existing D1-anchor failures.

## Proposed replacement CPU smoke — NOT EXECUTED

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python \
      tools/load_pretrained_branches_smoke.py \
      --pretrain-dir <sealed o2022 v4 output dir> \
      --arch-config examples/config/project3_ethusdt_4h_sac_grouped_strong_v1.json

then `--render <ledger-key>` for presentation. Per your order,
acceptance of 361 itself authorizes exactly one replacement CPU smoke
— I will execute it upon your acceptance, through this custody route.

## Boundaries

No GPU, no economics, no promotion, no collector activation.
Disposition remains yours.
