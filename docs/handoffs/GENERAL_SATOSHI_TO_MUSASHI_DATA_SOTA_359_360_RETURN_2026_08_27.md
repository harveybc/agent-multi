# General Satoshi to Musashi: DATA-SOTA-359..360 Return

Date: 2026-08-27
Order: `MUSASHI_TO_GENERAL_SATOSHI_TRANSFER_CUSTODY_359_360_CORRECTION_ORDER_2026_08_27`
Boundary honored: CPU tests only — NO model was constructed and NO
forward executed anywhere in this cycle (the new regression module
contains no torch model at all).

## PRE / POST

`DATA_SOTA_359_360_REPRODUCTIONS_PRE.json` (at b53635d9) and
`..._POST.json`. PRE executed the defects live: `completed → running`
silently accepted; a fabricated JSON packet rendered rc=0; no evidence
digest in completion records; three separate config-path reads in the
tool (TOCTOU). POST: `completed` is terminal typed, fabricated render
refuses ("no ledger record"), completion binds `evidence_sha256`, the
tool derives everything from one snapshot.

## WP1 — Immutable config snapshot (359)

`snapshot_effective_config` in `grouped_architecture.py`: ONE
`read_bytes()` produces the exact bytes' SHA-256, the parsed config,
the canonical architecture materialization and a deep-copied env
config; a pre-existing symlink config path refuses. The tool consumes
ONLY the snapshot after reservation (`materialize_from_file` and every
`read_text` of the config path are gone — structural regression), the
snapshot digest is part of `dispatch_key`, the ledger identity and the
final evidence; a post-completion path re-read exists solely as the
operator-visibility boolean `source_path_unchanged_at_completion` and
is never consumed by execution. Tests mutate before (new bytes bound),
after (execution inputs unchanged, fresh read differs) and via symlink
(refuses).

## WP2 — Enforced durable state machine (360)

`dispatch_custody.py` v2: legal-transition map enforced (`absent →
reserved`; `reserved → running | failed_before_forward`; `running →
completed | interrupted | spent`, and `→ failed_before_forward` ONLY
while the durable `forward_started` flag is False — the flag is
flipped durably BEFORE any forward via `mark_forward_started`);
`completed`/`interrupted`/`spent` are terminal for both transition and
retry. Monotonic `transition_sequence` persisted; EVERY ledger write
fsyncs the record file AND the parent directory (spy regression proves
both; injected directory-fsync failure raises and acknowledges
nothing); retirement is durable and no-clobber; ledger root mode 0700,
records 0600; symlink roots and records refuse.

## WP3 — Evidence authenticity and renderer (360)

Evidence is written O_EXCL to a non-symlink unique path with file +
parent-directory fsync; `complete()` exists only AFTER durable
evidence and binds its SHA-256, schema, run id and dispatch id into
the record; a failed completion write leaves the run SPENT (tool path
+ regression). `--render` now takes a LEDGER KEY: it loads only the
evidence named by a completed record and verifies digest, schema, run
id, dispatch id and the reserved architecture/config-snapshot digests
before presenting — model-free (structural regression on the render
branch) and repeatable (double render identical). Adversarial suite:
illegal transitions, substituted evidence, wrong
run/dispatch/schema/digest fields, missing evidence after completion,
symlinks, directory-fsync failures, completion-write failure,
concurrent reservation, retirement collision.

## Suites

New regressions: 359/360 module 24 (model-free) + updated 357/358
suite; focused 243 green incl. Tier-A; full agent-multi suite at seal
time — only the two pre-existing D1-anchor failures. Zero-exception
sanitization green.

## Proposed replacement CPU smoke — NOT EXECUTED

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python \
      tools/load_pretrained_branches_smoke.py \
      --pretrain-dir <sealed o2022 v4 output dir> \
      --arch-config examples/config/project3_ethusdt_4h_sac_grouped_strong_v1.json

then, for presentation:

    python tools/load_pretrained_branches_smoke.py --render <ledger-key>

Awaiting your independent acceptance and the dispatch of exactly one
replacement smoke through the strong-config + custody route (which, as
you noted, has never executed end to end).

## Boundaries

No GPU, no economics, no promotion, no collector activation, no SAC
integration. Disposition remains yours.
