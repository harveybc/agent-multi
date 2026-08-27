# Order: Transfer Loader Corrections 357--358

Date: 2026-08-27 America/Bogota
From: General Musashi
To: General Satoshi
Execution class: CPU-only; no additional model forward until acceptance

## WP1 -- Canonical effective architecture

Create one reusable materializer used by both the SAC construction path and the
transfer smoke. It must:

1. Read the complete grouped-extractor architecture from the effective config.
2. Apply the existing strict config merge and plugin topology validators.
3. Refuse absent or extra state-branch, fusion, family or feature declarations.
4. Bind the config file digest, complete effective-architecture digest, plugin
   names/parameters, ordered families, state keys and expected output dimension.
5. Construct branch, state and fusion modules only from that materialized
   object; remove all hardcoded architecture dictionaries from the smoke.
6. Prove that the SAC route and smoke route produce the same architecture digest
   and identical initialized non-transferred state/fusion tensors under one
   seed.

Add counterexamples for changed fusion heads/dimension, changed state branch,
changed family order, extra config key, config mutation after verification and a
same-shape but semantically different plugin.

## WP2 -- Loader accounting

Return an accounting object derived from offered state:

- tensors and bytes offered;
- tensors and bytes loaded per family;
- rejected keys by typed reason;
- adapter/optimizer/replay/calibration keys excluded;
- conservation assertion: offered = loaded + rejected.

A successful run must derive zero rejected keys; it may not print a literal.

## WP3 -- Single-use execution custody

Implement a durable dispatch ledger keyed by dispatch id + sealed generation
digest + effective architecture digest + data digest + code identity.

- Reserve the run atomically before constructing the model.
- Unique output path; no overwrite and no symlink destination.
- State transitions: reserved -> running -> completed/failed/interrupted.
- A completed or uncertain prior attempt refuses another execution.
- A renderer reads completed evidence and may be rerun without model execution.
- Record the historical double invocation as `DISCLOSED_PROTOCOL_DEVIATION` with
  only facts actually known.

Test concurrent reservation, parser/render failure after completion, crash
before/after forward, output collision, symlink, and attempted rerun.

## WP4 -- Return packet

Return PRE/POST reproductions, focused/full tests, canonical architecture object
and digest, SAC-vs-smoke parity proof, loader accounting, and ledger fixtures.
Propose one replacement CPU smoke command but **do not execute it**.

After independent acceptance, Musashi will decide whether the already executed
forward is sufficient with reconstructed architecture evidence or whether one
replacement smoke is scientifically necessary. No GPU or economic run is
authorized.
