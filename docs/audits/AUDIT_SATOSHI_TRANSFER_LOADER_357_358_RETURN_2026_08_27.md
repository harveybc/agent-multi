# Audit: Transfer Loader 357--358 Return

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@b53635d9`

## Verdict

**REVISE BEFORE REPLACEMENT SMOKE.** The canonical architecture correction and
the newly disclosed weak-route mismatch are accepted as valid. The proposed
single-use execution route still has two custody defects, so no replacement
forward is authorized yet.

Independent Tier-A reproduction: **55 passed**. No model forward was executed.

## Accepted Facts

- The prior v1 experiment config is incompatible with four of five pretrained
  branch plugins and now refuses transfer.
- `grouped_architecture.py` is shared by SAC and the smoke materialization paths.
- The strong v1 config materializes the five intended branches, explicit state
  branch and explicit cross-family fusion.
- SAC and smoke materialization produce the same architecture digest; seeded
  non-transferred state/fusion initialization is bit-identical in tests.
- Loader accounting is derived and conservation-checked.
- The historical double invocation is disclosed without invented first-run
  metrics.
- No replacement forward was run.

## Findings

### DATA-SOTA-359 (S2): execution config has a TOCTOU gap

The tool materializes the architecture from the config, reserves custody,
materializes it again, and then reads the config a third time to build the env.
A mutation after the second check but before the third read can change runtime
fields while the evidence retains the first config digest. Re-reading a mutable
path does not create an immutable execution identity.

Required: read config bytes exactly once into an immutable snapshot; derive the
file digest, parsed effective config, architecture materialization and env config
from that same snapshot. Bind the snapshot digest in the dispatch key and
ledger. Optionally verify the source path remains unchanged for operator
visibility, but execution must never consume a later read.

### DATA-SOTA-360 (S2): custody can acknowledge unverifiable evidence

The ledger does not enforce legal state transitions. Its records and output
directory entries are not parent-directory-fsynced. Completion records only an
evidence filename, not its digest. `--render` accepts arbitrary JSON without
proving a matching completed ledger record, run id, dispatch identity, schema or
evidence digest.

Thus a power loss can leave `completed` with missing evidence, and a fabricated
or substituted packet can be rendered as though it were the completed run.

Required: explicit state machine; durable file and parent-directory fsync for
reservation, transitions and output; evidence SHA-256 bound in the completed
record; completion only after durable evidence; renderer requires ledger key and
verifies completed state, evidence digest, schema, run id and dispatch identity.

## Disposition

Do not execute the proposed replacement smoke. Corrections 359--360 require
CPU-only implementation and tests without constructing or forwarding a model.
After independent acceptance, exactly one replacement CPU smoke will be
authorized because the actual strong-config/custody route has never executed
end to end. Previous hardcoded forwards cannot substitute for that fact.
