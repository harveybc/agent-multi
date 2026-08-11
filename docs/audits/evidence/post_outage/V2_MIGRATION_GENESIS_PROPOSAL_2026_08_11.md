# V2_MIGRATION: explicit-ID v2 genesis proposal — next distributed DOIN component job

Status: PREPARED EVIDENCE ONLY. Not deployed, not launched, no fleet host
touched, no legacy database opened for write, paused 2026-08-06 chain not
resumed. Launch requires a DOIN job boundary and Musashi's/owner ordering —
not granted by this document.

Prepared by: General Satoshi III (successor technical lead), 2026-08-11,
under WP5 (section 9) of
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_209_223_VERDICT_AND_PHASE1_LR_ORDER_2026_08_11.md`
and the deployment disposition in section 2.1 of
`docs/audits/AUDIT_SATOSHI_III_RETURN_209_220_2026_08_11.md`.

## 1. Reviewed migration manifest (validated this session)

- Canonical location: doin-node branch `fix/tx-content-binding-20260810`
  at commit `0821ec236e85040d9ab45c89b01437f4cbaeb9ab`, file
  `examples/fleet_shared_population_identity_template.json`
- sha256: `062ae19149d91ead710302e4aa119aeb1d9577eb324137753e32edcaa3e337cb`
- git blob: `6647d45579283567a319fdc30c40f8b8cfbfb5ce`
- Identity pinned by the manifest (the deterministic epoch genesis, i.e. the
  identity the existing fleet chains derive implicitly):
  - `chain_id`: `doin-4e19257e8941`
  - `genesis_hash`:
    `4e19257e8941caec2ec6a4d581a981a533ab5728c11e58c3d13040049e3cd6d5`

Validation of that manifest in a clean temporary state directory passed all
eight typed checks — see
`V2_MIGRATION_MANIFEST_VALIDATION_2026_08_11.json` (same directory) and the
reproducer `V2_MIGRATION_VALIDATION_SCRIPT_2026_08_11.py`.

## 2. Proposed NEW explicit-ID v2 genesis (this proposal)

Section 9 orders "one new explicit-ID v2 genesis for the next distributed
DOIN component job". A new job must not share the deterministic default
identity with the legacy fleet chains or with the paused 2026-08-06 chain
(finding 211: identical implicit identities are distinguished only later as
competing histories). The proposal therefore derives a NEW deterministic
genesis from an explicit, job-specific `generator_id`:

| Field | Value |
| --- | --- |
| generator_id | `doin-v2-component-job-001-20260811` |
| genesis_hash | `536f6234e5051018b2fdda956a1c535eead69917ae128d0b2837bb5e2aecfee0` |
| chain_id | `doin-536f6234e505` |
| protocol_version | `2` |
| doin-core commit | `00397f5390649280aab7ba9b6420e71ff299a9da` (branch `fix/tx-content-binding-20260810`) |
| doin-node commit | `0821ec236e85040d9ab45c89b01437f4cbaeb9ab` (branch `fix/tx-content-binding-20260810`) |
| domain_hash | `TBD_AT_JOB_MATERIALIZATION` (sha256 of the materialized domain spec) |
| config_hash | `TBD_AT_JOB_MATERIALIZATION` (sha256 of the exact per-machine node config JSON) |
| data_hash | `TBD_AT_JOB_MATERIALIZATION` (sha256/manifest digest of the immutable input dataset) |

Determinism: `Block.genesis("doin-v2-component-job-001-20260811")` at
doin-core `00397f5` uses the fixed Unix-epoch timestamp, so every machine
recomputes the identical genesis hash independently. Proven in check H of
the validation report: an independently instantiated chain from this genesis
in a second clean state directory fully verifies (10/10 checks, CLI exit 0)
under `--expect-chain-id doin-536f6234e505 --expect-genesis 536f6234e505…`.
The hash is distinct from the deterministic default
(`4e19257e8941…`), so the new chain can never be confused with any legacy
or paused chain; v2 peers refuse mismatched or unattested identity with
typed errors before any block exchange.

Boot mechanics at job materialization (required because
`UnifiedNode.start()` initializes an empty DB with the default
`"genesis"` generator): the job materializer pre-seeds each worker's fresh
state directory with `ChainDB.initialize("doin-v2-component-job-001-20260811")`
before first node start; startup verification then checks the chain against
the configured `genesis_hash` and stamps `(chain_id, genesis_hash)` into
database metadata inside the same connection.

## 3. Pre-claim identity check (every machine, before ANY candidate claim)

Each worker MUST print, and the job log MUST retain, the following line —
all seven values byte-identical across every participating machine — before
its first candidate claim; any mismatch or absence is a typed refusal, not
a warning:

```
DOIN_V2_IDENTITY chain_id=doin-536f6234e505 \
  genesis_hash=536f6234e5051018b2fdda956a1c535eead69917ae128d0b2837bb5e2aecfee0 \
  domain_hash=<sha256-of-materialized-domain> \
  config_hash=<sha256-of-node-config-json> \
  data_hash=<sha256-of-immutable-dataset-manifest> \
  doin_core_commit=00397f5390649280aab7ba9b6420e71ff299a9da \
  doin_node_commit=0821ec236e85040d9ab45c89b01437f4cbaeb9ab
```

The three `<...>` placeholders are bound at job materialization; the
materializer must fail closed if any placeholder survives into the printed
line.

Machine-readable proposal: `V2_MIGRATION_GENESIS_PROPOSAL_2026_08_11.json`
(same directory).

## 4. Constraints restated

- Every legacy blockchain database stays byte-for-byte untouched and
  read-only.
- The paused 2026-08-06 chain is not resumed and its identity is not
  reused.
- No deployment, no fleet dispatch, no DOIN optimization launch occurs
  under this document; the launch decision sits at the next DOIN job
  boundary under the ordered roadmap.
