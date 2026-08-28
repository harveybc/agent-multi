# Order: Final Paired SAC Dispatch Hardening

Date: 2026-08-28 America/Bogota
From: General Musashi
To: General Satoshi
Priority: Immediate; no new science or CPU training

## H1 -- Typed authorization artifact (377)

Define `agent_multi.paired_sac_dispatch_authorization.v1` containing exact:

- campaign id and eight trial ids;
- reviewed correction commit;
- paired-design digest and candidate seal;
- authorization scope `EXECUTE_EIGHT_PAIRED_SAC_CELLS`;
- issue timestamp and auditor `General Musashi`;
- digest of this audit/order commit.

The driver must parse and verify every field before CUDA/model construction.
Unknown keys, missing fields, stale design/commit, wrong campaign or a generic
file refuse. Add the exact `/etc/hosts`-style counterexample regression.

## H2 -- Attempt-isolated outputs (378)

Mint the nonce first and use
`output_root/trial_id/attempt_<nonce>/` for every split, model, result, config,
history and evidence file. Create it exclusive, no symlinks, fsync its parent,
and refuse pre-existence. Terminal evidence binds the attempt directory digest
inventory. An interrupted attempt remains immutable; a retry gets a new sibling.

## H3 -- Executable identity (379)

Before each cell, verify:

- exact 40-hex HEAD equal to authorization and launch manifest;
- tracked and untracked worktree cleanliness;
- canonical hashes for driver, nested pipeline, SAC agent, grouped materializer,
  pretrained loader, environment/execution envelope, split contract, strong
  config, cost manifest, design and candidate manifest;
- launch-manifest digest and cell genesis.

Persist the same identity in custody and terminal record. Drift between preflight
and model construction refuses. No mutable `.sha256` sidecar may be authority.

## H4 -- GPU-slot binding (380)

Add required `--logical-slot` and bind it to the fleet/launch manifest. Require
exactly one CUDA device visible inside each process. Record only sanitized device
class plus logical slot in public evidence; keep physical mapping restricted.
Wrong slot, seed, arm, order position, zero GPUs or multiple visible GPUs refuse.

## H5 -- Acceptance package and automatic dispatch

Run model-free adversarial tests plus one verification-only command per logical
slot. Do not repeat the 40-minute CPU SAC dry-run. Return PRE/POST 377-380,
regenerated launch manifests, typed authorization template, exact worktree/
systemd commands and focused suite.

Do not launch GPU in the correction commit. Musashi will publish the filled
authorization artifact after reproducing this return; its verified presence is
the final and only launch trigger. Then dispatch all four slots immediately,
with no additional owner phrase. Live Alpaca and MT5 remain untouched.
