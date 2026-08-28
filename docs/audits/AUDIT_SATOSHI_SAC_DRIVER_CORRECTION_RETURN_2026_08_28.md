# Audit: Paired SAC Driver Correction Return

Date: 2026-08-28 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@49f825aa`

## Verdict

**374-376 ACCEPTED. TRANSFER/PARITY ACCEPTED. GPU DISPATCH BLOCKED BY FOUR
OPERATIONAL FINDINGS.**

Independent evidence:

- 29 focused tests passed.
- Both-arm CPU dry-run completed through the real nested SAC pipeline.
- 112/112 non-branch tensors are identical and 219/225 temporal tensors change
  under transfer; sealed tensors load with bit parity.
- Actor and critic encoders are trainable and present in their optimizers.
- The corrected design has two arms, four paired seeds and eight regenerated
  genesis records.

## Findings

### DATA-SOTA-377 (S1): any existing file authorizes GPU execution

The driver accepts `--gpu-authorized-by-musashi` when `Path.is_file()` is true.
It does not verify schema, objective, reviewed commit, design digest, campaign
id or auditor identity. An unrelated file can satisfy the gate. Authorization
must be a typed, content-bound dispatch artifact, not path existence.

### DATA-SOTA-378 (S1): fresh attempts overwrite prior-attempt artifacts

Custody keys and evidence filenames include `attempt_nonce`, but `save_model`,
`results_file`, `save_config` and split paths live under the fixed
`output_root/trial_id`. A fresh non-resumable attempt can overwrite an
interrupted or completed attempt's model/config/results. Every attempt requires
an exclusive attempt directory created before pipeline construction; existing,
non-empty or symlink destinations must refuse.

### DATA-SOTA-379 (S1): execution identity does not bind the executable tree

The custody key binds only the driver's file hash as `code_identity`. Launch
manifests state "worktree pinned" but carry no exact commit or executable-file
digest set, and the driver verifies neither clean HEAD nor pipeline/agent/
environment/loader/config code. A modified trainer can run under the same cell
genesis. Bind a full commit plus clean-tree proof and a canonical allowlist of
executing file hashes before every cell.

### DATA-SOTA-380 (S2): logical GPU assignment is prose-only

The fleet plan assigns `gpu_slot_0..3`, but the driver receives no slot and only
checks that CUDA exists. It neither requires exactly one visible device nor
binds the physical/logical assignment into launch and terminal evidence. A cell
can run on the wrong device or see multiple GPUs.

## Accepted Scope

The exploratory treatment seal, nested data contract, execution envelope,
initialization parity, two-arm design and CPU mechanics remain accepted. No
scientific redesign or new proxy run is required. Correct 377-380 and dispatch.

