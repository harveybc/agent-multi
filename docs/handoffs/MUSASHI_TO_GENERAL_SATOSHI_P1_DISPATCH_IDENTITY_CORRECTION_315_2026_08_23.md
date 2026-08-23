# Musashi to General Satoshi: P1 dispatch identity correction 315

Date: 2026-08-23
Priority: immediate, non-destructive correction while first arms continue

## Finding AUD-F1-20260823-315 (S3, observed)

Dragon and Gamma execute immutable worktrees pinned to `6e7bd128`. Omega's
`run_seed101.sh` executes the mutable Satoshi worktree, whose HEAD advanced to
the documentation-only commit `08e0f724` after N started. Its launch manifest
does not persist a source commit. The current N process started before that
commit and must not be killed, but later arms could start under a different
repository identity and the report cannot prove otherwise.

## Required correction without wasting completed compute

1. Do not stop or mutate the active N process on Omega.
2. Prevent the current wrapper from launching EN-W/EN-F after N. Preserve its
   exit and report. Do not send a signal to the training child.
3. Create a clean detached worktree pinned exactly to `6e7bd128`, matching the
   Dragon/Gamma execution model.
4. Launch only the remaining Omega arms from that immutable worktree after N
   reaches an accepted terminal report.
5. Before any remaining arm starts, add an external launch-identity manifest
   containing full commit, clean-tree proof, hashes of the driver, pipeline,
   nested contract and effective command. Bind its digest into the arm return.
6. Verify all twelve arms share the same code-file hashes. A docs-only commit
   may be reported as provenance but must not silently substitute executable
   identity.
7. Add a regression ensuring a sequential wrapper refuses to launch its next
   arm when HEAD or an executable-file hash differs from the frozen launch
   identity.

The other three workers continue. This correction does not revoke campaign
authorization and must not discard any accepted arm.

