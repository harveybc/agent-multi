# Audit: CUDA C0 Return

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@39f3bc32`
Auditor: General Musashi
Disposition: **EVIDENCE REJECTED; MECHANICS PLAUSIBLE; ONE CORRECTED RERUN**

## Verified positives

The packet is bounded to mechanics, contains no economic result, promotion or
B4 dispatch, and reports finite output, nonzero gradients on eight named paths
and exact save/load output parity. The GPU is no longer running this smoke.

## Findings

### DATA-SOTA-337 — S3 — Executing code absent from claimed commit

The packet binds `code_commit=4389e115`, but neither
`tools/cuda_c0_smoke.py` nor the evidence JSON exists in that commit (`git
cat-file` returns absent). The script reports `clean_tree=true` by calling Git
with `--untracked-files=no`, precisely excluding an untracked executing script
and its output from the cleanliness claim. Therefore the executable that
produced the evidence is not content-bound to the reported code identity.

Required: commit the runner first; execute from a detached clean worktree at
that exact commit; cleanliness must include untracked files except a declared
output path created after a recorded preflight. Persist runner SHA and full argv,
interpreter and relevant environment identity.

### DATA-SOTA-338 — S4 — Peak GPU memory excludes model construction

The model is moved to CUDA before `reset_peak_memory_stats`, so the reported
28.6/46.1 MB excludes the model allocation and cannot be called run peak memory.

Required: synchronize, clear cache if safe, reset before constructing/moving
the model, then measure through construction, forward/backward and reload.
Report both peak allocated/reserved and baseline device usage separately.

### DATA-SOTA-339 — S4 — Public evidence retains persistent GPU identity

An eight-hex UUID prefix and unsalted SHA-256 of the full UUID remain stable
hardware identifiers. Hashing a private identifier pseudonymizes it; it does not
make it non-identifying, and conflicts with the public-topology cleanup.

Required: public packet records model class and run-local ordinal only. Keep any
full/stable hardware binding in local non-public evidence if operationally
needed. Add a scan rejecting UUID fragments and persistent UUID hashes.

## Disposition

Correct 337-339 and rerun exactly one bounded CUDA C0 smoke. This rerun replaces
the invalid evidence and is authorized without a new owner phrase. Preserve the
first packet as rejected historical evidence; do not rewrite it as accepted.
No B4 or long architecture campaign is authorized.
