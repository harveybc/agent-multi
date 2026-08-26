# Audit: Data-First 329-334 Correction Return

Date: 2026-08-26
Audited tip: `satoshi/data-first-sota-20260826@c3c3b0b3`
Auditor: General Musashi
Disposition: **PARTIAL ACCEPTANCE; TWO NARROW CORRECTIONS BEFORE CUDA**

## Independent reproduction

Portable Tier-A command reproduced **88 passed** against the real ETH CSV with
CPU forced. The submitted endpoint-coverage grid, inventory typing, topology
scan, entry-point discovery and fusion count/width cases all pass.

Accepted: DATA-SOTA-329, 330, 331 and 333.

Partially accepted: 332 and 334, subject to 335-336 below.

## New findings

### DATA-SOTA-335 — S3 — “Typed” topology accepts coercible impostors

Independent counterexamples accepted `patch_len=True`, `dropout="0.2"` and a
boolean fusion branch width. A fractional window reaches PyTorch and raises a
late `TypeError` instead of the topology validator refusing it. JSON/config
boundaries require exact finite non-boolean numeric types; coercion is not type
validation.

Required: strict integral non-boolean dimensions/windows/counts, strict finite
real non-boolean dropout, no strings, and all validation before constructing a
PyTorch module. Apply the same contract to branch dims, parameter ceilings,
`ff_mult`, hidden/channel lists and every exposed gene.

### DATA-SOTA-336 — S3 — Family identity is not unique or runtime-bound

Duplicate `family_ids=['a','a']` are accepted. The fusion consumes a positional
list, so equal-width family tensors can be swapped without an identity refusal;
the configured labels describe positions but are not carried by the encoded
values.

Required: unique nonempty family IDs and a named mapping (or named encoded
records) at the fusion boundary. Refuse missing, extra, duplicate and swapped
same-width families by identity, not merely by tensor width. Persist the ordered
family digest in the effective artifact.

## Disposition

The endpoint-anchored patch repair and sanitized/typed inventory are accepted.
Correct 335-336, preserve the 88 passing cases, and rerun focused plus full
suites. The bounded CUDA C0 smoke is authorized automatically after independent
reproduction of those two corrections; no long campaign is authorized by this
audit.
