# Musashi to General Satoshi: Final Two Bounded Corrections

Date: 2026-08-25
Execution: CPU/code only; continue P1 monitoring

## Owner/Auditor Disposition

The prospective v2 observation contract is recommended for owner ratification:
83 ordered features, no `typical_price`, no raw price window, four explicitly
bound agent-state fields, window 32, flattened dimension 2660.

## C1 — Canonical Observation Identity

- Remove the duplicate newline digest implementation.
- Reuse the canonical compact-JSON feature digest.
- Bind state-field names/order/digest and actual observation-space shape.
- Seal P1 with both the legacy diagnostic digest (labeled) and canonical digest;
  never rewrite historical feature lists.
- Add fixtures proving all producers and consumers compute one digest.

## C2 — Exact Update Conservation and Digest Syntax

- Materialize quotient/remainder refresh schedules summing exactly to the
  declared annual budget for 168h/24h/12h arms.
- Test exact sums and deterministic remainder placement.
- Require SHA-256 fields as 64 lowercase hex and Git commits as full 40 hex in
  release packets; bind ensemble schema/version.

## Return

Return correction commits, focused tests and the final prospective contract
artifact. Do not launch post-P1 screens. P1 continues to terminal unchanged.

