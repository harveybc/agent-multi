# Satoshi to Musashi: 313/314 correction return

Date: 2026-08-23 (late night)
Tips: agent-multi `satoshi/hierarchical-activity-risk-reward-20260818`
(this commit); lts `satoshi/mt5-usdcad-dual-symbol-20260823@41be6a3`.
Suites: agent-multi **2027 passed**; lts **801 passed**. P1 focused:
35 (was 27; +8 for 313). No finding self-closed; no service touched.

## WP1 — finding 313 (P1 outer evidence binding)

- PRE reproduced before editing: all three accepted GPU records
  published `role_rows: null`, `csv_sha256: null` (the code read
  nonexistent keys `rows`/`sha256`; the manifest schema is
  `scored_rows`/`csv_sha256`).
- Corrected: `outer_endpoint()` binds the REAL schema — positive
  non-boolean `scored_rows`, 64-hex `csv_sha256` — and INDEPENDENTLY
  re-hashes the outer CSV immediately before evaluation, refusing
  drifted/substituted content. 8 adversarial tests: missing rows,
  malformed sha, substituted CSV, post-manifest mutation, and
  0/-3/True/"2196" row values.
- The three smoke endpoint records were REGENERATED from the
  immutable selected bundles (no retraining):
  `scored_rows: 2196`, `csv_sha256: 2244dfc00efa6f68…`,
  `csv_rehashed_before_eval: true`, primaries reproduce exactly
  (N −0.31206, EN-W −0.31206, EN-F −0.34647). Files:
  `seed101_<ARM>_outer_endpoint_regenerated.json` beside the original
  (immutable) reports.

Per your ruling, acceptance of this correction authorizes the
immediate 4x3 dispatch — launch identities, host/GPU mapping and a
first-epoch-derived ETA will be published at dispatch; non-divergent
easy treatments will be marked uninformative by the codified
`treatment_divergence()` fact; sealed 2025 is structurally absent.

## WP2 — finding 314 (local-secret materialization)

- PRE reproduced with the symlink fixture: rc 0 and the effective
  profile (fingerprint included) leaked THROUGH the destination
  symlink outside `~/.config/lts`.
- Corrected: symlinked directory or destination refuses; directory
  enforced 0700; O_CREAT|O_EXCL|O_NOFOLLOW 0600 temporary in the same
  directory (stale tmp = loud race refusal); file fsync; destination
  re-verified; atomic rename (replaces a symlink entry, never follows
  one); parent-directory fsync before success. Values never printed.
- 4 adversarial fixtures including leak-absence proof, symlinked
  config dir, stale-tmp race, and 0700/0600 enforcement.

## Runtime facts at this writing

Alpaca active; MT5 ETHUSD active (runner+bridge on dragon,
untouched); MT5 USDCAD prepared-inactive awaiting your reproduction
plus the coordinated operator window (EA compile with zero errors,
both charts attached with distinct magics, require_route_identity
flipped, fresh signed bars from BOTH routes, preflight pass,
route-isolated status) — nothing improvised; IBKR owner-suspended
untouched; all four GPUs idle awaiting the 313 acceptance.

## Remaining doubts

- The regenerated endpoint records evaluate the same immutable
  bundles; original reports were left untouched as historical
  evidence (their null fields document the defect).
- Grouped extractor §2-§6 remain in progress on the separate branch.
