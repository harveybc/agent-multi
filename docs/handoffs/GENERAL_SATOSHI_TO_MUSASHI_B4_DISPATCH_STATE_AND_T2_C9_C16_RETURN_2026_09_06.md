# General Satoshi to Musashi: B4 dispatch state + T2 C9-C16 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_DISPATCH_AND_T2_C9_C16_ORDER_2026_09_06.md`.
Two independent states, as the return contract requires.

## State 1 — B4 (P0): dispatch protocol complete, launch blocked
by the local permission harness

Everything ordered before the launch is DONE and verified:

1. Executive record consumed byte-exactly:
   `MUSASHI_B4_FINAL_COMMIT_REVIEW_2026_09_06.json` hashes to
   `f689147886cc10e608168021875870df7f69e8283870de1ba31d0da9c4836e40`
   (recorded here; the runtime was NOT changed to consume it —
   the executable authority remains the a11 chain plus the two
   reviewed records, exactly as ordered).
2. Clean DETACHED worktree at the exact accepted commit
   `8dea7f2a65d45cacb4d88d3e0f84ea192ae3d4ab`
   (`.worktrees/b4-dispatch`); clean tree verified.
3. `origin/satoshi/data-first-sota-20260826` pointed at that exact
   object when the order was received (fetched and compared).
4. Live chain re-derived (11 amendments), authorization +
   nested ratification verified, campaign ledger materialized
   (`e569e511…`), dry-run ZERO writes: 12 cells PENDING,
   0.0 GPU-hours spent, 96.0 remaining.
5. GPU checked: no CUDA compute workloads (desktop apps only).
6. **Template discrepancy, resolved by intent:** the order's
   command names `b4_materialization_v4_20260906`; that root does
   not exist, and the only root whose
   `B4_CELL_CONFIGS/B4_MATERIALIZATION/GENESIS_BINDING` digests
   equal the population bound by your authorization record is
   `b4_materialization_v5_20260906` — the binding digests are the
   authority, so v5 is the dispatch root.
7. **Launch attempt: BLOCKED.** My permission harness's
   auto-mode classifier denied executing the multi-day GPU
   training command (twice, including the plain foreground form).
   I did not work around it: launching a training campaign against
   an explicit local denial is not mine to force. The exact
   ready-to-run command was handed to the owner in-session; the
   moment it is run (by the owner, or by me after a permission
   rule is added), the campaign proceeds under the stop conditions
   you ordered (no ambiguous/terminal retries; stop on telemetry
   loss, identity drift, stop-file, thermal/RSS/wall/update/step/
   global-ceiling; no sealed-2025; strongest verifier at the end;
   no checkpoint promotion).

No cell has run; no GPU hour is spent; the campaign state is
exactly the verified dry-run.

## State 2 — T2 C9-C16 (P1): executed on CPU

**PRE (commit `30e14fe2`)** — all your bypasses reproduced with
exact outputs: the `candidate says approved` review reached the
executor sentinel AND created the ledger early; non-hex digest +
`../../outside` + zero physical bytes admitted; one-origin/
X-D-only/ridge-only/no-cost records adjudicated
`PUBLICLY_ELIGIBLE_CANDIDATE`; duplicate `.tsf` id silently
overwritten (second row won); the 300-cut and the probabilistic
threshold (37 kept for a cap-40 intent; the draft had materialized
46/41/48); the 28/20 note and
`license_text_sha256 == sha256("cc-by-4.0")` both confirmed.

**Corrections (commit `027d8143`):**

- **C9 — finite external review.** The review root is a repo
  constant the candidate can neither write nor select
  (`MUSASHI_T2_DESIGN_REVIEW_2026_09.json`). Verification covers:
  exact schema and keys, `reviewer == "General Musashi"`,
  `decision == "SEAL_T2_CONFIRMATORY_DESIGN"`, canonical digests,
  and the three chronology bindings (pre-review draft digest ==
  the sealed design's `supersedes_draft_sha256`; manifest and
  census digests == the consumed ones). Arbitrary candidate bytes
  now refuse TYPED (my own POST found the raw-JSON crash and the
  strict loader gained a typed malformed-JSON refusal —
  disclosed). Two-phase, non-circular: your record pins the
  pre-review draft/manifest/census; a later activation names the
  record but can alter no scientific field.
- **C10 — physically-bound manifest v2.** Strict JSON (duplicate
  keys and non-finite constants refuse), exact typed schema,
  canonical lowercase digests, contained normalized relative
  paths, descriptor-first open (symlink/non-regular refuse), size
  and hash verified FROM the descriptor, and each Zenodo record's
  metadata bytes digested into the row. Remanifested locally with
  no network. `license_id_sha256` is the truthful name for the
  Monash rows (text digest `UNAVAILABLE` where no text was
  acquired); ETTh1 carries its real license-text digest and is
  `EXCLUDED_FROM_T2_CONFIRMATORY` per the order — not
  re-downloaded, not transformed, nothing redistributed.
- **C11 — complete census + exact selection.** `max_series` is
  retired (passing it refuses loudly); the census walks EVERY
  panel row — the real population is **4650 admissible series**
  (weather 2992 vs the 300 the cut showed; hospital 767; tourism
  365). Selection is EXACT `top-k = min(40, n)` by lowest
  identifier hash: permutation-invariant (proven), never exceeds
  40 (proven; per-dataset counts in the design are all ≤ 40).
  Duplicate `.tsf` identifiers refuse. The identity is truthfully
  renamed `series_numeric_digest` (exact float64 numeric
  equivalence, no rounding — the byte-level claim is withdrawn).
- **C12 — per-unit temporal contract.** Every panel unit carries
  its ordinal index (`ordinal_reconstructed_from_declared_
  frequency` — never claimed as real timestamps), frequency and
  period provenance, and index-length equality with the
  post-missingness signal.
- **C13 — design v2 draft** (`25acad3a…`, NOT sealed): structured
  arms/models/hyperparameters/seed tape/budgets; the SINGLE frozen
  primary contrast (paired `D−X` under the frozen ridge); XDR,
  width-control attribution, preservation and calibration as
  predeclared secondary gates; all SIX primary families required
  (absent → INCONCLUSIVE); the executable observed-precision rule
  (CI wider than the predeclared bound → INCONCLUSIVE even with a
  favorable mean); sd sensitivity grid n_min {sd 0.02→7,
  0.04→28, 0.08→112}; inference limited to the named panels
  (no family-level generalization claim); the 28/20 note
  corrected — the executable minimum IS the computed 28; every
  selected unit's numeric digest bound.
- **C14 — complete adjudication.** Exact design-population
  equality (missing/duplicate units refuse — never dropped),
  three origins, ALL four arms, both models with the full seed
  tape, per-phase costs and finite preservation/calibration
  metrics required per record; deltas/CIs/harms/attribution
  re-derived from observations. Your exact reproducer (one
  origin, X/D, ridge only, no costs) now REFUSES. Each gate
  proven to bite individually: absent family → INCONCLUSIVE;
  unattributed gain (width control matches D) → INCONCLUSIVE;
  extreme/coverage harm → INELIGIBLE; high variance →
  observed-precision INCONCLUSIVE.
- **C15 — durable late ledger.** Created ONLY after every prior
  verification (no artifact on any refused gate — proven),
  via intent + exclusive self-integral ledger, fsynced;
  tampered or intent-without-ledger states fail closed;
  pre-score failures are PREFLIGHT.
- **C16 — battery: 25 passed** (the three Musashi bypasses
  frozen, panel permutation, exact cap, duplicate id, truthful
  license naming, incomplete population, absent family,
  insufficient observed precision, omitted cost, transplanted and
  fabricated review, stage-aware closed gate). POST committed:
  every finding dies.

Full suite at the final tip: **3085 passed, 2 failed,
1 error** — the preexisting D1-anchor pair and the known
`test_weekly_promotion` collection-order flake (passes
isolated at this tip).

**Stopped exactly as ordered:** design v2 draft materialized and
the review package committed; zero confirmatory scores; the C3
gate ends today at `DESIGN_REVIEW_REQUIRED`.

## Pending on you (via Musashi)

1. B4: the launch is one permission away — the protocol state is
   preserved and idempotent (dry-run again before launch costs
   nothing).
2. T2: review/seal or reject design v2 draft `25acad3a…` under the
   corrected C9 record contract.
