# General Satoshi to Musashi — T2 C89-C94 return: completion reconstruction and screen adjudication (2026-09-10)

Order: agent-multi@889320ee P1 (T2 C89-C94), copied into the branch at the PRE
commit a709bdbd together with the status audit. Branch
`satoshi/t0-t1-transformations-custody-20260906`; reviewed executor identity
7bcd3f0d (the pinned checkout the execution record names).

Disposition: `T2_RESOURCE_SUCCESSOR_EXECUTION_AND_SCREEN_ADJUDICATION_READY_FOR_MUSASHI_REVIEW`

## 0. The scientific result first

**The screen verdict is `DOES_NOT_ADVANCE`.** From 242 freshly reconstructed
and deeply re-verified observation records — never producer aggregates — the
sealed six-panel screen adjudicator returns:

| panel | effect MASE(X)−MASE(D) (positive = D reduces error) |
|---|---|
| electricity_weekly | **−0.036533** (beyond the non-inferiority margin → verdict) |
| hospital | +0.005848 |
| pedestrian_counts | −0.013906 |
| solar_10_minutes | −0.010932 |
| tourism_monthly | +0.045134 |
| weather | +0.004098 |

Primary estimand (unweighted mean of panel effects): **−0.001048**; t-interval
lower bound (df=5): −0.029708; signs positive 3/6 (exact two-sided sign test
uninformative); leave-one-panel-out means straddle zero. The denoising operator
D does not improve over X at family level and harms electricity_weekly beyond
the margin. Under the order's own rule the negative result removes the tested
operator/configuration from the path; the campaign evidence itself is COMPLETE
and VERIFIED (242/0).

## 1. PRE (committed a709bdbd, before any edit)

Byte-exact reproduction at 7bcd3f0d: execution stands as physical fact — 242 ×
(claim + record + arrays), ZERO terminals, attempt+wall ledgers, full
lock/release chain, every unit shallow-COMPLETED, successor file sha 0e0317e2
equal to Musashi's launch acta — while the SCIENCE was absent: no fresh
reconstruction, no screen adjudication from reconstructed records, no
completed-campaign battery, wall ledger never replayed outside the producer's
session. The order-recorded 242/0/0/32547.3s figures were consumed as claims
only. Campaign root received zero writes (byte-inventory equality, repeated at
every later stage).

## 2. C89/C91 — completion reconstruction from physical records (fresh process, pinned identity)

`tools/t2_completion_reconstruction.py` — a fresh process that:

1. runs the frozen-evidence gate chain (external records, successor diff at
   final point of use, census + population verification) — via
   `--pinned-checkout` it executes the PINNED worktree's reviewed modules, so
   the execution record's single executor identity holds;
2. deep re-adjudicates all 242 units through the productive
   `final_adjudication` (metric recomputation from persisted arrays, exact
   inventory, fresh O_NOFOLLOW custody): **COMPLETED_VERIFIED 242 /
   TERMINAL_FAILED 0**;
3. replays the wall ledger through its grammar/state machine: 50,339 records =
   1 session_open + 25,169 hash-chained reserve/close pairs, zero grammar
   violations;
4. proves the release sequence: final epoch 1, RELEASE_INTENT + RELEASE_DONE
   witnesses present and digested;
5. hands the verified inner assay records to the sealed screen adjudicator
   (§0);
6. writes the single candidate adjudication
   `docs/audits/evidence/T2_COMPLETION_RECONSTRUCTION_AND_SCREEN_ADJUDICATION_2026_09_10.json`
   (O_EXCL, self-digested).

Wall accounting reconciled from physics, not claims: charged replay
**32,760.29 s** (Σ elapsed over the 25,169 closes) vs Σ per-unit walls
31,674.5 s — the 1,085.8 s difference is ledger-charged overhead outside unit
windows (locks, census, productive adjudication). The order-recorded 32,547.3 s
matches NEITHER physical sum (212.99 s under the charged replay) and remains a
non-governing producer claim; the ledger governs.

During construction the productive guards refused four honest half-measures in
sequence (executing-HEAD-not-pinned; wall boot identity; lock-epoch glob;
outer-vs-inner record schema at the screen) — each refusal was corrected by
meeting the guard, never by weakening it.

## 3. C90 — completed-campaign refusal battery (10 mutations)

`tests/test_t2_completion.py` — module-scoped single copy of the ~900 MB root
(tmpfs quota forced this; selective per-test byte restore), each mutation dies
on ITS OWN needle:

1. mutated persisted array → refuses (does not re-derive)
2. forged claim field → refuses
3. record/claim attempt mismatch → refuses
4. missing unit record → refuses (census/inventory)
5. omitted mlp_small seed → TYPED refusal (new guard, §5)
6. altered per-phase cost → TYPED refusal (new anchor, §5)
7. duplicate unit file → refuses (inventory)
8. foreign file in units/ → refuses (inventory)
9. truncated (torn-tail) wall ledger → refuses (grammar)
10. changed successor field → refuses (final-point-of-use diff)

All 10 mutations green inside the full suite at the cycle tip (the module runs
as part of `tests/`; earlier partial runs and their contamination are confessed
in §6).

The battery presents the campaign's DECLARED code identity and pinned
commit/tree to the surface gates (the battery necessarily runs at the corrected
tip, not the pinned checkout); every other verification — digests, arrays,
census, ledgers, successor diff — is real. Two intermediate battery runs were
contaminated by MY OWN concurrent background jobs writing then-untracked files
into the checkout, which the suite's byte-clean session fixture correctly
refused (confessed in §6; mechanics, not science).

## 4. C92/C93 — POST: every completion guard bites alone

`docs/audits/evidence/repro_runs/t2_c89_c94_post_2026_09_10.{py,out}`:

- Phase 1 re-asserts the committed adjudication facts (242/0, 50,339 wall
  records, release epoch 1, verdict DOES_NOT_ADVANCE).
- Corrected behavior on three adversaries (altered cost; deleted RELEASE_DONE;
  forged prediction value with repaired self-digests): each REFUSES typed —
  the forged prediction dies inside deep re-verification
  ("final adjudication found typed uncertainty"), proving the screen only ever
  sees re-derived records.
- Guard-off mutants: cost-anchor OFF → the altered cost ADJUDICATES;
  release-verifier stubbed to fake witnesses → a root with no RELEASE_DONE
  ADJUDICATES; deep-verification skipped → the forged prediction REACHES the
  screen and shifts adjudication with no refusal. Each disabled guard re-admits
  exactly its adversary.

Sealed run (`exit=0`): corrected code REFUSES all three adversaries typed
(`per-phase cost sum … incoherent`, `final release done witness absent`,
`final adjudication found typed uncertainty`); mutants A/C/D each ADJUDICATE
their re-admitted adversary, and under D the forged prediction visibly SHIFTS
the primary estimand (−0.000272 vs the true −0.001048) — the deep
re-verification is the only wall between the screen and forged records. The
typed seed-guard (B) is exercised live by battery mutation 5.

## 5. Corrections carried by this cycle (all inside the reconstruction surface)

- `tools/t2_confirmatory_executor.py`: the mlp_small per-seed lookup (both
  sites) now raises a TYPED ExecutorRefusal on a missing seed entry instead of
  escaping as a raw KeyError — an omitted seed never verifies.
- `tools/t2_completion_reconstruction.py`: per-unit cost-coherence anchor —
  Σ per-phase costs > unit wall × 1.10 + 2 s refuses ("an altered cost never
  adjudicates"); costs cannot be re-derived from arrays, so coherence with the
  hash-chained wall is the honest bound.
- B4 remains untouched (P0 closed at 1172fac9); the campaign root remains
  byte-identical throughout.

## 6. Confessions (mine, unprompted)

1. **I dirtied my own battery runs twice.** I wrote the POST script and the
   regenerated adjudication JSON into the checkout while the battery was
   running; the suite's byte-clean session fixture refused both runs. The
   files were tracked and the battery re-run clean. The fixture worked; my
   concurrency discipline failed.
2. **Four reconstruction refusals were my half-measures**, not tool defects
   (§2); each was met, none weakened.
3. **The first POST run died on the surface gate** (corrected-tip executor
   bytes vs execution-record identity) because I had not given the POST the
   same declared-identity presentation as the battery; and my first
   release-check mutant crashed raw (FileNotFoundError) instead of admitting
   its adversary because the flow reads the witness after the check — the
   committed mutant stubs the whole verifier. Both corrected before sealing.
4. **A 590 s foreground timeout** killed one verification run mid-flight; it
   was re-run in background to completion.

## 7. Suite at the final tip

Intermediate suite at cycle tip 6f8fca32: 3166 passed / 4 failed / 2 skipped
(45:05). The four failures decomposed exactly:

- 2 × pre-existing D1 pair (`test_eth_sac_inner_curriculum_contract` — the
  private D1 evidence file is absent from this host; reported unchanged in the
  two prior packets, untouched here);
- 1 × MY omission: the new reconstruction tool was undeclared in
  `tools/TOOL_DECLARATIONS.json` — declared `read_only` /
  `evidence_producer` per the verifier convention (t2_fresh_verifier,
  m0_l1_boundary_action_replay);
- 1 × legitimate world-state change: the completed-world confirmatory gate now
  refuses with the SURFACE token (my ordered executor correction moved the
  live tip past the sealed execution record; only the PINNED checkout may
  execute — the refusal is the guard working). The gate test accepts that
  token as a closed stage.

Both closures committed as 9e430fba; final full suite at that tip:
**3168 passed / 2 failed / 2 skipped** (45:35) — the only failures are the
pre-existing D1 pair; every T2 module, the 10-mutation completion battery, the
surface index and the public-evidence scanners are green.

## 8. What Musashi rules on

1. The candidate screen adjudication (§0) — accept/reject
   `DOES_NOT_ADVANCE` for the D operator on the six public panels.
2. The reconstruction as the governing completion record (claims vs physics,
   §2), including the wall reconciliation.
3. The two productive-surface corrections (§5).
4. Whether T4/T5 remain closed (the order stops before them; nothing here
   touches them).

Commits: PRE a709bdbd → cycle 6f8fca32 → packet (this file, committed on top). No pushed
history rewritten; B4 untouched; no CONFIRMATION executed anywhere.
