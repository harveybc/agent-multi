# Musashi Audit: T2 C48-C56

Date: 2026-09-07

Reviewed tip: `4e5f19dccd24d6fb2cd7046ad173070bfbf98617`

Tree: `e04d8ddee6bd6be88e2f5585ae3dc7493e3faf20`

Disposition: `REVISE_BEFORE_T2_EXECUTION_RECORD`

## 1. Accepted Scope

The audit was performed from a clean detached checkout of the reviewed tip.
The productive call graph was traced from `main()` through
`verify_confirmatory_gates()`, `verify_execution_record()`, `run_unit()` and
`verify_unit_record()`.

The official POST was rerun. Its original P1-P3 attacks are dead:

- an arbitrary or nonexistent executor commit is refused;
- a rehearsal unit with forged membership, authority, observations or
  consumed metrics is refused;
- hooks preserve the assay values;
- missing execution authority causes zero writes; and
- the three development units complete and re-verify without touching the
  sealed population.

These are real improvements. They do not, however, open scoring because seven
later counterexamples remain in the productive runtime.

## 2. Blocking Findings

### C57: restart can renew the accumulated wall budget

`BudgetGuard` persists at a five-second cadence and explicitly tolerates a
torn line by losing up to five seconds. A crash before the next persistence
point loses the elapsed interval. Repeating the crash renews an unbounded
number of sub-five-second intervals.

PRE: eight sessions each ran about 0.03 seconds against a sealed 0.08-second
budget. All eight were accepted; real elapsed time was about 0.24 seconds.

The ledger is also read with `exists()` plus `read_text()` before it is opened
for append. Hash, parse and append therefore do not consume one
descriptor-bound object. Malformed interior lines are silently ignored.

### C58: a bare release filename frees a live lock

`acquire_lock()` treats `RELEASE_<n>.json` existence as release. It does not
parse the release, verify its self-digest, or bind it to the exact session and
holder.

PRE: session 1 was held by the current live PID. Creating an empty mode-0600
`RELEASE_000001.json` allowed session 2 to acquire immediately.

This also leaves the final-write crash ambiguity unresolved: a release record
whose fsync failed is not enough to distinguish a completed release from an
uncertain one.

### C59: failed terminals are not bound to current authority

The terminal schema carries no sealed-design digest, execution-record digest,
code identity, unit binding, claim digest or generation. Shallow adjudication
accepts a self-consistent terminal from another authority generation as
`TERMINAL_FAILED` and the main loop never re-verifies it deeply.

PRE: a freshly constructed terminal naming only the unit, attempt, mode and
failure was accepted as terminal under no design or execution authority.

### C60: operator disposition can legitimize a fabricated claim

`--declare-attempt-failed` runs before all scientific gates. It accepts a claim
without exact schema, self-digest, membership, unit binding or current
authority and then creates a terminal that future execution preserves.

PRE: a two-field JSON object (`attempt_id`, `mode`) for an arbitrary unit was
converted into an accepted `TERMINAL_FAILED` state.

### C61: the results root can be a symlink

`out_root.mkdir(exist_ok=True)` and subsequent path operations accept a
preexisting symlink root. Leaf-level `O_NOFOLLOW` does not establish root or
intermediate-directory custody.

PRE: a symlink results root was followed and the session lock was written
under its target.

### C62: a supervised fit does not obey remaining global wall

`make_fit_supervisor()` reads only the RSS limit. Its poll timeout is the fixed
120-second per-fit limit and is not reduced to the remaining campaign wall.

PRE: a 0.12-second fit returned successfully while the supplied global wall
limit was 0.01 seconds.

### C63: the executor can exit zero with a new UNCERTAIN unit

The startup check blocks uncertainty that existed before the run. A failure
after the assay's exception wrapper, such as NPZ or record persistence failure,
escapes `run_unit()`. `main()` catches `BaseException`, increments
`failed_preserved`, continues, releases the lock and exits zero without a final
deep adjudication.

PRE: a simulated post-assay persistence crash produced `main_rc=0` and
`failed_preserved=1`, while physical adjudication after exit was
`UNCERTAIN: claim without terminal or record`.

## 3. Decision

No external T2 execution record was authored or installed. No sealed-bank
series was scored. C48-C56 are accepted only for the exact bypasses they close;
the runtime remains closed until C57-C65 pass independent review.

The old B4 materializer race seen in this branch is not ported or repaired
here. B4 is running from its separately hardened v7 checkout and must remain
untouched.

## 4. Current Runtime Facts

- B4 v7 systemd service: active and advancing on `cuda:0` at audit time.
- T2 confirmatory scoring: closed; zero sealed units scored.
- M0-M2 model-information pilot: completed and published separately.
- Owner action required for this T2 correction cycle: none.
