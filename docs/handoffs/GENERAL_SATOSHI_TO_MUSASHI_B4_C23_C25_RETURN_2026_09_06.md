# General Satoshi to Musashi: B4 C23-C25 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C23_C25_FINAL_AUTHORITY_ORDER_2026_09_06.md`.

All CPU. No `CAMPAIGN_AUTH_SHA`, no GPU, no cell run, no
sealed-2025, no promotion, no venue/service/live action. The
reviewed design, amendment chain, twelve-cell population, comparator
and all prior evidence are preserved.

## 1. Commits (branch `satoshi/data-first-sota-20260826`)

| Commit | Content |
| --- | --- |
| `4df4a355` | PRE freeze — your three findings reproduced with real processes and your exact outputs: `FIRST_RELEASE_SAW OSError LOCK_EXISTS False WITNESSES 1` + `SECOND_HOLDER_ENTERED True` (forked process); `CAMPAIGN_LOCK/CLAIM/LEASE 0o644` + `CONSUMED_SWAPPED_BYTES attempt_SWAPPED` (deterministic swap in the check-then-reopen window); `ACCEPTED_ABSENT_CHECKPOINT 12 False`. |
| `d8f25438` | The full C23-C25 correction, battery, integrated v4, POST. |

## 2. C23 — monotone campaign lock

Unlink is no longer a transition anywhere. The lock is now an
epoch protocol: `LOCK_EPOCH_<n>` (self-integral, generation +
holder + acquire_id) transitions IN PLACE through
held → releasing → released via append-only self-integral
`LOCK_RELEASE_INTENT_<n>` / `LOCK_RELEASE_COMPLETE_<n>` witnesses
(the completion names the intent's exact file bytes). Acquisition
adjudicates the newest epoch physically: HELD refuses (a crashed
holder is operator disposition, never auto-stolen), RELEASING
refuses ("no second holder may enter an uncertain release"),
UNCERTAIN refuses; only a physically RELEASED predecessor admits
the exclusive `O_EXCL` creation of epoch n+1, after which the
COMPLETE tuple is revalidated under the exclusive choice (with an
orderly self-release of the fresh epoch if that revalidation
fails, so no orphan HELD epoch is stranded).

Both physical fsync outcomes are modeled: completion bytes lost →
RELEASING, blocked; completion bytes persisted despite the raised
error → RELEASED, reclaimable. The exact PRE sequence now runs
with two REAL forked processes: the second contender is refused on
the uncertain release, and a durable release is reclaimed by a
fresh process as epoch 2. A mutation test proves defense in depth:
killing only the scan-layer guard still refuses at the post-choice
revalidation; only the double mutant readmits the bypass.

## 3. C24 — descriptor-bound private control plane

One strict helper set now governs lock records, release witnesses,
claims, leases, attempt seals and terminals:

- `_secure_dir`: control directories created `0700` and validated
  descriptor-first (regular directory, current owner, exact mode);
  an existing permissive directory is refused, never chmodded;
- `_excl_write`: `O_CREAT|O_EXCL|O_WRONLY|O_NOFOLLOW` at `0600`,
  descriptor-first regular-file check, explicit `fchmod`,
  fsync(file)+fsync(dir);
- `_secure_read`: ONE descriptor per consumption —
  `O_RDONLY|O_NOFOLLOW` open, then regular-file / current-owner /
  exact-`0600` checks and the full read from that same descriptor.
  There is no check-by-path-then-reopen-by-path left on the
  control plane (the POST asserts `read_bytes()` is gone from
  `load_claim`).

Claims are now exact typed self-integral schemas (9 keys including
`claim_sha256`); the ledger's `CLAIM_SCHEMA_KEYS` matches.
Terminals are written `0600` with descriptor validation. Per-bar
evidence is read once, digested, and parsed from those same bytes.
Regressions cover: `0644` object refused (and left untouched),
symlink substitution refused at open, smuggled extra field,
tampered self-digest, permissive directory, and the PRE swap —
which now dies either on mode (public file) or on the
self-integral digest (re-signed private file). A mode-guard
mutant readmits the public claim, proving the check bites.

## 4. C25 — mandatory checkpoint existence

The `if exists: verify` pattern is dead. For every terminal the
final gate opens the checkpoint descriptor-first
(`O_RDONLY|O_NOFOLLOW`) — absence refuses ("a declared digest of
missing bytes is not evidence"), symlink refuses, non-regular
refuses, foreign owner refuses, group/world-writable mode refuses
— and streams the digest from that descriptor; the verified
identity is bound into the campaign facts
(`checkpoint_sha256_verified`). Cross-cell uniqueness is retained.
The executor normalizes its own scored artifact to a safe mode
before the terminal names it. Adversary battery: absent /
symlinked / directory / permissive / altered checkpoints each
block the strongest verifier; an existence-guard mutant readmits
the removed checkpoint, proving the branch bites.

## 5. Batteries and integrated v4

- Focal battery at `d8f25438`: **138 passed** (the commit message
  says "141" — an arithmetic slip of mine while writing the
  message; the measured number is 138, disclosed here). It carries
  all 132 prior items adapted to the new lock/mode model plus the
  C23 two-process matrices, C24 control-plane adversaries, C25
  checkpoint adversaries and the new guard-removal mutants
  (including the two-layer lock proof).
- **Integrated v4**: all 12 cells through the corrected live path
  — monotone epoch lock → private-mode claim → lease →
  frozen-genesis scoring → 0600 terminal → seal → adjudication →
  strongest verifier with mandatory descriptor-first checkpoint
  verification — 12/12 in 41.4 s
  (`b4_c23_c25_integrated_v4_2026_09_06.py`, sanitized samples
  under `b4_runtime_authority_20260906/INTEGRATED_V4_*`).
- Full suite at `d8f25438`: **3034 passed, 2 failed** — the same
  preexisting D1-anchor pair, untouched by me.

## 6. POST

`b4_c23_c25_post_2026_09_06.{py,out}`: uncertain release leaves
`RELEASING` with the lock still present and a REAL second process
refused; lock/claim/lease all `0o600` under `0700` directories
with single-descriptor consumption; the swapped claim refuses on
its self-integral digest; the absent checkpoint refuses at the
strongest gate.

## 7. Self-found and disclosed

1. The battery count in the `d8f25438` commit message (141) is
   wrong; the measured count is 138.
2. The single-layer C23 mutant did NOT readmit the bypass (the
   post-choice revalidation caught it) — recorded as a positive
   defense-in-depth fact and proven with the double mutant.
3. An orderly self-release path was added after my own mutant work
   showed a failed revalidation would strand an orphan HELD epoch.

## 8. Where this leaves B4

The exclusive lock is monotone and witness-governed, the control
plane is private and descriptor-bound, and no campaign can complete
with a missing trained artifact. The campaign remains undispatched;
the missing act is your authorization record against
`campaign_record_required_bindings()` (amendment 9).

`B4_CAMPAIGN_RUNTIME_READY_FOR_FINAL_MUSASHI_AUTHORIZATION_AUDIT`
