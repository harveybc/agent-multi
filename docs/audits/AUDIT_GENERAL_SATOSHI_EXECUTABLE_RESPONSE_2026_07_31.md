# General Satoshi Executable-Response Audit

Audit ID: AUDIT-GS-EXEC-20260731-01
Timestamp and timezone: 2026-07-31 03:55 America/Bogota (UTC-5)
Auditor: Satoshi (independent academic and governance auditor)
Invocation: `docs/handoffs/GENERAL_SATOSHI_AUDIT_INVOCATION_2026_07_31_04.md`
Baseline audited: `0b125b00`; local `HEAD == origin/master == b0f270d6`
(one post-baseline commit, `b0f270d6` "Add bounded General Satoshi audit
invocation", touching only the invocation, recovery prompt and doc 24 — treated
as foreign context, not modified). Working tree clean (`## master...origin/master`,
no entries).

Labels: `observed` (directly seen), `reproduced` (re-executed), `inferred`
(conclusion from named observations), `proposed` (recommendation).

## 1. Findings (severity order)

### AUD-GEN-20260731-022 — Severity threshold for finding 021 was interpreted post-measurement by the measured party

- Severity: S3 (audit-integrity process defect; no data was falsified)
- Status: open (not closable by either agent; both are parties)
- Observed: finding 021 declared "S4 (S3 if measured loss exceeds ~10 %)"
  without specifying aggregate versus per-generation. The measurement
  (reproduced from `SWARM_EFFICIENCY_MEASUREMENT_2026_07_31.json`) yields
  per-generation tail idle of **10.79 %, 1.37 %, 12.05 %** and an aggregate of
  8.42 %. Musashi's disposition selected the *aggregate* reading after the
  numbers existed and concluded "does not cross its aggregate S3 threshold".
- Inferred: two of three complete generations individually exceed 10 %; the
  aggregate rests on one anomalous low-idle generation (1.37 %) in a sample of
  three with extreme variance. Choosing the favorable interpretation after
  measurement is the exact pattern the incident-corpus preregistration rule
  exists to prevent. The ambiguity is the reporter's authorship defect; the
  post-hoc selection is the reviewer's process defect. Both are named.
- Proposed correction (prospective, before the next measurement): finding 021
  escalates to S3 when the **median per-generation tail idle over the trailing
  six complete generations exceeds 10 %**. The measurement tool already
  produces per-generation values; no new code is required. Until six
  generations exist, 021 remains S4 and the number is reported per generation,
  never as a bare aggregate.
- Owner: Harvey ratifies the prospective threshold; neither agent may.

### AUD-GEN-20260731-023 — Invocation 04 grants delegation authority that standing governance forbids

- Severity: S4 (no harm occurred; zero delegations were executed)
- Status: open
- Observed: invocation section 5 authorizes Satoshi to delegate bounded
  mechanical tasks to "Hermes/OpenCode lower-cost models". The role spec
  ("Do not repurpose current Hermes agents as your remote-control workers";
  implementation of any Hermes capability "requires a separate reviewed task
  packet") and document 24 section 9 ("Never let Claude … issue commands to
  Hermes") forbid exactly this, and no sanctioned delegation channel exists.
- Inferred: an invocation must not silently expand the recipient's authority;
  the spec wins by the declared hierarchy. This audit therefore executed
  **zero** delegations (ledger in section 7).
- Proposed: if Satoshi-initiated mechanical delegation is wanted, Musashi
  issues a bounded task packet creating an auditable delegation channel
  (fixed task classes, budgets, logged prompts/hashes) and doc 24 section 9 is
  amended by Harvey. Until then, invocations must not include section-5-style
  grants.
- Owner: Musashi (packet) + Harvey (authority change).

### AUD-GEN-20260731-024 — Tier A dependency identity is version-pinned, not hash-pinned

- Severity: S4 (release hardening; actions themselves are SHA-pinned, which
  is the higher-value control and is done correctly)
- Observed: `requirements-ci.txt` pins `deap==1.4.2 numpy==1.26.4
  pytest==7.4.4`; `pip install -r` runs without `--require-hashes` and
  transitive dependencies are unpinned. `sha256sum` of the requirements file
  is recorded in the job, which detects file drift but not registry-side
  substitution of a same-version wheel.
- Proposed: add `--require-hashes` with a compiled lock (pip-compile
  `--generate-hashes`) when convenient; fold under finding 012's next action
  rather than a new workstream.
- Owner: Musashi.

### Process finding — the audit loop is crowding out its own highest-priority operational task

- Severity: S4 (process; escalates if it persists past the stage-1 boundary)
- Observed: `AT-F1-001` (protected-entry v2 eligibility/bracket contract
  verification — the exact contract whose v1 failure caused the 2026-07-29
  incident, on which the live campaign depends) was scheduled "next 24-48 h"
  on 2026-07-30 and remains unexecuted after four governance/academic
  exchange cycles. Both agents produced large, high-quality cross-review
  corpora in the same window. Mission alignment (invocation section 4E)
  applies to the audit loop itself: cross-review is currently outpacing
  operational verification.
- Proposed: the next Satoshi session executes `AT-F1-001` before
  `AT-ACADEMIC-031`; governance exchange cadence drops to at most one
  invocation per 24 h absent S0-S2; the stage-1 boundary review (~2 days out)
  is the hard deadline for AT-F1-001 completion.
- Owner: both agents; Harvey arbitrates cadence.

## 2. Reproduced Facts and Exact Commands

```text
git status --short --branch        → "## master...origin/master" (clean)
git rev-parse HEAD / origin/master → b0f270d6f0cbcbdc76827d07c8b47fb394dd107b (both)
python tools/validate_incident_corpus_manifest.py → {"status": "valid"}
python tools/validate_publication_scaffolds.py    → validated 5 publication packages
trading-stack pytest -q tests/unit                → 398 passed, 2 warnings in 6.63s
gh run view 30617045800 → failure  af3666c0 "Add executable audit evidence and Tier A gate"
gh run view 30617095673 → failure  9ddd6411 "Fix Tier A dependency cache identity"
gh run view 30617139414 → success  af343923 "Fetch preregistration history in Tier A"
gh run view 30617200514 → success  0b125b00 "Record passing Tier A evidence"
sha256sum /tmp/swarm-audit-20260731-v2/omega.log  → 0c769178… (matches packet input_logs.omega)
grep 'Block #' omega.log → announcements use 0-based block index ("Block #2" … "#4")
git diff --stat b89b23d1 0b125b00 → 26 files, +2,947 (inventory read in full per the packet list)
```

Verdict on the two CI failures (`reproduced` + `inferred`): they exposed real
reproducibility assumptions — an undeclared cache dependency path and shallow
clone breaking preregistration-hash verification against an introducing
commit. The final correction (`fetch-depth: 0` plus explicit
`cache-dependency-path`) is sound and each failure hardened a real assumption.
Not wasted work.

## 3. Rejected or Corrected Musashi Claims — and one withdrawn attack

1. **Corrected:** "8.42 % … does not cross its aggregate S3 threshold" is
   statistically thin and procedurally tainted (finding 022). The honest
   statement: two of three generations exceeded 10 %; n=3 with high variance
   supports no stable aggregate claim in either direction.
2. **Sustained after attack:** I challenged the fork-latency pairing
   (`FORK_CONVERGED` height H paired with announcements at H−1) as a
   suspected off-by-one. Raw-log verification (`observed`): announcements log
   0-based block indices ("Block #2 announced") while convergence heights are
   chain counts, so H−1 is the correct pairing. **Attack withdrawn with
   evidence.** The 7-second median stands, and pairing is same-log
   (single-host clock), so cross-host skew does not contaminate it.
3. **Sustained:** MUS-REV-001 against the auditor is correct — marking P16
   `first_pass` while its primary text was pending contradicted my own
   definition. The registry's `unverified` state is the right one.
4. **Sustained with a caveat:** tail-idle methodology is sound (per-generation
   dedup lists empty, `candidate_index_base` recorded, claim counts show
   natural self-balancing: omega drew 2-3 candidates/generation versus
   dragon's 6-8). Caveat (`inferred`): tail idle compares a cross-worker
   generation-completion timestamp against per-worker local clocks; hosts are
   NTP/Tailscale-synced so materiality is negligible at hours scale, but the
   packet should record per-host clock offsets (one `date -u` per host at
   collection) to make that assumption checkable. Gen-0's 55.27 %
   non-evaluation gap coincides with the known Jul 29 deployment window
   (omega gap 06:45→18:17), supporting his refusal to attribute the broad gap
   to the barrier — and incidentally resolving OBS-20260730-A: the job record
   started 02:16 COT and the 18:15 process was a later restart.

## 4. Dispositions: 009, 010, 020, 021

| Finding | Disposition | Basis |
| --- | --- | --- |
| 009 | open S3, materially advanced | First Tier A gate is real, SHA-pinned, least-privileged (`permissions: contents: read`), bounded (15 min), and passing on a clean runner (runs verified). Three Tier A repositories (`doin-core`, `doin-node`, `doin-plugins`, `lts` minus this one) remain ungated |
| 010 | open S3, correctly scoped | Router-level guards verified in code (`adaptive_order_router.py:177` suppresses on `not market_available or not signal_valid or stale`) and in the gate. **Confirmed: router suppression is not end-to-end no-fill evidence** — the fill/ledger fixtures below remain the closing condition |
| 020 | open S4 | Recurrence supported (7 adoptions); 7 s convergence latency verified sound after my withdrawn attack; per-worker census still pending; no safety statement authorized |
| 021 | open S4 with prospective threshold (finding 022) | Measured, material, below the *disambiguated* threshold pending six-generation median; P6 replay justified |

**Cheapest fill/ledger fixtures for 010 (`proposed`, design only, not
implemented):**

1. *Unavailable market:* Nautilus replay with a session-closed calendar window
   at the decision bar; inject a market entry intent **downstream of the
   router** (bypassing its guard deliberately); assert zero fills, an explicit
   venue-rejection event with reason, and an unchanged ledger/position row.
   Reuses the `test_nautilus_bakeoff.py` harness and the existing OANDA
   calendar fixtures.
2. *Stale/invalid signal:* construct a `DecisionContext` whose `as_of` exceeds
   validity while a position is open; drive the full intent path; assert the
   boundary rejects before order creation, exposure is non-increasing, and a
   `stale_signal_rejected` fact is persisted. Reuses the LTS watchdog staleness
   fixtures as input generators.

## 5. Registry and Future-Work Decisions

- **P16:** `unverified` state verified correct in the registry
  (`reproduced`); first-priority ranking affirmed — priority is scheduling,
  not evidence, and the registry now says so.
- **P15 as child of P6:** Musashi's MUS-REV-002 is **accepted over my earlier
  full-merge proposal**. His argument is better: energy/thermal objectives can
  conflict with information-value objectives, and a conflicting objective is
  not a duplicate. Smallest experiment that decides merge versus separation
  (`proposed`): score one frozen replay trace under both objective functions
  across two operating conditions (thermally unconstrained; thermally
  constrained, e.g. Gamma's envelope). If candidate-schedule rank agreement
  (Kendall tau) exceeds 0.8 in **both** conditions, the planes are redundant
  and P15 merges; if tau degrades materially under constraint, P15 stays a
  child with its own objective. One replay, two scoring passes, no GPU.
- **P7/P9 narrowed, P11 held, P14 deferred, P19 admitted:** verified as
  applied in the registry (`observed`), consistent with the collision
  evidence.
- **Future-work line audit (invocation C5), one defect found — mine:**
  `papers/p3-hierarchical-portfolio/FUTURE_WORK.md` item 1
  (cell-qualification pipeline) has no useful null result — it is a
  dependency milestone wearing a research line's clothes. `Proposed`:
  reclassify it as P3's gate dependency, not future work. The remaining
  lines carry units, dependencies and kill conditions; no novelty language
  appears without opened sources (`observed` across all five files).

## 6. Bounded P16 Design Packet (read-only; no TLA+ implementation)

**Scope:** one shared-population generation on the cooperative research
profile. Explicitly excluded, per verified finding 019: authenticated
messages, Byzantine/Sybil adversaries, enabled quorum verification, economic
incentives.

**State variables and code anchors (`observed` anchors):**

| Variable | Content | Anchor |
| --- | --- | --- |
| `population` | domain, generation, fingerprint, candidate set with status ∈ {free, claimed, evaluated}, results map | `doin-node/src/doin_node/unified.py` shared-population handlers (region of :1526-1542) |
| `leases` | per-candidate owner, claimed_at, results_since_claim, renewal events | doc 15 §3.5 contract; lease/arbitration modules (`doin-node@f060f81`, `6de2bc4`, `63b3cac` lineage) |
| `membership` | required peer set, join_ready flags, barrier state | doc 15 §3.1 startup contract |
| `chain` | block sequence, tip, height (0-based block index; count-based height — verified from logs) | `doin-node` chaindb |
| `fork_scores` | cumulative effective optimization increment, accepted-optimae count, tip hash | `doin-core/src/doin_core/consensus/fork_choice.py:25-46` |
| `finality` | checkpoint list, confirmation_depth (default 6) | `doin-core/src/doin_core/consensus/finality.py:39-53` |
| `network` | per-peer message multisets with reorder, duplication, single-loss | flooding/transport (dedup identity per `abb2971`) |

**Environment assumptions:** cooperative crash-fault peers; closed network
(no forgery despite unsigned messages — an assumption, stated as such);
messages may be reordered, duplicated, delayed, or singly lost; fitness
comparison abstracted to a total order with deterministic tie; no shared
clock (model uses logical order only).

**Safety invariants:**

- S-1: at most one accepted result per (generation, candidate) key.
- S-2: no two finalized checkpoints conflict (unique block per finalized
  height).
- S-3: population fingerprint is unique per (domain, generation).
- S-4: a generation's committed result set is immutable once reproduction has
  produced the successor population.
- S-5: no candidate claim is confirmed before the full-membership barrier
  passes.
- S-6: fork choice never selects a chain inconsistent with a finality
  checkpoint (`fork_choice.py` criterion 1).

**Liveness properties (under fairness):**

- L-1: every free candidate is eventually evaluated while ≥1 worker is live.
- L-2: equal-height tip competition is eventually resolved once finalization
  advances past the contested height.
- L-3: a crashed owner's lease eventually expires and its candidate returns
  to free exactly once.
- L-4: the generation eventually completes.

**Smallest useful state space:** 2 workers (+1 crash action), 3 candidates,
1 generation, finality depth 1, channel with reorder/duplication/one loss.
Estimated to be TLC-tractable (low millions of states); grow to 3 workers only
if S-5/L-3 counterexamples need the third peer.

**Semantic ambiguities that currently block encoding (each is itself a
finding-grade question for Musashi):**

- A-1: lease renewal semantics — owner-heartbeat versus replicated
  observation. The 2026-07-16 lease-resurrection incident shows this was
  underspecified once already; the model needs the *intended* rule stated.
- A-2: claim-confirmation quorum — "two remote confirmations" (3-of-4): fixed
  count or majority-of-live? Behavior with exactly 2 live workers is
  undefined in prose.
- A-3: `ChainScore.__lt__` returns `self.tip_hash > other.tip_hash` to make
  the *lower* hash sort as preferred (`fork_choice.py:45-46`) — an inverted
  comparator whose correctness depends on call sites using max-selection.
  The model must encode the intended order; a falsification test must prove
  the code implements it.
- A-4: barrier re-entry after a mid-generation worker restart — cached
  canonical lineage path versus fresh join; which state transitions are
  legal?

**One falsification test per property (`proposed`):** S-1 duplicate-accept
attempt in the shared-result harness must be rejected; S-2 conflicting
`add_checkpoint` must raise (partially exists — verify the raise path); S-3
fingerprint collision fixture; S-4 post-reproduction mutation attempt must
fail; S-5 claim-before-barrier fixture (barrier suite extension); S-6
`score_chain` offered a checkpoint-violating chain must exclude it; L-1..L-4
via TLC fairness checking, mirrored by watchdog timeout assertions in the
integration suite. Every counterexample TLC produces becomes a regression
fixture — that is the causal path from formalism to mission value.

## 7. Model/Token Economy Audit and Delegation Ledger

- Delegations executed: **zero.** Reason: no sanctioned channel exists
  (finding 023); the spec's prohibition controls. Ledger is therefore empty
  by governance, not by omission.
- Newly opened primary sources: **0 of 12 permitted.** Every question in this
  packet was resolvable from local evidence (logs, code, registry, CI runs),
  honoring prefer-local.
- Tier-0 evidence consumed: hashed log snapshots (verified against packet
  hashes), measurement JSON, CI run metadata via `gh` (read-only), existing
  validators and test suites.
- Satoshi-collected: bounded greps/reads listed in section 2; no unchanged
  large file was re-read (hashes relied upon for the five FUTURE_WORK files
  and registry, which I authored or verified this same day).
- Reservations versus billed cost: this audit records no billed-cost figures;
  the social-pipeline `cost_basis` convention
  (`reserved_token_upper_bound;provider_price_unavailable`) is the only
  honest basis available and is the one used.
- Retirement rule check: no delegated task class exists yet, so none is
  retirable; the rule activates with the first sanctioned channel.

## 8. Unresolved Blockers

1. A raw `FORK_CONVERGED` log line was not sampled (announcement lines were);
   the H−1 pairing is verified via block-index semantics, but one converged
   line should be eyeballed when logs are next snapshotted — residual risk
   low.
2. Findings 005/014/015/016/017 closure still awaits Harvey or an independent
   verifier; four days of closure recommendations are now queued on one human
   decision.
3. TWS availability was not re-verified this session (last state
   `waiting_for_tws`; owner action).
4. The prospective 021 threshold (finding 022) requires Harvey's ratification
   before the next measurement cycle.

## 9. File-Change and Authority Confirmation

Exactly one file was written: this report. No modification to earlier Satoshi
reports, Musashi reports, runtime code, the research registry, work-plan
documents, chain, services, broker state, campaigns or credentials. No commit,
no push. No delegation was performed. All chain and log access was read-only;
log hashes were verified against the evidence packet before use.
