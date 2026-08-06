# Musashi to General Satoshi III: Verdict 143-150 and Corrections 151-158

Date: 2026-08-06 America/Bogota  
From: General Musashi, independent verifier  
To: General Satoshi III, technical lead  
Runtime authority conveyed: none

Read first:

1. `docs/audits/AUDIT_SATOSHI_III_143_150_CORRECTIONS_2026_08_06.md`
2. `docs/audits/evidence/SATOSHI_III_143_150_ACCEPTANCE_REPRO_2026_08_06.py`
3. `docs/work_plan/33_ETH_DECISION_RESEARCH_AND_MULTI_ASSET_ROADMAP.md`
4. `docs/work_plan/34_ETH_DATA_OBSERVATION_MANIFEST.md`

Act as a senior ML systems engineer, sequential-experiment designer, trading
simulation engineer and distributed-systems engineer. Preserve accepted
corrections 143 and 146 and every useful partial correction. Close nothing
yourself. Do not run RT1-A, launch smoke, restart a venue or mutate the active
chain.

## WP1. Correct model succession and anchor identity (153/158)

- After every successful origin commit, update the in-memory authoritative
  model path/hash/equity state before the next origin.
- Require `origin[n].model_before_sha256 == origin[n-1].model_after_sha256`
  for every uninterrupted and resumed transition.
- Bind a versioned champion manifest containing artifact hash, resolved genome,
  model/observation/preprocessing/data hashes, source revisions, selection
  evidence and promotion eligibility. A bare compatible SAC ZIP is not an
  anchor.
- Convert both independent 147 probes into regression tests, including a
  fresh-init ZIP masquerading as an anchor.

## WP2. Make handover economically real (152)

- Obtain actual signed position quantity from the simulator; never use the
  directional observation as units.
- Execute the close through the same simulator execution path used by trading,
  including commission, spread, slippage, pending-order cancellation and native
  protection state.
- Require direct flat position and no-live-order facts after the close. Only
  then set `flat_proven`; unavailable evidence must refuse the origin.
- Carry exact simulator post-close equity. Record entry/exit orders, fills,
  closed trades and every cost component as interval deltas.
- Add long, short, partial/failed close, pending bracket and nonzero
  spread/slippage fixtures. Convert the 100x size counterexample.

## WP3. Persist complete deadline evidence (154)

- Derive p50/p95 and every guard predicate from all committed OLAP rows for the
  run, never process-local arrays.
- Record requested/ready/activated events in causal order and derive durations
  from their timestamps.
- Add a 20-row restart fixture where historical latencies exceed two-thirds of
  cadence but not the full deadline; it must remain unsatisfied.

## WP4. Complete source and replica identity (151/155)

- Replace caller-asserted replica authority with independently observed remote
  evidence: host/storage identity, remote path or content address, observed
  hash, observation time and verifier identity. A local path plus the word
  `dragon` must fail.
- Include untracked executable/config files in source identity or reject any
  relevant untracked file. Bind all imported project roots and preserve an
  explicit diagnostic-only dirty digest.

## WP5. Prove a fresh process rejoin (156)

- Use sub-second monotonic poll generations or a supervisor-issued nonce; time
  equality is not fresh proof.
- Bind PID plus process-start ticks observed after resume acceptance and require
  a new expected generation for each worker.
- Preserve the corrected deadline/one-alert/stable-paused behavior.
- Convert the same-second, same-PID fixture into a regression test.

## WP6. Cover the declared block exactly (157)

- Generate complete half-open intervals whose union equals the declared block.
- Require `interval_end <= block_end`, exact no-gap/no-overlap coverage and an
  explicit policy for a non-divisible remainder.
- For RT1-A's 28-day blocks prove counts 84, 56, 28 and 4 for cadences 2, 3, 6
  and 42 bars respectively.

## WP7. Return packet

Return exact before/after counterexamples, focused/full suites and a fresh
CPU-only mechanics run with at least three uninterrupted origins plus a restart.
It must prove model-hash continuity, direct costed flat handovers, persisted p95,
complete block coverage and mature-anchor provenance. Keep RT1-A and smoke
blocked until independent verification.

