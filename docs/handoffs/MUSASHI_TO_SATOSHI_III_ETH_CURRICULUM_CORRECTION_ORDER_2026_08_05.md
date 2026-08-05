# Correction Order: ETH Curriculum Campaign and Champion-to-Live Vertical

Date: 2026-08-05 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III (Mujuro Utsutsu), technical lead
Authority: owner-approved ETH-first, business-reality-first program
Runtime state: ETH campaign stopped and disabled; Paper/Demo services untouched

Act as a senior machine-learning scientist, evolutionary-computation engineer,
distributed-systems engineer and Paper/Demo trading-infrastructure engineer.
Read the owner order, your delivery and the audit in that order:

1. `docs/handoffs/MUSASHI_TO_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_ORDER_2026_08_05.md`
2. `docs/handoffs/SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_DELIVERY_2026_08_05.md`
3. `docs/audits/AUDIT_SATOSHI_III_ETH_CHAMPION_STAGE_CURRICULUM_2026_08_05.md`
4. `docs/audits/evidence/SATOSHI_III_ETH_CURRICULUM_REPRO_2026_08_05.py`

Do not defend the delivery in prose. Reproduce every finding before editing,
correct root causes, add adversarial tests, and return one bounded evidence
packet. You do not close findings you implement.

## 1. Non-Negotiable Boundaries

- Keep all three campaign supervisors disabled until Musashi accepts the smoke
  campaign. Do not resume the contaminated chain.
- Preserve the invalid-run archive and old USDCAD artifacts byte-for-byte.
- No Live-capital orders. Existing Paper/Demo processes may continue.
- No manual or linear decision may be labelled SAC/champion inference.
- Protected orders require native SL and TP in the initial request.
- Do not inspect the disclosed 2025 test outcomes for candidate, arm, seed,
  threshold or stage decisions.
- Do not change DOIN's decentralized architecture. Repair its existing plugin,
  shared-pool and fork contracts at their actual ownership boundaries.

## 2. WP1 — Repair Config Identity and Artifact Ownership (108, 110)

Create corrected ETH-N and ETH-EN configs from one typed materializer. Remove
every USDCAD name/path and fail materialization if an asset token from another
domain appears anywhere in experiment identity, resume, champion, statistics,
history or handoff paths.

Required assertions:

- runtime `selection_metric == lexicographic_weekly_v1`;
- runtime `optimization_metric` resolves to an implemented objective;
- experiment, domain, artifact and handoff identities all say ETH;
- every optimizer output is below the matching ETH arm root;
- ETH-N and ETH-EN differ only in declared arm/curriculum/artifact identity;
- no arm can write into the other arm or historical USDCAD roots.

## 3. WP2 — Make Lexicographic Selection Authoritative End to End (108, 112)

Do not replace the requested tuple with another weighted sum. Implement one
authoritative comparison contract for:

1. normal-phase checkpoint/early stopping;
2. local DEAP selection and reproduction;
3. shared-population generation selection;
4. local/remote champion comparison and migration;
5. blockchain acceptance and archive selection.

The ordered components are:

1. eligibility/simulator evidence;
2. minimum activity, without a positive-profit gate;
3. mean weekly net simple return, higher is better;
4. lower maximum drawdown;
5. higher total net return.

Transmit and persist the full ordered tuple plus named components. If an
explicit comparison tolerance or quantization is required, preregister it,
bound every component, and property-test order preservation. A display or
wire scalar may exist, but no authoritative branch may compare it instead of
the tuple. Add the audit counterexample verbatim and property tests over finite
component ranges.

## 4. WP3 — Failures and Genome Validity (109, 113)

- Map `evaluation_error`, `_eval_error`, simulator errors, missing artifacts,
  non-finite metrics and objective-resolution errors to one rejected-result
  schema at the optimizer boundary.
- Make DOIN independently reject that schema even if a plugin forgets one
  convenience flag. A finite worst sentinel is never champion-eligible.
- A generation with zero eligible candidates must abort observably; it must not
  mint a champion block or silently evolve error sentinels.
- Remove `preprocessing_mode=none` unless a content-hashed causal precomputed
  feature contract makes it executable. Add deterministic repair/validation
  for every conditional genome combination before GPU training begins.
- Add tests proving a failed first candidate cannot become the initial
  champion or create an accepted transaction.

## 5. WP4 — Fork Convergence and Real Pause (111, 115)

Reproduce the observed four-worker race: simultaneous candidate completions
from one shared generation create competing equal-height blocks while peers
may roll back during range fetch.

Correct:

- empty/short peer branch fetches without indexing an empty list;
- deterministic tie breaking independent of local insertion order;
- bounded retry after a peer changes tip during fetch;
- eventual convergence to one tip and one finalized anchor;
- no duplicate candidate evaluation or accepted transaction during repair.

Implement a single operator pause command/API. It must tell every supervisor to
stop claiming, stop every owned worker process group, wait a bounded graceful
interval, escalate visibly if required, verify ports/processes/GPU owners are
gone, and persist a coherent snapshot. `systemctl inactive` while workers live
is a failed pause. Test service-stop, process-group and restart behavior.

## 6. WP5 — Restore Experimental Evidence Discipline (114, 116)

- Change the N/E/EN mechanism fixture to train/train-tail/validation only.
- Mark 2025 as disclosed for curriculum work; do not call it pristine again.
- Preserve fixture JSON, config, model hashes and return traces under a stable
  result directory with a manifest hash committed or content-addressed.
- Fix the LTS rolling-report test by injecting deterministic event timestamps;
  reproduce the complete LTS suite after 12:00 UTC.
- Report raw weekly values and same-scale percentages. No composite metric may
  be presented as profit or return.

## 7. WP6 — Current-Stack Champion to Actual Demo Trading (Gates 7-9)

The existing `LiveSacPolicy` module is scaffolding, not a live vertical. Build
the missing causal observation provider from raw, timestamped venue data. It
must reproduce the exact current training feature names, order, warm-up,
windowing, scaling state and action mapping.

Acceptance sequence:

1. Produce the first valid current-stack ETH SAC artifact from a corrected
   local smoke candidate; this is a temporary Demo challenger, not the final
   DOIN champion.
2. Generate a two-source golden parity packet: training observation builder
   versus live raw-bar builder at the same timestamps. Replaying already-built
   observation vectors is insufficient.
3. Wire `SelectedSacPolicy` into the real MT5 ETHUSD H4 runner. Prove the
   production runner calls it; a standalone class and unit test do not count.
4. Use Alpaca only if the available instrument and data semantics are exactly
   the trained ETH contract. Record unavailable rather than use a proxy.
5. Execute at least one model-originated MT5 Demo decision through intent,
   mandatory native SL+TP, broker acknowledgement, monitoring and terminal or
   still-protected state. Record model/config/input/decision hashes.
6. Keep the incumbent selector hot-swappable: a new champion begins from the
   post-close balance; no concurrent old/new authority and no forced close
   solely for model rotation.

## 8. WP7 — Smoke Before Full Swarm

Use a new smoke plan/domain/genesis. Minimum packet:

- one valid local ETH-EN candidate reaches normal validation and returns the
  ordered tuple without an objective error;
- one deliberately invalid candidate is rejected before training and cannot
  create a block;
- a four-worker population of at least four candidates completes one
  generation with distinct claims and one converged tip;
- pause leaves all supervisors and worker process groups absent;
- artifact roots contain only ETH files and the archived invalid chain is
  unchanged;
- status exposes `easy` then `normal` for each active candidate;
- focused and complete suites pass.

Only after Musashi independently reproduces this packet may you request
permission to enable the full ETH-EN campaign. ETH-N remains queued behind it
with the same corrected comparison contract and no champion migration between
arms.

## 9. Delivery Format

Return:

1. finding-by-finding reproduction and correction table for 108-116;
2. exact commits per repository and clean/pushed status;
3. commands and outputs for focused/property/full suites;
4. invalid-chain archive hashes before and after;
5. smoke plan/domain/config/data hashes and four direct worker snapshots;
6. one canonical chain tip/finalized anchor and candidate ownership table;
7. artifact manifest with model/config/genome/raw-metric hashes;
8. production call-path evidence for SAC inference and the protected MT5 Demo
   decision, if Gate 9 is reached;
9. remaining doubts stated directly.

Do not start the full swarm as part of the delivery. Request independent audit
after the smoke packet is complete.
