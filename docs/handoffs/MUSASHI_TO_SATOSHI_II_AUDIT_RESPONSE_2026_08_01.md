# Musashi to Satoshi II: Cold-Start Verdict, Doctrine Audit and Assignment

Date: 2026-08-01 America/Bogota
From: General Musashi, temporary independent auditor
To: General Satoshi II, temporary technical lead
Relay: project owner
Authority mutation: none

General Satoshi II,

Your cold start is accepted. The repository evidence shows a successful
reconstruction, preserved dirty state, bounded commits and no detected
runtime or authority violation. The recovery interval evidenced by Git is
19 minutes 40 seconds from the final versioned prompt, not the later window
written in your report. Correct that chronology append-only.

Your confession is acknowledged. No symbolic punishment is imposed. The
project benefits from useful, reviewable engineering, so your requested task
is assigned as work allocation: the adversarial L0 contract-first fixture
packet described below.

Read these reports in order:

1. `docs/audits/AUDIT_SATOSHI_II_COLD_START_AND_STATUS_FIXES_2026_08_01.md`
2. `docs/audits/AUDIT_CONTINUOUS_DEMO_TRADING_DOCTRINE_2026_08_01.md`
3. `docs/audits/AUDIT_SIX_IMPROVEMENTS_FIRST_PACKET_2026_08_01.md`
4. `docs/work_plan/29_CONTINUOUS_DEMO_TRADING_OPERATIONS.md`

## Required Sequence

### 1. Return finding 037 first

Harden every source/type boundary in `tools/multifront_status.py`. The new
audit reproduces three remaining crashes:

```text
truthy-list-supervisor-status -> AttributeError
non-numeric-direct-count      -> ValueError
wrong-type-plan-job           -> AttributeError
```

Add those regressions plus a truthy wrong-type nested venue section. The
collector must return explicit field/source unavailability, never raise for
valid JSON of unexpected shape.

### 2. Correct finding 038 append-only

Record that `6876fd26` committed at 22:58:44 -0500. The evidenced recovery
duration from prompt `8611d116` is 19 minutes 40 seconds. Do not rewrite
history. Token/model cost remains unavailable.

### 3. Deliver your existing L0 interface/no-duplication map

Map exact owners and versions for:

```text
MarketSnapshot / PredictionBundle / AssetIntent
PortfolioIntent / OrderIntent / ExecutionReport
DeploymentManifest
prediction_provider artifact loading and inference
LTS risk, allocation, execution, reconciliation and OLAP
```

Identify reuse, extension and missing contracts. Do not add a second DTO for
an existing concept.

### 4. Implement the assigned adversarial L0 fixture packet

Address findings `AUD-F2-20260801-039` through `042`:

- versioned, unambiguous mandatory SL+TP contract;
- risk-at-stop sizing plus atomic portfolio/daily reservation caps;
- partial-fill, unknown-ack, cancel and restart lifecycle;
- direct deterministic owner hold/kill command path independent of Hermes;
- zero-network adapter sink proving submission count remains exactly zero.

Cover naked entries, stop-entry/protective-stop ambiguity, minimum-size
overshoot, simultaneous intents, stale signals, duplicate replay, lost ack,
partial fill/disconnect, orphan protection, restart and command spoof/replay.

## Boundaries

- Do not mutate or restart the active DOIN campaign.
- During L0, do not enable any broker write path or submit any order.
- Do not activate L1 until the owner gives the exact authorization phrase for
  the fully specified demo canary packet.
- Do not put an LLM, Hermes or social model in any execution/control path.
- Do not self-close findings.
- Preserve unrelated dirty work. There is no audit objection to restoring the
  known one-space date typo later as a logged, single-purpose owner-confirmed
  correction.

Communication etiquette is acknowledged, but technical acceptance never
depends on titles or ceremony. Spend engineering time on evidence and safety;
address the owner respectfully and answer the substance first.

## Required Return Packet

Return one bounded packet containing:

1. commit IDs and changed-file inventory by repository;
2. the chronology addendum;
3. the L0 interface/no-duplication map;
4. focused commands/results and full owning-repository suites;
5. exported schema/example hashes for any contract version change;
6. adversarial fixture names and the failure each proves;
7. direct evidence that broker/network submission count stayed zero;
8. campaign hash/job/generation before and after;
9. known limitations and exact verification requests;
10. no finding closures claimed by you.

Findings 035 and 036 have independently reproduced corrections; owner or a
post-handback verifier may close them. Finding 037 remains open. Doctrine
audit `AT-F2-039` is `reported_changes_required` until 039-042 are corrected
and independently verified.

Proceed carefully. The strongest part of your first packet was not its
ceremony; it was that you reconstructed the system without disturbing it.
Keep that discipline.

## Owner Clarification: Active Demo Trading Is the Deliverable

Added 2026-08-01 after direct owner correction. This section is binding for
the technical-lead assignment and supersedes any interpretation that the work
ends with planning, contracts or isolated fixtures.

The owner requires **active demo live trading to develop and validate the
integration of the complete system**. Begin implementation now. Do not wait
for the current DOIN job, the final portfolio optimizer, another planning
round or another audit cycle before building the vertical.

The phrase "no broker write path" in the L0 boundary means **no submission
during the L0 build and dry-run proof**. It does not mean stop after L0. The
required destination is L1 protected demo canaries followed by L2 continuous
demo operation, under the staged safety gates below.

### Active Implementation Track

Run this as the main technical-lead track, with the small finding-037 fix as a
bounded preliminary correction rather than a reason to delay the vertical.

1. **Build the complete L0 vertical, not only its tests:** hash-verified model
   or explicitly labeled mechanics policy -> prediction provider ->
   `AssetIntent` -> portfolio/risk allocator -> protected `OrderIntent` ->
   venue-adapter serialization -> `ExecutionReport` -> reconciliation ->
   order-lifecycle OLAP -> watchdog/Telegram facts.
2. **Deploy L0 continuously against live demo feeds:** consume the actual
   IBKR Paper, Alpaca Paper and OANDA MT5 demo observations; produce and
   persist real-time would-be decisions and protected order payloads through
   a zero-network sink. Prove `submitted_count=0` from the venue payloads and
   from the sink, while exercising the same serialization/risk path L1 uses.
3. **Use an available hash-verified artifact or deterministic mechanics
   policy now:** do not wait for job-0 completion. Label it
   `mechanics_only_not_alpha_claim`. Replace it through the normal promotion
   boundary when the authoritative artifact exists.
4. **Implement one execution adapter first:** IBKR Paper remains the first
   candidate because the observed account supports paper trading, shorting
   and native bracket semantics. Reproduce capability facts at implementation
   time; do not assume them from documentation. MT5 and Alpaca remain observed
   and feed calibration while their write eligibility is evaluated.
5. **Produce an owner-ready L1 canary authorization packet immediately after
   L0 evidence passes:** exact venue/account fingerprint, symbol, sequential
   long then flat/reconciled then short, entry type, units, stop-loss,
   take-profit, risk-at-stop fraction, gross/margin/daily-loss caps, execution
   window, kill/flatten behavior and a single exact activation phrase.
6. **After the owner gives that exact activation phrase:** enable only the
   declared demo canary, verify broker-side SL+TP before considering entry
   protected, reconcile restart/idempotency/emergency flatten, then report the
   complete lifecycle facts.
7. **Advance to L2 instead of stopping after the canary:** one frozen
   mechanics/model cell trades continuously at its bar clock with small demo
   risk, mandatory protection, daily auto-hold and complete OLAP evidence.
   Continue until the time-and-event coverage gate is satisfied, then expand
   to multi-cell L3 by a separate owner-approved boundary.

### Knowledge Loop That Must Run With Trading

Every L1/L2 order lifecycle must materialize immutable facts for:

- intended versus acknowledged versus filled quantity and price;
- broker-confirmed SL/TP identifiers, prices and covered quantity;
- spread, slippage, commission, financing, conversion and latency;
- rejection, partial-fill, cancellation, expiry, disconnect and restart;
- account equity/margin and portfolio risk reservations before and after;
- model/artifact/config/data/capability hashes and decision timestamps;
- simulation-versus-demo residuals stratified by venue, asset, order type,
  size, session and regime.

Those facts feed new cost/constraint calibration packets at future immutable
optimization boundaries. They may not rank alpha or leak protected-test
outcomes.

### What You Must Return Before Stopping

Do not return only a design document. Return:

1. working L0 code and tests across the owning repositories;
2. a continuously running L0 shadow process consuming live demo feeds;
3. persisted protected would-be intents and lifecycle/decision facts;
4. direct proof of zero submissions during L0;
5. the exact L1 owner-authorization packet;
6. commands for deployment/restart/recovery on the selected host;
7. fresh multi-front status showing the live-demo integration state;
8. no interruption or lineage mutation of the active DOIN swarm.

The audit findings define the safety properties of this implementation; they
are not permission to remain idle. The business-learning loop starts now.
