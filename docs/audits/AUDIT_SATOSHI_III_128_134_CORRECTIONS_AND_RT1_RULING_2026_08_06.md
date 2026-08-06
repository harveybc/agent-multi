# Audit: Corrections 128-134 and RT1 Design Ruling

Date: 2026-08-06 America/Bogota
Auditor: General Musashi, independent verifier
Delivery:
`docs/handoffs/SATOSHI_III_J0_J4_RT_CORRECTION_PACKET_2026_08_06.md`
Reviewed commits: `agent-multi@4ab460e1`, `lts@4738bc5`
Runtime mutation by this audit: none
Broker submissions: zero

## 1. Verdict

The packet is **not accepted yet**. It contains useful corrections, and the
real pipeline now does produce distinct best and terminal SAC files, but eight
contract gaps still defeat the claims needed for campaign rejoin, curriculum
promotion, exact live-controller inventory and rolling-origin adaptation.

The active `full-v2` campaign remains running and untouched. Corrections 123,
124 and 127 still require the bounded runtime smoke already ordered. No N/EN/E
decision campaign or RT1 performance sweep is authorized by this audit.

Canonical independent reproducer:

```text
docs/audits/evidence/SATOSHI_III_128_134_CORRECTION_REPRO_2026_08_06.py
```

It reports `all_counterexamples_reproduced=true`, `network_used=false` and
`runtime_mutated=false`.

## 2. Findings

### AUD-F1-20260806-135 - S2 - rejoin does not prove code or tip ancestry

The pause binding records component revisions, semantic domain hash and
pre-pause worker tips, but `request_resume()` compares only plan/profile and
`verify_rejoin()` compares only domain, genesis and generation-zero population
fingerprint. A worker with changed component revisions and an unrelated tip
sharing those three initial identifiers returns `rejoin_proven=true`.

Evidence: `app/campaign_supervisor.py:3267`, `:3465`, `:3536`; reproducer
`inexact_rejoin`.

Correction 128 is partial. Rejoin needs current component/domain equality and
proof that every observed tip descends from the bound tip or finalized anchor.

### AUD-F1-20260806-136 - S2 - exact-id reuse and terminal artifacts are incomplete

`run_arm()` reuses a matching execution id when only `splits_raw` is truthy.
A record with no artifacts, traces, config body or replica is returned as
complete. New records evaluate the terminal weights, but the arm artifact map
contains only `final`/`post_easy`; terminal evaluation stores a hash without a
retrieval path. The reported real-pipeline proof lives only under a temporary
Claude scratch directory and is not a durable decision record.

Evidence: `tools/eth_curriculum_decision_experiment.py:213-233`, `:326-343`,
reproducer `incomplete_exact_reuse` and `terminal_reference_gap`.

Corrections 129 and 130 are partial. Best and terminal must both be typed,
hashed, retrievable, replicated and load-proven in the arm record. Reuse must
run the same complete-record validator used before aggregation.

### AUD-F1-20260806-137 - S2 - duplicate and identity-empty packets can promote

Packet discovery assigns `seeds[packet["seed"]] = packet`; a second physical
packet for the same seed silently overwrites the first. Common identity is only
tested for difference, so four packets whose data, base and lineage are all
empty pass. The reproducer supplies five physical packets, one duplicate seed
and no identity fields; aggregation exits zero with
`promotion_eligible=true`.

Per-arm execution lineage is also not persisted. A long sequential seed run
can therefore cross a code revision while the packet records only the lineage
observed after every arm has finished.

Evidence: `tools/aggregate_curriculum_decision.py:37-52`, `:118-123`;
`tools/eth_curriculum_decision_experiment.py:421-445`; reproducer
`duplicate_seed_empty_identity_promotion`.

Correction 131 is partial. Reject duplicate physical seeds, empty/invalid
hashes and any per-arm lineage drift; bind packet lineage to each execution
record and assert it did not change before/after the arm.

### AUD-F1-20260806-138 - S3 - repair validation still fails open

`validate_repair_rules()` validates the target only when
`mixed_genome_schema` happens to be a list. A missing schema is accepted. It
also accepts a forbidden value that is not a member of the categorical domain,
making a typo a valid but inert rule.

Evidence: `optimizer_plugins/project3_full_genome_optimizer.py:405-467`;
reproducer `repair_validation_fail_open`.

Correction 132 is partial. Require a typed schema, unique choices and declared
membership of the forbidden value before accepting a rule.

### AUD-F2-20260806-139 - S2 - J4 can grant authority to stale mismatched control

J4 collects config/input hashes but neither loads them from manifests nor
compares them. `_join_manifest()` grants SAC authority from artifact hash plus
`live_execution_eligible` alone. It ignores heartbeat freshness, model id,
`live_inference_eligible`, observation parity and unit state. The reproducer
uses a stale heartbeat with mismatched model/config/input and failed inference
and parity eligibility; J4 still returns authority `true`.

Evidence: `lts/tools/controller_inventory.py:74-111`, `:114-141`, `:145-189`;
reproducer `incomplete_authority_join`.

Correction 133 is partial. Authority requires a fresh running seat and exact
model/artifact/config/input/feature/preprocessing/manifest hashes plus all
declared eligibility predicates. Missing facts are unavailable, never true.

### AUD-F1-20260806-140 - S2 - RT0/RT1 scores the wrong interval and resets the account

The evaluation CSV contains 256 warm-up bars plus the next interval, while
`_score_interval()` scores every environment step. In the independent fixture,
the intended next interval gains 10%, but the runner reports -12% because it
includes warm-up history.

The supplied RT0 OLAP independently shows `equity_before=10000` at both origins.
The runner creates a new environment from the base config every interval and
does not carry the prior after-close balance, protected exposure or handover
cost. It therefore cannot estimate the deployment contract the business uses.

Evidence: `tools/rolling_origin_adaptation.py:122-152`, `:245-254`; reproducer
`warmup_in_interval_score`; supplied RT0 rows 0 and 1.

Correction 134 is not accepted. Exclude warm-up from metrics and preserve
within-block account/effect continuity. Reset is legal only at a declared
independent block boundary.

### AUD-F1-20260806-141 - S2 - RT restart and execution identity are not reproducible

RT identity omits `initial_steps`, device, base-config hash, code revisions and
model/feature contract. A code or initialization change can reuse the same
run id. One mutable `incumbent.zip` is overwritten before the OLAP transaction;
a crash between those operations causes an unrecorded update to be applied
again on restart. Existing rows are skipped without proving the incumbent hash
matches their after-state.

The supplied dry-run summary itself has `p50=15.4778 s` and `p95=6.4517 s`.
The percentile code was changed after that run without changing
`RUNNER_VERSION`, so the same output directory remains reusable under changed
semantics.

Evidence: `tools/rolling_origin_adaptation.py:46`, `:174-188`, `:209-240`,
`:293-320`; reproducer `rt_identity_and_split_collision`.

Correction 134 is not accepted. Store immutable before/after checkpoints per
origin and commit an atomic state pointer with the OLAP row. Bind every
decision-bearing input and code revision into a bumped runner identity.

### AUD-F1-20260806-142 - S3 - the executable decision config still contradicts the manifest

Document 34 says the dormant year-count fields were removed, but the actual
N/EN/E runner loads the old base and `_base_config()` leaves
`train_years=4` and `test_years=1` beside explicit dates covering about 6.25
training years. The materializer removes them; the decision runner does not.

Evidence: `tools/eth_curriculum_decision_experiment.py:126-143`; reproducer
`rt_identity_and_split_collision`.

Remove the shorthand in the executable runner and hash the resolved data and
observation manifest into each arm and RT execution identity.

## 3. What Was Verified

- The RL pipeline saves terminal weights before reloading best weights.
- The supplied real pipeline created distinct loadable SAC files:
  `best=373e050ce805...`, `terminal=67f97325b016...`.
- Current source topology correctly locates MT5 on Dragon and reports
  unreachable remote evidence as unavailable.
- Seeded repair selection is ordering-invariant after validation succeeds.
- Explicit ETH rows remain 13,699 train, 2,196 validation and 2,190 disclosed
  test bars; 83 features, 32-bar observation, 256-bar scaler and 2,724 input
  values are accurately labeled as used but not evidence-selected.

These are retained; the correction order is limited to the remaining gaps.

## 4. RT1 Reduced-Factorial Ruling

Satoshi's exact proposal `{3,6,42} x {1y,expanding}` is **not accepted**: it
omits the 8-hour boundary that motivates the business question and leaves a
large 24-hour-to-weekly hole. The original 160-cell interpretation is also
unnecessary.

Use a sequential screen after corrections 140-142:

1. RT1-A: cadence `{2,3,6,42}` bars (8 h, 12 h, 24 h, 168 h) x lookback
   `{1y, expanding}` x four fixed 28-day blocks x two paired seeds = 64
   block-runs, plus a frozen/no-update control in every block.
2. RT1-B: test cadence 18 bars (72 h) and rolling lookbacks 2y/4y only for the
   two best RT1-A cadence regions, or whenever a boundary winner, non-monotone
   cadence curve or cadence-lookback interaction appears.
3. Pair seed, initial weights, block, environment and account starting state.
   Preserve capital inside each block; score warm-up-free next intervals.
4. Hold update work per new bar and the operational deadline contract fixed.
   Report raw interval/weekly profit, risk, trades, costs, failures and
   handovers; do not replace them with one opaque score.

This design is cheaper than a full factorial while retaining the high-frequency
boundary, daily control and weekly incumbent.

## 5. Owner Ratification Advice

### Deadline budget: ratify only the amended rule

Recommended owner wording:

> I ratify the deadline rule provisionally as end-to-end p95 latency no greater
> than two thirds of cadence, with zero deadline misses and zero unreconciled
> handovers. Latency begins at data cutoff and ends only after the new artifact
> is durably saved, validated, replicated and activation-ready. A cell needs at
> least 20 measured updates for p95 eligibility; below that it is provisional
> and its maximum latency must also remain within the budget.

Training-only elapsed time does not satisfy this rule and the current RT0 run
does not qualify.

### RT2 gene bounds: do not ratify yet

Approve the gene *families* as a draft registry, but no numeric bounds until
corrected RT0/RT1 evidence exists. Add conditional `handover_max_delay_bars`
and an explicit end-to-end update wall-budget gene/constraint. Bound update
steps, replay half-life and lookback from measured sensitivity, not the current
invented ranges.

### Heartbeat restart window: conditional authorization only

Do not ratify an immediate fleet restart. First land and independently verify
the complete manifest join from finding 139. Then restart one seat at a time:
Alpaca first while direct-flat; MT5 only next-flat or after direct proof of the
open position and both native protections. Preserve a pre-restart heartbeat,
rollback command and post-restart hash-joined proof within the freshness
budget. A service restart must never remove broker-native SL/TP.

## 6. Direct Runtime Snapshot

At 2026-08-06 12:55 COT:

- `full-v2`: running, stage 1/3, generation 0, 1/20 evaluated;
- Omega RTX 4070: 47 C, 16%; Dragon RTX 4090: 50 C, 37%; Gamma RTX 5070 Ti:
  45 C, 5%; Gamma RTX 5090: 56 C, 39%;
- Alpaca Paper: runner active, broker flat, linear controller, fresh heartbeat;
- IBKR Paper: runner active, broker flat, `halt=hold`, fresh heartbeat;
- MT5 Demo on Dragon: runner active, one position, zero open pending orders,
  native execution enabled, but controller hashes remain incomplete;
- social collection: 5,496 posts, zero drafts.

GPU utilization is sampled and phase-dependent; service/process/chain facts,
not one utilization sample, establish worker participation.

## 7. Disposition

- 128-133: partially corrected, remain open through findings 135-139.
- 134: manifest description partially accepted; RT0/RT1 evidence rejected
  through findings 140-142.
- 123/124/127: bounded runtime smoke still pending.
- reduced RT1: accepted only as the amended sequential design in section 4.
- owner action: ratify only the amended deadline wording; defer RT2 numeric
  bounds and the venue restart until their stated evidence gates pass.

## 8. Verification Evidence

```text
independent 128-134 reproducer: all_counterexamples_reproduced=true
network_used=false; runtime_mutated=false
agent-multi complete suite: 583 passed, 2 convergence warnings
lts complete suite: 655 passed, 1 framework deprecation warning
doin-node complete suite: 409 passed
```

The passing suites confirm delivery stability; the new reproducer demonstrates
that the remaining defects are uncovered contract cases, not unrelated test
breakage.
