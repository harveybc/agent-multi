# Musashi to General Satoshi: Explicit Close, Parity and Routing Work Order

Date: 2026-08-18
Priority: Front 1 corrected smoke, Front 2 exact-model Demo parity, then
execution-routing evidence
Authority: owner-approved existing Paper/Demo and anti-idle program; no real
capital, secret handling, destructive history change or sealed-test access

Read in order:

1. `docs/audits/AUDIT_EXPLICIT_EARLY_CLOSE_AND_ORDER_ROUTING_2026_08_18.md`
2. `docs/work_plan/39_MODEL_CONTROL_EARLY_CLOSE_AND_ORDER_ROUTING.md`
3. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
   sections 20-22
4. `examples/config/phase_3_eth_sac_dynamics/p1_difficulty_lr_factorial_v2.json`
5. `gym-fx@1a606df`
6. `lts@1ec2889`

Do not resurrect any pre-2026-08-18 P1LR identity. Do not report a linear
canary as the ETH SAC champion. Review runs in parallel with compatible compute;
it does not create an idle fleet.

## WP0. Urgent Independent Live-Control Reproduction

Before WP3/WP4 expansion, independently reproduce `lts@1c16d2d`:

1. prove Alpaca `{"qty":"-1","side":"short"}` resolves to exposure `-1`,
   never `+1`;
2. prove a close signal while the equity clock is closed calls neither cancel
   nor flatten and returns `protection_preserved=true`;
3. for Alpaca, IBKR and MT5, persist `model_close_requested`, then present the
   same `model_id/timeframe/bar_close` once with exposure and once flat;
4. prove both replays submit zero broker calls and return a typed consumed-bar
   state;
5. prove a strictly later closed bar may act again;
6. inspect direct Paper/Demo facts after the 2026-08-18 live replay: Alpaca's
   queued flatten, IBKR's protected USD.CAD short and MT5's protected ETH long;
7. report the IBKR historical-data refusal as unavailable, not as zero bars or
   a model hold.
8. prove a historical-data request has a finite timeout, restores the prior
   client timeout and reaches heartbeat/backoff rather than an unbounded call;
9. terminate during reconnect refusal and prove a clean inactive unit, with no
   `None.close()` or systemd failed state.

Open a finding for any route that can cancel protection before an executable
flatten, reverse within one decision bar, or treat signed broker quantities
inconsistently. Do not close findings implemented by Musashi.

## WP1. Independent Contract Reproduction

Reproduce before changing code:

- resolved observation dimension 2,660 and exact ordered feature digest;
- 18,085-row, 83-column live-formula parity against
  `predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`;
- same action mapping in gym-fx and LTS at entry/exit thresholds, including
  open long, open short, explicit close, opposite close-first and pending cancel;
- same four live-state meanings in simulator and MT5 runner;
- all four genesis artifacts load, have 2,660 inputs, match pinned container and
  tensor hashes, are identical within seed/cell and distinct across seeds;
- P1 materialized config fixes `entry_order_mode=market` and
  `pending_order_ttl_bars=0` in both phases.

Return a machine-readable packet with commands, exit codes, hashes and direct
counterexamples. Do not close findings authored or implemented by yourself.

## WP2. Corrected Mechanics Smoke and Decision Dispatch

After WP1 reproduction:

1. deploy exact agent-multi/gym-fx revisions to Omega, Dragon and Gamma;
2. replicate and hash-check the per-seed genesis artifacts;
3. run preflight and one-pass mechanics smoke on seeds 101/202/303/404;
4. keep all four workers on the one declared identity and collect 16/16 records;
5. seal and independently load replicas on Dragon;
6. require non-zero actor liveness, non-constant selected behavior and
   `selected_equals_genesis=false` for at least one trained region;
7. dispatch the full decision automatically only if the typed smoke verdict is
   viable; otherwise stop that identity and return the measured cause.

Report current cell, phase, epoch/pass, best raw metrics, completion fraction,
throughput, ETA, GPU utilization/memory/temperature and exact identity per GPU.

Before unmasking decision units, independently inspect their *effective*
systemd properties on Omega, Dragon and Gamma. They must bind:

- working directory at immutable runtime `agent-multi@8758273f`;
- contract `p1_difficulty_lr_factorial_v2.json` in both gate check and runner;
- the newly sealed `bfbfd6443b849275` v2 screen verdict;
- explicit `--mode decision`; and
- seed 101/202/303/404 to the contract GPU UUID.

The installed base template still carried a v1 default and old gate (EC-13).
Musashi deployed a local v2 override; upstream the installer/pinning script so
future contracts cannot inherit that stale default. Add a test that fails when
`ExecStartPre` and `ExecStart` name different contract files.

Also correct the checkpoint source enum that currently labels normal cells
`easy_training_epoch` (EC-14). Preserve the current screen as-is because
phase/config/hash evidence disambiguates its actual dynamics; the enum fix
applies to the next content-addressed identity.

## WP3. SAC Manifest Publisher and MT5 Readiness

Implement a local, fail-closed publisher in `lts/tools/` that accepts an exact
trained artifact and resolved config, recomputes hashes and emits
`prediction_provider.live_sac_manifest.v1`. It must refuse unless:

- artifact input shape is 2,660;
- config feature list digest matches the observation contract;
- action contract is `target_exposure_hysteresis_v2` with entry 0.10 and exit
  0.02 for normal Demo inference;
- state is `live_stationary_v2` and raw price window is false;
- parity evidence names train-side and LTS action/input hashes;
- research-validation evidence is explicit; and
- no account identifier, credential, private address or secret enters Git.

Do not publish from zero-update genesis, smoke-ineligible, constant, dead,
genesis-equal or pre-2026-08-18 artifacts.

Prepare but do not counterfeit the human MT5 step: the updated
`LtsMt5ModelBridge.mq5` must compile with zero errors and be reattached with
`InpClosedBarHistory=800`. Direct heartbeat must show the new snapshot field and
800 closed bars before the SAC runner can replace the linear controller.

## WP4. Live/Simulation Early-Close Evidence

Extend the due-bar OLAP join so every ETH decision records:

- model raw action and mapped target;
- current exposure, entry price, holding bars and unrealized PnL input;
- enter/monitor/model-close/opposite-close/protection-close outcome;
- native SL/TP geometry and direct venue acknowledgement;
- corresponding synchronized simulation action and discrepancy reason;
- realized spread/slippage/latency and model-close PnL avoided/forgone when the
  counterfactual is observable.

The first report is descriptive after one complete lifecycle; the primary
operational comparison uses at least one week. Missing due bars are coverage
failures, not removed rows.

## WP5. Order-Routing Data Readiness and O1 Design

Do not start a broad GPU search. Build CPU-first evidence:

1. inventory causal ETH H1/15m/tick bid/ask coverage and venue order-family
   capabilities;
   record separately that gym-fx already has market/limit/stop mechanics while
   current Alpaca, IBKR and MT5 risk-increasing parents are market-only;
2. identify where H4 OHLCV cannot determine fill ordering;
3. specify deterministic O1 market/limit/stop routing with offsets and TTL;
4. add pending cancel-on-model-close and restart-idempotency fixtures;
5. define paired replay holding signal, size and SL/TP fixed;
6. emit fill, non-fill, opportunity-cost, adverse-selection, slippage,
   protection and compute metrics.

The design packet must also:

7. keep buy/sell as directional-policy output, not separate order families;
8. distinguish entry stop from protective stop-loss in schemas and metrics;
9. make risk-reducing model close market-by-default and prove a pending entry is
   cancelled before flat is acknowledged;
10. define side-normalized offset/TTL features so one router can serve long and
    short before any model split is considered;
11. specify O2 as a small counterfactual utility model or contextual bandit,
    with the directional SAC, size and risk geometry frozen;
12. include an ablation showing whether separate side/family models add robust
    out-of-sample value beyond conditioning, including trial-count and compute
    penalties; and
13. reject any scalar-Box encoding that gives categorical order families an
    arbitrary continuous geometry.

`stop-limit` stays an explicit later arm because it is unsupported by the
current gym plugin and adds trigger-plus-non-fill risk. A learned router or
separate model is not authorized until O1 demonstrates material headroom.

## WP6. Parameter Registry and Architecture Artifact

Generate a machine-readable report, not prose copied from defaults:

- every held, factored, optimized, excluded and future parameter by domain;
- type, units, active condition, source/bound provenance and current value;
- historical best values qualified by their now-invalid observation/action
  contracts;
- corrected best value as unavailable until the new decision completes;
- PyTorch module tree and trainable/target parameter counts for every selected
  SAC artifact; and
- exact config/model/feature/data/code hashes.

The report must state plainly that P1LR is a factorial calibration, not DOIN,
and that no corrected DOIN champion exists yet.

## Return Standard

Return one audit request with exact commits, tests, runtime identities, output
paths, hashes, direct metrics, residual doubts and any proposed improvement.
Never turn an absent fact into zero or success. Never wait on audit while a
compatible approved CPU/GPU job can run without contaminating the active
identity.

## Runtime Facts for Independent Reproduction

- screen identity: `bfbfd6443b849275`;
- screen outcome: `SCREEN_VIABLE_REGION`, 16/16 records and loads;
- sealed digest:
  `61a59e2121b158239d8cd31d94167796b9eb301463228e8aee2cf0717fe48594`;
- activity: 7 active, 9 inactive, six viable handoffs;
- decision identity: `ac0941e7bdb1a163`;
- decision start: 2026-08-18 08:09 America/Bogota;
- first cells: seed101 `P1N_LR1E4`, seed202 `P1N_LR3E5`, seed303
  `P1E_LR1E4`, seed404 `P1E_LR3E5`.

Audit and installer hardening run in parallel with these workers. Do not stop,
restart or mutate them to inspect code. Reproduce EC-13 from effective unit
properties and the refused first-start journals, then fix the generic installer
and add the contract-coherence test outside the immutable runtime worktree.
