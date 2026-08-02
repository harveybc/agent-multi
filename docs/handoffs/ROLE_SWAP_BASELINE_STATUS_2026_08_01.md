# Role-Swap Baseline Status

Observed: 2026-08-01 approximately 20:18 America/Bogota
Collector: Musashi while still technical lead
Use: transition comparison only; every runtime fact must be refreshed

## 1. Executive State

- Four DOIN workers participated in one shared `USDCAD@4h` job.
- No duplicate campaign, version mismatch or active LTS watchdog event was
  observed.
- A finalized-anchor warning remained between Omega and the other nodes.
- LTS social-lab finding 034 was corrected, tested and pushed at
  `lts@11d8958` after this baseline was initially summarized.
- Satoshi-owned changes to the audit findings register and audit recovery
  prompt remained uncommitted in the `agent-multi` worktree.

## 2. Hosts And Workers

| Worker | Observed work | GPU snapshot | Operational interpretation |
| --- | --- | --- | --- |
| omega | candidate 20 | RTX 4070, 49 C, 32 %, 2.3/8.2 GB | active |
| dragon | candidate 19 | RTX 4090, 46 C, 0 % at sample; node CPU about 90 % | active candidate setup/CPU phase, not hung |
| gamma-5070ti | no candidate after finishing 15 | RTX 5070 Ti, 41 C, 0 % | generation-tail barrier wait |
| gamma-5090 | candidate 18 | RTX 5090, 57 C, 52 %, 0.9/32.6 GB | active |

Resource risks:

- Omega: about 11 GB RAM available, 2/8 GB swap used, root 44 % used.
- Dragon: about 8.5 GB RAM available, 1/8 GB swap used, root 39 % used.
- Gamma: about 7.7 GB RAM available, 2.4/4 GB swap used, root 88 % used with
  about 47 GB free. Cleanup is required before sustained growth crosses the
  operational threshold.

Temperatures were below the 78 C alert threshold.

## 3. Front 1: DOIN Optimization

- Plan: `phase-1-protected-execution-fleet-v2`.
- Executable queue: exactly two jobs.
- Current job 0:
  `usdcad-4h-protected-easy-sac-shared-v2`.
- Domain: `trading-asset-policy-usdcad-4h-protected-easy-v2`.
- Stage: 2/4, `model_training`; global generation 6.
- Population at final sample: 17 evaluated, 3 claimed, 0 free of 20.
- Full-budget progress: 137/480, 28.5 %.
- Recent throughput: approximately 1.76 candidates/hour versus the documented
  1.73 candidates/hour expectation.
- Full-budget job-0 estimate at that rate: about 8.1 days remaining; early
  stopping may shorten it.
- Job 1, protected easy-to-nominal-to-stress robust weekly-RAP curriculum,
  was queued and blocked on exact job-0 champion/elite archival.

Lineage:

- canonical seed 2703 and shared population matched;
- component versions matched:
  `agent-multi@6a7bf5a`, `doin-core@8573a87`,
  `doin-node@7c400f9`, `doin-plugins@f5fedf8`,
  `gym-fx@40a5c84`, `trading-contracts@534b034`;
- Omega finalized height 6 while Dragon and Gamma reported height 7. This did
  not stop candidate evaluation but must converge before archive/transition.

Current champion evidence at the sample:

- network displayed fitness proxy: `0.0006247008569073586`;
- stored metric evidence for the available champion artifact:
  full-validation-period total return `0.0019967774` (+0.1997 %), maximum
  drawdown `0.0006068970` (0.0607 %), 166 trades and Sharpe `-0.6228`;
- artifact SHA-256:
  `892fbee0f00ecf7b93ca6ed2bea05ca4db3cd09aa557a8cd16b9698fa7e09842`;
- artifact format: Stable-Baselines3 ZIP, 48,801,983 bytes.

The current fitness is the owner-ratified job-0 full-period proxy. Job 1 must
select with the robust weekly-RAP contract. Do not compare these values as if
they used the same unit or horizon.

Broader Front-1 dependencies not yet in the executable campaign:

1. archive job-0 champion and five elites;
2. run job 1 and freeze the complete USDCAD cell;
3. produce execution-state fidelity evidence and auxiliaries;
4. compare market-only, deterministic router and shared entry/exit policy;
5. materialize real context bundles for subsequent selected assets;
6. run one coordinated campaign per asset;
7. confirm winners across three seeds and build the cell library;
8. optimize the static portfolio;
9. add rush/event activation and weekly retraining.

## 4. Front 2: Live And Paper Business Reality

All paths were read-only and reported zero submitted orders, open orders and
positions. The consolidated watchdog reported zero active events.

| Path | Continuous evidence at sample | 24-hour gate |
| --- | ---: | ---: |
| multi-venue shadow | 45.0 h | passed |
| Alpaca Paper | 21.5 h | about 90 % |
| OANDA MT5 demo | 18.2 h | about 76 % |
| IBKR Paper | 6.8 h after TWS recovery | about 28 % |

MT5 reported more than 4,360 heartbeats, six symbols, fresh connectivity and
zero exposure. Observed symbols were `ADAUSD`, `BTCUSD`, `DOGEUSD`, `ETHUSD`,
`EURJPY` and `SOLUSD`.

External/platform state:

- cTrader Copy catalogue verified by owner;
- cTrader Open API application status `submitted`;
- eToro Virtual catalogue, Buy/Sell and CopyTrader controls verified;
- Darwinex Zero held before recurring spend;
- MQL5 Signals provider path classified live-only;
- OANDA REST-v20 Practice unavailable for the owner's OGM account division;
  MT5 demo bridge is the active OANDA observation path.

Interesting evidence: a DOGE/USD one-hour move of -2.01 % was recorded as a
rush-research discussion event and recovered without an order.

Primary blockers: incomplete continuous windows, weekend market closure,
Spotware approval, runtime feature parity and protected canary acceptance.

## 5. Front 3: Social Intelligence

- Moltbook identity available and claimed as `Dragon_DOIN`.
- 93 collection runs and 1,741 stored posts at the final sample.
- Latest collection inserted 18 posts with no error.
- Collector cadence: 30 minutes.
- Flash triage cadence: 120 minutes; last run about 19:40 COT.
- Flash senior review cadence: 360 minutes; last run about 19:30 COT.
- Live-business review cadence: 720 minutes.
- Model reserved-token use: 20,335/250,000 daily (8.1 %) and
  185,038/6,000,000 monthly (3.1 %).
- No drafts or autonomous publications existed.

Omega intentionally owns the only bidirectional Hermes Telegram gateway.
Remote machines use outbound notification rather than competing Telegram
pollers. A Moltbook connection reset at 17:40 recovered on the next timer run;
same-run retry/backoff remains a useful improvement.

Publishing is blocked on a source-backed draft, explicit human approval,
seven clean draft-only days and a credential/threat review.

## 6. Front 4: Audit And Academic Program

- Satoshi independently verified findings 030-033 closed.
- Satoshi opened finding 034: social CLI `--database` did not control the
  scenario persistence destination.
- Musashi fixed 034 at `lts@11d8958`.
- Verification: 18 focused social tests and 235 complete LTS tests passed.
- Finding 034 awaited independent Satoshi verification.
- Satoshi's findings-register and recovery-prompt edits were present locally
  but not committed or pushed at the baseline.
- P1-P5 paper scaffolds existed; P5/P1 citation verification and frozen
  evidence remained pending.
- P6-P18 continuous-research lines were registered but not all materialized as
  executable work.

## 7. Immediate Transition Risks

1. Treating the two-job executable campaign as the complete multi-asset queue.
2. Mutating the active chain to repair an unfinalized-anchor lag.
3. Losing Satoshi's uncommitted documentation changes.
4. Allowing Gamma disk use to cross a recovery-hostile threshold.
5. Confusing accumulated venue observations with continuous 24-hour evidence.
6. Comparing the job-0 proxy fitness with job-1 weekly robust fitness without
   units and horizon labels.
7. Enabling any order path before feature parity and mandatory SL/TP canary
   acceptance.

