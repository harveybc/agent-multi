# 393: calendar-first scenarios (supersedes 392)

Never resume the two-account 392 arms for this question. The downstream
contract is one synthetic 2016-2019 prefix followed by untouched real
2020-2023, one account, ordinary replay, unchanged real validation 2024.
Synthetic data are training augmentation, not reconstructed history.

Development gates (in order):

1. Fit macro dynamics on TRAIN. Score a no-regime control on the same
   chronological data. Keep completed and censored regime spells separate.
2. Emit a seeded, dated script of variable-length spells and alternating
   pivot segments. Controls must have joint historical support, not be sampled
   independently. Draw profiles, not donor bars. Record the calibration dates,
   data fingerprints, source profile and terminal censoring. The provisional
   hourly quote index is not proof that every tradable hour is present.
3. Conditional price decoder: adapt published conditional financial models,
   not a generic VAE. Compare on the same script, inputs, and dates.
   Reference: Market-GAN (AAAI 2024), authors' repository
   XiaHaochong98/Market-GAN at f0f201829e5d2b5c5df51658fe994ce0d41bcfd1;
   controllable diffusion contrast: CoFinDiff (IJCAI 2025), with the
   condition-alignment refinement NSE (ICAIF 2025). Distinguish adaptations
   from reproductions. All generated hourly input channels and producer
   states, not just the close, must carry forward across segment changes.
4. Re-extract pivots on generated OHLC. Compare the scripted and recovered
   paths, drawdown timing, correlations and seam behavior against real
   between-segment reference distributions. Test restart parity and that a
   deliberately time-shifted cross fails. Reject calendar gaps, missing
   producers and invalid candles. Controls known only to the simulator never
   enter SAC. A good script alone cannot release any training job.
5. Recompute original 61 inputs or all 71 including their additional
   upstream sources. Compare each width with its own real control.

First-stage acceptance tests: no rows beyond TRAIN fit; timestamps follow
the target clock exactly once; durations sum to the input calendar length;
directions alternate at seams, lengths and profile controls share a donor;
no resetting the count at weekends; identical seed replays the same plan.
The generator must distinguish market closure/reopening from an hourly step.
No publication or inference eligibility is granted by this CPU development
artifact.

## Executed CPU diagnostic

The implemented one-step decoder is a small conditional autoregressive
diffusion probe, not an official reproduction of Market-GAN or CoFinDiff.
Upstream methods: [TimeGrad](https://proceedings.mlr.press/v139/rasul21a.html),
[CoFinDiff](https://ijcai.org/proceedings/2025/1040),
[NSE](https://doi.org/10.1145/3768292.3770342), and
[Market-GAN](https://github.com/XiaHaochong98/Market-GAN).
The signed-root candle wicks, 13-observation history, regime/profile controls,
epsilon vs velocity and remaining-extremum contrast are specific to this probe.

The HMM was fitted to signed efficiency (48 observations), and 12/48
observation volatility at the pivot origin; macro generation uses empirical
completed sojourn lengths and transitions, not the geometric HMM dwell time.
Adjacent context windows overlap: likelihood is descriptive, not evidence of
independent trading opportunities. First and last censored sojourns are
excluded from the dwell estimator. Script dates derive only from the observed
clock. Five timestamps disagree across instruments; none were silently filled.
3,122 profiles through 2022; 2023 descriptive log likelihood per origin:
-3.105 versus -3.849 for the iid five-component mixture and -3.925 for one
state. Seed 7: 24,915 observed hours, 4,399 segments, 745 spells. Segment
profiles sampled jointly from empirical donors conditional on state and side;
continuous conditioning on the previous segment's magnitude is not implemented.

There were 17,720/17,442 eligible decoder examples (61/71). Short decoder
fits used 2020-21 targets and 2022 for engineering checks. The production
feature scaler and regime classification are fixed upstream, so no blind
assessment is implied. Returns to the pivot extreme are not close-to-close
returns. Both width-specific probes used 12 epochs; the 71-output-market
variant generates the EURUSD candle plus six FX returns, not macro releases.

The stage compared 16 generated paths (24 hours each, two script locations,
two noise seeds, two widths, plain versus remaining-distance conditioning).
At each hour the original producers reconstructed the 61 or 71 columns before
the next denoising step. For the 71 scenario probe only, exogenous releases
were from the real seed calendar. Re-extracted ZigZag H6, identical
implementation and parameters to the historical catalog, tolerance 0/1 bar:

| Inputs | Decoder | Mean absolute move error (bps) | Exact / 12 | Within 1 / 12 | Detected pivots |
|---|---|---:|---:|---:|---:|
|61|plain|23.41|5|7|14|
|61|remaining distance|11.16|7|9|13|
|71|plain|31.79|5|9|13|
|71|remaining distance|9.45|5|8|10|

The small panel repeats scripts; it is not 48 independent segments and does
not establish generalization. Denoising loss alone hid an inflated single-bar
body magnitude for epsilon parameterization. Remaining distance helped
amplitude in both widths, not timing for 71. **No SAC release.**
Next decision gate: compare pivot-timing/extra-extrema control under the same
train-only calendar, including fresh spells, rare profiles, independent
starts, weekends, cross-rate integrity and release-time/clock reconciliation.
Do not tune on real 2024. Full-prefix rollout requires a verified streaming
equivalent of the producers; prefix reconstruction here tests correctness,
not four-year throughput.

Reproduction: `tools/probe393_sequence.sh` documents the ordered CPU stages
and required environment variables. `tools/clock393.py` audits timestamps.
Outputs: `macro/SCRIPT.parquet`, `data/{61,71}/EXAMPLES.npz`, each
`PROBE.pt`, per-path OHLC and features, `PIVOTS.csv`, `SUMMARY.csv`.
