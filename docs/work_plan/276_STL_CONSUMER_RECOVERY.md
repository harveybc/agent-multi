# Consumer activity and real source recovery

2026-11-11. This follows steps273-275. No training dataset, frozen champion,
live service, feature count, model window or calendar has been changed.

## Correction and completed recovery

Step275 recovered a RETIRED global activity classifier, not the producer used
by selected69. Its exact comparison against the old score archive was real but
insufficient: the consumer manifest substitutes `act.p_hit` with cross-fitted
values. The old classifier differs from the current column by as much as
0.6963831437035639. It is explicitly prohibited for consumer or live use.
The old receipt is retained; this correction supersedes its handoff claim.

`tools/recover_consumer_activity.py` now verifies the consumer's manifest,
scope report, input files and producer identities before reproducing the
original cross-fit. It captures and exports all six estimators: the full model
and five purged fold models. Each model's fit indices and fold prediction
indices are retained. It checks exact current-store probabilities, serializes,
reloads and repeats the complete comparison.

| Check | Result |
| --- | ---: |
| Actual consumer rows replayed | 31116 |
| Maximum absolute probability difference | 0 |
| Reload difference | 0 |
| Full-model fit rows | 17128 |
| Allowed scope bars | 21809 |
| Input lookback | 121 observed bars |
| Purge/embargo | 96 bars |
| Estimators exported | 6 |

Scope: 2020-01-01 to 2023-07-01 by the frozen availability rule. This recovery
preserves the historical `purged_kfold` mode; it does not relabel it as a new
forward-only experiment or establish real-time source availability.

Private artifacts relative to the operator home:
`.local/share/agent-multi/stl-recovery-20261111/activity/consumer_recovered/`.

- Weights: `CONSUMER_ACTIVITY.joblib`, SHA256
  `fdc95a0cdabf9a5327c991187f62c8be928476f385551c7e8db3972045167d6b`.
- `RESULT.json` records source identities and replay evidence.
- Private `activity/CONSUMER_SPEC.json` SHA256
  `85e720a4ff647a553a08d520f651a55b9d0b98b655148a5daec3891fddaf5da5`.
- Actual consumer store SHA256
  `cd0eb8d20c0d0888f7814cda410557a79a927e5cc125e9de5c15dbe073d2f543`.

## Real observations

`tools/qualify_stl_recovery_sources.py` produced the private
`stl-recovery-20261111/SOURCE_CANDIDATES.json`, with source hashes and actual
values. It does not write into a source or bypass coverage gates.

The six 2023-11-29T13:00Z candidate H1 bars agree EXACTLY in all four OHLC
fields with aggregation of their twelve M5 and four M15 rows. Further boundary
inspection identified differences in the native12:00 and14:00 observations
as well. In every pair, all524 other observed November bars match exactly in
all OHLC fields. The alternative has exactly one extra timestamp,13:00.

`repair_interval` now requires exact outside-interval identity, both boundary
anchors, matching union grids and complete M5/M15 aggregation for ALL THREE
affected hours. All six pairs pass:18 H1 observations verified, six inserted
and twelve replaced in separate candidate files. No price adjustment or
interpolation is used. These resolutions may share upstream provenance; they
are not claimed as independent market feeds. The source pattern supports an
outage affecting three hours, not just one absent timestamp.

Private `SOURCE_CANDIDATES_V2.json` records hashes and differences; separate
`november_candidates/*.parquet` contain the repaired pre-2025 observed prices.
The original files and admitted store are untouched. This closes the measured
November boundary disagreement, but does not by itself release a new69-feature
store or prove quote availability for live execution.

For USDCHF2022-05-19T21:00Z:

- Downloaded the official FXCM week20 M1 archive: zero observations inside the
  missing hour. It cannot fill the gap.
- Downloaded Forexite's May19 archive: 60 actual observations under its recorded
  Europe/Berlin clock, ten distinct closes. Direct-archive bid/ask and minute
  start/end semantics remain unqualified; no offset or price correction applied.
- HistData's current download page produced an empty token and its submitted
  form returned no data. No recovered ZIP is claimed.
- Forex Tester's official data page explicitly identifies its Forexite-derived
  download as Bid/GMT, but downloading requires browser verification. The owner
  was asked to download the USDCHF ZIP to Downloads. No credential or charge is
  required by this request, and browser verification was not bypassed.

Sources: [FXCM candle format](https://github.com/fxcm/MarketData/blob/master/CandleData/README.md),
[Forex Tester download contract](https://desktop.forextester.com/data/datasources).

## Warmup and release conditions

Real recovered EURUSD history contains 6226 observations from 2019-01-01T22:00Z
through 2020-01-01T23:00Z. That is raw history, not 69 reconstructed features.
The admitted closure calendar begins2019-12-27 and cannot silently be treated
as a complete2019 calendar. The original generator was located: `required_hours`
uses the America/New_York17:00 weekly grid plus explicit holiday exclusions,
then derives sessions and closures. The source business calendar explicitly
covers2020-2025, not2019. In particular `px.hours_since_session_open` and
the activity inputs need the same session semantics. Do not derive a holiday
closure from an arbitrary source outage or fill missing calendar values.

Remaining work is to qualify the USDCHF source,
extend the real calendar/history contract, and rebuild all69 columns with
frozen states and transformations. Then compare the entire changed region,
run prefix/future-perturbation and streaming checks, and verify monthly common
support. Any repaired store and STL run have NEW identities; sealed inputs are
not patched in place. Four STL training arms remain unreleased.

## Traceability

| Requirement | Evidence |
| --- | --- |
| Reject retired global estimator | Actual consumer comparison; unit negative control |
| Preserve fold assignment and full-model application | Recorded selectors; replay unit tests |
| Retain input order and exact timestamps | Column/grid rejection tests |
| No invented observations | Complete-window, duplicate and OHLC rejection tests |
| Exact real source aggregation | Six candidates, two finer resolutions each |
| Neighboring bars not silently retained |18 interval observations;524 outside bars per pair exact |
| Preserve original data and source identities | No source mutations; private hash receipts |

21 focused tests pass across activity recovery and source-coverage tools.
The bounded actor-transfer results in274 are unchanged. No new trading return
is attributed to this data-recovery work, and the main refinement lane does
not require waiting for the external ZIP.
