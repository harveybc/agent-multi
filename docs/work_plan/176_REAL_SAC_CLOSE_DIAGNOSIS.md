# Frozen SAC Close Diagnosis

The owner flags concentration of real-policy early closes at the first hourly
opportunity. Diagnose completed probes without changing the running source,
weights, reward, decision thresholds or native business contract.

For every recorded CLOSE, replay the identical baseline prefix and clone its
actual open position. Compare executable close now, passive HOLD until the
unchanged SL/TP/forced-flat boundary, and every executable intermediate close
before that boundary. Use the existing tested native `fixed_position_exits`.
Future outcomes are retrospective diagnostics only, never policy observations.

Requirements/tests before implementation:
- authenticate campaign and every completed receipt;
- reproduce recorded timestamps, phases, durations, rewards and whole ledgers;
- prove native counterfactuals do not mutate the baseline;
- count protective preemption separately from actual agent closes;
- check both directions, helpful/harmful/tied closes, empty support and an
  intermediate optimum better than both immediate and passive close;
- report signed close-minus-HOLD net dollars and regret to the retrospective
  optimum. Risk normalization uses original price-to-SL cash risk, explicitly
  excluding fees; both compared net outcomes include actual costs;
- never interpret summed fixed-position differences as a feasible portfolio
  result: alternative positions overlap and change subsequent opportunities.

Publish TRAIN and INNER_A separately. Show earliest-close fraction, win rates,
cost attribution, passive exit reasons, harmful/helpful counts and risk-normalized
regret. This is not a change to model selection, not new financial training,
not a claim of full-account improvement and does not access2024/TEST.

A bounded CPU observer processes each authenticated completed unit once and
keeps following the existing collector until all twelve terminal results exist.
Artifacts are written outside campaign roots and pinned by source/result hashes.

## First Exact Replays

18 targeted tests pass, including complete native replay in both directions,
mutated action/ledger rejection and isolation of fixed-position counterfactuals.
Seven completed models were replayed in TRAIN and INNER_A with exact archived
ledgers. Six models have early closes in INNER_A;12h seed7 has none and is not
silently assigned a zero harmful-close rate.

| Horizon/seed | Early closes | Harmful versus passive HOLD | Sum close-minus-HOLD USD |
|---|---:|---:|---:|
| 3h/7 | 48 | 13 (27.1%) | +703.42 |
| 3h/43 | 47 | 21 (44.7%) | -224.40 |
| 6h/7 | 56 | 18 (32.1%) | +244.79 |
| 6h/43 | 66 | 33 (50.0%) | -425.88 |
| 12h/43 | 107 | 36 (33.6%) | -660.47 |
| 24h/43 | 54 | 20 (37.0%) | -182.22 |

The last column is an attribution sum over independent frozen positions,
NOT the account profit achievable by replacing all closes with HOLD. Both
helpful and harmful cases exist: a blanket minimum holding time is not justified.

Median chosen CLOSE policy probabilities are about0.54-0.58 in these runs.
These are action-selection probabilities, NOT calibrated probabilities that
closing improves profit. The controller currently uses categorical argmax.
The next controlled diagnosis must separate weak action preferences, the
entropy term's dependence on legal-action count (flat3 versus occupied2), and
the learned value of continued holding. None is yet established as the sole
cause. No ad hoc confidence threshold or forced holding interval is added.

The observer remains automated for subsequent completed cells, separate from
the immutable training fleet. This table is a dated intermediate observation;
its output SUMMARY and final CLOSE cover the full probe when it finishes.
