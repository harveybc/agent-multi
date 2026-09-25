# WP21(b) — Laya as a first-layer decision evaluator, 2026-09-25

What was run, what came out, and what none of it establishes. Every number below is in a file in this directory;
nothing here was typed from memory.

## The one-line result

Over the 256-bar ETHUSD 4h development sample the fitted policy's example ships, the classification checkpoint was
asked one `choice` per bar — `long` / `flat` / `short` — and **abstained on all 256**: its top probability ran from
0.5754 to 0.7046 (mean 0.6337) against the 0.8 threshold WP09 measured. The decision series is therefore all `flat`
and is, by construction, the naive arm. Its environment-return total is identical to `flat`'s, and the comparison is
`COMPARABLE` because it is literally the same series.

The threshold is not what flattened it. The checkpoint's own argmax, recovered from the 256 abstention records, was
`flat` on **every one of the 256 bars**. Without the gate the series would have been the same series.

## The table (see `reports/reward_table.md` and `closure_table.md`)

| stage | environment_return_training_reward_total | naive (`flat`, same rows) | difference | comparability |
|---|---|---|---|---|
| `laya_first_layer` | 0.000000 | 0.000000 | 0.000000 | COMPARABLE (rank 1, tied) |
| `flat` | 0.000000 | itself | 0.000000 | COMPARABLE (rank 1, tied) |
| `fitted_sac` | -0.003888 | 0.000000 | -0.003888 | COMPARABLE (rank 3) |

One seal `f6b715d0…` over 256 rows covers all three arms, protocol `65013b74…`.

`evaluation/compare_stages.py`'s own closure table (`closure_table.md`) carries all three stages on that seal and
`NO_NEW_MEASUREMENT` in every row, because `policy_profitability` is refused by name: *a proposed action is not a
realised return*. That refusal is the "why not" the done-when allows, and it is not a gap in this run.

## What it is not

`environment_return_training_reward_total` is the gym-fx environment's own reward under its configured reward plugin,
summed over the replayed steps of a simulation with the campaign's own commission, slippage and execution rules.
**A training-reward total over 256 development bars of one instrument is not evidence about any market.** It is not
profit, not a backtest, and every report here flags itself `UNDERPOWERED` against a minimum declared before the run
(500 rows) precisely so the number cannot be quoted as one. `execution_authorized` is false throughout; no key in any
file names profit, P&L or an order; nothing here reached a broker.

The 0.8 threshold was measured on a CLASSIFICATION corpus (which economy a calendar release names), not on any trading
question. Carrying it here is an assumption about this checkpoint's calibration, named as one in every record.

## Calibration (see `decision_calibration.md`, `outcome_linking.json`)

256 decision records of kind `trading_decision`, question `exposure`. **0 linked**, and the linking was attempted on
every one: `m5phet.decide.outcome` refused all 256 with `ABSTENTION_HAS_NO_OUTCOME` — an abstention is not a choice, so
no row can rank it. Independently, the closure row for `laya_first_layer` is `NO_NEW_MEASUREMENT` / rank `null`, so it
could have labelled nothing even had a choice been made. `NO_NEW_MEASUREMENT`; 30 linked scorable outcomes are
required for this (kind, question) and 30 are missing.

## Files

| file | what it is |
|---|---|
| `confidence_bands.json` | `m5phet.abstention_source.v1`: WP09's 450 scored answers counted into confidence buckets. Below 0.8 the checkpoint is right 0.3266 of the time against a chance rate of 0.3333; at or above, 0.8727 |
| `decisions.jsonl` | the decision series, one line per bar, in the shape `sweep.py` reads |
| `decisions.manifest.json` | the sidecar: the header (`asked` 256, `answered_above_threshold` 0, `abstained` 256, `refused` 0), the 256 decision-record digests, the bars' digest, the citation the threshold came from |
| `ask.log` | one line per decision point, as it was produced |
| `sweep.json` | the three arms replayed on the same rows (`m5phet.policy_sweep.v1`) |
| `reports/` | one `m5phet-evaluation-report/1` per arm, the shared `corpus_seal.json` and `protocol.json`, and `reward_table.md` |
| `closure_table.md` / `.json` | `evaluation/compare_stages.py` over the three reports |
| `link_outcomes.py`, `outcome_linking.json` | every decision record offered to `m5phet.decide.outcome`, and what it refused |
| `decision_calibration.md` / `.json` | `evaluation/decision_calibration.py` over the outcomes and the record inventory |

## How it was produced

```
set -a; source ~/.config/m5phet/chat.env; set +a      # CPU only: CUDA_VISIBLE_DEVICES=""

# the threshold's citation, counted from WP09's own scored answers
crispdm-run -m 2G -t 300 -n wp21-bands -- python -m agent_multi_m5phet.abstention \
    --answers <wp09>/answers.jsonl --quality <wp09>/quality.json --out confidence_bands.json

# 256 questions to the real checkpoint (~20 s each; records are content-addressed, so --resume loses nothing)
crispdm-run -m 6G -t 7200 -n wp21b-ask -- python -m agent_multi_m5phet.first_layer \
    --bars sample_256.csv --out decisions.jsonl --every 1 --max-declared-columns 12 \
    --min-confidence 0.8 --abstention-source <wp09>/report_laya_zero_shot.json \
    --records ~/.local/state/m5phet/decisions --resume

crispdm-run -m 6G -t 1800 -n wp21b -- python -m agent_multi_m5phet.sweep \
    --bars sample_256.csv --decisions decisions.jsonl --out sweep.json
crispdm-run -m 4G -t 600 -n wp21b-rep -- python -m agent_multi_m5phet.sweep_report \
    --sweep sweep.json --bars sample_256.csv --decisions decisions.jsonl --out-dir reports/
python -m evaluation.compare_stages --report laya_first_layer=... --report flat=... --report fitted_sac=...
python -m evaluation.decision_calibration --outcomes ~/.local/state/m5phet/decision_outcomes \
    --inventory ~/.local/state/m5phet/decisions
```

The bars are the last 256 rows of the operator's declared policy sample
(`predictor/examples/data/project3/ethusdt_4h_tech_stat_full_model_ready.csv`, 18085 rows, 4h, ETHUSD),
2025-11-19T08:00:00 .. 2025-12-31T20:00:00, sha256 in `decisions.manifest.json`. They are not copied here.

`--max-declared-columns 12`: the classification provider budgets head, options and state against one token limit and
refuses `TOKEN_BUDGET_EXCEEDED` rather than truncating silently. The whole list of 83 fitted column names does not fit
beside the summary, so it is cut where the cut can be declared — the state carries the true count (83) and the number
shown (12), and says in its own text that it was cut.
