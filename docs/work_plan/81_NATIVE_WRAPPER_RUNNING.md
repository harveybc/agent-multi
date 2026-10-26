# Native wrapper deployment

The owner authorized execution without intermediate approvals. This is the
additional native-model feature-selection phase from plan 80, not a declaration
that the full selection is finished.

## Pinned release

- Agent evaluator/queue: `e18d00d4`.
- Corrected business engine: `cd6489f`, accounting `event_costs_v2`.
- Generic selector: `bec66cca`, `feature_selector` 0.2.1.
- Campaign spec: `aa56540dce43d42f906ccecaae42fe290cef967e46ae3c8ca75f0cab2ba64f48`.
- Input candidates: 120 columns, 31 nonempty branches, window 13 / stride 2.
- Common valid support: TRAIN 21,221; INNER_A 2,805. TEST is closed.

The split is explicit: TRAIN 2020-01-01 through 2023-07-01, INNER_A through
2024-01-01. This is the internal development selection contract, not a claimed
four-year training / independent annual validation comparison. No 2024 or TEST
score is used in the wrapper. Upstream selection used 2023, so INNER_A is not an
independent confirmation of the complete pipeline.

Thirty-three focused agent tests passed, including real PPO updates/reload and
the corrected business engine. The selector suite passed 38 tests with four
explicit legacy skips. GPU admission passed for both TCN and CNN, including
reload and finite weights. Admission is a synthetic engineering test, not a
claim of profitable learning.

## Automated execution

The primary GPU worker runs one heavy child at a time. Its enabled user service
restarts after host restarts, uses a queue lock, and caps aggregate service RAM
at 6 GiB with no swap. Completed results are authenticated before reuse; partial
attempts are never used as donors. There are at most two fresh attempts per task.
Unknown persistent errors terminate explicitly instead of being hidden as scores.

Controls: TCN with configured costs, TCN without costs, and prior shallow CNN
with configured costs, three seeds each. Every control is evaluated with both
cost modes by replaying the business engine. Configured-cost TCN results alone
enter the generic backward-floating wrapper. The first allocation is 128 unique
subsets with three seeds. A budget stop is provisional, not final selection.

Each task retains the effective recipe, scaler, support identity, fresh training
receipt, final model ZIP, TRAIN and INNER_A trades, equity curves, native metrics,
cost components, and an authenticated result. A separate two-minute archival
timer copies evidence to the archive volume, only when that volume is mounted.
An in-progress mirror is not independently authenticated until its completed
result's files pass hash verification.

First live observation: 2026-10-26 08:26:59 UTC, configured-cost TCN seed 7 at
87,552 / 102,400 steps. Service had zero restarts; observed service memory about
2.7 GB. This verifies actual financial training, not just launch. First-result
evaluation and receipt authentication were still pending at that observation.

## Read-only status

From the exported runtime, with its selector source on `PYTHONPATH`:

```bash
python -m tools.rl_selection_run status --root <private-campaign-directory>
```

`STATUS.json` identifies the unit; `PROGRESS.json` records training/evaluation;
`CONTROLS.json` records complete three-seed controls; `SEARCH.json` and
`CLOSE.json` describe the bounded wrapper outcome. No global RELEASE is issued.
