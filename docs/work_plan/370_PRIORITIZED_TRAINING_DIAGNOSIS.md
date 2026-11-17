# 370: prioritized training diagnosis

## Contract and decision

Owner approved cheap evidence-first diagnosis, then a data-quantity learning
curve, before synthetic augmentation or broad architecture search. Phase 2
remains open. No live, STL, feature changes or promotion.

Keep the four existing GPU workers running through their declared stopping
rules. The first diagnostics are CPU-only and do not need those GPUs. Do not
interrupt an incomplete contrast merely because its current return is poor.
Do not launch more GPU work over occupied devices.

## Requirements and tests designed before implementation

| Requirement | Acceptance test | Evidence |
| --- | --- | --- |
| Preserve workers and weights | Collector only reads explicit JSON files; no process or model mutation API | runner source; fleet snapshot |
| Authenticate observations | Match SPEC and PROGRESS identity; hash raw bytes; record capture time | unit tests; snapshot receipts |
| Never report missing work as zero | Missing or inconsistent inputs produce explicit error per run | negative tests |
| Measure completed epochs only | Monotone history; no inferred return from partial epoch | history tests; tables |
| Separate exposure from samples | Report updates/decision, not effective independent sample count | numeric tests |
| Analyze all selected inputs | Fixed source and support digest; per-column/year finite coverage and dependence | DATA.json |
| No test or new validation inference | Only existing account histories; data store must end before 2025 | date guard test |
| Resume and avoid duplication | Data report input/code identity; content-addressed snapshots; exclusive lock | cache test; service |
| Bounded unattended collection | Poll 30 minutes, at most 72 hours; failures explicit, no promotion | timer status and FINAL.json |

## Sequential CPU work

1. Authenticate source, data files and support. Inventory all 71 columns and
   all TRAIN years; report lag-one dependence on adjacent hourly rows, not an
   invented number of independent samples. Report TRAIN-to-2024 input shift
   descriptively, never optimize on this analysis.
2. Fetch atomic JSON snapshots from the four active runs. Report completed
   epochs, patience, rates, updates per decision, TRAIN/validation gap, action
   concentration, activity, evaluation cost and time to the minimum epoch
   budget. This ETA is not a guaranteed completion time.
3. Reuse authenticated369 nonqualification and existing366/368 parity and
   weight probes. Do not rerun the same critic calibration. Missing replay
   distributions remain unmeasured; policy signal counts are not replay counts.
4. Refresh only cheap histories every30 minutes. Publish one results JSON and
   PNG per changed history. Stop at all terminal runs or the72-hour bound;
   a failed source stays visible and cannot count as successful completion.

## Next experimental gate

First long campaign: nested6/12/24/48-month TRAIN suffixes ending2024-01-01,
same2024 validation, all selected columns and full window, seeds7/19/43.
Compare fixed optimizer-update budgets, with a separate exposure-matched
analysis. This is a design, NOT a launched training campaign.

Crucial confound: the champion encoder and scaler already saw four years.
Reusing them cannot establish the effect of having only six months of data.
A causal data-quantity contrast requires fresh per-subset learned preprocessing
and initialization, with publication-time features and full lookback support.
Keep the selected71 set fixed and label conclusions conditional on that set;
do not claim end-to-end selection from each shorter history.

Before GPU dispatch, implement/test that subset contract and expose exact sample
and update budgets. If historical learned feature producers cannot be refitted,
report that limitation and separate frozen-representation augmentation from
the strict data-quantity question. Do not silently replace inputs or truncate
windows. Synthetic generator work follows this measurement, not an assumption
that it cures critic instability or position aliasing.

## Decisions requiring owner input

Only changes to the observation/action contract, use of unavailable paid data,
or changing the agreed business rules require a new question. Failed numerical
qualification does not authorize endless unchanged retries. The runner writes
the next unresolved object explicitly; it cannot certify causes from curves.

## First executed pass

13 focused tests and scoped lint pass. The authenticated source inventory has
71 selected columns,24026 valid TRAIN windows and6208 validation windows; zero
TRAIN-constant columns. Four active histories were collected without errors.
The largest absolute mean shift measured in TRAIN standard deviations was
1.2715 for cot.eur.net_share. This is a descriptive distribution difference,
not proof of the cause of failed generalization.

| Run | Complete epoch | TRAIN return % | Validation return % |
| --- | ---: | ---: | ---: |
| 368 refinement | 39 | 72.0030 | -11.3850 |
| 367 balanced monitor | 7 | -4.3714 | -1.3617 |
| 367 strict monitor | 53 | 8.0392 | 6.3140 |
| 365 control | 46 | 10.8944 | 9.2472 |

These are latest completed epochs, not newly selected champions or final
outcomes. Updates per collected decision are approximately0.244-0.249; this
does NOT measure unique historical data, minibatch reuse, or independent
samples. The observer produces plots without rerunning account evaluation.

Deployment: pinned runtime21a23c90, enabled user service with lingering already
enabled. Verified active with zero restarts, four PNGs and no collection errors.
CPU capped at one core, low priority, memory high4G/max5G; no GPU requested.
Combined370/369 regression tests:21 passed. Poll30minutes; stop after72hours or
all four observed arms terminal. Existing GPU supervisors continue their own
queues; observing an arm ending does not cancel its queued successor.
