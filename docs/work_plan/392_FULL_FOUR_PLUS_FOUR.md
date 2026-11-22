# 392: four real TRAIN years plus four synthetic TRAIN years

Historical contract only. Superseded by [393: calendar-first scenarios](393_SEGMENT_SCENARIO.md).
Do not launch the paired-account or balanced-replay experiments below.

Supersedes the 10% augmentation arms of 391. Do not resume or use those arms
in this comparison. Keep the original 61- and 71-input real-only controls.

Each augmented epoch runs the same policy online through two independent,
complete 2020-2023 accounts: real, and one hierarchical synthetic trajectory
(seed 7, price or contextual according to input width). Alternate eligible
decisions without forcing identical missingness. Record each successor,
terminal, elapsed time and reward in the acting account; only then switch
the observation for the next action. Never trade across accounts. Require
all eligible decisions of both calendars at the end of each epoch.

Replay samples exactly 64 real plus 64 synthetic per 128-sample update.
Store origin outside the model inputs. Four pilot epochs; preserve optimizer,
learning rate, cadence, recovery, window, starting weights and real controls.
At unchanged cadence per decision a doubled training traversal entails more
optimizer updates than real-only. Report both counts, not a matched-compute
causal claim. Evaluation and checkpoint selection use exactly the existing
real TRAIN and real 2024 validation accounts; no synthetic validation or TEST.

Runtime evidence: CONTRACT, EPOCH_ZERO, PROGRESS, HISTORY, RESULT for each
arm, and FLEET / COMPARISON at the new coordinator. Final comparison checks
the common support and starting model per width, complete trajectories and
matching real/synthetic replay counts. No promotion. Coverage and ring-buffer
wrap / recovery tests in tests/unit/test_full392.py.
