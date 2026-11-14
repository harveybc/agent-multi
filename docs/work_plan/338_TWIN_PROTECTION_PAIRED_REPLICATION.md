# 338: Paired Twin-Protection Replication

Status: COMPLETE_VERIFIED_REPAIR_REPLICATED_TRANSFER_MIXED.

337 fixes a measured B43 nonsmooth protection defect:256/256 updates accepted
versus0/256 legacy, stable-added regret0.304704942 ->0.293662736. Internal TRAIN
stop/audit regret worsens. Do not infer generalization or actor/trading gains.

## Frozen Contrast

Use all four334 cost-only1024 endpoints (foldA/B, seeds19/43), same optimizer
moments,256 further scheduled attempts1025..1280 per arm. Eight runs in total.
Both arms use the SAME high-accuracy primal solver; the only treatment is
active-minimum versus both-chosen-twins constraints. This separates the337
combined solver/constraint change. Do not reuse mismatched dual controls.

No data, features, losses, temperature, batch, learning rate, original margins,
2x interior target,3 relinearizations,8 scales or4x norm-cap changes. Freeze each
stable-added mask from that source's unanimous five-view actions, intersected
with the previous label-free stable mask. Never select checkpoints using held
labels. Full original guards and actual weight motion remain mandatory.

## Execution And Verification Contract

- Authenticate parents, source checkpoint and support identity before fitting.
- Restore source model/AdamW/RNG separately for each paired arm.
- Save model/AdamW/RNG every32 attempts, endpoints64 and256. Resume from them.
- Guard each saved state across GPU/CPU/NumPy and batch/single execution.
- Apply those same batch AND single guards INSIDE every update transaction;
  a failed single evaluation rolls back weights and AdamW immediately.
- Assert optimizer step count equals source accepted count plus new accepts.
- Cold-reload both endpoints and reproduce three transitions after64 exactly.
- Seal each unit and root. Reuse completed units without overwriting evidence.
- Independent NumPy endpoint scoring/old-decision protection and paired support
  checks after completion. Existing analytic twin-crossing and derivative tests
  plus53 focused regressions must pass before launch.

Report every case at fixed256, including unchanged/worse cases. Primary outcome
is stable-added decision regret, alongside acceptance, full-FIT cost/MSE and
internal TRAIN stop/audit diagnostics. The encoder was pretrained on whole
TRAIN; these are not independent end-to-end generalization estimates.

No automatic budget extension,actor update,live promotion,2024/TEST read or
unchanged floating pass. Preserve299/71inputs/28groups and49 cumulative return
candidates;STL remains paused. At closure choose the next diagnostic from
measured attribution, not from loss improvement alone.

Prelaunch:53 focused CPU tests in3.87s, CLI smoke and scoped Ruff pass. The
runtime includes all imported plugin packages, following the337 deployment
correction. Each complete unit is sealed; restart reuses it, while partial
units recover full optimizer/RNG state with exact metric checks.

## First-Launch Witness And Correction

First runtime `013813fa` stopped at its first checkpoint guard, A19/active.
The last persisted state was offset0,accepted0 and safe. No completed unit or
model was promoted. Replaying its first32 attempts locates the first violation
at offset10: batch guards pass, GPU single-observation margin is
-4.602432250976644e-8, with argmax unchanged. Thus the final checkpoint guard
worked, but the transaction evaluator omitted a required execution geometry.

Revision2 evaluates original batch AND CPU/GPU single guards on EVERY proposed
finite step in both arms. Rejection restores weights and AdamW atomically.
No margin/tolerance change. A regression supplies a passing batch evaluator
and failing single evaluator and verifies24 rejected trials and exact rollback
of both states.54 focused tests pass in3.82s; scoped Ruff passes. Preserve the
failed first root; start a new identity from the original334 sources rather
than continuing an unsafe intermediate state. This is a common safety repair,
not a treatment-specific advantage or a change to data/learning objective.

Revised runtime `19f70c6d`, identity
`43b9e5c253c63d58a68e378b099364c04ae3bedf96b18038f1e620acc62fe9da`.
Separate CPU verification waited for the root seal (bounded30min), then checked
every saved endpoint, source/support/schedule identity, optimizer counter and
NumPy metrics, emitting the four-case table automatically. The producer and
verifier both exited0 with0 restarts in revision2. Both are now finished.

## Completed Paired Results

Eight arms completed in569.114s. Both treatments use the same primal solver,
batch/single guards and256 scheduled attempts. Scores below are native
stable-added FIT decision regret; lower is better, not trading returns.

| Case | Source | Active-only | Twin-complete | Active accepted | Complete accepted |
| --- | ---: | ---: | ---: | ---: | ---: |
| A19 |0.208440852|0.201129450|0.190651389|54|256|
| A43 |0.188652227|0.188652227|0.182815917|183|256|
| B19 |0.252695854|0.246494942|0.246494942|237|256|
| B43 |0.304704942|0.304704942|0.293662736|0|256|

Twin-complete wins3/4 and ties1/4; mean0.235245390 ->0.228406246, a2.9072%
relative reduction versus the matched active-only control.1024/1024 accepted
updates versus474/1024. Every saved original-decision guard passes and every
stable-added execution-geometry comparison agrees. Full-FIT cost falls further
in all four complete arms. Do not equate lower surrogate loss with universal
decision gains: A43 regret rises late while cost continues falling.

| Internal TRAIN scope, fixed256 | Active mean regret | Complete mean regret | Change |
| --- | ---: | ---: | --- |
| Stop |0.263998496|0.264439621|0.1671% worse|
| Audit |0.228386774|0.225302945|1.3503% better|

Audit improvement comes from A19; B43 worsens. Stop improves in B19 and worsens
in B43; the other two tie versus active controls. These are four descriptive
paired cases, not a significance claim. The encoder used whole TRAIN. No2024,
TEST,actor update,model promotion or new floating pass occurred.

## Verification And Attribution

Nine producer/unit seals,16 cold endpoints and24 resumed transitions verified.
Independent local and producer-host NumPy verification checks all16 endpoints,
paired schedules/support,source hashes,optimizer counters and original margins.
Decision-regret values agree exactly. MSE/margin numerical differences between
machines are at most2.220446049250313e-16; not a byte-identical arithmetic claim.
Manual derivative regression and57 focused CPU tests pass in4.44s; scoped Ruff
passes. Curves render all8 runs through256 attempts without truncating axes.

Compared with their own frozen source, the complete arms change43 FIT action
occurrences:24 beneficial,9 harmful,10 outcome-equivalent; all aggregate regret
deltas reconcile exactly. These counts are across cases, not unique market
timestamps. Internal stop changes:6 beneficial and1 harmful. Audit changes:
3 beneficial,3 harmful,2 equivalent. B43's harmful held changes are all
flat/both-allowed states (one bad added opening and two suppressed profitable
openings), while its changed open-position decisions improve. This localizes
the remaining transfer error; it does not justify fitting those held rows.

Evidence: [table](generalization/twin_replication_20261114/RESULTS.md),
[producer receipts](generalization/twin_replication_20261114/RESULT.json),
[independent scoring and row attribution](generalization/twin_replication_20261114/INDEPENDENT_VERIFY.json),
[curves](generalization/twin_replication_20261114/CURVES.png).

## Adopted Practice And Next Action

Adopt twin-complete constraints plus per-transaction batch/single guards for
new protected critic diagnostics. Keep legacy paths solely as explicit replay
controls. The nonsmooth protection stall is repaired in all four measured
cases; no blanket claim about full SAC convergence follows.

Next use saved64/256 weights and FIT-only action-change attribution to separate
surrogate-cost improvement from harmful greedy-decision crossings, including
gradient contributions by row/role. Keep held rows descriptive, not fit targets.
Specify the next bounded loss/update contrast only after that witness. No
unchanged floating repeat or long refit; source299/71inputs/28groups and49
return candidates remain intact. Value calibration335 is available but cannot
change greedy regret by itself. No owner decision is required for this next
read-only diagnosis. No training worker remains running after338.
