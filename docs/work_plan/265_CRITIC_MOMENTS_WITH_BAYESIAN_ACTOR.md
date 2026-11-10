# 265. Critic residual moments with unchanged Bayesian actor

Owner approved 2026-11-10 after the bounded Bayesian last-layer probe (264).
The previous six arms finished in 235.67 seconds; 475 files were authenticated.
Bayesian endpoint validation equaled the paired deterministic endpoint in all
three seeds. Neither replaced the selected69 champion.

## Hypothesis and fixed contract

Change only the critic calibration objective, retaining the same Gaussian
last-layer actor, source weights, 69 features, fixed TRAIN scaler, frozen
representation, native risk/strategy, all TRAIN rows and validation support.
No feature selection, synthetic inputs, MMD, output-variance minimization,
framework switch or live promotion belongs to this contrast.

For each twin critic independently, define residual e = Q(s,a) - y, with
the identical detached duration-aware realized soft-return target used in264:

- Control: SmoothL1, reuse the authenticated completed Bayesian arms of264.
- M1: mean(e)^2 + population_variance(e); exactly MSE.
- M05: mean(e)^2 + 0.5 * population_variance(e).

Sum the two critic losses, matching the native twin reduction. Do not pool
the two critics before squaring: opposite biases must not cancel. Compute
all moments on CUDA tensors for the batch; no Python tick-level loss loop.
This experiment changes the critic fitting loss, not the SAC actor objective.
Lower posterior uncertainty is not automatically better accuracy or calibration.

Each new objective calibrates fresh critic heads for4096 updates on the same
TRAIN replay, batch128 and schedule as264. Then freeze Q and train only the
Bayesian last actor layer for256 updates, seeds7/19/43, with the same optimizer,
4 training draws,32 inference draws, prior, posterior initialization and KL.
Primary endpoint256; intermediate0/8/32/128 are diagnostic, not winner selection.
These are six new bounded actor refinements, not2000-epoch full SAC retrainings.

## Measurement and acceptance design

| Requirement | Proof before/after launch |
|---|---|
| CM1 Correct residual objective | M1 value AND gradient equal native MSE; M05 algebra; twin-bias noncancellation; singleton/constant errors |
| CM2 Production loss actually changes | Real TorchRL qvalue_loss hook, distinct loss and gradient on the same batch; native hook restored even on exception |
| CM3 No incidental representation changes | Existing actor/encoder freeze checks and native-account parity, source checkpoint hashes |
| CM4 Paired comparisons | Same source/scaler/support/replay checksum, actor initialization, optimizer, schedules and seeds; mismatches reject aggregation |
| CM5 Uncertainty not confused with accuracy | Report critic bias/MAE/RMSE/residual std separately from actor posterior probability std/entropy/MI and native trading results |
| CM6 Durable sequential run | Independent sealed objectives; bounded worker restart; authenticated control reuse; final table/plots and collector |

Report TRAIN/VALIDATION return, Sharpe, drawdown, trades/win rate, direction and
harmful closes from the native accounts. Report the fitted critic's errors on
its frozen TRAIN policy trajectory and a separate frozen VALIDATION policy
trajectory with no optimizer access. These are return-estimation errors,
not classification accuracy or proofs of counterfactual action ranking.
Uncertainty calibration to action correctness remains unestablished: the
probability standard deviation alone must never be presented as calibrated.
No threshold on uncertainty is introduced into execution.

## Orchestration

Secondary RTX4090: M1 then M05, then aggregation against sealed SmoothL1.
Completed arms are hash-verified and skipped; interrupted arms restart from
their pinned seed, without mixing evidence. Preserve selected69 and all264
artifacts. Aggregate every seed, including inactive/regressing outcomes.
Current master-plan position remains phase2 post-selection refinement;
return to selection only after a qualified change to the preserved model.
