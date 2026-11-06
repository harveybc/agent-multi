# Structural76 Refit

Owner request2026-11-06: use the idle primary GPU for the outstanding fresh fit
after the completed93 ->76 ->76 ordered selection in[208](208_R1_FROZEN_76_TWO_PASS_RESULTS.md).

## Frozen Contract

- Instantiate only76 selected input channels and28 groups, not a93-input mask.
  Fresh actor, critics, optimizers and replay. Preserve the retained TRAIN scaler
  constants, source files, support,13 observations/stride2 and original calendar.
- Preserve R1: two TCN blocks per group, branch width8, core width16/two blocks,
  causal patch attention with positional encoding, head8, critic-core dropout0.1,
  weight decay0.01, no BN and no agent-state inputs. No synthetic or G2 context.
- Four TRAIN years2020-2023, VALIDATION2024. Same costs, risk rules, categorical
  decisions, early closes, batch128, replay65536, update_every4, warmup4096,
  learning rate0.0003 and original SAC settings. No TEST access.
- Seed7, max2000 complete TRAIN epochs, patience20, min_delta1e-8, validation
  return/week as monitor. Save the best checkpoint from epoch1; begin counting
  patience after epoch20. Full native evaluation once per epoch. Persist reusable
  inference weights, recovery state and TRAIN/VAL diagnostic plots.
- File-backed replay is the already-tested storage implementation, not a smaller
  replay or altered sampling distribution. Source R1 used the in-memory backend;
  interrupted/uninterrupted parity tests cover the storage substitution.

## Acceptance and Automation

| Requirement | Check |
|---|---|
| Only selected channels | exact selection receipt,76-feature recipe and28 groups |
| Same support/scaler/business | parent hash authentication, projected scaler, row/month checks and native synthetic controls |
| Correct trained architecture | actual full-shape CUDA canary before training; EFFECTIVE.json |
| Valid learning and inference | focused structural/critic tests, full-batch update, exact checkpoint reload |
| Durable execution | systemd restart, epoch recovery, shared GPU lock and memory protection |
| No babysitting | coordinator collection; resource-only fallback to spare worker, no automatic change to batch/data |

The comparison reference is the frozen masked76 policy (+11.2871% validation),
with the original93 R1 checkpoint also retained. Reproducing that return is an
experimental question, not a promised consequence of removing channels.
No additional selection or unbounded hyperparameter search is chained to this fit.
G2 continues independently on its unchanged93-input contract.

## Deployment Receipt

Started2026-11-06 at17:55UTC on the primary RTX5090, runtime`9e8c1528`.
Run identity:`8145612ef9f01c9de669fcb31189744cac20f538eafd2b6803ebba4a8a432f72`.

- CPU regression:59 passed,12 CUDA skips. Worker acceptance:43 passed,6 skips.
- Separate real-TRAIN CUDA canary passed:128 transitions, input`13 x76`, finite
  SAC losses and gradients. This throwaway update is not experiment performance.
- Effective model:76 channels,28 groups,92,320 trainable parameters; confirmed
  RTX5090. First native training update observed after4,096 warmup decisions.
- Initial worker RAM approximately2.9GB; no service restarts. Replay is file-backed.
- Dedicated enabled supervisor authenticates/collects results, diagnostic plots
  and reusable checkpoints. Epoch recovery and a resource-only fallback are
  installed. The fallback shares the secondary worker's GPU lock with G2, so it
  waits rather than overlapping training on that GPU.
- No epoch or validation result exists at launch. Completion ETA will be based
  on measured full epochs and patience progress, not the warmup throughput.

The private deployment configuration remains outside the repository. No new
replication, synthetic-data arm or selection pass is chained to this refit.
