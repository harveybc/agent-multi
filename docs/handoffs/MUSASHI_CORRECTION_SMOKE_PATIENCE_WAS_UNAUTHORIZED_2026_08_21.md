# Musashi Correction: Smoke Patience Was Unauthorized

Date: 2026-08-21 America/Bogota
Severity: S2 scientific-contract contamination
Target: `tools/wp4_cpu_smoke.py`, introduced at `agent-multi@92828f8e`

## Finding

The smoke derives scientific stopping semantics from a runtime budget:

```python
"l1_patience": max(2, args.max_epochs // 5),
"l1_patience_start_epoch": 0,
```

Consequently `--max-epochs 50` silently became patience 10 starting at epoch
1. The owner never authorized that setting. Seed 303 stopping at epoch 13
means only that the smoke's invented patience found no monitor improvement
after epoch 3; it is not evidence of learning convergence.

The implementation lead introduced the rule in `92828f8e`. Musashi failed to
detect it before ordering four-seed replication. This correction records both
facts without transferring responsibility to the owner.

## Immediate Disposition

- Let the already-running bounded replicas finish; mutation now would destroy
  their comparability and their infrastructure/rank evidence.
- Classify every result as `MECHANICS_RANK_DIAGNOSTIC_ONLY`.
- A patience stop under this contract is never labelled convergence.
- A max-epoch stop is `RIGHT_CENSORED_BY_SMOKE_BUDGET`.
- No checkpoint from this replication may be promoted as a trained champion.

## Required Correction Before Further Training

1. Delete all derivation of patience or patience-start from `max_epochs`.
2. Require explicit CLI/config values for `l1_patience` and
   `l1_patience_start_epoch`; missing values must refuse.
3. Persist requested and effective values plus their provenance.
4. Add regression tests proving `max_epochs` changes neither patience field.
5. Preserve a dedicated mechanical-smoke profile, but name every reduced
   stopping parameter explicitly in its contract.
6. For the long easy experiment use the owner-approved starting contract:
   `max_epochs=2000`, `l1_patience=60`,
   `l1_patience_start_epoch=40`. These remain experimental parameters, not
   claimed optima.
7. Implement and pair-test the separately ordered plateau-LR controller before
   deciding whether fixed LR or adaptive LR enters the long campaign.

Correction and tests proceed immediately after the current replication packet,
without another owner phrase.
