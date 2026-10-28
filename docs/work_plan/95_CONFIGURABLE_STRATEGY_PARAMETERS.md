# Configurable strategy parameters

## Actual interface

`configs/validation/directional_strategy_midpoint.json` names the six active
execution coefficients, independently for long and short:

| Parameter | Domain | Execution meaning |
|---|---|---|
| `tp_fraction` | [0,1] | interpolation within the currently feasible TP distance band |
| `sl_fraction` | [0,1] | interpolation within the currently feasible SL distance band |
| `order_fraction` | [0,1] | fraction of the maximum order capacity; zero opens nothing |

One means the maximum permitted allocation, not the full account balance:
capacity uses `rel_volume * current_balance` and the frozen leverage/lot rules.
Direction fixes the signs of TP/SL; coefficients never change side semantics.

`DirectionalRisk.from_json` validates schema, exact field names, numeric types
and domains. Unknown business overrides, Boolean numbers, NaN, infinity and
out-of-range genes cannot be silently accepted. Its ordered six-gene tuple is
the same representation already used by the DEAP evaluator. No JSON key can
relax the calendar, leverage, daily cap, required protections or acceptance
criteria. Market scenarios and score expectations are not genes.

The scenario runner consumes the file, binds its hash and effective genes to
the evidence identity, and applies the candidate to both the crossed market
cells AND the 22 broker-event witnesses:

```bash
CUDA_VISIBLE_DEVICES="" python -m tools.systematic_strategy_matrix \
  --recipe <sealed-recipe.json> --source-spec <parent-SPEC.private.json> \
  --engine-root <pinned-engine> --output <new-evidence-directory> \
  --strategy-config configs/validation/directional_strategy_midpoint.json
```

Add `--limit 12` only for a labelled smoke, never a full-grid qualification.
The absent flag retains the previous baseline interface. Completed evidence
is verified and returned without overwriting it. Changed config needs a new
identity/output root. Closed source runtimes are not modified.

## Optimization rule

Compare candidates against unchanged scenarios, acceptance rules and controls.
Each gene must demonstrate an effect on the relevant simulation outcomes
before interpreting it as identified/optimized; the earlier ineffective SL
genes remain a counterexample. Compare the proposed fitness variants as
separate frozen objectives, not tunable weights that can inflate their own
score. Correctness precedes fitness. No new DEAP search is launched by adding
this interface and none of the rejected historical optima is reinstated.

Verification: the initial JSON test failed because no loader existed; after
implementation, 48 CPU tests passed with zero skips. Native integration proved
the JSON midpoint preserves the prior case results, asymmetric fractions reach
both directions in all 22 witnesses, and zero-volume/endpoints retain their
contract. The actual JSON CLI completed a new 12-cell smoke with the same
70 comparisons and decisions as the baseline smoke. The 5832-cell completed
baseline remains bound to its original pre-JSON runtime, not relabelled.
