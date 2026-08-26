# Gap analysis: existing code vs the Data-First SOTA Multibranch order

Date: 2026-08-26. Branch: satoshi/data-first-sota-20260826 (from
7886de39). Method: direct inspection of the working tree.

## EXISTS and satisfies the order

| Order requirement | Existing artifact | State |
|---|---|---|
| Grouped extractor, plugin-resolved, strict unknown-key refusal | `agent_plugins/grouped_features_extractor.py` (schema grouped_features.v1) | §1 DONE: real-env adversarial layout proof + order-binding refusal (94451a62); every-feature-claimed-once enforced |
| Branch plugin system | entry group `feature_branch.plugins`: mlp, gru, tcn, transformer | causal-TCN and GRU branches EXIST |
| Fusion plugin system | `feature_fusion.plugins`: concat, gated | gated fusion EXISTS |
| Semantic families | `agent_plugins/feature_families.py` (5 families, ambiguity = contract failure) | maps the 83 exactly |
| Pretraining objective plugins | `pretraining_objective.plugins`: next_step_huber, direction_bce | scaffolding only (the order says exactly this) |
| Architecture diagram exporter | `tools/export_sac_architecture.py` | exists for SAC; needs grouped-model segmentation |
| Structured action groundwork | envelope plugin (target exposure + close/hold inside native SL/TP), doc 40 A2 | execution side DONE on gym-fx branch |
| C0 mechanics evidence | `tests/test_grouped_extractor_real_env_adversarial.py` (gradient-to-every-branch, layout adversaries) | C0-grade smoke EXISTS on CPU |

## MISSING vs the order (the build list)

1. **Branches**: PatchTST (continuous return/range), TFT-style variable
   selection + temporal attention (trend/known covariates),
   TimesNet-style (volatility/distribution). TCN/GRU exist as the
   declared alternatives.
2. **Fusion**: cross-family attention (typed gated fusion exists).
3. **Heads**: configurable actor + twin-critic over the fused latent as
   PLUGINS (today SB3 default heads consume the extractor).
4. **WP-PRETRAIN runner**: executing branch-pretraining runner with
   artifact contract + resume; objectives masked-patch, hierarchical
   contrastive, multi-horizon quantile, volatility, barrier-hit.
5. **Multiresolution windows**: extractor consumes ONE (32, F) window;
   multi-window inputs (16/64 intraday once collected; 180/540 H4;
   90/365 daily) need observation-contract + preprocessor extension.
6. **Diagrams/param tables from the EFFECTIVE model** for the grouped
   route + README links.
7. **COP CDT hurdle reporting**: nothing exists; requires the USDCOP
   collector (inventory) + a reporting-layer converter with FX
   provenance. NEVER enters fitness.
8. **Screen C amendment** per §5 (C0 mechanics-only; strong arms; MLP
   as bounded matched baseline only; hierarchical gene stages).

## Collisions / design notes

- The 32-bar window is baked into the v2 observation identity (2,660,
  flattened MLP contract). Multiresolution inputs CHANGE the
  observation contract → new system identity (v3) with its own digest;
  the seam refusals we built make silent drift impossible — the v3
  contract must be authored, not patched.
- The grouped extractor consumes Dict obs `features` (window, F);
  multi-window needs either separate Dict keys per resolution (clean,
  matches the seam) or ragged packing (rejected: violates strict
  shapes).
- Pretraining artifacts must bind the SAME identity fields the P1
  aggregation now demands (316/317 lessons): digest-authoritative,
  embedded state maps.
