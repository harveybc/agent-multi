# Joint Genome And Weight Diagnostics After Phase2

Owner direction,2026-11-13: add this analysis to the later DEAP campaign,
after phase2 feature selection. This is a deferred design, not an experiment
launch or a claim of having exhausted weight-space diagnostics. The immediate
next task remains323's native counterfactual-label coverage contrast.

## Purpose

Trace the chain from candidate genes to effective configuration, trained
components, functional outputs and trading behavior. Consider topology,
module structure and training hyperparameters as possible contributors to
instability rather than attributing every failure to input data or capacity.

## Candidate Record

- Raw genome, decoded configuration, repaired constraints, active/inactive
  genes, parent identities, mutation/crossover history and effective runtime
  values. Hash both genome and effective phenotype; different genomes can
  decode to the same candidate and must not count as distinct exploration.
- Feature/support/scaler identities, initialization and training seeds,
  environment/code identities, optimizer exposure, stopping decision, runtime
  and memory. Never mix candidates trained on different supports unnoticed.
- Checkpoints by component, optimizer/RNG recovery state and fixed diagnostic
  probes. Store selected and final endpoints separately.
- Training and internal-validation performance, activity, BUY/SELL balance,
  early-close consequences, calibration/regret, drawdown and seed dispersion.
  Keep the eventual search objective distinct from these diagnostic measures.

## Analysis And Verification

1. Test each gene adapter before search: endpoint values, integer rounding,
   logarithmic transforms, categorical choices and conditional activation.
   Instantiate boundary/interior candidates and inspect actual layer sizes,
   parameter counts, optimizer groups and strategy values. Detect fixed,
   ignored, saturated or excessively quantized genes before expensive fits.
2. Summarize parameter coverage, duplicate phenotypes, boundary accumulation
   and fitness distributions. A neutral mutation can be valid; distinguish
   inactive genes from accidental failure to propagate configuration.
3. Estimate local parameter sensitivity and interactions from candidate
   records. DEAP samples adaptively and changes several genes at once; these
   associations are hypotheses, not causal estimates. Confirm promising
   explanations with paired single-factor or small factorial contrasts using
   the same data, comparable exposure and repeated seeds.
4. Link those contrasts to per-component activation drift, gradient/update
   norms, effective rank, action margins and critic error on fixed probes.
   Direct tensor differences require compatible shapes and aligned units;
   permutations and scaling symmetries can change weights without changing
   function. Across topologies, compare functional responses first.
5. Confirm any proposed weight edit or component transplant by cold replay,
   full-account evaluation and matched controls. Weight inspection alone does
   not certify why a model fails or that an edited model will improve.

## Deliverables And Gates

Produce a gene-to-effective-parameter coverage table, a parameter/interaction
diagnostic table with uncertainty across seeds, component-level functional
comparisons and a ledger of confirmed versus unconfirmed explanations.
Preserve failed and duplicate candidates with explicit reasons.

Do not expand the current search, modify the selected feature set or launch
DEAP for this task now. After phase2 closes, freeze the resulting input recipe
and declare the search space, objective, stopping and comparison protocol.
Instrumentation should accompany that campaign from its first candidate.
