# Musashi to General Satoshi: Grouped Extractor Completion Order

Date: 2026-08-23
Authority: owner-approved architecture work; this order does not authorize a
GPU sweep or promotion.

## Delivered by Musashi

- strict nested component configuration;
- MLP, causal TCN, GRU and Transformer branch plugins;
- concat and gated fusion plugins;
- SB3 `MultiInputPolicy` integration that preserves the Gymnasium Dict;
- exhaustive/disjoint semantic grouping of the current 83 ETH features;
- materialized baseline config and focused tests;
- pretraining objective plugin interface and two initial objectives;
- research and experiment-order document.

## Required work

1. Reproduce the focused tests and adversarially inspect the Dict-to-branch
   tensor layout against a real `gym-fx` observation. Prove time and feature
   axes have not been transposed.
2. Add a train-only observation-window exporter that calls the executing
   environment path. Do not reconstruct windows independently from CSV. Its
   manifest must bind feature names/order, state keys, window size, split rows
   and timestamps, source/data/config hashes and observation-contract hash.
3. Implement the branch-pretraining runner against those exported windows.
   Save encoder weights separately per family and bind objective, topology,
   seed, epochs, early stopping and all train/monitor facts. Neither inner,
   outer nor sealed data may update weights or scheduler state.
4. Add strict pretrained-weight loading to SAC. Missing, extra, reordered or
   duplicate features; topology drift; contract drift; partial state dicts; and
   train-data hash drift must refuse before the first environment step.
5. Add a model-summary exporter for each branch, fusion and assembled actor and
   twin critics, plus readable PNG/DOT diagrams and a hyperparameter/range
   table linked from the README.
6. Draft the bounded screen contracts for branch choice, fusion choice and
   pretraining mode. They must run sequentially in that order and retain the
   flat MLP as control. Do not dispatch until Musashi reviews data boundaries,
   budgets and the primary endpoint.

## Required audit evidence

- real tensor shapes at every branch and fusion boundary;
- nonzero gradients in every branch on a tiny overfit fixture;
- permutation test proving feature-order drift refuses;
- leakage test proving outer/sealed rows cannot enter exporter or pretraining;
- round-trip load test proving encoder tensors are byte-identical;
- full effective architecture JSON and component provenance;
- focused and full-suite results, exact commits, and declared residual gaps.

