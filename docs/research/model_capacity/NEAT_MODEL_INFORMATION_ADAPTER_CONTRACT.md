# NEAT Model-Information Adapter Contract

Status: interface specified; no NEAT experiment is authorized or claimed here

Date: 2026-09-07

## Purpose

The M0-M2 pilot uses fixed-topology threshold neurons and one-hidden-layer
MLPs. Graph measures such as diameter, eccentricity and modularity are static
or uninformative there. A later NEAT experiment may expose them only through
this adapter, because NEAT changes topology during learning.

## Input identity

Every observation must bind:

- run, population, generation, genome and seed identities;
- task/generator identity and train, calibration and evaluation roles;
- NEAT implementation and complete effective configuration digests;
- node and enabled-edge inventories with stable innovation identifiers;
- activation, aggregation, bias, response and recurrent-edge attributes; and
- cumulative unique observations, repeated exposures and fitness evaluations.

Two observations with different topology, attributes or temporal roles are
different objects even if their scalar fitness is equal.

## Required output

At each predeclared generation and at selection, the adapter emits:

1. predictive performance on train, calibration and untouched evaluation;
2. sample-specific memorization where the generator exposes exceptions;
3. nodes, enabled edges, recurrent edges, components and cycle status;
4. directed and undirected graph measures, each naming its convention;
5. component-aware diameter and eccentricity, never a fabricated finite value
   for a disconnected graph;
6. modularity with algorithm, resolution and random seed bound to identity;
7. weighted-graph sensitivity over a predeclared absolute-weight threshold
   grid, including the unweighted enabled-edge graph;
8. canonical genome serialization and estimator-specific compressed lengths;
9. activation effective dimension on a fixed calibration probe; and
10. CPU/GPU time and memory overhead attributable to the adapter.

## Selection and failure rules

- Evaluation data cannot select a generation, topology or threshold.
- A post-stop diagnostic lineage cannot overwrite the selected genome.
- Missing, disconnected or undefined graph measures remain typed unavailable;
  they are not coerced to zero.
- Repeated weights or compressed genomes do not imply unused capacity.
- A graph metric becomes an optimizer feature only after incremental utility
  over parameter count, edge count, validation fitness and learning-curve
  controls on unseen tasks.
- Failure to improve a decision is a valid negative result and closes the
  admission gate.

## Scope boundary

This document defines the later adapter surface required by work plan 44 M2
and M5. It does not execute NEAT, modify an optimizer, license a DOIN gene or
change B4, T2, financial, live or venue behavior.
