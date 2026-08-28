# Audit: Routing Common-Probe Return

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@4b610586`

## Verdict

**R0-R5 ACCEPTED AS A DIAGNOSTIC RUN; ROUTE VERDICTS ARE NOT YET SELECTION
AUTHORITY. NO NEW GENERATION WAS CORRECT.**

The common surface fixes finding 369 and the reported table follows the
predeclared rule. It establishes that self-supervised-only encoders transfer
poorly to forward predictive probes under this protocol, and that M2 auxiliary
loss degradation is not a sufficient proxy for downstream representation
quality.

## Findings

### DATA-SOTA-371 (S1): adapter convergence is not demonstrated

`_fit_adapter` calls an adapter converged when the final stochastic minibatch
loss is merely lower than the first. It has no held-out probe-fit validation,
plateau test, best-state restoration or multi-seed stability. Fixed 300 steps
may underfit different probes by very different amounts. Ratios as large as 69x
may be real, but this predicate cannot distinguish representation failure from
an incompletely fitted adapter.

### DATA-SOTA-372 (S1): solo specialists are an inappropriate hard gate

The 1.2 gate requires one shared encoder to remain within 20% of five separate
encoders, each optimized solely for its own task. Those solos are useful upper
references, not a neutral eligibility baseline. A multi-task representation can
be useful to SAC while losing more than 20% to a specialist on one auxiliary
probe. The present `NO_ACCEPTABLE_ROUTE` therefore means "no route matches all
specialists under this probe protocol", not "no useful route exists".

### DATA-SOTA-373 (S2): the new probe surface has no dedicated regression suite

The design commit adds the five-way split and over 200 lines of probe logic, but
adds no tests under `tests/`; repository search finds no test of
`common_probe_surface`, `five_way_split` or the routing ranker. Evidence from one
successful execution is not adversarial coverage. Claims about frozen encoders,
partition isolation, convergence refusals and cardinality invariance require
permanent tests.

## Accepted Scientific Reading

- `predictive3` is a promising route for oscillators, not yet promoted.
- Full5 is the strongest observed route in the three failing families.
- Self-supervised2 is unsuitable as a stand-alone predictive representation
  under the current protocol.
- No full routed generation should be minted from this table.

This is the final proxy-method correction cycle. After a validated neutral
probe screen, representation utility must be decided by a bounded paired SAC
experiment, not by more layers of auxiliary proxy optimization.

