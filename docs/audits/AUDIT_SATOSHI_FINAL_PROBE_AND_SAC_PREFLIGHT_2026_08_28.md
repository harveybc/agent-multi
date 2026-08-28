# Audit: Final Probe and SAC Preflight

Date: 2026-08-28 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@224061a0`

## Verdict

**SEAL/GENESIS/DRY-RUN ACCEPTED. PROBE SELECTION REJECTED. GPU DISPATCH REVISE
BEFORE LAUNCH.**

Independent facts:

- 29 focused tests passed.
- The candidate manifest has five complete families, ten epochs each.
- The real candidate seal and per-family encoder digests are bound into all
  eight trial genesis records.
- Independent CPU dry-run reproduced cell genesis
  `3c1c8af0...` and remained `NOT_LAUNCHED`.
- The GPU path is currently a refusal by construction.

## Findings

### DATA-SOTA-374 (S1): a refused route is selected by fallback

For `volatility_distribution`, `oscillators` and `volume_flow`, the report's
`full5_control` route is `ROUTE_REFUSED` (seed instability or adapter fit
failure). The no-ranked-route branch nevertheless writes full5 into `selected`
as `CONSERVATIVE_DIAGNOSTIC`. A refused candidate cannot be produced by the
selector. The fallback must distinguish "valid but worse than random" from
"not evaluable".

### DATA-SOTA-375 (S1): marginal random floors can inflate skill

The first addendum permits a random-floor adapter that failed the declared fit
gate to contribute its best-restored score. Underfitting the random floor raises
its loss and can make route skill look artificially positive. A failed or
unstable floor must make that task diagnostic-invalid; it cannot anchor skill.
The final report also discards the per-task `floor_fit_marginal` provenance when
reducing the floor to raw scalars.

### DATA-SOTA-376 (S1): missing predictive probes are treated as passing

Eligibility uses `all()` over whichever predictive skills remain valid. Thus a
route can be eligible with only two of quantile/volatility/barrier. The declared
rule says no predictive probe may be materially worse than random; it requires
all three valid, or an explicit incomplete-evidence result. Returns/momentum's
`0.6736` is based on quantile and volatility while barrier is invalid.

## Disposition Without Another Proxy Cycle

Do not rerun the probe screen. Preserve it as
`DIAGNOSTIC_PROTOCOL_INVALID_374_376`; retain its raw measurements.

The full5+PCGrad generation is mechanically sound and may be used as an
**auditor-chosen exploratory treatment**, not as a probe-selected winner. The
paired downstream SAC experiment itself will decide whether it provides useful
initialization. Relabel its eligibility accordingly and keep every economic
claim prohibited until that experiment completes.

