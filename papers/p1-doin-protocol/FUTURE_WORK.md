# P1 Future Work

Ranked. Every line carries: limitation → falsifiable question → prior-art
state → required implementation/data → cheapest discriminating experiment →
decision metric (unit) → dependency / kill condition → registry ID.

## 1. Authenticated peer messages

- Limitation: identity primitives exist but network messages carry no
  signature (verified: `messages.py:43-62`); P1 cannot claim authenticated
  channels.
- Question: does per-message signing with the existing ECDSA identities keep
  claim/announcement latency overhead below 10 % at four workers?
- Prior art: candidate_unverified (PBFT anchor seeded; BFT lit unopened).
- Required: signed-envelope prototype behind a feature flag; no protocol
  redesign.
- Experiment: sign/verify microbenchmark plus a bounded four-worker replay
  with and without signatures.
- Metric: added latency per message and per candidate claim (ms, %).
- Dependency: none. Kill: overhead >20 % with no change in the declared
  cooperative threat model's needs.
- Registry: P17 (this is its first concrete step).

## 2. Verification-to-generation cost ratio as designed measurement

- Limitation: the headline "verification cheaper than production" claim is
  operational anecdote, not measurement.
- Question: what is the measured verification/production cost ratio across
  candidate classes, and is it stable across worker hardware?
- Prior art: candidate_unverified (proof-of-useful-work line unopened).
- Required: instrument verification path timing in the existing quorum-enabled
  profile; no new protocol.
- Experiment: enable verification on a bounded replayed candidate set; record
  paired production/verification durations.
- Metric: cost ratio (dimensionless) with per-class dispersion.
- Dependency: quorum profile enabled on a bounded run (Musashi packet).
  Kill: ratio ≥1 for the dominant candidate class.
- Registry: P8/P18.

## 3. Barrier-idle economics and claim ordering

- Limitation: generational barrier idles fast workers behind stragglers
  (finding AUD-F1-20260731-021); magnitude unmeasured but bracketed at
  6–14 % of fleet capacity.
- Question: does slowest-worker-first claim ordering (or bounded
  cross-barrier lookahead) recover ≥half of measured idle without harming
  population semantics?
- Prior art: first_pass (Hyperband verified; adaptive allocation is developed
  in centralized settings; decentralized shared-population EA residue).
- Required: none for measurement (logs already carry start/finish pairs);
  scheduler variants for replay only.
- Experiment: idle extraction from completed generations, then counterfactual
  replay of claim orderings.
- Metric: GPU-hours idle per generation; fraction of fleet capacity (%).
- Dependency: log extraction (Musashi). Kill: measured idle <3 % of capacity.
- Registry: P6.

## 4. Recurring minority-tip propagation study

- Limitation: equal-height competitions recur with Dragon repeatedly on the
  minority tip (finding AUD-F1-20260731-020); convergence works, latency and
  cause unknown.
- Question: is minority-tip incidence uniform across peers, or concentrated by
  route/generation timing?
- Prior art: not applicable (system-specific); fork-choice lit anchor via
  Nakamoto row.
- Required: none; node logs suffice.
- Experiment: per-peer minority-tip census plus announcement-to-adoption
  latency by route across observed heights.
- Metric: per-peer minority incidence (count/height) and adoption latency (s).
- Dependency: none. Kill: incidence uniform within noise — then it is plain
  propagation jitter and P1 reports it as such.
- Registry: P1 core evidence (feeds section VII results).
