# P2 Future Work

Format per line: limitation → falsifiable question → prior-art state →
required implementation/data → cheapest discriminating experiment → decision
metric (unit) → dependency / kill condition → registry ID.

## 1. Model-only and fixed-data control campaigns

- Limitation: P2's central claim (data-first joint search beats model-only)
  has no controls yet; without them there is no result.
- Question: does the mixed genome beat a model-only genome and a fixed-data
  genome on identical asset/seed/protocol under the same activity floors?
- Prior art: candidate_unverified (evolutionary AutoML rows seeded; unopened).
- Required: two control campaign configs reusing the existing evaluator; no
  new code path.
- Experiment: bounded control runs on USDCAD@4h after job 0 completes, same
  seeds and budget class.
- Metric: validation robust weekly RAP delta (fraction/week) with three-seed
  dispersion.
- Dependency: job-0 champion archived. Kill: controls match or beat the mixed
  genome — which is itself a publishable negative result.
- Registry: P2 core (H0).

## 2. Data-contract transfer across assets

- Limitation: a searched data contract may be asset-idiosyncratic; the paper
  currently cannot distinguish discovery from overfitting-to-asset.
- Question: does the USDCAD-searched contract, frozen, outperform each new
  asset's default contract when transplanted (and vice versa)?
- Prior art: candidate_unverified.
- Required: none beyond config transplantation.
- Experiment: cross-asset transplant matrix over the next two campaign assets.
- Metric: validation RAP delta under transplant (fraction/week).
- Dependency: second asset campaign. Kill: transfer deltas indistinguishable
  from seed noise.
- Registry: P2 ablation section.

## 3. Genome-family attribution

- Limitation: a winning mixed genome does not reveal which gene families
  earned the fitness; reviewers will ask.
- Question: which genome families (data/preprocessing/observation/training/
  risk) carry the improvement, measured by frozen-family ablations?
- Prior art: candidate_unverified (ablation methodology standard; specific
  application unopened).
- Required: family-freeze evaluation mode (small evaluator flag).
- Experiment: leave-one-family-frozen ablations on the archived champion.
- Metric: per-family RAP contribution (fraction/week) with sign stability
  across seeds.
- Dependency: item 1 controls exist. Kill: attribution unstable across seeds —
  reported as a limitation, not hidden.
- Registry: P2 ablations (H0).

## 4. Activity-floor sensitivity

- Limitation: the 12-trade annual validation floor (incident-derived) shapes
  the search landscape in unmeasured ways.
- Question: how do champion selection and fitness ranking change across floors
  {6, 12, 24}, and does the floor create boundary-gaming candidates?
- Prior art: not applicable (system-specific guard; incident evidence in
  doc 20).
- Required: re-scoring of already-evaluated candidates under counterfactual
  floors — no retraining.
- Experiment: offline re-ranking of the completed candidate set per floor.
- Metric: rank correlation across floors; count of floor-boundary candidates.
- Dependency: ≥1 completed generation set (exists). Kill: rankings invariant —
  then the floor is a pure safety guard and the paper says so.
- Registry: P2 sensitivity (H0).
