# 43. Causal Transformation Evidence and DOIN Admission

Status: T0 v3 mechanics accepted; T1 v4 externally reviewed; T2 public
utility execution under correction; B4 v7 runs separately; work plan 44 active

Date: 2026-09-07

Independent audit update: T0 is `T0_V3_MECHANICS_ACCEPTED`; T1 v4 is
`T1_V4_EXTERNALLY_REVIEWED_FOR_T2_ONLY`. Musashi independently passed the 57
T1 adversarial tests and re-derived all 1,116 records from the bound arrays;
the complete quantitative publication matched exactly. The review record is
`../audits/evidence/MUSASHI_T1_V4_EXTERNAL_REVIEW_2026_09_06.json`, SHA-256
`a0414440bda4785618943d5d7958b203e7747bde05e317804fb2bb4407eb1a8b`.
T2.0-T2.4 are active under
`../handoffs/MUSASHI_TO_GENERAL_SATOSHI_T2_PUBLIC_UTILITY_GATE_ORDER_2026_09_06.md`.
T3, T4, DOIN transformation genes, T5 and live use remain closed.

Owner decision: keep the representation-selection proposal as the doctoral
submission; incorporate the useful transformation research into the trading
and DOIN program as an early, separately falsifiable evidence track

Superseding integration note (2026-09-07): this plan is the transformation
admission ladder, not the complete data-centric program. The accepted
model-capacity, memorization, description-length and per-feature information
work is binding work in
`44_DATA_CENTRIC_SIGNAL_MODEL_INFORMATION_AND_CAPACITY.md`. It is not an
optional discussion or a doctoral-proposal-only appendix. No later optimizer,
feature-selection, topology-search or live-candidate plan may claim that this
work was absent from the roadmap.

## 1. Decision

The project will test whether causal input transformations improve downstream
learning before enlarging feature, representation or topology searches. A
transformation is not accepted because it smooths a series, improves
reconstruction, resembles a communications operation or appears in a doctoral
proposal. It must pass a sequence of increasingly applied gates.

The operational meaning of the project's data-centric objective is not that a
single universally perfect representation exists. It is that every retained
input and transformation has explicit provenance, timing, noise and
missingness facts, a measured information/utility profile, an identity control
and a falsifiable reason to be present. Paid data receives no exemption from
these requirements.

The sequence is:

```text
T0 operator and observation contracts
  -> T1 known-truth CPU calibration
  -> T2 public out-of-family utility
  -> T3 optional selector and abstention
  -> T4 bounded DOIN admission
  -> T5 financial confirmation
  -> L0 shadow parity
  -> L1 Paper/Demo canary under separate approval
```

A null result closes the affected branch and preserves the identity input. It
does not block unrelated optimization. No active or completed campaign is
retrofitted. In particular, B4 retains its frozen observation, cost, execution
and result contracts byte for byte.

## 2. Scope and Ownership

| Repository | Responsibility |
| --- | --- |
| `preprocessor` | Reference operator protocol, deterministic CPU implementations, known-truth generators and focal tests |
| `agent-multi` | Designs, materialization, trial ledger, adjudication, applied evaluation and campaign bindings |
| `financial-data` | Versioned source bytes, timestamp/availability contracts and later financial materializations |
| `doin-plugins` | Thin adapter with byte-parity to the accepted reference implementation, only after T2 |
| `doin-domains` | Licensed operator genes and bounded parameter domains, only after T2 |
| `feature-extractor` | STEP 11 corruption training if its separate gate opens; never the T0-T2 owner |
| `gym-fx` | Consumes already materialized causal observations; never fits a transformer during evaluation |
| `lts` | Shadow/release consumption of a frozen transformation artifact after live gates |

`predictor` is not the implementation owner of this track. Frozen forecasting
models may be used as assays, but the production transformation contract does
not live in that repository. The standalone `preprocessor` executable is not
co-installed into a DOIN runtime environment merely because both projects use
similarly named plugin groups.

## 3. Evidence States

Each operator/configuration has one typed state:

| State | Meaning |
| --- | --- |
| `REGISTERED_UNTESTED` | Schema-valid candidate; no utility claim |
| `LAB_CALIBRATED` | Causal mechanics and known-truth diagnostics pass in the regimes named by the evidence |
| `LAB_REJECTED` | Known-truth, causality, determinism or preservation gate failed |
| `PUBLICLY_ELIGIBLE` | Incremental out-of-family utility is demonstrated on the public bank |
| `PUBLICLY_INELIGIBLE` | Public result is null/harmful or cost exceeds the predeclared value margin |
| `INCONCLUSIVE` | Support, precision or resource completion is insufficient |
| `FINANCIAL_CONFIRMED` | A new, post-B4 financial identity confirms incremental value |
| `LIVE_SHADOW_ELIGIBLE` | Batch/incremental parity and read-only shadow behavior pass |
| `LIVE_RELEASE_ELIGIBLE` | Separate Paper/Demo release evidence and approval exist |

The producer may emit measurements and a candidate disposition. Admission is
always re-derived by the consuming gate from observation-level records. No
operator may self-assert an eligibility label as authority.

## 4. T0: Contracts Before Algorithms

T0 defines an operator as a versioned directed acyclic graph whose nodes have:

- stable operator and implementation identities;
- exact input/output column schemas and ordering;
- fit role, fit interval and immutable fitted-parameter artifact;
- causal lookback and availability/finalization requirements;
- applicability constraints and a typed refusal path;
- deterministic batch and incremental transformations;
- resource measurements and failure facts; and
- a canonical digest over topology, parameters, code and input contract.

Required invariants:

1. Fitting uses only the declared training role.
2. The same accepted prefix produces the same output bytes in batch and
   incremental replay.
3. Chunking, row order and process restart cannot change accepted output.
4. A missing, duplicate, reordered, non-finite or temporally unavailable input
   is rejected or represented by an explicitly licensed missingness policy.
5. Identity is always an available no-op control.
6. Invalid graph compositions are absent, not assigned a bad score.
7. Fitted artifacts round-trip exactly and never depend on the current working
   directory or machine topology.

STEP 01 belongs here as an observation contract, not as a trainable transform.
It distinguishes point observations, interval aggregates, timestamp labels,
finalization times and asynchronous availability. A bar is an interval
aggregate; a missing bar is not automatically a closed market.

## 5. T1: Known-Truth CPU Bank

### 5.1 Signal families

The minimum bank contains independently generated, seeded families:

- canonical piecewise/smooth denoising signals such as Blocks, Bumps,
  HeaviSine and Doppler;
- sinusoids, chirps and amplitude-modulated signals with known frequency and
  phase structure; and
- multivariate state-space signals with known common and variable-specific
  components.

The bank varies white Gaussian, temporally correlated, cross-variable
correlated, heteroscedastic, impulsive, missingness and delayed-observation
perturbations. Per-variable SNR includes at least
`{infinity, 20, 10, 5, 0, -5}` dB and both homogeneous and heterogeneous
assignments. A generator used by a strategy or prediction-noise experiment is
not accepted as this bank merely because it has a `noise_std` setting: T1
requires known clean signal, realized perturbation and exact provenance at the
input-variable level.

### 5.2 Initial controls and candidates

Keep the first comparison deliberately small:

- identity;
- trailing mean;
- trailing robust median;
- exponential moving average; and
- forward-only local-level/Kalman filtering whose parameters are fit on the
  training role.

A centered or two-sided filter may appear only as a non-deployable diagnostic
ceiling. It cannot enter an eligibility comparison.

### 5.3 Required measurements

Where clean signal is known, report per variable and regime:

- SNR estimation error and ordering accuracy across variables;
- reconstruction error and SNR gain;
- delay, phase and amplitude distortion;
- discontinuity, extreme and tail preservation;
- missingness behavior and false structure introduced;
- residual whiteness plus the incremental forecasting/detection value of the
  residual; and
- CPU wall time, peak RSS and output expansion.

The downstream comparison uses the same frozen assay, tasks, splits and
budget for:

```text
X
D(X)
[X, D(X), X-D(X)]
```

The third arm uses matched model capacity or an explicit capacity control. A
reconstruction or SNR improvement is never sufficient if downstream held-out
utility, calibration or tail preservation worsens. If the residual retains
target-relevant information, the result is reported as a transformation, not
as verified noise removal.

### 5.4 T1 decision

An operator becomes `LAB_CALIBRATED` only for the named perturbation regimes
when it is causal, deterministic, parameter-custodied and does not create a
material preservation failure. T1 does not authorize DOIN genes, financial
claims or live use.

## 6. T2: Public Utility Gate

T2 uses public, non-financial tasks and leaves out entire datasets or task
families. It starts with a bounded census and pilot, then freezes the matrix,
unit of generalization, margins and compute budget before scoring.

The initial bank should remain small enough to complete:

- a representative subset of the Monash forecasting archive;
- ETTh1 and Weather as multivariate confirmations; and
- a bounded M4 subset only if it adds task-family support rather than more
  correlated seeds.

Identity, a best fixed development transform, random search, Hyperband/ASHA
and BOHB or SMAC-HB are the required search controls where applicable. Frozen
downstream assays begin with a seasonal naive method, lagged linear model and
small neural model. A more complex model is confirmatory, not the gatekeeper.

`PUBLICLY_ELIGIBLE` requires all of the following:

1. causal and artifact-contract gates remain valid;
2. incremental utility over identity survives out-of-family evaluation;
3. the effect is not explained only by extra input dimensions or model
   parameters;
4. diagnostic, failed-attempt and transformation costs are included;
5. tails/extremes and calibration do not cross predeclared harm margins; and
6. the result has sufficient task-level support and precision.

Seeds quantify run variability; they do not replace independent tasks. A null,
negative or underpowered result is retained and closes or narrows that
operator family.

## 7. Disposition of the Original Thirteen Steps

The communications analogy is a source of hypotheses, not a serial pipeline.
This table is the controlling disposition.

| Step | Disposition | Work-plan use |
| --- | --- | --- |
| 01 sampling | **Keep now as contract** | Sampling, aggregation, timestamp and availability semantics in T0; no Nyquist slogan applied blindly to OHLC bars |
| 02 noise/SNR | **Keep now as calibrated diagnostics** | Known-truth SNR in T1; indicators only, never claimed true SNR, on natural data |
| 03 denoising | **Execute now** | T1's first falsifiable operator family and the only initial transformation code |
| 04 quantization/companding | **Conditional after T1** | CPU scalar-resolution sweep; enters T2 only if a nontrivial precision region preserves or improves utility and tails |
| 05 source coding | **Split** | Keep entropy rate, surprisal and innovation as diagnostics; reject Huffman/ANS bitstreams as model inputs; no MDL generalization claim without evidence |
| 06 time-frequency | **Conditional** | Trailing magnitude/complex representations after T1/T2; phase separate; no centered Hilbert transform or whole-series leakage |
| 07 matched detection | **Separate and deferred** | Detector contract, not another autoencoder; synthetic matched-filter recovery first; enters applied work only after its representation gate |
| 08 equalization | **Conditional** | Only for a named operational distortion above existing z-score normalization; null or tail damage closes it |
| 09 common/private | **Conditional** | Train-only decomposition with `[X,C,U]`; shared correlation is not deleted as interference |
| 10 synchronization | **Keep metadata now, defer correction** | Availability age/finalization belongs in T0; estimated lag carries uncertainty and never pulls future observations backward |
| 11 controlled corruption | **Separate robustness track** | Train-only masking/corruption of the existing extractor after T2; not a preprocessor operator and not part of the initial bank |
| 12 adaptive routing | **Blocked by evidence** | Opens only with at least two eligible modes and a measured oracle routing gap; identity/abstention remains a mode |
| 13 branch allocation | **Deferred** | Static width/budget allocation only after multiple useful branches; `64:32:32:32` is a local configuration, not an information-theoretic law |

Compression-derived sparse coding, progressive representations and distributed
source separation remain references for STEP 07, the doctoral
representation-selection program and STEP 09 respectively. They do not create
additional immediate modules.

## 8. T3-T5 and DOIN

### 8.1 T3 selector

T3 opens only after T2 contains enough task/operator outcomes to support a
non-vacuous holdout. Start with nearest-task retrieval and a tree-based
surrogate. It may recommend identity, a short list, an additional evaluation
or abstention. It cannot declare an applied champion. Advanced meta-learning
is conditional on simple baselines and an out-of-family gain.

### 8.2 T4 DOIN admission

Only `PUBLICLY_ELIGIBLE` operators enter a new versioned DOIN domain. The first
genome is bounded to:

```text
operator_id
per_variable_mask
licensed_parameter_choice
```

No arbitrary graph, executable expression or unlicensed numeric interval is a
gene. Start with identity versus one eligible operator. Composition opens only
after at least two operators each pass separately and a bounded interaction
pilot demonstrates headroom.

The `doin-plugins` adapter must reproduce reference outputs byte for byte from
the same input prefix and fitted artifact. DOIN re-evaluates candidates under
the financial contract; public eligibility is admission to search, not a
winner declaration. L3 may later recommend warm starts or top-k candidates but
cannot bypass L2 or held-out evidence.

### 8.3 T5 financial confirmation

T5 starts only after B4 closes and uses a new campaign, observation-contract,
dataset and result identity. It compares the current financial input contract,
the best fixed public operator and a selector short list under paired budgets.
No B4 score, checkpoint or population is reinterpreted.

## 9. Live and Weekly-Flat Promotion

A financial result cannot move directly to execution. The promotion ladder is:

1. reproduce the fitted operator and transformed financial dataset from
   content-addressed artifacts;
2. prove batch versus incremental output parity on identical closed bars;
3. prove feature order, scale, missingness, latency and availability parity
   between research and the read-only live collector;
4. run raw-control and transformed-candidate paths in shadow on the same bars,
   with every candidate command suppressed;
5. compare feature drift, action divergence, inference latency and overlay
   behavior without changing the active seat; and
6. request separate owner approval for a bounded Paper/Demo canary with a
   manifest rollback.

Any accepted live transformation creates a new observation contract and model
artifact. It cannot hot-patch an existing model. The weekly-session overlay,
native protection, flatten/reopen duties and venue reconciliation retain
priority over model output. Preprocessing evidence cannot authorize or replace
the MT5 session collector, its key ceremony, its coordinated window or its
direct venue evidence.

## 10. Orchestration

1. Preserve B4 under its own authority chain; T2 cannot alter its materialization or results.
2. Do not alter B4 materialization, cells, observation contract, comparator or
   result lineage.
3. Retain T0 v3 and T1 v4 under their reviewed identities. If a B4 GPU campaign
   is later active, cap T2 CPU/RSS/I/O so it cannot starve the campaign.
4. T0-T1 independent review is complete; preserve its external record by exact digest.
5. Execute T2.0-T2.4 before feature selection, new representation searches or
   any DOIN transformation gene.
6. Only after T2, implement T4 adapters and then T5 under a new identity after
   B4 closure.
7. Keep STEP 04-13 branches parked until their individual prerequisites in
   section 7 are satisfied.
8. Execute the model-information program in work plan 44. Its bounded CPU
   calibration may run beside T2 and B4, subject to resource isolation; its
   results cannot modify either frozen campaign.
9. Before any new post-B4 feature, representation or topology search, publish
   the per-feature evidence cards and the model-capacity/description pilot
   disposition required by work plan 44. A null result remains a valid
   disposition and cannot be bypassed by calling the work exploratory.

No GPU, venue connection, service restart, key generation, live command,
position change, checkpoint promotion or sealed-data read is authorized by
this document.

## 11. Acceptance

This integration is complete when:

- T0 defines one tested operator/artifact protocol;
- T1 yields observation-level, reproducible known-truth evidence and a typed
  result for each initial denoiser;
- T2 publishes a complete public matrix and re-derived eligibility list;
- no ineligible operator reaches the DOIN gene space;
- T4 parity is exact and T5 uses a new post-B4 identity;
- live shadow parity is demonstrated before any Paper/Demo canary; and
- every null result removes work instead of being renamed as progress.
