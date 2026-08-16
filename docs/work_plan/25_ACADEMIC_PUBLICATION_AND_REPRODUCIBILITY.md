# Academic Publication and Reproducibility Program

Status: specified; evidence collection and continuous research active; no submission authorized
Version: 1.2.0
Date: 2026-08-15 (1.2.0: publication order separated from paper identifiers;
P1 verification wording aligned to the bounded-evaluation contract; P1
threat-model limitation clarified as a technical decision, not an owner
slogan — doctrine alignment order §7, Musashi verdicts Q20/C9/C10)

## 1. Purpose and Authority

The system should survive beyond its current machines, operators and software
sessions because its useful contributions are independently understandable,
testable and reproducible. This program turns validated engineering and
experimental results into a series of concise scientific papers and,
eventually, a synthesis book.

The intended publication path is:

1. an IEEE-compatible LaTeX manuscript and reproducibility package;
2. an arXiv preprint after evidence and disclosure gates pass;
3. submission to a suitable peer-reviewed venue selected from the completed
   contribution, not selected in advance to shape the evidence.

An arXiv posting is a preprint and preservation mechanism. It is not peer
review, proof of correctness or evidence of commercial performance.

Human and review responsibilities are:

| Role | Responsibility |
| --- | --- |
| Harvey | Human author, domain owner, final claim/submission/licensing authority |
| Claude ("Satoshi") | Academic research lead: literature program, research questions, novelty analysis, paper architecture, scholarly drafting and reviewer simulation |
| Codex ("Musashi") | Experimental and technical lead: implementation, experiment design, artifact integrity, reproducibility execution and technical claim verification |

AI systems are tools and reviewers, not authors. Harvey retains responsibility
for every submitted claim. AI-generated text, figures, code or analysis must be
disclosed according to the target venue's current policy.

Harvey alone selects the venue, accepts publication fees, signs copyright or
license agreements and authorizes public release.

Academic and technical authority use cross-review:

- Satoshi leads literature search, contribution framing, paper decomposition,
  manuscript structure, venue analysis and academic quality.
- Musashi leads executable hypotheses, experimental implementation, data/code
  lineage, metric reconstruction and reproducibility packages.
- Musashi verifies the technical and evidentiary content of Satoshi's proposed
  claims.
- Satoshi reviews the novelty, validity, baselines and scholarly framing of
  Musashi's experiments.
- Neither agent may approve its own material claim. Harvey resolves priorities
  and authorizes release or submission.

## 2. Non-Negotiable Scholarly Rules

1. No result is promoted merely because it is profitable or visually
   impressive.
2. Protected-test observations cannot select a method, ablation, title,
   narrative or claim.
3. Every empirical claim maps to immutable data, configuration, code commit,
   seed, split, metric and artifact hashes.
4. Null, negative and contradictory results remain visible.
5. Appropriate baselines, ablations, uncertainty, limitations and threats to
   validity are mandatory.
6. Every financial result declares period, unit, split, execution-cost regime,
   data coverage and whether it is simulated, paper or live.
7. Simulated and paper results cannot be presented as expected live profit.
8. Related work is searched systematically. Every cited source is opened and
   checked; search snippets and citation counts are discovery aids only.
9. Satoshi may reject or revise a claim, citation or paper structure within
   the academic program. It cannot manufacture evidence, decide authorship or
   submit a manuscript.
10. A paper may remain `outline`, `evidence_incomplete` or
    `negative_result`. Academic-looking prose never upgrades its state.

## 3. Common Manuscript and Evidence Contract

Every paper uses this IEEE-compatible structure unless a target venue requires
a documented change:

1. Title
2. Abstract: one self-contained paragraph, no more than 250 words
3. Index Terms: three to five terms
4. Introduction: problem, research questions and explicit contributions
5. Related Work: systematic search and claim ledger
6. Problem Formulation and Assumptions
7. Methodology
8. Experimental Design: baselines, ablations, splits and protected-test rules
9. Results
10. Discussion
11. Limitations and Threats to Validity
12. Conclusion
13. Data, Code and Artifact Availability
14. AI Disclosure, Acknowledgments and Conflicts
15. References
16. Supplementary Material

The source package for each paper is:

```text
papers/<paper-id>/
  paper.tex
  references.bib
  claims.csv
  search_protocol.md
  artifact_manifest.json
  figures/
  tables/
  supplement/
  README.md
```

`claims.csv` records claim ID, manuscript location, claim type, supporting
artifact or citation, verifier, verification date and state. Bibliographic
records are imported or entered structurally; prose generation must never
rewrite DOI, author, title, venue or year fields.

## 4. Paper Series

### 4.0 Publication Order Versus Paper Identifiers (2026-08-15)

Paper identifiers (P1, P2, P3, ...) are stable names. They are never
renumbered and they do not encode publication order.

The intended first two publication outputs are:

1. **First publication output — P1:** the DOIN protocol with bounded
   verification evidence, phrased as "a bounded verification evaluation
   compared with the search process that produced the candidate"; the
   measured verification-to-generation ratio (with workload, hardware,
   domain, metric and uncertainty) replaces that phrase only after the
   sealed measurement exists.
2. **Second publication output — the P5/P13 adversarial cross-audit method
   paper:** role-separated evidence review evaluated on the enumerated
   incident/audit corpus.

Ordinal phrases such as "first paper" and "second paper" always denote
publication order and never a paper identifier. In particular, the "second
paper" (the P5/P13 cross-audit method) is not paper P2. P2 remains the
data-first mixed-genome trading search paper regardless of when its
evidence completes or when it publishes.

### P1. DOIN Protocol and Verifiable Optimization

Working title:

> DOIN: Decentralized Evolutionary Optimization with Verifiable Champion
> Migration and OLAP-on-Blockchain

Primary question:

> Can heterogeneous peers collaborate on evolutionary optimization, avoid
> duplicate evaluation, verify improvements through a bounded verification
> evaluation compared with the search process that produced the candidate,
> and preserve queryable lineage without a central coordinator?

Until the designed verification-to-generation measurement exists (with its
workload, hardware, domain, metric and uncertainty), P1 may not claim cheap,
free or asymptotically cheaper verification (work-plan document 40 §8).

Table of contents:

1. Abstract and Index Terms
2. I. Introduction
3. II. Requirements, System Model and Threat Model
4. III. Related Work
   - Distributed evolutionary computation
   - Federated and peer-to-peer optimization
   - Blockchain and verifiable computation
   - Experiment provenance and append-only analytical ledgers
5. IV. DOIN Architecture and External Plugin Boundary
6. V. Shared Population, Candidate Identity and Duplicate Prevention
7. VI. Proof of Optimization and Champion Verification
8. VII. Controlled Flooding, Migration, Fork Handling and Finalization
9. VIII. OLAP-on-Blockchain and Content-Addressed Artifacts
10. IX. Experimental Method
    - Single-node control
    - Heterogeneous swarm
    - Node loss and rejoin
    - Message loss and competing tips
11. X. Results
    - Throughput and scaling
    - Duplicate rate
    - Champion propagation latency
    - Verification-to-generation cost ratio
    - Recovery and availability
12. XI. Security Analysis and Failure Modes
13. XII. Limitations and Threats to Validity
14. XIII. Conclusion
15. Availability, Disclosure, Acknowledgments and References

Required exhibits include a protocol sequence diagram, trust-boundary diagram,
chain/OLAP schema, scaling curves, migration-latency distribution, fault
matrix and protocol-comparison table.

Publication gate: deterministic multi-node reproduction, frozen protocol
version, explicit threat model, fault-injection experiments and verified
artifact bundle.

The initial P1 threat model is deliberately narrow. It may claim only
properties demonstrated under cooperative heterogeneous peers with crash,
omission, stale, duplicate, lazy and malformed-result behavior. Current code
contains hash-chained blocks, deterministic candidate identity and evaluator
selection, commit-reveal, quorum logic, checkpoints and deterministic fork
choice. Network messages are not presently cryptographically authenticated,
the deployed research profile can accept reported candidate results without
independent re-evaluation, and one operator controls the observed fleet.
Byzantine tolerance, Sybil resistance, collusion resistance, permissionless
economic security and externally anchored finality are non-goals until
separate implementations and adversarial experiments support them.

This scope limitation is a technical threat-model decision of the research
program, retained on auditor review (2026-08-15, verdict C10: technically
honest). It is not an owner slogan and was not authored as owner speech; it
must not be quoted as one. The enumerated behavior list and non-goals above
are the authoritative statement of P1's claim boundary.

### P2. Data-First Mixed-Genome Trading Search

Working title:

> Leakage-Safe Data-First Genome Search for Multi-Asset Trading Policies

Primary question:

> Does jointly searching point-in-time data, preprocessing, observation,
> policy, risk and execution choices outperform model-only optimization under
> leakage-safe chronological evaluation?

Table of contents:

1. Abstract and Index Terms
2. I. Introduction and Research Questions
3. II. Related Work
   - Algorithmic-trading evaluation
   - Automated feature and preprocessing search
   - Evolutionary AutoML and reinforcement learning
   - Leakage and backtest overfitting
4. III. Point-in-Time Data Model and Leakage Threat Model
5. IV. Mixed Genome
   - Data sources and feature groups
   - Preprocessing and observation windows
   - Policy/model choices
   - SL/TP, activity and order-routing genes
6. V. Hierarchical Screening and DOIN Level 2 Optimization
7. VI. Fitness, Activity Eligibility and Protected-Test Firewall
8. VII. Experimental Design
   - Assets and timeframes
   - Naive and model-only controls
   - Fixed-data versus searched-data controls
   - Easy, nominal and stress execution regimes
9. VIII. Results
10. IX. Ablations and Sensitivity
11. X. Cross-Asset Transfer and Failure Analysis
12. XI. Limitations and Threats to Validity
13. XII. Conclusion
14. Availability, Disclosure, Acknowledgments and References

Required tables include the feature inventory, publication-lag contract,
genome schema, chronological splits, ablations, coverage/activity and metric
units.

Publication gate: completed asset series, archived champions, model-only and
fixed-data controls, one-time protected-test opening and reconstruction of
reported metrics from atomic OLAP facts.

### P3. Hierarchical Portfolio Control

Working title:

> Hierarchical Multi-Horizon Portfolio Control with Asset Policies, Regime
> Gating and Risk-Constrained Allocation

Primary question:

> Do frozen asset specialists, probabilistic opportunity gates and a
> risk-constrained allocator improve portfolio behavior over equal-weight and
> static-risk baselines?

Table of contents:

1. Abstract and Index Terms
2. I. Introduction
3. II. Related Work
   - Portfolio optimization
   - Hierarchical and mixture-of-experts control
   - Regime detection and change points
   - Multi-timescale reinforcement learning
4. III. Portfolio Problem, Information Set and Constraints
5. IV. Layered Architecture
   - Asset policy cells
   - Rush/opportunity detector
   - Allocator and risk governor
   - Venue and execution constraints
6. V. Training and Optimization Order
7. VI. Baselines and Fitness Contract
8. VII. Experimental Design
9. VIII. Results
   - Return, RAP and drawdown
   - Concentration, turnover and diversification
   - Opportunity-gate calibration
10. IX. Ablations
    - No gate
    - No hierarchy
    - Static versus weekly allocation
    - Short- or long-horizon subsets
11. X. Stress, Counterfactual and Causal Analyses
12. XI. Limitations and Threats to Validity
13. XII. Conclusion
14. Availability, Disclosure, Acknowledgments and References

Publication gate: at least three qualified short-horizon and three qualified
long-horizon cells, frozen artifacts, non-overlapping evaluation, calibrated
opportunity gate and complete baseline set. This paper is not yet
evidence-complete.

### P4. Simulation-to-Paper Execution Parity

Working title:

> From Backtest to Multi-Venue Paper Trading: Protected-Order Parity and
> Portfolio Reconciliation

Primary question:

> Which simulator assumptions survive authenticated paper execution across
> broker APIs, and can fail-closed order and reconciliation contracts bound the
> discrepancy?

Table of contents:

1. Abstract and Index Terms
2. I. Introduction
3. II. Related Work
4. III. Execution and Safety Contracts
   - Mandatory SL/TP
   - Order types
   - Idempotency and reconciliation
5. IV. Simulation and Cost Model
6. V. Global Portfolio Ledger and Venue Adapters
7. VI. Capability Discovery and Protected Canaries
8. VII. Experimental Design
   - Alpaca Paper
   - IBKR Paper
   - OANDA/MT5 when available
   - Replay, disconnect and stale-data scenarios
9. VIII. Simulation-versus-Paper Results
10. IX. Fills, Spread, Slippage, Rejection and Staleness
11. X. Fault Injection and Recovery
12. XI. Regulatory and Product Boundary
13. XII. Limitations and Threats to Validity
14. XIII. Conclusion
15. Availability, Disclosure, Acknowledgments and References

Publication gate: at least seven days of paper observation, protected canaries,
full order/position/cash reconciliation, clock and currency checks, and no live
profit claim.

### P5. Continuous Audit and Swarm Recovery

Working title:

> Evidence-Driven Continuous Audit and Recovery for Heterogeneous
> Decentralized Machine-Learning Swarms

Primary question:

> Can deterministic evidence packets, bounded independent audit and
> fail-closed watchdogs detect distributed, resource and broker failures
> without granting a central auditor runtime authority?

Table of contents:

1. Abstract and Index Terms
2. I. Introduction
3. II. Related Work
   - ML observability and provenance
   - Distributed health and incident management
   - Continuous assurance and audit
   - Agent trust boundaries
4. III. System and Failure Model
5. IV. Deterministic Evidence Plane
6. V. Watchdogs, Telegram Reporting and Alert State
7. VI. Independent Auditor and Authority Separation
8. VII. Isolation and Test-Evidence Economy
9. VIII. Incident Corpus and Experimental Method
10. IX. Detection, Recovery and Cost Results
11. X. Ablations
    - Heartbeat-only
    - Port-only
    - No independent audit
    - No content-addressed evidence
12. XI. Security, Privacy and Social-Input Boundaries
13. XII. Limitations and Threats to Validity
14. XIII. Conclusion
15. Availability, Disclosure, Acknowledgments and References

Publication gate: versioned incident taxonomy, reproducible fault scenarios,
detection latency, false-alert rate, runtime overhead and recovery evidence.
Personal and operationally sensitive security details are excluded.

P5 uses the following conflict controls:

1. Its incident corpus is enumerated before analysis from every incident in
   work-plan documents 13, 15, 16 and 20, every S0-S2 audit finding, and every
   deterministic alert escalated to an incident. Exclusions require a dated
   reason and remain visible.
2. Musashi reconstructs every quantitative effectiveness claim from raw
   timestamps and immutable evidence packets; Satoshi does not verify its own
   effectiveness claims.
3. P5 requires external human or independent technical review before any
   preprint. Every cited Satoshi audit artifact carries an explicit
   self-evaluation conflict disclosure in addition to the AI disclosure.

## 5. Later Synthesis Book

Provisional title:

> Verifiable Adaptive Intelligence: From Data and Optimization to Portfolio
> Execution

Provisional parts:

1. Foundations and trust models
2. Point-in-time data and causal context
3. Trading policies and execution
4. DOIN and verifiable decentralized optimization
5. Hierarchical portfolio control
6. Multi-venue execution and reconciliation
7. Operations, audit and recovery
8. Deployment, incentives and business boundaries

Book drafting does not begin until at least three papers have frozen evidence
packages and one paper has received external technical review.

The book is not the end of the research program. New research questions,
replications, negative results and protocol extensions enter through
`26_CONTINUOUS_RESEARCH_AND_INNOVATION_PROGRAM.md`. Papers P1-P5 are the first
registered series, not a finite backlog.

## 6. Related-Work and Citation Ledger

Every candidate source records:

- canonical title, authors, year and venue;
- DOI, arXiv identifier and canonical URL when available;
- search query and access date;
- contribution and exact section/page supporting the manuscript claim;
- source type: primary paper, standard, official documentation or survey;
- correction/retraction state;
- Satoshi, Musashi and Harvey review state.

Preferred discovery and verification sources are IEEE Xplore, ACM Digital
Library, arXiv, Crossref, Semantic Scholar and official project/standards
documentation. Original method papers are preferred over summaries.

## 7. Reproducibility Package

Each evidence-complete paper publishes or documents:

1. exact repository commits and environment lock;
2. content-addressed artifact manifest;
3. licensed-data hashes and lawful reconstruction procedure;
4. resolved configurations, seeds, splits and fitness definitions;
5. compact OLAP export and exact figure/table queries;
6. figure and table generation scripts;
7. negative, failed and excluded-run register;
8. deterministic tests and package hashes;
9. data/code/model licenses and AI-use disclosure.

Paid or restricted raw data is never redistributed. The package records
vendor, product, as-of timestamp, permitted hashes and a reconstruction path
when licensing allows it.

## 8. Lifecycle and Review Gates

```text
idea
  -> registered research question
  -> related-work search
  -> outline
  -> claim/evidence matrix
  -> evidence complete
  -> independent red-team
  -> clean-room reproducibility replay
  -> human approval
  -> arXiv preprint
  -> peer-reviewed submission
  -> correction and maintenance
```

No stage transition is automatic. Satoshi records academic readiness, Musashi
records technical/reproducibility readiness, and neither substitutes for the
other. Harvey authorizes public release and submission.

When no paper gate or urgent audit is ready, Satoshi continues the bounded
research queue defined in document 26. Completion of P5 never places the
academic role in an idle state.

## 9. Authoritative Publication Guidance

Policies must be rechecked at submission time:

- IEEE conference manuscript structure:
  <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/structure-your-paper/>
- IEEE authoring tools and templates:
  <https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/>
- IEEE research reproducibility:
  <https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/research-reproducibility/>
- IEEE submission, peer-review and AI-use policies:
  <https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/guidelines-and-policies/submission-and-peer-review-policies/>
- arXiv submission guidance:
  <https://info.arxiv.org/help/submit/index.html>
