# P5 Future Work

Conflict notice: Satoshi operates the audited process and Musashi built the
monitored infrastructure; every line below inherits the document 25 conflict
controls (enumerated corpus, raw-timestamp reconstruction, external review).
Format per line: limitation → falsifiable question → prior-art state →
required implementation/data → cheapest discriminating experiment → decision
metric (unit) → dependency / kill condition → registry ID.

## 1. Incident-corpus manifest and latency extraction

- Limitation: the incident record is prose across documents 13/15/16/20 and
  audit reports; no machine-readable corpus with measured latencies exists.
- Question: what are the actual detection and recovery latencies per incident
  and per detecting tier, reconstructed from raw timestamps?
- Prior art: not applicable (internal corpus construction).
- Required: `incidents.csv` per the Packet E schema; Musashi extraction
  scripts with published hashes; Satoshi re-run.
- Experiment: none — measurement task.
- Metric: detection latency (s), recovery latency (s), per-tier attribution.
- Dependency: enumeration-rule hash pinned. Kill: n/a.
- Registry: P5 core (H0); roadmap 032c/032f.

## 2. Monitoring ablations on replayed incidents

- Limitation: the claim that layered monitoring beats simple heartbeats is
  currently architecture prose.
- Question: which incidents in the corpus would heartbeat-only, port-only and
  no-independent-audit configurations have detected, and how late?
- Prior art: candidate_unverified (ML observability rows seeded).
- Required: replay/tabletop evaluation harness over the corpus — no live
  fault injection required for the first pass.
- Experiment: per-incident counterfactual detection walk-through with blinded
  labels (doc 25 controls).
- Metric: detection coverage (%) and latency delta (s) per configuration.
- Dependency: item 1 corpus. Kill: no configuration separates — cross-layer
  claim is dropped and the null is published.
- Registry: P5 ablations (H0).

## 3. Cross-audit effectiveness under blinded scoring

- Limitation: "role-separated review reduces false closure" is motivated
  (verified: Huang et al., ICLR 2024 — intrinsic self-correction is weak) but
  unproven for this setting; both candidate reviewers are conflicted.
- Question: does cross-agent review reduce false-closure and
  unsupported-claim rates versus single-agent review on the enumerated corpus
  after controlling for model and evidence access?
- Prior art: first_pass (Huang et al. opened; governance/audit lit unopened).
- Required: blinded scoring protocol with external labeler (doc 25 control 3).
- Experiment: P13's registered design; the mandatory negative cases (Satoshi's
  withdrawn 011, Musashi's Arendt designation) are in-corpus and may not be
  excluded.
- Metric: false-closure rate, unsupported-claim rate (per decision), detection
  rate for seeded defects.
- Dependency: external reviewer availability. Kill: P13's registered kill
  condition — no advantage after controls.
- Registry: P13.

## 4. Audit token-economy cost curve (with P19 tie-in)

- Limitation: the economy layer records reservations, but cost-per-finding
  and the tier delegation ratio are not yet analyzed as results.
- Question: what does a finding cost by tier (deterministic, cheap-model,
  Satoshi), and how did the cost curve move as tier-0 infrastructure landed?
- Prior art: not applicable (self-measurement; methodology standard).
- Required: none — packets and reports already carry the data.
- Experiment: none — longitudinal extraction across the audit corpus.
- Metric: reserved tokens and CPU-seconds per verified finding, per tier.
- Dependency: item 1. Kill: n/a.
- Registry: P5 results (H0); probe-cost dimension shared with proposed P19.
