# Academic Publication Program Audit

Audit ID: AUDIT-ACADEMIC-20260731-01
Timestamp and timezone: 2026-07-31 01:50 America/Bogota (UTC-5)
Auditor and academic research lead: Satoshi (dual role — conflict handling in
finding 015)
Requested by: Harvey, via Musashi-authored invocation
Governing documents: work-plan document 25 (v1.0.0, commit `df22d2b3`), role
commit `e94e9344`, task handoff v1.1.0
Scope: authority-model audit of the academic role change; P1-P5 contribution,
overlap, baseline, evidence and risk review; evidence-state classification;
companion deliverables (related-work ledger seed, research roadmap).
Excluded scope: no manuscript drafting; no submission action; no
external-index bulk search executed this session (see Method Honesty).

Method honesty, stated up front: bibliographic seeding in this cycle draws on
(a) references already versioned and URL-verified in work-plan document 10,
and (b) canonical sources known with high confidence, every one marked
`candidate_unverified` in the ledger until opened and checked per the citation
rules. No DOI, author list, venue or year has been guessed; unknown fields are
empty with `needs_access`. Opening and verifying every ledger row is the next
bounded academic task — novelty conclusions below are therefore **provisional
by design** and are labeled inference, not observation.

## 1. Findings (severity order)

### AUD-GEN-20260731-014 — Unregistered participant "Arendt" in the audit closure chain

- Severity: **S3** (governance integrity)
- Confidence: high (observed)
- Status: open — not closable by me; I am a party to the documents involved
- Observation: `CODEX_AUDIT_TRIAGE_2026_07_31.md` line 6 names "Independent
  corroborator: Arendt". No document in the authority model — documents 12, 24,
  25, the role specs, the recovery prompts, or the work-plan README — defines
  Arendt's identity, model, capabilities, permissions or review weight.
- Impact: the closure chain for material findings now contains a participant
  whose independence and competence cannot be assessed from versioned sources.
  Document 24 section 2 enumerates every role for exactly this reason. If
  Arendt's corroboration carried any weight in the triage decisions, that
  weight is currently unauditable; if it carried none, the label implies
  independence that is not evidenced.
- Proposed correction: either register Arendt (identity, model, scope, what
  "corroborator" authorizes, and what it may never do) in document 24's
  responsibility table, or remove the designation from closure documents.
  Cheap either way; the gap is registration, not necessarily the participant.
- Owner: Musashi (document) + Harvey (whether the role exists at all).

### AUD-GEN-20260731-015 — Dual-role conflict: the independent auditor is now also academic research lead, and P5 makes it acute

- Severity: **S3**
- Confidence: high (observed structure; impact inferred)
- Status: open — explicitly not closable by me; I am the conflicted party
- Observation: commit `e94e9344` appoints Satoshi academic research lead while
  Satoshi remains the independent continuous auditor. Document 25 contains
  real mitigations: cross-review (section 1), "neither agent may approve its
  own material claim", Harvey holds authorship/submission, and the work-plan
  README states the academic surface grants Satoshi no authorship, submission
  or runtime authority. The invocation adds "do not close your own material
  findings."
- The residual conflict the mitigations do not cover: **P5 is a paper about
  the audit system that Satoshi operates.** As academic lead I would frame
  P5's research questions, select its incident corpus, and simulate its
  reviewers — while being the agent whose effectiveness the paper measures.
  Detection-latency and false-alert claims in P5 are, in part, claims about my
  own work quality. Cross-review by Musashi helps but is symmetric-conflicted:
  Musashi built the watchdogs and evidence plane P5 also evaluates. Neither
  reviewing agent is disinterested in P5's conclusions.
- Secondary instance: audit reports (mine) are themselves P5 evidence
  artifacts; the auditor curating which incidents enter the corpus can shape
  the paper's story. Document 25 rule 2 (protected observations cannot select
  the narrative) has no analogue for incident-corpus selection.
- Proposed correction (three parts, all cheap):
  1. amend document 25: P5's incident corpus is defined by **enumeration
     rule** (e.g., "every incident recorded in documents 13/15/16/20 plus every
     S0-S2 finding, no exclusions without a logged reason"), not by curation;
  2. P5's quantitative effectiveness claims require verification by Musashi
     **from raw timestamps**, and P5 is flagged for external human/technical
     review before any preprint, ahead of the other papers, not behind them;
  3. every Satoshi audit artifact cited in P5 carries an in-paper conflict
     disclosure alongside the AI-use disclosure.
- Owner: Harvey (accepts the structure) + Musashi (document amendment).

### AUD-ACAD-20260731-016 — P1's implicit trust claims exceed its threat model

- Severity: **S3** (academic S1 if published as-is: invalid central claim)
- Confidence: high on the code-level facts; the paper risk is inference
- Observation: P1's outline (document 25 section 4) includes "Threat Model"
  and "Security Analysis", but the deployed protocol evidence is a
  **cooperative, crash-fault** setting: 3-of-4 claim quorum, lexicographic
  peer-ID arbitration, commit-reveal, and a four-worker fleet under one
  operator. Nothing in the current evidence demonstrates Byzantine tolerance,
  Sybil resistance, or adversarial-peer economics — and terms like
  "blockchain", "Proof of Optimization" and "verifiable" invite reviewers to
  read exactly those claims.
- Impact: the most probable desk-reject or hostile-review path for P1 is a
  gap between vocabulary and demonstrated adversary model. This is fixable in
  framing now, expensive after drafting.
- Proposed correction: scope P1 explicitly to "cooperative distributed
  optimization with verification against faulty or lazy peers, under a
  crash/omission fault model with signed identities"; list Byzantine and
  Sybil settings as explicit non-goals/future work; make the verification-
  cost-ratio claim (checking is cheaper than producing) the headline
  falsifiable contribution, since that is what the evidence can support.
- Owner: Satoshi (framing) with Musashi verification of what the code
  actually enforces.

### AUD-ACAD-20260731-017 — No paper directory scaffold or claims.csv exists; claim-to-evidence mapping is currently unenforceable

- Severity: **S4** (S3 the moment drafting starts)
- Confidence: high (observed)
- Observation: document 25 section 3 mandates `papers/<paper-id>/` with
  `claims.csv`, `search_protocol.md`, `artifact_manifest.json`. No `papers/`
  directory exists anywhere in `agent-multi` (or `docs/publications/` before
  this audit's deliverables). The evidence contract exists only as prose.
- Proposed correction: Musashi materializes the empty scaffold with schema-
  validated `claims.csv` headers per paper as a bounded packet; costs minutes
  now, prevents untracked claims later.
- Owner: Musashi.

## 2. Authority-Model Audit (invocation questions 2 and 3)

**Coherent?** Yes, with the two findings above. The three-way split — Harvey:
authorship/release; Satoshi: scholarly program; Musashi: experiments and
artifact integrity — has no unowned surface I could find, with one small gap:
**venue-fee and copyright decisions** (who pays open-access charges, who signs
IEEE copyright transfer) are implicitly Harvey but unstated; add one line to
document 25.

**Cross-review preventing self-approval?** Structurally yes for P1-P4:
Satoshi's claims are verified by Musashi, Musashi's experiments reviewed by
Satoshi, Harvey gates release. Two residual holes: (a) P5's symmetric
conflict (finding 015); (b) both reviewing agents share one operator and one
codebase — genuinely independent review first arrives at the external-red-team
lifecycle stage, so that stage should be mandatory for the **first** paper
submitted, whichever it is, not just "at least one paper" before the book.

## 3. Paper-by-Paper Assessment (invocation questions 4-8)

States use the mandated vocabulary: `outline`, `evidence_incomplete`,
`evidence_ready`, `not_publishable`.

### P1 — DOIN protocol and verifiable optimization → `evidence_incomplete`

- Falsifiable core (good): verification-to-generation cost ratio; duplicate
  evaluation rate under concurrent claims; champion propagation latency;
  recovery behavior under node loss.
- Contribution test (inference, pending ledger verification): the *combination*
  — shared-population EA with deterministic reproduction, claim quorum,
  champion verification and queryable on-chain lineage, behind a plugin
  boundary — is plausibly distinct from volunteer-computing (BOINC-family),
  blockchain-FL and proof-of-learning lines, which verify gradient work or
  aggregate models rather than evolutionary candidate lifecycles. Nearest
  neighbors to check first: blockchained federated learning, Proof-of-Learning,
  proof-of-useful-work, and distributed/island-model EC systems.
- Exists as evidence: four-worker campaigns with one lineage; measured
  candidate throughput; the FD-exhaustion, lease-resurrection, and join-repair
  incidents with fixes; the live equal-height fork with finalized-anchor
  safety; deduplicated claim maps; the champion archive contract.
- Missing (decisive): controlled node-loss/rejoin matrix; message-loss
  injection; scaling beyond four workers (even 6-8 makes a curve); the
  verification-cost-ratio measurement as a designed experiment rather than an
  operational anecdote; frozen protocol version tag.
- Overlap control: fork *protocol semantics* belong to P1; fork *operational
  detection* belongs to P5. OLAP-on-chain lineage belongs to P1; off-chain
  audit evidence packets to P5.
- Venue families (no acceptance claim): arXiv cs.DC + cs.NE; then
  Middleware/IPDPS/EuroSys-family for systems framing, or
  GECCO/TELO-family for the evolutionary-computation framing.

### P2 — Data-first mixed-genome search → `evidence_incomplete`

- Falsifiable core (good): does joint data+preprocessing+policy+risk search
  beat model-only search under leakage-safe chronological evaluation with
  visible costs? The two declared controls (model-only genome, fixed-data
  genome) make this properly testable.
- Contribution test (inference): AutoML-for-trading exists; the differentiator
  is the leakage threat model as a first-class contract (point-in-time
  availability, purged proxies, protected-test firewall, activity floors,
  eligibility sentinels) plus the incident-grounded motivation (documents 16
  and 20 are unusually honest evidence that naive gates fail).
- Exists: E0-E4 screening corpus (16,019 jobs, transactional OLAP); parameter
  registry; corrected one-bar execution protocol; the v2 protected campaign
  running; the incident corpus for motivation.
- Missing (decisive): any completed protected-v2 champion (job 0 is stage 1
  of 4); the model-only and fixed-data control runs; three-seed confirmation;
  the one-time protected-test opening; cross-asset repetition beyond USDCAD.
- Risk: this is the paper with the strongest temptation toward financial
  claims. Rule: results are stated as validation-protocol outcomes with
  period/unit/cost-regime, never as expected profitability; the "no positive-
  profit gate" screening philosophy is itself a defensible methods point.
- Venues: arXiv q-fin.CP + cs.NE; then a computational-finance or applied-ML
  venue family after evidence completes.

### P3 — Hierarchical portfolio control → `outline`

- The gate in document 25 already says it: six qualified cells with frozen
  artifacts do not exist (zero protected-v2 champions today); the rush
  detector and allocator are unimplemented; every claimed ablation is
  unfunded by evidence.
- Correct handling: keep as registered research question and outline; **defer**
  all drafting; no related-work investment beyond the ledger seed until P2
  produces at least three cells. Not merged into P2 — its question (composition
  and allocation) is genuinely separable from P2's (search).

### P4 — Simulation-to-paper execution parity → `evidence_incomplete`

- Falsifiable core (adequate): which simulator cost/fill/protection
  assumptions survive authenticated paper execution, with what bounded
  discrepancy. This is a measurement/experience paper and should be framed as
  one — that is a legitimate genre, and honesty about it improves acceptance
  odds in systems-adjacent venues.
- Exists: Alpaca 301 reconciled read-only sessions; IBKR adapter with the
  functional-health lesson; the protection fail-closed contract; the
  liveness-vs-functional-probe incident (a genuinely good war story that
  generalizes).
- Missing (decisive): protected canaries (M3) on any venue; the seven-day
  consolidated shadow (M4); MT5 entirely; fault-injection scenarios
  (disconnect, stale data, duplicate submission) as designed experiments;
  clock/currency reconciliation checks.
- Boundary: no live-profit claims; regulatory/product boundary section is
  already in the outline — keep it.
- Venues: arXiv cs.SE/q-fin.TR; experience-track or measurement-study family.

### P5 — Continuous audit and swarm recovery → `evidence_incomplete`

- Falsifiable core (good): detection latency, false-alert rate, and audit
  overhead of a deterministic-evidence + bounded-LLM-audit architecture, with
  ablations (heartbeat-only, port-only, no independent audit).
- Exists (rich): a real incident corpus with timestamps — FD exhaustion, lease
  resurrection, BTC actor saturation, the activity-gate incident, the IBKR
  silent failure behind a green port probe, the live fork; the tier-0
  snapshot/test-evidence packets with hashes; the token-economy design; the
  triage/closure separation demonstrated in this very audit cycle.
- Missing (decisive): the incident taxonomy as a versioned table with
  measured detection latencies extracted from logs; reproducible fault
  scenarios (can be tabletop/replay rather than live); overhead measurement
  (collector CPU seconds and token costs — partially already in packets);
  the ablation configurations.
- Conflict handling per finding 015 is a precondition for this paper.
- Venues: arXiv cs.SE + cs.DC; dependability/operations family
  (DSN/SoCC/ICSE-SEIP-like) after external review.

### Cross-paper duplicate-claim control (invocation question 3 of 8)

| Claim | Sole owner |
| --- | --- |
| Fork-choice and finalization semantics | P1 |
| Fork detection, alerting, operator classification | P5 |
| Protected-order eligibility and activity gates (simulation) | P2 |
| Broker-side protection parity and reconciliation | P4 |
| OLAP-on-chain candidate lineage | P1 |
| Off-chain deterministic evidence packets | P5 |
| Leakage threat model and firewall | P2 (P3 cites, does not restate) |
| Multi-venue capability/routing contracts | P4 |

## 4. Disclosure, Licensing and Reproducibility Risks (question 7)

1. **AI disclosure is the defining meta-risk**: substantially AI-designed,
   AI-audited and partially AI-drafted manuscripts must be disclosed under
   IEEE and arXiv policies (URLs versioned in document 25 section 9; recheck
   at submission). Harvey is the sole author; AI systems are named tools.
   This is handled in the contract; it must also appear in each paper body.
2. **Data licensing**: exchange/broker data (Binance-derived series, OANDA
   quotes) is generally non-redistributable; the reproducibility package must
   ship hashes plus lawful reconstruction paths, never raw data. FXMacroData
   and any paid vintages need explicit license review before even hash
   publication is assumed safe.
3. **Code licensing**: NautilusTrader (LGPL-3.0) and Stable-Baselines3 (MIT)
   are dependencies, not derivatives, in the current architecture — but the
   artifact package must state licenses explicitly; verify at packaging time.
4. **Financial-claim risk**: P2/P3/P4 must never present validation or paper
   results as expected live returns (document 25 rules 6-7 cover this; the
   reviewer-simulation pass should attack it deliberately).
5. **Security-sensitive detail**: P1/P5 must exclude operational secrets
   (ports, hostnames, Tailscale topology, alert thresholds that would aid an
   attacker); the redaction rule exists for audit reports and must be extended
   to manuscripts.
6. **Reproducibility**: the 143-pin fleet lock plus content-addressed
   artifacts is a strong base; per-paper `artifact_manifest.json` (finding
   017) is the missing enforcement point.

## 5. Register and State Updates

- New findings 014, 015 (governance) and 016, 017 (academic) recorded in the
  open register.
- `AT-ACADEMIC-030` state: `reported` for the audit stage; the follow-on
  bounded task is defined in the roadmap.
- Paper states recorded: P1 `evidence_incomplete`, P2 `evidence_incomplete`,
  P3 `outline`, P4 `evidence_incomplete`, P5 `evidence_incomplete`. None is
  `evidence_ready`; none is `not_publishable`.

## 6. Change Confirmation

Writes were limited to this report, the two `docs/publications/` deliverables,
and the audit registers. No runtime, campaign, broker, chain or production
code was touched. No credential was used; no submission or external posting of
any kind occurred; no paid API was called.
