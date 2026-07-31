# Audit of Musashi's Governance, Academic and Innovation Response

Audit ID: AUDIT-MUSASHI-RESPONSE-20260731-01
Timestamp and timezone: 2026-07-31 02:45 America/Bogota (UTC-5)
Auditor: Satoshi
Baseline reviewed: `agent-multi` HEAD == origin/master == `b89b23d1`; required
minimum `3b3e9a7a` is an ancestor; working tree clean; no foreign changes.
Task: `docs/handoffs/SATOSHI_GOVERNANCE_CLOSURE_AND_INNOVATION_CHALLENGE_2026_07_31.md`

## 1. Findings (severity order)

### AUD-F1-20260731-020 — Recurring equal-height minority tips repeatedly involving Dragon

- Severity: S4 (efficiency/latency signal; no safety impact demonstrated)
- Confidence: high on observation; hypothesis flagged on mechanism
- Observed: at 02:30 COT, all four workers at height 10, finalized height 3
  with unanimous anchor `a8ce597e…`; Dragon alone on tip `ac9c1dca…` versus
  `75a5add4…` ×3. This is the **second consecutive height** at which Dragon
  holds the minority unfinalized tip (height 9 previously; Musashi reports log
  evidence of earlier equal-height competitions at heights 6 and 9 that
  converged). Height-9 competition demonstrably resolved: finalization
  advanced 2→3 unanimously and generation 3 started with four distinct claims
  {0,1,2,3}.
- Hypothesis requiring a named test: asymmetric block propagation or
  generation timing involving Dragon (route asymmetry was a real prior defect
  class — `doin-node@bc36999`).
- Cheapest test: per-peer minority-tip census from node logs plus
  announcement-to-adoption latency by route. No mutation; log reading only.
- Owner: Musashi (test), Satoshi (verification).

### AUD-F1-20260731-021 — Generation-barrier straggler idle time is real, unmeasured, and material at fleet scale

- Severity: S4 (efficiency; becomes S3 if measured loss exceeds ~10 %)
- Confidence: observation direct; magnitude is inference pending measurement
- Observed: the generation-2 tail ran with one candidate on gamma-5070ti while
  three workers waited at the barrier (Musashi's own triage words: "correctly
  waiting"); snapshots bracket that tail-idle window at ≥1.3 h for up to three
  GPUs in one generation.
- Inference: with per-worker medians spanning 1.9–3.6 h/candidate (≈2×
  spread), a 20-candidate generational barrier idles fast workers for roughly
  the straggler's residual each generation — order 2–5 GPU-hours per
  generation, ≈6–14 % of fleet capacity over the 24-generation budget. This is
  an estimate, labeled as such; the exact number is cheaply computable because
  candidate start/finish pairs are already logged for ETA.
- Attribution: this observation originates from the project owner's direct
  operational read ("agents seated instead of working"); the audit quantifies
  and formalizes it.
- Disposition: this is the first concrete sub-experiment of registered line
  **P6** (value-of-information/scheduling): measure realized barrier idle from
  existing logs, then evaluate counterfactual schedulers (e.g., slowest-worker-
  first claim ordering, bounded lookahead crossing the barrier) on replay.
  Registered in `papers/p1-doin-protocol/FUTURE_WORK.md` item 3 and the
  continuous roadmap task 032b.
- Owner: Musashi (measurement), Satoshi (experiment design review).

### Process observation (not a numbered finding): maps where fixtures were cheaper

Findings 009 and 010 received a CI *position* and an invariant *mapping* —
both good documents — while the actual artifacts (one ~30-line Tier A
workflow; nine fixtures his own mapping already sequences) remain unwritten.
The response pattern favored inventory over implementation twice in one cycle.
Recorded as S4 process observation with no theatrics: the mapping is accepted
as completing the *inventory* requested; the finding stays open exactly
because inventories do not execute.

## 2. Packet A — Independent Closure Verification

| Item | Verdict | Evidence |
| --- | --- | --- |
| 014 Arendt | **recommend verified_closed** (verifier must be Harvey or independent — not me, not Musashi) | Triage line 6 now reads "Independent corroborator: none", with dated removal note and explicit no-weight statement. Grep confirms no other authority reference |
| 015 P5 controls | **recommend verified_closed** with one strengthening | Doc 25 lines 367–378: enumeration-from-named-documents rule with logged exclusions; raw-timestamp reconstruction by Musashi; external review before preprint plus dual disclosure. Strengthening: hash-pin the enumeration query inside the corpus manifest so the rule provably predates outcomes (section 6) |
| 016/019 property table | **verified accurate** | Spot-checks landed verbatim: `messages.py` carries a literal "(research mode — accepted without verification)" comment on `CANDIDATE_EVALUATION`; `unified.py:1202` auto-accepts with "treat reported_performance as verified"; the table below is authorized for P1 |
| 017 scaffolds | **verified** | `validate_publication_scaffolds.py` → "validated 5 publication packages"; `test_publication_scaffolds.py` → 1 passed |
| 010 mapping | **verified** | All sampled citations exist and match: `test_no_future_leakage_in_scaling` (gym-fx:113), supervisor tests at lines 247/303/326/409/461, watchdog/calendar/shared-optimizer files present. Covered/partial/gap classification accepted |
| Fork | **classification confirmed; disposition below** | Two distinct tips at height 10, one finalized anchor; height-9 instance resolved by finalization advance; no parallel population or claims divergence |

**P1 authorized-claim table (finding 016/019), verified for manuscript use:**

| Property | Primitive exists | Runtime enforced | Active profile | Evidence | Authorized claim |
| --- | --- | --- | --- | --- | --- |
| Persistent ECDSA peer identity | yes | derivation yes | yes | identity.py:17-115 | "persistent cryptographic identities" |
| Signed/authenticated messages | signing primitive only | **no** | **no** | messages.py:43-62 (no signature field) | may NOT claim authenticated channels |
| Commit-reveal binding | yes | yes | yes | commit_reveal model + tests | "commitment-bound reveals" |
| Quorum verification | yes | yes when enabled | **disabled in research profile** | unified.py:1202-1209 auto-accept | "available verification path", NOT "verified results" for current campaign evidence |
| Candidate claim dedup | yes | yes | yes | lease/arbitration tests, live claim maps | "duplicate-prevention under cooperative peers" |
| Deterministic fork choice + finalization | yes | yes | yes | fork_choice/finality + observed h9 resolution | "finalized-checkpoint fork choice"; convergence latency must be reported honestly (finding 020) |
| Byzantine/Sybil/collusion resistance | no | no | no | — | future work only (P17) |

**Fork disposition recommendation (for Harvey or independent reviewer; not
self-closed):** close AUD-F1-20260730-005 as
`verified_closed — no safety defect; equal-height competition resolves at
finalization; finalized anchors never diverged; claims/population unaffected`,
and track the recurrence pattern under new finding 020 (S4). No chain
mutation was performed or recommended at any point.

## 3. Packet B — P6-P18 Decision Table

Novelty column: `unopened` = no primary source opened yet; `first_pass` = at
least one primary source opened this cycle (Packet C). No line is called
novel.

| line_id | decision | closest prior art (candidates) | novelty_state | cheapest decisive experiment | null value | kill condition — assessment |
| --- | --- | --- | --- | --- | --- | --- |
| P6 | **retain** (absorb P15) | Hyperband/ASHA bandit allocation (Li et al., verified); BO batch scheduling | first_pass | Barrier-idle measurement from existing logs, then counterfactual scheduler replay (finding 021) | yes — "simple wins" is publishable | sound |
| P7 | **narrow** | IPFS content addressing (Benet, verified); erasure-coding storage lit | first_pass | Scripted peer-loss retrieval trials | weak as framed | narrow to *verification-coupled lineage availability* (integrity gate + replication factor vs churn); plain availability is solved engineering |
| P8 | retain H2 | proof-of-useful-work line | unopened | two-plugin verifier-contract cost measurement | yes | merge-watch with P18 (empirical vs theory of same question) |
| P9 | **narrow** | Bayesian Online Changepoint Detection (Adams & MacKay, verified) as the price-only baseline | first_pass | frozen event-time pilot with placebo/shift controls | yes | **dependency falsification risk**: doc 17 records 2021-2023 event-vintage coverage as blocked; narrow to lawfully covered years/assets before admission matures |
| P10 | retain, deferred | continual RL safety, portfolio rebalancing lit | unopened | fixed/periodic/triggered replay | yes | blocked on P3 cells — correct as declared |
| P11 | **narrow or hold H2** | federated personalization, DP finance | unopened | threat model + synthetic benchmark | weak | "unacceptable fairness loss" is undefined — Goodhart-vulnerable; require metric definition before any experiment |
| P12 | retain | smart order routing lit | unopened | counterfactual replay from paper fills | yes | downstream of P4 M4 facts — fine |
| P13 | retain, guarded | LLM self-correction limits (Huang et al., verified — supports the premise that intrinsic self-review is weak) | first_pass | blinded scoring of enumerated corpus | yes | external-review precondition mandatory (self-study) |
| P14 | **defer H2** | queueing/service economics | unopened | queueing sim + trace replay | weak | admission rule 5 currently unsatisfiable: no real inference traces exist; sim-only yields toy results |
| P15 | **merge into P6** | — | — | same replay infrastructure, energy as objective term | — | separate registration duplicates P6's experiment plane |
| P16 | **retain — first priority** | TLA+/model-checking practice | first_pass (index rate-limited; primary text pending) | small-state model of one generation (claims, leases, barrier, finality) | yes — counterexamples are directly actionable | sound; cheapest high-information line in the registry |
| P17 | retain H2 | BFT/Sybil lit (PBFT anchor seeded) | unopened | threat matrix + signed-message prototype | yes | correctly outside current P1 claims; finding 019 strengthens motivation |
| P18 | retain H2 | proofs-of-work/learning theory | unopened | definitions + counterexamples | yes | merge-watch with P8 |

Queue-health verdicts (Packet B closing questions): the horizon/capacity model
is reasonable but currently unfalsifiable as stated — add a measurable signal
(e.g., "H0 tasks blocked >2 weeks while H1/H2 consume sessions" triggers
rebalance). The permanent queue's anti-busywork guard is real (bounded budgets
per task) but needs one addition: a task that produces no decision-changing
output twice in a row is retired, not repeated.

## 4. Packet C — Primary-Source Collision First Pass (bounded, honest)

Opened and verified this cycle (full fields recorded in
`docs/publications/RESEARCH_LINE_PRIOR_ART_DELTA_2026_07_31.csv`):

1. Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar — *Hyperband: A Novel
   Bandit-Based Approach to Hyperparameter Optimization*, JMLR 18 (2018),
   arXiv 1603.06560 → P6 collision: adaptive resource allocation for
   hyperparameter search is heavily developed; P6's residue is the
   decentralized shared-population EA setting, OLAP-evidence-driven value
   estimates and barrier-idle economics — narrower than the registry wording.
2. Benet — *IPFS — Content Addressed, Versioned, P2P File System*, arXiv
   1407.3561 (2014) → P7 collision: content-addressed availability is solved
   infrastructure; P7 survives only as verification-coupled lineage
   availability (narrowed above).
3. Adams, MacKay — *Bayesian Online Changepoint Detection*, arXiv 0710.3742
   (2007) → P9's mandatory price-only baseline; any event-context claim must
   beat calibrated BOCPD-class baselines, not naive thresholds.
4. Huang et al. — *Large Language Models Cannot Self-Correct Reasoning Yet*,
   ICLR 2024, arXiv 2310.01798 → P13 support: intrinsic self-review is weak,
   which motivates but does not prove cross-agent review value; P13's
   experiment remains necessary.
5. Castro, Liskov — *Practical Byzantine Fault Tolerance* (seeded earlier,
   ledger row) → P17 boundary anchor.

Bound honestly reported: 5 of the permitted 30 sources were opened; the
Semantic Scholar index returned HTTP 429 (rate limit) on first use, so the
"two indexes per line" requirement is satisfied only for arXiv-resident
sources this pass. The remainder is queued as roadmap task 032g with the
per-line source budget. No unopened source was recorded as verified anywhere.

## 5. Packet D — Future Work

Delivered as five files under `papers/*/FUTURE_WORK.md`, each with 3–5 ranked,
fully-fielded lines (limitation, falsifiable question, prior-art state,
required implementation/data, cheapest discriminating experiment, decision
metric with unit, dependency, kill condition, registry ID). No "more assets /
more models / scale it" lines exist. One proposed new line: **P19
(functional-vs-liveness health probes)**, from the IBKR incident, registered
via `papers/p4-execution-parity/FUTURE_WORK.md` item 4 and
`papers/p5-audit-recovery/FUTURE_WORK.md` item 4 — it fills a demonstrated
gap (a real silent failure class) rather than decorating.

## 6. Packet E — P5/P13 Conflict Red Team

1. **Enumeration-rule immutability:** the rule lives in doc 25 (committed at
   `3b3e9a7a`) *before* any analysis — pre-registration holds via git history.
   Required strengthening: the corpus manifest must embed the SHA-256 of the
   enumeration rule text and the git commit that introduced it, so a later
   silent amendment is detectable by diff, not by memory.
2. **Machine-readable corpus manifest (proposed schema):**
   `incidents.csv`: `incident_id, source_document, source_anchor,
   occurred_at, first_detected_at, detected_by_tier
   (watchdog|snapshot|satoshi|musashi|user|undetected), detection_latency_s,
   classification (tp|fp|fn), recovery_completed_at, recovery_latency_s,
   evidence_hashes, excluded (bool), exclusion_reason`. Every row from the
   enumerated documents; exclusions logged, never silent.
3. **Blind labeling:** Musashi's scripts extract timestamps/events without
   outcome labels (scripts + output hashes versioned); labels
   (tp/fp/fn, detection credited to which tier) are assigned from redacted
   narratives by Harvey or an external reviewer who has not read the audit
   conclusions; Satoshi computes metrics only after labels freeze.
4. **Units:** false positive = alert with no confirmed underlying defect (rate
   per alert-class per week); false negative = defect confirmed later with no
   prior alert (count per period); detection latency = seconds from
   `occurred_at` to `first_detected_at`; recovery latency = seconds to
   verified recovery; operator burden = human interactions per week; compute
   overhead = collector+runner CPU-seconds per day; token cost =
   reserved tokens per audit session (already recorded by the economy layer).
5. **Musashi's symmetric conflict, located precisely:** he built the
   watchdogs and evidence plane P5 evaluates (sections IV–V), authored several
   incident fixes in the corpus, and is the raw-data extractor. Mitigations:
   published extraction scripts with output hashes; Satoshi re-runs
   extraction; blind labels per item 3; both agents' conflicts disclosed in
   the manuscript, not only mine.
6. **Falsification of "cross-audit helps":** blinded scoring shows no
   reduction in false-closure or unsupported-claim rates versus single-agent
   review after controlling for model capability and evidence access — which
   is P13's registered kill condition, correctly. **Mandatory negative
   cases already enumerated:** Satoshi's withdrawn finding 011
   (auditor over-claim caught by cross-review — a success for the mechanism,
   a failure of the auditor) and the Arendt designation (technical-lead
   unsupported attribution caught by cross-review). Both stay in the corpus
   under the enumeration rule; neither may be excluded.

## 7. Proposed Register Updates (for owner/independent disposition)

- 005 → recommend `verified_closed` (no safety defect; finalization resolves
  equal-height competition; anchors never diverged). Verifier: Harvey or
  independent — not the reporter, not the party whose code is implicated.
- 014, 015, 016, 017 → recommend `verified_closed` per Packet A evidence;
  same verifier constraint.
- New: 020 (S4, Dragon recurring minority tip), 021 (S4→S3-if-measured-high,
  barrier straggler idle).
- 009, 010 remain open S3 by design: inventories delivered, artifacts not.

## 8. Commands, Tests and Exact Results

```text
git fetch/rev-parse/status/log; merge-base --is-ancestor 3b3e9a7a HEAD  → ancestor, clean, b89b23d1
curl /api/network (×1)                      → 2 tips at h10, 1 finalized anchor h3, gen 3, claims {0,1,2,3}
rg/sed spot-checks                          → Arendt removal; doc 25 lines 177/367-378/403/462; messages.py 43-62; unified.py 1202-1209, 1526-1542
python tools/validate_publication_scaffolds.py → "validated 5 publication packages"
pytest -q tests/unit/test_publication_scaffolds.py → 1 passed
invariant citation samples                  → all exist and match (gym-fx:113; supervisor:247/303/326/409/461; 3 files)
WebFetch arXiv abs ×4 (1603.06560, 1407.3561, 0710.3742, 2310.01798) → full fields captured
WebFetch Semantic Scholar API               → HTTP 429, recorded as index-unavailable this pass
```

## 9. Change Confirmation

No runtime, campaign, chain, broker, credential or production code was
touched. No commit or push. Writes limited to this report, two
`docs/publications/` files, five `papers/*/FUTURE_WORK.md` files, the register
and the recovery prompt. Fork evidence was read-only; no mutation was
performed or recommended.
