# Academic Research Roadmap

Version: 1.1.0
Date: 2026-08-15
Academic research lead: Satoshi (Claude) — program design and scholarly review
Experimental and technical lead: Musashi (Codex) — experiments and artifacts
Human author and release authority: Harvey
Companion audit: `../audits/AUDIT_ACADEMIC_PUBLICATION_PROGRAM_2026_07_31.md`

## 1. Disposition and Recommended Order

| Paper | Disposition | Current state | Order rationale |
| --- | --- | --- | --- |
| P1 DOIN protocol | **retain — first intended output** | evidence_incomplete | Core contribution and direct continuation of the 2018 thesis; decisive fault/scaling and verification-cost experiments remain required. |
| P5 audit and recovery | **retain — second intended output** | evidence_incomplete | The P5/P13 adversarial method has a rich incident corpus, sealed nulls and role-separated review; conflict controls and external review remain mandatory. |
| P2 data-first genome search | **retain — evidence accrues from the running campaign** | evidence_incomplete | Blocked on protected-v2 champions plus model-only and fixed-data controls; nothing to draft before at least one asset completes with controls |
| P4 execution parity | **retain — measurement-paper framing** | evidence_incomplete | Passive observation already accumulating; blocked on protected canaries (M3) and the seven-day consolidated shadow (M4); MT5 optional, not gating |
| P3 hierarchical portfolio | **defer** | outline | Gate honestly unmet (needs ≥6 qualified frozen cells, rush detector, allocator); revisit when P2 yields three cells |

Paper IDs are stable subjects, not ordinals in the drafting queue. In
particular, the second intended output is P5/P13; P2 continues to identify the
data-first mixed-genome paper. No paper is merged or rejected: the five
research questions remain separable, and the duplicate-claim table in the
companion audit keeps their boundaries enforceable.

## 2. Dependency Graph

```text
protected-v2 campaign (running)
      │ champions + controls
      ▼
     P2 ────────────► P3 (deferred until ≥3 cells)
      │ frozen cells
      ▼
broker canaries + 7-day shadow ──► P4
DOIN fault/scaling experiments ──► P1
incident corpus + packets (exists) ──► P5   ← conflict controls (finding 015)
P1, P2 evidence frozen ┐
P5 external review     ├──► synthesis book (per doc 25 §5, ≥3 frozen + 1 reviewed)
one more frozen paper  ┘
```

## 3. Decisive Missing Experiments (by information value)

| # | Experiment | Paper | Owner | Why decisive |
| --- | --- | --- | --- | --- |
| 1 | Model-only and fixed-data control campaigns on the same asset/seed/protocol | P2 | Musashi | Without them the central claim (data-first > model-only) is untestable — the paper has no result |
| 2 | Verification-to-generation cost ratio measured as a designed experiment | P1 | Musashi | Converts the headline claim from anecdote to measurement |
| 3 | Node-loss/rejoin and message-loss fault matrix (tabletop/replay acceptable where live injection is unsafe) | P1 | Musashi | Reviewers will not accept recovery claims from incident prose alone |
| 4 | Incident taxonomy table with measured detection latencies extracted from existing logs | P5 | Musashi extracts, Satoshi structures | Turns the audit narrative into quantitative results; data already exists |
| 5 | Protected canaries + 7-day consolidated shadow with full reconciliation | P4 | Musashi + user (account actions) | The paper's entire results section |
| 6 | Scaling run beyond four workers (6-8 suffices for a first curve) | P1 | Musashi + user (hardware) | Distinguishes "works on my fleet" from a scaling claim |
| 7 | Ablations: heartbeat-only / port-only / no-independent-audit monitoring configs | P5 | Musashi | The IBKR incident already demonstrates the port-only failure mode; ablations generalize it |

## 4. Venue and Category Proposals (no acceptance claims)

| Paper | arXiv | Peer-review family |
| --- | --- | --- |
| P1 | cs.DC + cs.NE | distributed-systems (Middleware/IPDPS-like) or evolutionary computation (GECCO/TELO-like) |
| P2 | q-fin.CP + cs.NE | computational finance / applied ML |
| P3 | q-fin.PM + cs.LG | portfolio/quant venues — decide after evidence |
| P4 | cs.SE + q-fin.TR | measurement/experience track |
| P5 | cs.SE + cs.DC | dependability/operations (DSN/SoCC/SEIP-like) |

## 5. Responsibility Assignment

- **Satoshi:** related-work verification and ledger maintenance; falsifiable
  question wording; manuscript architecture; reviewer simulation; academic
  state tracking; venue analysis. Never: evidence manufacture, authorship,
  submission, closing own findings.
- **Musashi:** experiment implementation from packets; artifact manifests and
  `claims.csv` scaffold (audit finding 017); metric reconstruction;
  reproducibility replay; verification of every Satoshi technical claim.
- **Harvey:** authorship; conflict-control acceptance (finding 015); Arendt
  registration decision (finding 014); licensing/fee/copyright decisions;
  release and submission authority.

## 6. Next Bounded Academic Task

`AT-ACADEMIC-031 — Ledger verification pass, P5+P1 rows first`: open every
seeded source for P5 and P1 (≤20 rows), complete/correct all bibliographic
fields from the primary source, record exact supporting sections, check
retraction/correction state, and flip rows to `verified` or `rejected`. Output:
updated CSV plus a one-page delta note. Bounded: no new searching beyond
replacing rejected rows; no drafting. After it: P5 claim/evidence matrix
(`claims.csv` seed) against the enumeration rule from finding 015.

## 7. Standing Rules Reaffirmed

Protected-test data selects nothing in any paper. Null and negative results
stay visible. Simulated/paper results are never expected live profit. AI
involvement is disclosed in every manuscript. Readiness states change only on
evidence, never on prose quality.
