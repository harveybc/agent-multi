# Continuous Research Roadmap

Version: 1.0.0
Date: 2026-07-31
Academic lead: Satoshi | Experimental lead: Musashi | Authority: Harvey
Supersedes nothing; extends `ACADEMIC_RESEARCH_ROADMAP_2026_07_31.md` with the
permanent queue demanded by document 26. Preemption: any S0/S1 finding, and
the read-only fork check class (AT-F1-011 successors), preempt every task
below.

## 1. Next Ten Bounded Academic Tasks (after AT-ACADEMIC-031)

| # | Task ID | Task | Trigger | Dependency | Budget | Output |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | AT-AC-032a | Complete P16 collision test and specify the small-state model scope (generation, claims, leases, barrier, finality) | next academic session | none | ≤8 sources | model-scope memo + ledger rows |
| 2 | AT-AC-032b | P6 barrier-idle measurement protocol + counterfactual scheduler replay spec (finding 021) | Musashi extracts candidate start/finish pairs | log extract | 0 sources; spec only | experiment protocol with decision metric (GPU-h idle per generation; fraction of fleet capacity) |
| 3 | AT-AC-032c | P5/P13 machine-readable incident-corpus manifest + blind-labeling protocol (Packet E schema) | doc 25 controls accepted by Harvey | enumeration rule hash-pinned | 0 sources | `incidents.csv` schema + labeling procedure |
| 4 | AT-AC-032d | P9 lawful-coverage audit: which assets/years have point-in-time event vintages (doc 17 constraints); narrow the hypothesis accordingly | next academic session | none | ≤6 sources | narrowed P9 registry revision |
| 5 | AT-AC-032e | Seed `papers/p1-doin-protocol/claims.csv` from the verified authorized-claim table (audit section 2) | scaffolds committed | none | 0 sources | populated claims.csv rows, each mapped to code/evidence |
| 6 | AT-AC-032f | P5 incident taxonomy extraction review: verify Musashi's raw extraction scripts/hashes and structure the corpus | 032c manifest exists | Musashi extraction | 0 sources | verified corpus + latency table |
| 7 | AT-AC-032g | Ledger verification continuation: open remaining `candidate_unverified` rows for P5+P1 and retry the rate-limited index | weekly literature slot | none | ≤20 sources | ledger delta with verified/rejected states |
| 8 | AT-AC-032h | P7 narrowing memo: availability-with-integrity-gate claim and replication-factor experiment, or reroute to engineering roadmap | after 032g P7 rows | none | ≤6 sources | retain/reroute decision |
| 9 | AT-AC-032i | Adversarial reviewer simulation on P1 sections II and VI | first P1 draft of those sections exists | 032e | 0 sources | reviewer report with attack list |
| 10 | AT-AC-032j | Quarterly retirement dry-run: apply every registry kill condition, propose retire/promote/split for Harvey | month-end or quarter boundary | registry current | 0 sources | retirement decision table |

## 2. Permanent Non-Idle Fallback

When no task above is triggered and no S0-S2 work exists: **weekly
primary-literature delta** for registered H1 lines (P6, P9, P13, P16, P12) —
maximum 10 sources, output is ledger rows plus at most one registry state
change. If two consecutive fallback runs produce no decision-changing output,
the fallback itself is retired for that line per the anti-busywork rule and
the line is queued for the next quarterly review instead.

## 3. Retirement Decisions

- **Monthly:** dependency review — every blocked line either gains a concrete
  unblocking event/date or moves one step toward retirement; lines P10, P12,
  P14 are the current dependency-blocked set.
- **Quarterly:** apply kill conditions registry-wide (032j); publish the
  internal synthesis; require at least one retire-or-promote decision so the
  registry provably breathes. Current merge/narrow slate from this audit:
  P15→P6 merge; P7 and P9 and P11 narrowed; P14 deferred.

## 4. Standing Constraints

No compute-heavy experiment before collision review (doc 26 §4). No novelty
claims without opened primary sources. No task authorizes runtime mutation,
submission, or claim promotion. Satoshi's conflict on P5/P13 and Musashi's
symmetric conflict are disclosed wherever their artifacts become evidence.
