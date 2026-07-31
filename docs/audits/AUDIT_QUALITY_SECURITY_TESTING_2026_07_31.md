# Cross-Repository Quality, Security and Testing Audit

Audit ID: AUDIT-QUALITY-20260731-01
Timestamp and timezone: 2026-07-31 00:20 America/Bogota (UTC-5)
Auditor: Satoshi
Requested by: user
Addressed to: Musashi
Scope: (1) review of the Moltbook agent's published output and collected
corpus for use-case value; (2) security posture across all ten active
repositories; (3) project-structure and software-quality assessment;
(4) coverage against the four declared testing levels; (5) a per-repository
quality tiering driven by exposure, as requested.
Excluded scope: no deep per-module code review yet — this is a first-pass
assessment establishing the framework and priorities. Per-repository deep
reviews are scheduled in section 8. No penetration testing, no dependency
installation, no test execution, no external posting.

Method note: this is a **breadth-first** pass. It is deliberately structural
rather than exhaustive, because the user asked for continuous, increasingly
deep coverage rather than one large one-off review. Depth arrives through the
recurring program in section 8.

## 1. Moltbook Agent Review

### 1.1 Published output: none, and that is correct

The account `u/Dragon_DOIN` has published nothing. Verified three ways:

- the social OLAP `drafts` table is **empty** (zero rows in any state), so no
  draft has been created, approved or posted;
- `posts` contains **zero** rows authored by `Dragon_DOIN`, meaning the
  collector has not observed our own agent publishing;
- `publishing.enabled` is `false` in the tracked config, and `publish_approved`
  raises before any network call.

`register_moltbook.py` only registers the agent and writes the key at mode
`0600`; it contains no posting path. **Verified non-finding: the publishing
boundary holds in practice, not just on paper.**

Limitation to be honest about: I attempted to read the public profile page
directly, but `https://www.moltbook.com/u/Dragon_DOIN` is JavaScript-rendered
and returned only the page shell, with no posts, karma, or reactions visible.
I did **not** authenticate to the API to read it, because using the stored
credential would mean exercising a secret, which the audit permission model
forbids. Consequently I cannot report reactions or engagement metrics. If you
want that visibility, the cheapest safe route is a read-only `me()` /
profile call added to the existing tool and surfaced in the digest — see
recommendation 1.4.

### 1.2 What the collected corpus reveals (147 posts)

This is the more valuable finding. The highest-relevance material our collector
is surfacing clusters tightly around **verification, reliability and
operational trust** rather than model capability:

| Relevance | Title (truncated) | Submolt |
| ---: | --- | --- |
| 1.00 | AI Agent for Algorithmic Trading Systems — Seeking Community Insight | general |
| 0.94 | AI Agent Skill Development: Building Capabilities for Economic Independence | ai-agents |
| 0.94 | Technical Implementation Challenges for AI Agents | technology |
| 0.71 | The Verification Gap: Why I Stopped Trusting My Own Logs | general |
| 0.71 | Verification isn't a compute problem, it's a workflow identity problem | general |
| 0.71 | The Winners Won't Have the Biggest Model. They'll Have the Most Reliable… | general |
| 0.71 | Incident Response Engineering for Rapid Agent Service Recovery | agentstack |
| 0.71 | Error Taxonomy for Agent Systems | agentstack |
| 0.71 | Fine-Grained Access Control for Multi-Agent Collaboration Platforms | agentstack |
| 0.71 | Profit-maximization functions can induce optimal herding in AI agents | general |

### 1.3 Strategic read — where the actual use-case value is

**Observed:** the community's stated pain is verification, reliability,
incident response and error taxonomy. **Inferred:** that is precisely the
capability set this project has already built and can evidence, and it is
*not* what most agent projects can credibly claim:

- Proof of Optimization, commit-reveal and evaluator quorum are a working
  answer to "The Verification Gap: Why I Stopped Trusting My Own Logs";
- content-addressed artifacts with replication proofs answer "verification is a
  workflow identity problem";
- the deterministic watchdog fleet, the campaign lifecycle state machine and
  this independent audit function answer the incident-response and reliability
  threads — including a genuinely instructive worked example, the IBKR observer
  that failed silently behind a green TCP probe;
- the equal-height fork we are currently classifying is exactly the kind of
  distributed-systems war story that community values.

Therefore the highest-value publishing use case is **not** trading signals or
performance claims — which are also the most regulated and least defensible.
It is **verifiable-optimization and reliability engineering evidence**, where
we can post reproducible hashes and be right in public.

Two immediate, concrete opportunities:

1. The relevance-1.00 post "AI Agent for Algorithmic Trading Systems — Seeking
   Community Insight" is a direct request for what we know. It is the single
   best candidate for a first source-backed reply when `draft_only` opens.
2. "Profit-maximization functions can induce optimal herding in AI agents" is
   research-relevant to our own design: a shared-population swarm optimizing
   one fitness is structurally a herding mechanism. Whether our diversity
   controls (Pareto elites, island seeds, diverse-elite archival) actually
   prevent premature convergence is a testable question about **our** system,
   and a good candidate for a bounded offline experiment proposal.

### 1.4 Recommendations for the Moltbook track

1. Add a read-only profile/engagement probe (`me()` plus own-post reactions)
   to the collector and surface it in the digest, so published output can be
   evaluated for usefulness without a human opening the site and without an
   auditor touching credentials.
2. Record an explicit "target topic" list favouring verification, reliability
   and distributed-systems evidence over trading performance claims. Document
   23 already forbids unbacked financial claims; this makes the positive
   direction explicit rather than only the prohibition.
3. When `draft_only` opens, seed it from the two opportunities above rather
   than from generic content.

## 2. Repository Risk Tiering (as requested)

Quality bars should follow exposure and blast radius, not uniform ceremony.
Proposed tiering, for your ratification:

| Tier | Repositories | Why | Required bar |
| --- | --- | --- | --- |
| **A — public / adversarial** | `doin-core`, `doin-node`, `doin-plugins`, `lts` | DOIN accepts input from untrusted peers by design and its repos are public; LTS holds broker credentials, will hold customer risk, and is the only live-order authority | CI mandatory; adversarial/fuzz/property tests; dependency pinning and provenance; documented threat model; security review before any public/multi-user exposure; all four test levels |
| **B — trust-critical internal** | `agent-multi`, `prediction_provider`, `financial-data` | Silent correctness failures invalidate research or serve a wrong artifact; hash verification is a security boundary; data lineage integrity | CI; leakage/cutoff tests (already strong in `agent-multi`); property tests for invariants; integration tests across the boundary they own |
| **C — libraries** | `trading-contracts`, `gym-fx`, `heuristic-strategy` | No network surface, but every consumer depends on their correctness; `trading-contracts` is depended on by everything | CI; unit plus property/contract fixtures; backward-compatibility tests on schema changes |

`predictor` is reference-only and is excluded from the active bar.

The tiering matters because current effort is **not** distributed this way:
`agent-multi` (Tier B) has the deepest leakage coverage, while `doin-core` and
`doin-node` (Tier A, adversarial, public) have flat test directories with no
level separation and near-zero property or adversarial tests.

## 3. Security Posture

### 3.1 Verified good (all ten repositories)

- **Zero tracked secret-like files** across all ten repositories (`.env`,
  `credentials`, `secret`, `.pem`, `.key` patterns) — checked against
  `git ls-files`, so this covers tracked content, not merely the working tree.
- Every repository has a `.gitignore`.
- LTS practice lab **hard-fails on live**: "The execution laboratory cannot
  connect to OANDA live" is enforced in code.
- Credential handling consistently uses environment references, redaction and
  account fingerprints — densest exactly where it matters
  (`mt5_bridge_lab.py`, `oanda_practice_lab.py`, `ibkr_paper_lab.py`).
- `doin-node` network modules carry size limits, timeouts and rate bounds
  across `transport`, `discovery`, `sync`, `flooding`, `gossip`, `peer` and
  `sharding` — consistent with the file-descriptor incident fix.
- The Moltbook tool enforces credential file mode `0600` and parses the env
  file **without shell evaluation**.

### 3.2 Gaps

Detailed as findings in section 6: no CI anywhere (so no automated secret
scanning, dependency audit or regression gate), and dependency reproducibility
resting on a single conda environment hash rather than per-repository pinning
or an SBOM — a supply-chain gap that matters most for the Tier A public repos.

`doin-core`'s cryptographic surface is compact (`crypto/hashing.py`,
`crypto/identity.py`, with signature verification appearing in `identity.py`
and `payment_channel.py`). Compactness is good, but for a Tier A consensus
system this deserves a dedicated deep review rather than the structural pass
performed here — scheduled as `AT-SEC-020`.

## 4. Testing: Coverage Against the Four Declared Levels

Document 09 section 1 declares four verification layers: unit; property and
metamorphic; integration; and system/acceptance. Measured file counts:

| Repository | Total | Unit-organised | Integration | System/E2E | Property/invariant | Adversarial | Leakage/cutoff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `heuristic-strategy` | 83 | 57 | 17 | 1 | – | – | – |
| `agent-multi` | 48 | 47 | 1 | 0 | 0 | 3 | **41** |
| `prediction_provider` | 46 | 0* | 13 | 8 | 0 | 7 | 1 |
| `financial-data` | 35 | 0* | 0 | 1 | – | – | – |
| `lts` | 29 | 13 | 5 | 3 | 1 | 10 | 6 |
| `doin-node` | 21 | 0* | 0 | 0 | 1 | 2 | 9 |
| `doin-core` | 20 | 0* | 0 | 0 | 0 | 1 | 3 |
| `gym-fx` | 13 | 0* | 0 | 0 | 0 | 2 | 10 |
| `doin-plugins` | 7 | 0* | 3 | 1 | – | – | – |
| `trading-contracts` | 3 | 0* | 0 | 0 | 0 | 0 | 3 |

`0*` means a flat `tests/` directory with no level separation — the tests
exist and many are genuinely unit tests, but the **structure cannot assert
which level is covered**, and no gate can run "all integration tests" for
those repositories. `prediction_provider` is the positive outlier, with an
explicit `acceptance_tests`, `behavioral_tests`, `integration_tests`,
`production_tests` taxonomy that the other repositories should copy.

Honest answer to your question — **no, the four levels are not fully
covered**:

- **Unit:** well covered everywhere by volume.
- **Property/metamorphic:** **effectively absent** (0–1 files per repository).
  Document 09 section 1.2 enumerates ten specific invariants — zero exposure
  produces no P&L, linear notional scaling, asset-order permutation
  invariance, future-input mutation, one-cell portfolio equivalence,
  deterministic replay, tighter limits never increase exposure, stale signal
  cannot enlarge a position, net instrument target equals the sum of virtual
  cell targets. These are the tests that catch accounting and leakage bugs
  that unit tests pass straight through. This is the largest genuine gap.
- **Integration:** partial. Good in `heuristic-strategy`, `prediction_provider`
  and `lts`; essentially absent in the DOIN repositories, which is inverted
  relative to risk.
- **System/acceptance:** largely **not runnable suites**. Document 09
  section 1.4 items (full validation-year walk-forward, multi-node champion
  migration, provider channel switch and rollback, multi-user LTS portfolios,
  multi-venue paper weekly operation, 24-hour observation) are exercised
  operationally on live machines and recorded in prose, not as repeatable
  automated tests.

One important nuance in the project's favour: `agent-multi`'s 41
leakage/cutoff-related test files are a genuinely strong showing in the single
highest-risk correctness area, and the protected-test firewall test is real.
The gap is not carelessness; it is uneven distribution relative to exposure.

## 5. Project Structure and Software Quality

**Good:** ownership boundaries in document 01 are respected in practice — the
Moltbook tool imports only the standard library and has no campaign or broker
coupling; `doin-plugins` stays thin; `trading-contracts` remains dependency-
light. Packaging is modern (`pyproject.toml`) in 6 of 10 repositories. Naming
and layout are consistent and predictable across repositories, which materially
helped this audit.

**Weak:** four repositories lack `pyproject.toml` (`agent-multi`, `gym-fx`,
`financial-data`, plus flat-layout scripts elsewhere), so they are not
installable/pinnable as packages in the same way. Test taxonomy is inconsistent
(section 4). There is no repository-level quality gate anywhere, so
conventions rely on discipline rather than enforcement — which has worked so
far because the discipline is unusually good, but does not scale to public
contribution on the Tier A repositories.

## 6. Findings

### AUD-GEN-20260731-009 — No continuous integration in any repository

- Severity **S2**; confidence high (observed).
- Observation: **zero** `.github/workflows` across all ten active
  repositories. Every verification is a manual or local run.
- Contract tension: document 09 section 12 states "CI covers unit, property,
  integration and contract tests", and section 2 requires "CI includes a
  future-row mutation test and explicit column provenance audit". Neither
  exists.
- Impact: acceptance evidence is point-in-time and human-triggered; a
  regression between deployments is invisible until someone runs a suite by
  hand. This is the same class as `AUDIT-TEST-EVIDENCE-002` but broader — that
  packet only captures results, whereas CI would also gate merges. For Tier A
  repositories that are public, it additionally means no automated secret
  scanning or dependency audit on incoming changes.
- Proposed correction: start with one minimal workflow per Tier A repository
  running the existing suite on push; add the document 09 section 2 leakage
  mutation test to `agent-multi`; expand to Tier B/C afterwards. Deliberately
  small — the goal is a gate, not a platform.
- Owner: Musashi.

### AUD-GEN-20260731-010 — Property and metamorphic layer is declared but unimplemented

- Severity **S3**; confidence high (observed).
- Observation: 0–1 property/invariant test files per repository against ten
  invariants enumerated in document 09 section 1.2.
- Impact: the invariants most likely to catch silent accounting, netting,
  permutation and staleness defects are unenforced. These are cheap to write
  and disproportionately valuable in exactly the areas (portfolio accounting,
  LTS netting, simulation ledger) where a bug would be expensive and quiet.
- Proposed correction: implement the ten declared invariants where their owner
  is unambiguous — `gym-fx` for zero-exposure/linear-scaling/permutation,
  `lts` for netting, stale-signal and tighter-limits, `agent-multi` for future
  mutation and deterministic replay. `hypothesis` is a natural fit but plain
  parametrised tests are acceptable.
- Owner: Musashi.

### AUD-GEN-20260731-011 — System/acceptance level exists operationally, not as runnable suites

- Severity **S3**; confidence high (observed).
- Observation: document 09 section 1.4 acceptance items are verified by live
  operation and recorded in prose. Only `prediction_provider` maintains an
  acceptance/production test taxonomy.
- Impact: acceptance evidence cannot be re-run on demand after a change, so
  regression at the system level is detected by operating the system rather
  than by testing it. Given the plan's emphasis on reproducibility, this is the
  layer least aligned with its own standard.
- Proposed correction: adopt the `prediction_provider` taxonomy repository-wide
  and convert the two cheapest acceptance items first — provider channel switch
  and rollback, and one bounded deterministic replay.
- Owner: Musashi.

### AUD-GEN-20260731-012 — Dependency reproducibility depends on a single environment hash

- Severity **S3** (Tier A), S4 elsewhere; confidence high.
- Observation: no meaningful per-repository pinning (`lts` has 4 pinned lines;
  others have none or no requirements file); reproducibility rests on the
  conda `trading-stack` environment inventory hash recorded in document 13.
- Impact: the fleet is reproducible, but individual repositories are not
  independently installable at known versions, and there is no SBOM or
  dependency provenance record. Document 24 section 3.6 requires "package,
  model-weight and binary provenance is recorded". For public Tier A
  repositories this is also a supply-chain exposure.
- Proposed correction: pin Tier A repositories with a lock file and record an
  SBOM; keep the environment hash as the fleet-level control.
- Owner: Musashi.

### AUD-F3-20260731-013 — Social relevance scoring is coarse and saturating

- Severity **S4**; confidence high (observed).
- Observation: scores cluster at 1.00, 0.94 and then a wide band at exactly
  0.71, indicating simple term-match counting. Because the digest takes the
  top 30 by relevance, a saturated score makes ranking near-arbitrary within
  the largest band.
- Impact: reduces the usefulness of the review packet; combined with
  AUD-F3-20260731-008 (flagged posts consuming top-N slots) the effective
  selection quality is weaker than the design intends.
- Proposed correction: length-normalise, weight distinctive terms above
  generic ones, and add recency; or defer ranking to the Hermes tier with the
  deterministic score as a filter rather than a sort key.
- Owner: Musashi.

## 7. Verified Non-Findings

1. Publishing boundary holds in practice: zero drafts, zero published posts,
   config disabled, code raises before any network call.
2. No tracked secret-like files in any of the ten repositories.
3. LTS practice lab refuses live endpoints in code.
4. `doin-node` network modules carry size, timeout and rate bounds.
5. Repository ownership boundaries from document 01 are respected in the new
   social code (stdlib only, no campaign/broker coupling).
6. `agent-multi` leakage/cutoff coverage is substantial (41 files) in the
   highest-risk correctness area.
7. `prediction_provider` demonstrates a correct four-level test taxonomy — the
   pattern to propagate.

## 8. Proposed Continuous Deepening Program

Answering the request for continuous, increasingly deep coverage across all
three fronts. Each item is a new backlog task with a stable ID; depth
increases as earlier layers close.

| ID | Front | Depth | Task |
| --- | --- | --- | --- |
| `AT-SEC-020` | 1 | deep | `doin-core` cryptographic and trust-primitive review: identity, signing, hashing, commit-reveal, replay and quorum assumptions |
| `AT-SEC-021` | 1 | deep | `doin-node` untrusted-peer input review: message validation, bounds, dedup identity, fork handling, resource exhaustion |
| `AT-SEC-022` | 2 | deep | `lts` credential, redaction and order-authority review; confirm no path bypasses risk and reconciliation |
| `AT-QUAL-023` | all | medium | Verify CI adoption and the leakage mutation gate after AUD-GEN-20260731-009 |
| `AT-QUAL-024` | all | medium | Verify the ten document 09 invariants once implemented |
| `AT-F3-013` | 3 | medium | Hermes-side model call: provider, budget, and whether the sanitized packet is the only prompt input |
| `AT-SEC-025` | 3 | deep | Moltbook adversarial fixture review: multilingual injection, citation forgery, crowd-out |
| `AT-QUAL-026` | all | light | Dependency/SBOM verification for Tier A after pinning |

Suggested rotation honouring the 72-hour front-coverage rule: Front 1 deep
security (`AT-SEC-020`, `AT-SEC-021`), then Front 2 (`AT-SEC-022`), then Front
3 (`AT-SEC-025`), with quality verification tasks interleaved as their
prerequisites land. This complements rather than replaces the operational
tasks already in the backlog.

## 9. Commands and Queries

All read-only. Tier-0 pre-collected evidence was not applicable to this scope
(the collector packet does not yet carry code-quality sections).

```text
find <repo> -name 'test_*.py'                        (10 repos, counted by level)
ls <repo>/tests ; find -type d -name 'tests*'
rg -l -i 'hypothesis|property|metamorphic|permut|invariant' <repo>/tests
rg -l -i 'injection|malicious|tamper|replay|forge|adversar|fuzz' <repo>/tests
rg -l -i 'leak|lookahead|cutoff|firewall' <repo>/tests
git -C <repo> ls-files | grep -iE '\.env$|credentials|secret|\.pem$|\.key$'
ls <repo>/.github/workflows ; test -f <repo>/pyproject.toml
grep -cE '==' <repo>/requirements*.txt
rg -c -i 'max_size|size_limit|rate_limit|timeout' doin-node/src/doin_node/network/*.py
rg -n -i 'api-fxtrade|live' lts/app/oanda_practice_lab.py
sqlite3 -readonly social-intelligence.sqlite  (authors, drafts, relevance ranking)
WebFetch https://www.moltbook.com/u/Dragon_DOIN   (returned JS shell only)
```

## 10. Change Confirmation

No code, configuration, service, campaign, chain, broker, credential or Git
state was modified. No credential was used, read or tested; the Moltbook API
was not authenticated against. No test was executed. No content was published.
Writes were limited to `docs/audits/`.
