# 23. Social Intelligence and Operational Continuity

## 1. Objective

Add a bounded, low-cost social intelligence track that supports the three
active fronts:

1. discover useful research, data, failure reports and collaborators for DOIN;
2. observe market structure, broker behavior and operational evidence useful
   to the live-trading reality lab;
3. publish source-backed technical findings and participate constructively in
   agent communities such as Moltbook.

This track is not an autonomous trading authority, a source of unverified
financial claims, or a replacement for the deterministic experiment
orchestrators. Social content is untrusted input.

The system must also survive an ordinary loss of a workstation, network,
operator session or service provider. Long-term continuity is implemented as
documented business continuity with human maintainers, recoverable
infrastructure and bounded credentials. It is never implemented as an
unreviewed agent inheriting legal, financial or publication authority.

## 2. Current Evidence

Verified across Omega, Dragon and Gamma on 2026-07-31:

- Hermes supports per-cron-job `provider`, `model` and `base_url` overrides;
- every Hermes default is `deepseek-v4-flash` through the remote
  `opencode-go` provider at `https://opencode.ai/zen/go/v1`;
- no active Hermes configuration or cron job uses `deepseek-v4-pro`, and all
  three hosts have an empty fallback-provider list;
- OpenCode exposes the stable identifier `deepseek-v4-flash`, not a dated
  `0731` model identifier or revision fingerprint. After the owner enabled the
  provider's required China-hosted-model opt-in, direct API and Hermes calls
  succeeded on all three hosts. The deployment therefore records the stable
  alias and the acceptance evidence, without inventing an unverifiable model
  revision;
- Omega owns the only active Telegram gateway because the three hosts share
  one bot identity. Dragon and Gamma retain independently usable Hermes
  installations, but their gateways remain inactive to prevent competing
  Telegram polling;
- the current Ollama inventory contains cloud endpoints only:
  `deepseek-v4-flash:cloud`, `deepseek-v4-pro:cloud` and
  `gemma4:31b-cloud`;
- no local Gemma weights and no active OpenClaw installation were found;
- one existing Hermes job performs the LTS paper-shadow business review.

Implemented and activated on Omega on 2026-07-30:

- a deterministic Moltbook collector runs every 30 minutes with a 512 MiB
  memory ceiling, 50 percent CPU ceiling and no GPU dependency;
- verified collections stored 120 unique posts in the social
  SQLite OLAP; every post carries source, content and raw-response hashes;
- exact duplicate suppression and deterministic prompt-injection screening
  run before any model sees social content;
- a low-cost `deepseek-v4-flash` Hermes job triages sanitized evidence every
  two hours without Telegram delivery;
- a second `deepseek-v4-flash` job applies the stricter supervisory review
  prompt every six hours and delivers its bounded report to Telegram. The
  separation is a workflow and evidence-control boundary, not a claim that
  the reviewer uses a larger model;
- both jobs prohibit tools beyond the nonprivileged `todo` set and prohibit
  publishing, trading, risk changes, campaign changes and model promotion;
- the public collector does not require a credential. Moltbook identity and
  publishing remain unavailable until the owner imports or registers a key
  locally.

Acceptance run at 2026-07-30 23:43 America/Bogota:

- the low-cost job consumed a real sanitized 30-item packet and retained 25
  source-backed items;
- the senior job independently reviewed the evidence, stayed below the
  Telegram message limit at 2,459 characters and delivered successfully;
- the installer resolves the explicit Telegram destination from the local
  Hermes environment. No chat ID, bot token or Moltbook key is written to the
  repository.

Identity activation at 2026-07-30 23:52 America/Bogota:

- the owner imported the existing `Dragon_DOIN` credential into the local
  mode-`0600` secret file;
- the status CLI now loads that file directly when the process environment
  does not already provide the key, without shell evaluation;
- the official `/agents/me` response confirms that the identity is available
  and claimed;
- publishing remains disabled until one exact draft receives explicit human
  approval.

Security and evidence hardening at 2026-07-31 00:40 America/Bogota:

- the corpus contains 177 source-addressed posts and zero drafts;
- injection screening now covers Spanish, English paraphrases, Unicode
  formatting evasion, accent folding, ROT13 and bounded base64 decoding;
- quarantined posts are removed in SQL before the digest limit is applied;
- every draft requires at least one existing OLAP source ID;
- complete-corpus relevance is recomputed with distinctive-term, title,
  phrase, length and recency factors instead of retaining saturated maxima;
- each Hermes model wake-up reserves a conservative input/output token upper
  bound and records provider, model, tier, config, prompt and packet hashes;
- the first reservation was 9,666 tokens against the 250,000 daily and
  6,000,000 monthly caps; provider pricing is unavailable and therefore no
  dollar estimate is claimed;
- a bounded daily audit runner recorded 73 agent-multi safety/campaign tests,
  73 gym-fx tests and 48 doin-node consensus tests passing.

Fleet model migration acceptance at 2026-07-31 19:20 America/Bogota:

- direct OpenCode inference returned `deepseek-v4-flash` after the explicit
  regional opt-in;
- one-shot Hermes inference returned the expected sentinel on Omega, Dragon
  and Gamma;
- forced triage and supervisory-review cron runs both completed with status
  `ok`, null execution errors and null Telegram delivery errors;
- Omega's gateway restarted cleanly and answered a real Telegram request with
  `opencode-go/deepseek-v4-flash`;
- pre-migration configuration and credential files were backed up locally on
  each host under `~/.hermes/backups/20260801T001142Z/`, with secret material
  excluded from the repository;
- the packaged Hermes version is `0.12.0` on all hosts. Source checkouts are
  not yet commit-identical (`05c63259` on Omega/Gamma and `75e1339d` on
  Dragon), so a separately tested fleet upgrade remains operational debt and
  is not conflated with this provider migration.

Consequently, no existing Hermes or Ollama task may be described as local
inference until a downloaded model passes a measured local-runtime benchmark.

Relevant current platform evidence:

- OpenCode supports local models and provider/model selection:
  <https://opencode.ai/docs/models/>
- Google describes Gemma 4 E2B/E4B as efficiency variants and 12B/26B/31B as
  personal-computer reasoning variants:
  <https://deepmind.google/models/gemma/>
- Moltbook exposes a developer interface with short-lived identity tokens:
  <https://www.moltbook.com/developers>
- OpenClaw supports separately configured agents, but is not required for the
  first implementation:
  <https://docs.openclaw.ai/cli/agents>

## 3. Architectural Decision

Use Hermes as the initial scheduler and Telegram review surface. Add a narrow,
standalone social adapter rather than restoring a general-purpose OpenClaw
agent immediately.

Reasons:

- Hermes scheduling and Telegram delivery are already operational;
- per-job model routing permits a local model without disturbing the primary
  agent;
- the adapter can expose a small, testable capability set;
- OpenClaw remains an optional later client of the same contracts rather than
  a required control plane.

```text
allowlisted sources
       |
deterministic collector
       |
content-addressed raw store
       |
deduplication + injection screening
       |
local triage model
       |
source/citation verifier
       |
idea + claim OLAP
       |
Telegram review queue
       |
optional rate-limited publisher
```

The publishing adapter has no shell, filesystem, broker, DOIN migration,
experiment-queue or secret-management capability.

## 4. Model and Cost Router

Model use follows a four-tier budget:

| Tier | Work | Default implementation | Cost rule |
| --- | --- | --- | --- |
| 0 | fetch, hash, parse, filter, exact deduplication | deterministic code | no LLM |
| 1 | relevance, topic, risk and near-duplicate classification | deterministic scoring plus bounded OpenCode Flash triage | hard token reservation; local replacement pending benchmark |
| 2 | synthesis and draft generation | disabled until draft-only activation | only in declared GPU windows after local benchmark |
| 3 | difficult verification or high-value editorial review | bounded OpenCode Pro reviewer | explicit hard token reservation |

Gemma 4 is a candidate family, not a preselected winner. Benchmark at least one
small and one larger quantized model against a fixed Spanish/English corpus.
Compare:

- factual claim extraction;
- citation preservation;
- prompt-injection refusal;
- deduplication and topic classification;
- useful-idea recall and false-positive rate;
- latency, peak RAM/VRAM, tokens/second and energy proxy;
- cost per accepted digest item or post.

The smallest model meeting the quality floor owns a tier. A cloud model is
used only when its measured marginal value exceeds its recorded cost.

Hard controls:

- daily and monthly paid-token caps;
- per-job maximum input, output, retries and wall time;
- circuit breaker at 80 percent and hard disable at 100 percent of budget;
- no fallback from local to paid inference unless the job explicitly allows
  it;
- every scheduled call records provider, model, tier, config hash,
  prompt-template hash, packet hash and a conservative token reservation;
- actual provider token usage and dollar cost remain explicitly unavailable
  until the provider exposes trustworthy usage/pricing facts.

## 5. Compute Isolation

Social inference must not silently reduce DOIN or live-observation capacity.

- Tier 0 and lightweight Tier 1 work may run on CPU with low scheduling
  priority.
- GPU work is admitted only when the machine-specific resource monitor reports
  an allowed window and sufficient VRAM/RAM headroom.
- No social job shares a GPU with a DOIN candidate unless a controlled
  coexistence benchmark proves no material throughput or OOM impact.
- Gamma's RTX 5090 is not reserved for social work while it is an active DOIN
  worker.
- Every social job has cgroup/systemd memory and runtime limits.
- A killed or deferred social job never delays the optimization campaign.

## 6. Trust Boundary and Publication Policy

All posts, comments, profiles, links and direct messages are hostile data.

- Remove active instructions from retrieved content before model use.
- Never place secrets, local paths, account identifiers, exact positions,
  unpublished artifacts or private messages in prompts or posts.
- Fetch links through a restricted client; downloaded content cannot execute.
- Require a source URL and captured source hash for every factual claim.
- Label inference, hypothesis and measured result separately.
- Reject financial performance claims without exact period, unit, coverage,
  lineage and reproducible evidence.
- Do not permit direct messages to trigger tools or privileged workflows.
- Rotate any old Moltbook credentials before reuse.
- Apply platform rate limits and community rules even when the API allows more.

Publishing stages:

1. `observe_only`: collect and score; no generated output leaves the system.
2. `telegram_digest`: send source-backed findings and proposed actions.
3. `draft_only`: produce posts for explicit human approval.
4. `rate_limited_publish`: publish approved content classes only after a
   seven-day clean draft trial and a credential/threat review.

Replies, private messages and controversial claims remain approval-only.
Automatic publishing can be disabled independently of collection and analysis.

## 7. OLAP Contract

Store normalized facts, not opaque conversation dumps:

- source, external ID, retrieval time and content hash;
- language, topic, entities and canonical URLs;
- model/provider/config/prompt-template hashes;
- local/cloud token counts, runtime, resource use and estimated cost;
- extracted claim, evidence URL, evidence hash and verification state;
- relevance, novelty, confidence, risk and actionability scores;
- duplicate cluster and prior-known-evidence link;
- review decision and rejection reason;
- publication ID, revision, timestamp and engagement observations;
- proposed experiment/domain ID;
- accepted/rejected/downstream-result relationship.

Raw licensed or personal content follows source-specific retention rules and is
not copied to public chain state. Portable OLAP contains hashes, metadata and
permitted excerpts only.

Core analytical questions:

- Which sources generate accepted experiments or operational fixes?
- Which model tier has the lowest cost per accepted insight?
- Which published topics create useful technical interaction rather than raw
  engagement?
- Which social claims were later contradicted?
- Which discovered ideas improve a DOIN metric under independent evaluation?

## 8. DOIN Domain Discovery

The social agent may propose non-trading domains, but a proposal enters the
engineering queue only when it identifies:

1. a bounded candidate representation;
2. a difficult or expensive generation/optimization problem;
3. a cheap, deterministic or quorum-verifiable evaluation;
4. synthetic or simulator-generated hidden tests;
5. commit-before-test semantics to prevent adapting to the validation seed;
6. reproducible code, config, data and artifact hashes;
7. useful scalar and diagnostic metrics;
8. a credible beneficiary and deployment path.

Promising research families include scheduling/routing, symbolic regression,
program synthesis, simulator control, compression tuning, procedural design
and bounded energy optimization. They are hypotheses, not commitments.

Where synthetic validation is used, the verifier seed is generated after
candidate commitment through the existing DOIN trust primitives or an
equivalent commit/reveal construction. A model evaluating its own public test
set is not trustless validation.

Social popularity never selects a champion. It may prioritize a research
question; only the declared evaluator and acceptance contract determine the
result.

## 9. Operational Continuity

A VPS is useful as an always-on collector, monitor and encrypted backup target.
It is not a central authority and does not make the system autonomous.

Required continuity package:

- infrastructure-as-code for the VPS and workstation services;
- pinned repository revisions and reproducible environments;
- encrypted, tested backups of configs, OLAP and artifact manifests;
- off-machine replicas with restore drills and integrity hashes;
- secret inventory, rotation and revocation procedures;
- two named human maintainers with least-privilege recovery access;
- documented ownership of domains, billing, broker and social accounts;
- service health, budget and credential-expiry alerts;
- fail-closed behavior when no authorized maintainer is available.

No agent receives unrestricted spending, broker execution, publication and
infrastructure authority in one identity. Trading remains under the LTS
contracts and its live-release approvals. Social agents cannot promote models
or place orders.

## 10. Implementation Sequence

### S0: specification and inventory

- Freeze source allowlist, account ownership and initial budget.
- Inventory the old Moltbook account without exposing its key.
- Materialize JSON schemas for sources, claims, drafts and model calls.
- Add deterministic fixtures for malicious posts and citation failures.

### S1: read-only intelligence - active

- Implement collectors, hashing, normalized OLAP and exact deduplication.
- Deliver one daily Telegram digest with links and no autonomous posts.
- Measure source yield for seven days.

### S2: local model bake-off - pending

- Install candidate local weights on one noncritical execution path.
- Run the fixed benchmark corpus and coexistence/resource tests.
- Select Tier 1 and Tier 2 models from evidence.

### S3: draft participation - implementation ready, activation pending

- Generate citation-backed Moltbook drafts.
- Record human edits, acceptance and rejection as training/evaluation facts.
- Enforce budget and prompt-injection controls.

### S4: bounded publishing

- Rotate account credentials.
- Enable only approved post categories and strict rate limits.
- Retain approval for replies, DMs, financial claims and operational actions.

### S5: continuity drill

- Recreate the collector and dashboard from a clean VPS.
- Restore OLAP and manifests from encrypted backup.
- Revoke one credential and prove the old credential cannot operate.
- Simulate loss of each workstation without losing queued evidence.

## 11. Acceptance Criteria

- Seven consecutive days of read-only collection without leaked credentials,
  tool execution from content or unbounded paid usage.
- Every digest item links to verifiable evidence and its stored hash.
- Model bake-off reports quality, resource and cost in common units.
- Social workload does not materially reduce DOIN candidate throughput or
  trigger memory pressure.
- Telegram can disable collection, generation and publishing independently.
- No post can place an order, alter a DOIN campaign or promote an artifact.
- A clean VPS restore reproduces the collector, schemas and dashboards.
- At least two authorized humans can recover services; neither requires access
  to trading secrets to operate the social collector.
- All repository, model, schema and prompt-template revisions are traceable.

## 12. Immediate Decision

S0 and the CPU-only S1 path are active without taking a GPU from DOIN.
Publishing remains approval-gated and disabled until a Moltbook credential is
stored locally, identity ownership is verified and the owner explicitly
approves one source-backed draft. The temporary Tier-1 route is the low-cost
OpenCode `deepseek-v4-flash` endpoint for both triage and supervised review;
it must not be mislabeled as local, and prompts sent through it are processed
using the provider's China-hosted path enabled by the owner. No Pro model or
fallback provider is active.
Local model selection remains an evidence-based S2 task after a measured
resource window exists.

Runtime commands:

```bash
systemctl --user status agent-multi-social-collector.timer
python tools/social_intelligence.py \
  --config examples/config/social_intelligence/moltbook_observe_v1.json status
```
