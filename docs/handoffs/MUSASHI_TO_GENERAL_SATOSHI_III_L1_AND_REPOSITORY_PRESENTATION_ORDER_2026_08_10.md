# Execution Order: L1 Decision Run and Multi-Repository Presentation Refresh

Date: 2026-08-10 America/Bogota  
From: General Musashi, independent auditor  
To: General Satoshi III, technical lead  
Owner status: both work packages are approved; no additional phrase is needed

## 1. Role and Operating Standard

Act as a senior Python packaging architect, decentralized-systems engineer,
machine-learning/trading-platform maintainer, open-source technical writer and
GitHub repository curator. Treat executable code, packaging metadata, tests,
deployed manifests and current examples as sources of truth. Existing READMEs
are evidence to audit, not authority to repeat.

Be proactive and complete both work packages. Ask the owner only for actions
that genuinely require owner authority: real capital, paid services, secrets,
visibility/archive changes, destructive history changes or mission changes.
README, About and topic corrections requested here require no repeated owner
confirmation.

Never stop, restart or mutate the running L1 decision identity for
documentation work. Never overwrite an unattributed dirty tree. Never delete
historical results, OLAP databases, champion artifacts or legacy repositories.

## 2. Required Reading

Read these first, in order:

1. `docs/audits/AUDIT_SATOSHI_III_L1_ROUND3_ACCEPTANCE_2026_08_10.md`
2. `docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_III_L1_ROUND3_ACCEPTANCE_AND_LR_PAIR_PLAN_2026_08_10.md`
3. `docs/work_plan/38_NEAT_LESSONS_L1_L2_CURRICULUM_AND_FEATURE_SELECTION.md`
4. `docs/work_plan/05_DOIN_TRADING_DOMAIN_INTEGRATION.md`
5. `docs/work_plan/08_IMPLEMENTATION_ROADMAP.md`
6. `docs/work_plan/13_IMPLEMENTATION_STATUS_AND_TASK_LEDGER.md`
7. `docs/work_plan/15_DISTRIBUTED_CAMPAIGN_LIFECYCLE.md`
8. `doin-node/docs/shared_population_semantics.md`

The first three files are available at commit `a0b8f18a` on branch
`musashi/l1-round3-acceptance-audit-20260810` until integrated. Read that exact
revision; do not substitute the older round-2 documents.

## 3. Work Package A: Continue the Accepted L1 Decision Run

Decision identity `2de49ea9225e2baf` is active on four workers. Preserve it.

1. Monitor fresh heartbeats, exact identity, seed/host/GPU binding, current
   cell, terminal state, temperature and unresolved failures.
2. Do not restart healthy workers. A failed worker resumes only through the
   existing durable launcher and immutable attempt contract.
3. After all four seeds terminate cleanly, collect, seal, replicate to Dragon,
   load all sixteen terminal artifacts from the replica and aggregate only
   through the collection envelope.
4. Return raw, consistently scaled metrics and paired-seed contrasts for
   difficulty, normal-phase LR and interaction.
5. Apply the conditional `LR_easy x LR_normal` plan exactly as specified in
   the preceding Musashi handoff. Do not broaden the factor set or open the
   sealed 2025 split.

This runtime work has priority over documentation if an unresolved operational
alert occurs. Normal healthy training requires observation, not interference.

## 4. Work Package B: Establish the Repository Inventory

Create one machine-readable inventory before editing any README:

`agent-multi/docs/audits/evidence/REPOSITORY_PRESENTATION_INVENTORY_2026_08_10.json`

For every repository record:

- local path, GitHub `owner/name`, visibility and default branch;
- HEAD, upstream HEAD, dirty files and whether the tree is safe to edit;
- package name/version, Python requirement, entry points and plugin groups;
- role proven from code/imports/tests/deployment manifests;
- lifecycle: `active-core`, `active-component`, `experimental`,
  `legacy-superseded`, `historical-reference` or `third-party-excluded`;
- README path and identified obsolete claims;
- current/new GitHub description, homepage, topics and archive state;
- current consumers and replacements, with file references.

Use codebase-memory graph tools for code architecture and call relationships.
Use structured packaging/config parsers for `pyproject.toml`, `setup.py`, JSON
and YAML. Use `rg` for README/config literals. Do not infer activity merely
because a directory exists.

### 4.1 Presumptive active scope to verify

1. `agent-multi`
2. `doin-core`
3. `doin-node`
4. `doin-plugins`
5. `gym-fx`
6. `trading-contracts`
7. `lts`
8. `financial-data`
9. `prediction_provider`
10. `predictor`
11. `heuristic-strategy`

### 4.2 DOIN legacy scope

- `doin-optimizer`
- `doin-evaluator`

These are presumptively superseded by unified role execution in `doin-node`.
Verify that from current code/history. Preserve them and clearly label their
status; do **not** archive, delete, make public/private or rewrite history
without a separate owner decision.

### 4.3 Supporting candidates to classify

- `causal-inference`, `feature-eng`, `feature-extractor`, `preprocessor`
- `rl-optimizer`, `synthetic-datagen`, `timeseries-gan`, `trading-signal`

Update a candidate only after proving its current public role or explicitly
classifying it experimental/historical. `causal-inference` currently has an
unattributed dirty tree: inventory it read-only and do not edit, stage or reset
those files.

Exclude `TradingAgents`: it is a third-party checkout and is not owned by
`harveybc`.

## 5. DOIN Architecture That Every Relevant README Must State Correctly

The READMEs must agree on these boundaries:

- **`doin-node` is the unified participant runtime.** Optimizer, evaluator /
  inference worker and network-node responsibilities that were once separate
  repositories now run as configured roles/capabilities in this repository.
- **`doin-core` owns shared decentralized protocol primitives**, including
  blockchain/consensus-related contracts and OLAP-on-blockchain structures;
  describe only behavior actually present in current code.
- **`doin-plugins` owns reusable DOIN extension contracts and plugins**. It is
  not the home of every domain model.
- **Domain optimizers/models remain external installable packages**. They must
  work locally without DOIN first and implement the plugin interface consumed
  by `doin-node`. DOIN extends local optimization/inference collaboratively;
  it does not absorb domain concerns.
- Per-machine JSON selects roles/plugins and points to the same experiment,
  seed/genesis/domain and shared population/chain contract. Champion migration,
  candidate leasing and duplicate-evaluation avoidance must be described from
  current tested semantics, not recreated or guessed.
- Do not claim that retired `doin-optimizer` or `doin-evaluator` services are
  required for a current deployment.

## 6. README Contract

For every in-scope owned repository, produce an accurate root README with:

1. project name and a one-paragraph concrete purpose;
2. a prominent lifecycle/status declaration;
3. its exact role and explicit non-responsibilities;
4. current architecture and relationship to sibling repositories;
5. supported Python/runtime prerequisites derived from packaging metadata;
6. installation commands that were executed in a clean environment or clearly
   marked as unverified when external services prevent execution;
7. the smallest working local example using a repository-owned config;
8. distributed/DOIN usage only where genuinely supported;
9. configuration and plugin entry-point reference with links to real files;
10. test/validation commands and their observed results;
11. artifact/data/output locations and reproducibility expectations;
12. safety/security/credential handling appropriate to the repository;
13. limitations and legacy migration notes;
14. links to the correct sibling repositories and current deeper docs;
15. license/citation/contribution information when those files actually exist.

Additional constraints:

- Keep setup instructions copy-pasteable and relative links valid on GitHub.
- Do not invent badges, support promises, performance numbers, package releases,
  hosted services or downloadable artifacts.
- Do not expose machine paths, account identifiers, API keys, Telegram data or
  private topology details.
- Trading repositories must distinguish simulation, Paper/Demo and real-capital
  status and state that examples are not financial advice.
- Private `financial-data` topics are still public GitHub metadata; include no
  sensitive provider/account information.
- Legacy repositories begin with a clear supersession notice and link to
  `doin-node`; retain historical usage below it when still reproducible.

## 7. GitHub About and Topics

GitHub permits no more than **20 topics**, each lowercase, at most 50
characters, using letters, numbers and hyphens. For every owned in-scope repo:

1. set a non-empty, concrete About description that states what the repository
   actually does now;
2. set **exactly 20 relevant topics**, because the owner requested the maximum;
3. remove stale topics that describe retired architecture or unrelated claims;
4. use a homepage only when a real maintained page exists; otherwise leave it
   empty rather than inventing one;
5. preserve visibility, archive state, default branch, issues, wiki and other
   repository features unless separately authorized.

Topics must favor discoverability and truth: domain, architecture, language,
framework and interoperability. Legacy repos include `legacy`, `deprecated`
and `superseded-by-doin-node`; active repos must not use those labels.

Apply metadata through authenticated `gh repo edit`/GitHub API without logging
tokens. Snapshot before and after with:

```bash
gh repo view harveybc/REPO \
  --json name,description,homepageUrl,repositoryTopics,isArchived,visibility
```

Official topic constraints:
https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/classifying-your-repository-with-topics

## 8. Git and Execution Discipline

- One clean worktree and one bounded commit per repository.
- Never mix existing user changes. A dirty repo is read-only until attributed.
- Run README link checks, packaging metadata validation and the repository's
  proportionate test/CLI smoke before committing.
- Push every completed repository commit. Record branch and remote commit.
- Metadata can be applied after its matching README commit is pushed.
- Do not mass-edit version numbers or dependencies merely to make prose look
  current.
- Preserve consolidated results and OLAP databases; generated runtime outputs
  do not belong in README commits.

## 9. Required Acceptance Packet

Return:

`agent-multi/docs/handoffs/SATOSHI_III_REPOSITORY_PRESENTATION_DELIVERY_2026_08_10.md`

and a CSV/JSON evidence table containing, per repository:

- lifecycle and role evidence;
- old/new README hash and commit;
- README sections present;
- tested commands/results;
- broken-link count (must be zero for edited links);
- old/new About description;
- old/new topic lists and exact topic count;
- visibility/archive/default-branch unchanged proof;
- push/upstream equality;
- explicit refusals or unresolved ambiguity.

Acceptance requires:

- no active README names retired DOIN role repositories as deployment
  requirements;
- `doin-node` documents unified role configuration and external domain plugins;
- every completed repository has a non-empty About description and exactly 20
  relevant topics;
- all edited README links resolve;
- no secrets or private operational identifiers appear;
- no runtime, chain, model artifact or OLAP state changed;
- the L1 decision run remains healthy throughout the documentation work.

General Musashi will independently review architecture claims, links, commands,
GitHub metadata and runtime non-interference. Do not close your own findings.

