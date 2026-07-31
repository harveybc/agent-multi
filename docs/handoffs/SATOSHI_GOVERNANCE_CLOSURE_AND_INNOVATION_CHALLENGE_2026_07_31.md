# Satoshi Governance Closure and Innovation Challenge

Date: 2026-07-31
Author: Musashi, relayed by Harvey
Recipient: Satoshi
Mode: independent academic and governance audit
Required minimum audited baseline: `3b3e9a7abc4e5b1d83df039e7079e23b1bfcd78f`

Give Satoshi this entire file.

## Invocation

Satoshi, execute this specification completely. Do not substitute a summary,
an informal review or a new plan for the required evidence and deliverables.

Begin with:

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
git fetch origin
git rev-parse HEAD
git rev-parse origin/master
git status --short
git log -1 --oneline --decorate
```

Work from the latest `origin/master`, which must include the required minimum
audited baseline above. Record the exact commit reviewed in every report. If
the local checkout is behind, update it without overwriting unrelated local
work. If the tree contains changes you did not create, preserve them and
report their paths before continuing. Do not silently audit a stale commit.

Complete every packet in this file. P5 and AT-031 are milestones, not the end
of the assignment: the permanent P6+ queue in
`26_CONTINUOUS_RESEARCH_AND_INNOVATION_PROGRAM.md` remains mandatory. Every
claim must resolve to code, executable evidence, an authoritative source or an
explicitly labeled hypothesis. A polished document without that evidence is a
failed delivery.

---

Satoshi, your audit was useful, but its academic roadmap ended where a durable
research institution should begin. Musashi has responded to findings 014-017,
audited your P1 security framing against code, materialized the paper
contracts, and created a continuous P6+ research program. Your task is to
attack that response rigorously, correct it where evidence demands and leave a
permanent research queue rather than a finite checklist.

Act as:

- a senior machine-learning and data-science researcher;
- a distributed-systems and security reviewer;
- an algorithmic-trading methodology reviewer;
- a reproducibility and scientific-publication editor;
- an independent auditor who values falsification over agreement.

Do not perform martial rhetoric in the report. Be precise, concise and
evidence-led. Admit corrections immediately when code or primary literature
contradicts you. Do not reward Musashi for effort; evaluate the result.

## 0. Authority and Prohibitions

Your authority remains defined by:

```text
/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/CLAUDE_CONTINUOUS_AUDIT_AGENT_SPEC_2026_07_30.md
/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/CLAUDE_AUDIT_RECOVERY_PROMPT_2026_07_30.md
```

You may read repositories, runtime status and redacted evidence. You may write
audit/publication deliverables and proposed register updates. You may not:

- mutate a chain, campaign, broker, credential, service or runtime;
- close your own material findings;
- promote a paper or research line on prose quality;
- invent citations, bibliography, novelty, empirical results or security
  guarantees;
- commit or push your own output;
- expand this task into unrelated repository exploration.

## 1. Mandatory Read Order

1. Musashi response:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/audits/CODEX_GOVERNANCE_ACADEMIC_AND_INNOVATION_RESPONSE_2026_07_31.md`
2. Corrected publication contract:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md`
3. Continuous program:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/work_plan/26_CONTINUOUS_RESEARCH_AND_INNOVATION_PROGRAM.md`
4. Research registry:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/publications/RESEARCH_LINE_REGISTRY_2026_07_31.csv`
5. Invariant mapping:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/audits/CODEX_TEN_INVARIANT_TEST_MAPPING_2026_07_31.md`
6. Paper packages:
   `/home/harveybc/Documents/GitHub/agent-multi/papers/`
7. Your prior reports and roadmap:
   `/home/harveybc/Documents/GitHub/agent-multi/docs/audits/AUDIT_ACADEMIC_PUBLICATION_PROGRAM_2026_07_31.md`
   `/home/harveybc/Documents/GitHub/agent-multi/docs/publications/ACADEMIC_RESEARCH_ROADMAP_2026_07_31.md`

Read code only for claims under dispute. Required P1 surfaces are:

```text
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/crypto/identity.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/protocol/messages.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/models/commit_reveal.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/models/quorum.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/consensus/deterministic_seed.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/consensus/finality.py
/home/harveybc/Documents/GitHub/doin-core/src/doin_core/consensus/fork_choice.py
/home/harveybc/Documents/GitHub/doin-node/src/doin_node/unified.py
```

## 2. Packet A: Independent Closure Verification

Verify separately:

1. finding 014: the unregistered Arendt designation was removed and had no
   closure weight;
2. finding 015: all three P5 controls are unambiguous and enforceable;
3. finding 016 / Musashi 019: produce a table with columns
   `property`, `primitive_exists`, `runtime_enforced`, `active_profile`,
   `evidence`, `authorized_claim`;
4. finding 017: run
   `python tools/validate_publication_scaffolds.py` and the focused unit test;
5. finding 010: sample every cited test before accepting the covered/partial/gap
   classification;
6. fork: query the read-only network status and classify the generation-3
   equal-height tips. No mutation. Report finalized-anchor agreement,
   population identity, claims, winner rule and whether convergence occurred.

Do not mark 014-017 closed yourself. Output a verification recommendation for
Harvey or a genuinely independent reviewer.

## 3. Packet B: Audit Musashi's Research Program

Review P6-P18 as adversarial hypotheses:

- detect duplicates, aliases and lines that should be merged;
- identify unfalsifiable wording or metrics vulnerable to Goodhart effects;
- expose dependencies that make a cheap experiment impossible;
- reject ideas whose null result would teach little;
- distinguish scientific contribution, engineering roadmap and business
  feature;
- test whether the horizon/capacity model can actually prevent backlog growth;
- verify the permanent queue cannot create busywork.

Produce a decision table:

```text
line_id
decision: retain | narrow | merge | split | reject
closest_prior_art_candidates
novelty_state
central_hypothesis
cheapest_decisive_experiment
success_metric
null_result_value
dependency
kill_condition
next_bounded_task
```

Do not call any line novel before primary sources are opened.

## 4. Packet C: Primary-Source Collision Tests

Select the five highest-information lines from P6-P18. P6, P7, P9, P13 and
P16 are the current default, but replace one when evidence justifies it.

For each selected line:

1. search at least two scholarly indexes;
2. open and verify the original primary sources;
3. record canonical title, complete authors, year, venue, DOI/arXiv ID and URL;
4. record exact section/page supporting the overlap;
5. record correction/retraction state when discoverable;
6. identify what remains distinct, if anything;
7. recommend retain/narrow/merge/reject.

Bound: at most 30 primary sources in this pass. Add rows to a new dated
research-line ledger; do not rewrite verified history silently.

## 5. Packet D: Future Work That Cannot Be Decorative

For every P1-P5 package, create `FUTURE_WORK.md` containing three to six
ranked lines. Every line must include:

- the limitation/evidence gap that creates it;
- a falsifiable question;
- prior-art verification state;
- required implementation or data;
- cheapest discriminating experiment;
- decision metric and unit;
- dependency and kill condition;
- corresponding P6+ registry ID or a proposed new ID.

Future work cannot be “use more assets”, “try more models” or “scale the
system” without a scientific question and discriminating design.

Then propose additional P19+ lines only when they fill a demonstrated gap.
New line count is not a success metric.

## 6. Packet E: P5 and P13 Conflict Red Team

Because you operate the audit process being studied:

1. verify that the enumeration rule cannot be altered after outcomes;
2. propose a machine-readable incident-corpus manifest;
3. specify blind labeling and external-review steps;
4. define false positive, false negative, detection latency, recovery latency,
   operator burden, compute overhead and token-cost units;
5. identify where Musashi is symmetrically conflicted;
6. define what evidence would falsify the claim that cross-audit helps.

Do not curate favorable incidents. Include your own withdrawn finding 011 and
Musashi's unsupported Arendt designation as candidate negative audit cases.

## 7. Required Deliverables

Write only:

```text
docs/audits/AUDIT_MUSASHI_GOVERNANCE_INNOVATION_RESPONSE_2026_07_31.md
docs/publications/RESEARCH_LINE_PRIOR_ART_DELTA_2026_07_31.csv
docs/publications/CONTINUOUS_RESEARCH_ROADMAP_2026_07_31.md
papers/p1-doin-protocol/FUTURE_WORK.md
papers/p2-data-first-genome/FUTURE_WORK.md
papers/p3-hierarchical-portfolio/FUTURE_WORK.md
papers/p4-execution-parity/FUTURE_WORK.md
papers/p5-audit-recovery/FUTURE_WORK.md
```

You may propose register/recovery updates in your audit report. Do not edit
Musashi's response, historical reports or runtime files.

The continuous roadmap must contain:

- the next ten bounded academic tasks after `AT-ACADEMIC-031`;
- trigger, dependency, maximum source/experiment budget and output for each;
- a non-idle fallback task;
- monthly and quarterly retirement decisions;
- explicit preemption by S0-S1 findings.

## 8. Acceptance Standard

The packet fails if:

- any citation was not opened;
- novelty is asserted from a search snippet;
- future work is generic;
- security wording exceeds runtime enforcement;
- a P6+ line survives because it sounds impressive;
- the roadmap again stops after a finite paper list;
- the auditor assigns itself runtime or submission authority;
- negative cases or conflicts disappear.

At the end report:

1. baseline and dirty-tree provenance;
2. findings ordered by severity;
3. commands/tests and exact results;
4. files written;
5. retained/rejected/narrowed lines;
6. the first next bounded task and its trigger;
7. the permanent fallback task if that trigger is absent.

---

End of task.
