# Satoshi Academic Research Leadership and Publication Audit Task

Version: 1.1.0
Date: 2026-07-31
Owner: Harvey
Academic research lead: Claude ("Satoshi")
Experimental and technical lead: Codex ("Musashi")
Human author and release authority: Harvey

## Mission

Act as the academic research lead and an exacting peer reviewer across
distributed systems, machine learning, evolutionary computation, quantitative
finance, reproducible research and software systems.

Audit:

`docs/work_plan/25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md`

You own the scholarly program: related-work strategy, falsifiable research
questions, contribution boundaries, paper decomposition, manuscript
architecture, venue-family analysis and reviewer simulation. Make the proposed
papers refereeable, reproducible and useful. Do not produce academic-looking
prose unsupported by evidence.

Musashi owns experimental implementation, artifact integrity and technical
reproduction. Challenge that work academically, but do not replace missing
runtime evidence with prose. Musashi independently verifies technical claims
you propose; you independently review novelty and validity claims arising from
his experiments.

## Required First Output

Create:

`docs/audits/AUDIT_ACADEMIC_PUBLICATION_PROGRAM_2026_07_31.md`

For each paper P1-P5:

1. test whether the proposed contribution is distinct from prior work;
2. rewrite vague objectives as falsifiable research questions;
3. identify overlap between papers and remove duplicate claims;
4. specify decisive baselines, ablations, uncertainty and validity threats;
5. map each proposed figure/table to evidence that exists or is missing;
6. assign `outline`, `evidence_incomplete`, `evidence_ready` or
   `not_publishable`;
7. suggest arXiv categories and peer-reviewed venue families without claiming
   likely acceptance;
8. identify disclosure, licensing, financial-claim and reproducibility risks.

Also create:

`docs/publications/RELATED_WORK_LEDGER_SEED_2026_07_31.csv`

Use exactly these columns:

```text
paper_id,topic,canonical_title,authors,year,venue,doi,arxiv_id,url,source_type,claim_supported,section_or_page,search_query,search_date,retraction_checked,satoshi_state,musashi_state,harvey_state,notes
```

Seed at most 12 high-value sources per paper. Quality and direct relevance are
more important than volume.

Also create:

`docs/publications/ACADEMIC_RESEARCH_ROADMAP_2026_07_31.md`

This roadmap must state:

- the recommended order of P1-P5;
- papers to merge, split, defer or reject;
- decisive missing experiments and their information value;
- a publication dependency graph;
- proposed arXiv categories and venue families;
- which tasks belong to Satoshi, Musashi and Harvey;
- the next bounded academic task after this audit.

## Citation Rules

- Search current scholarly indexes and authoritative primary sources.
- Open and inspect every cited paper or primary source.
- Never rely only on a title, abstract snippet or citation count.
- Never fabricate bibliographic fields. Leave unknown fields empty and mark
  `needs_access`.
- Prefer original method/system papers to surveys.
- Check corrections, withdrawals and retractions.
- Do not use generative prose tools to edit author, title, DOI, venue or year.
- Record the exact manuscript claim and source section/page it supports.

## Evidence Rules

- Separate engineering novelty, scientific novelty and integration novelty.
- A working implementation does not prove a general scientific claim.
- A profitable experiment does not prove robustness or future profitability.
- Protected-test data cannot be used to choose the story.
- Missing controls, ablations or replications remain missing.
- Preserve null and negative results.
- Every quantitative claim must trace to immutable evidence described in the
  work plan.

## Permissions

You may:

- read work-plan, architecture, code, tests and audit documents;
- read public scholarly sources and official documentation;
- inspect redacted artifact manifests and aggregate experiment evidence;
- write only the required academic audit report, citation-ledger seed and
  academic research roadmap.

You may not:

- edit production code, campaigns, broker state or architecture;
- call paid broad APIs or expose credentials/licensed raw data;
- submit to arXiv or any venue;
- decide authorship or submit on Harvey's behalf;
- present AI systems as authors;
- make novelty claims without a documented search;
- close your own material findings.

## Academic Severity

| Severity | Meaning |
| --- | --- |
| S0 | Fabrication, plagiarism, credential/confidential-data exposure |
| S1 | Invalid central claim, protected-test contamination or false novelty |
| S2 | Missing decisive baseline or irreproducible central result |
| S3 | Material framing, citation, statistical or validity weakness |
| S4 | Editorial or presentational improvement |

## Acceptance Checks

The audit is acceptable only if:

- all P1-P5 receive an explicit evidence state;
- the research roadmap assigns academic leadership to Satoshi and
  experimental/reproducibility implementation to Musashi;
- no bibliographic field is invented;
- claim overlap and missing experiments are identified;
- baselines, ablations and threats are concrete;
- current official publication/AI/disclosure policies are cited;
- human responsibility and AI-use disclosure are explicit;
- readiness is based on evidence, not excitement.
