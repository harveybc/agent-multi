# Technical-Lead Triage of the 2026-07-31 Audit

Date: 2026-07-31
Reviewer: Musashi
Reporter: Satoshi
Independent corroborator: none. The prior unregistered `Arendt` designation
was removed on 2026-07-31 and carried no evidentiary or closure weight.

## 1. Decision Summary

Satoshi produced four files, not two: two reports plus changes to the audit
backlog and finding register. All four were reviewed. The reports are retained
as immutable audit history; this triage records corrections where the
breadth-first audit inferred too much from directory names or omitted existing
environment evidence.

Accepted findings:

- `AUD-F3-20260731-006`: multilingual/evasive injection coverage was too
  narrow;
- `AUD-F3-20260731-007`: Hermes model-call budgets and facts were not
  materialized;
- `AUD-F3-20260731-008`: flagged posts could consume the digest SQL limit;
- `AUD-F3-20260731-013`: relevance scores saturated and ranked poorly.

All four have an implementation and focused tests in the current change set.
They remain `implemented_pending_independent_verification` until Satoshi runs
the handoff in
`../handoffs/SATOSHI_POST_FIX_VERIFICATION_TASK_2026_07_31.md`.

Corrected findings:

- `AUD-GEN-20260731-009` is confirmed but re-triaged from S2 to S3. There is
  no automated merge gate, but no active regression or unsafe deployment was
  found.
- `AUD-GEN-20260731-010` is partially confirmed. The declared invariant set is
  incomplete, but it is not unimplemented. Future-data mutation,
  deterministic replay and independent accounting checks already exist.
- `AUD-GEN-20260731-011` is rejected as written. LTS has runnable acceptance
  and system suites; DOIN has multi-node and full-lifecycle suites; agent-multi
  has a coordinated multi-supervisor integration suite. Specific operational
  scenarios remain candidates for automation and must be named individually.
- `AUD-GEN-20260731-012` is re-triaged to S4. Reproducibility does not rest on
  one hash: the fleet environment has 143 exact package pins, pinned
  Python/pip and a documented reconstruction procedure. Per-repository locks
  and a CycloneDX/SPDX SBOM remain useful release-hardening work.

## 2. Implemented Corrections

The social pipeline now:

- screens English and Spanish injection patterns after Unicode normalization,
  zero-width removal, accent folding, ROT13 inspection and bounded base64
  decoding;
- filters quarantined posts in SQL before applying the digest limit and
  reports the total withheld count for the period;
- re-scores the complete corpus with distinctive-term weighting, title and
  phrase bonuses, length normalization and recency decay;
- requires at least one existing OLAP source for every draft;
- reserves a conservative input/output token upper bound before Hermes runs;
- records tier, provider, model, config hash, prompt hash, packet hash and
  budget state in SQLite;
- blocks model wake-up at the per-tier, daily or monthly token cap and warns at
  80 percent;
- labels provider cost as unavailable rather than manufacturing a price.

The deterministic audit infrastructure now runs three bounded suites when
Omega is not evaluating a DOIN candidate and has GPU, RAM and CPU headroom.
The first packet recorded:

- agent-multi safety/campaign: 73 passed;
- gym-fx full: 73 passed;
- doin-node consensus-focused: 48 passed;
- all suites exit 0, with duration, command/output hashes and repository
  revisions recorded.

## 3. Runtime Triage

The four DOIN workers remain on one plan, job, domain, seed, generation and
population. One candidate is owned by Gamma's RTX 5070 Ti; the other workers
are correctly waiting at the generation barrier. The equal-height tip split
is still unfinalized and has not produced divergent finalized anchors,
populations or fitness. No chain mutation is authorized. Re-sample when the
generation advances.

The IBKR observer is healthy but TWS is not listening on
`127.0.0.1:7497`. Its state is `waiting_for_tws`, not an adapter crash. The
last reconciled shadow state remains usable only until its configured stale
boundary.

## 4. Evidence

- `tests/unit/test_social_intelligence.py`
- `tests/unit/test_audit_test_evidence.py`
- `~/.local/state/agent-multi/audit-test-evidence/latest.json`
- `~/.local/state/agent-multi/audit-snapshots/latest.json`
- `docs/environment/UBUNTU26_TRADING_STACK.md`
- `gym-fx/tests/test_nautilus_bakeoff.py`
- `lts/tests/acceptance/test_acceptance.py`
- `lts/tests/system/test_system.py`
- `doin-node/tests/test_multinode.py`
- `doin-plugins/tests/test_e2e_lifecycle.py`
- `tests/integration/test_campaign_supervisor_swarm.py`

## 5. Omitted Academic Dimension

The audit did not cover academic preservation, publication-quality
reproducibility, novelty search or claim-to-evidence traceability. That is a
valid scope gap: a useful system is more likely to persist when its distinct
contributions can be independently examined and reproduced.

The corrective contract is now:

- `../work_plan/25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md`
- `../handoffs/SATOSHI_ACADEMIC_PUBLICATION_AUDIT_TASK_2026_07_31.md`

The contract defines five short papers rather than one monolithic manuscript,
an IEEE-compatible structure, a verified citation ledger, immutable
claim/artifact mappings, protected-test and negative-result rules,
reproducibility packages, AI-use disclosure and human submission authority.
The existence of that contract does not make any paper evidence-ready.
Satoshi, as academic research lead, must classify each one and define the
scholarly roadmap; Musashi independently verifies its executable evidence.
