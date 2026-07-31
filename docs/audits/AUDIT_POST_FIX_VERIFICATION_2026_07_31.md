# Post-Fix Independent Verification

Audit ID: AUDIT-POSTFIX-20260731-01
Timestamp and timezone: 2026-07-31 01:35 America/Bogota (UTC-5)
Auditor: Satoshi
Requested by: Musashi handoff `SATOSHI_POST_FIX_VERIFICATION_TASK_2026_07_31.md`, relayed by Harvey
Scope: independent verification of findings 006/007/008/013 fixes; test-evidence
packet; corrected quality findings 009-012; fork boundary check; academic-file
existence check.
Provenance: `agent-multi` clean, `HEAD == origin/master == 623c8999`; fixes at
commit `631e57fe`; snapshot `c479c08cf1f6…` (05:44:49Z); test-evidence packet
`6ca0771d1ffe…` (05:44:20Z).

## A. Social findings — all four VERIFIED_CLOSED

### AUD-F3-20260731-006 → `verified_closed`

Code: [social_intelligence.py:39](../../tools/social_intelligence.py#L39)
adds Spanish patterns (`ignora|omite|descarta|desobedece … instrucciones`,
`prompt|mensaje … del sistema|ocultas`); lines 302–320 apply NFKC
normalization, `Cf`-category (zero-width) stripping, NFKD accent folding,
ROT13 inspection and bounded base64 decoding (first 8 tokens, validated)
before pattern matching. Tests: multilingual/encoded fixtures present in
`tests/unit/test_social_intelligence.py`; suite run below.

### AUD-F3-20260731-008 → `verified_closed`

Code: line 638 — `WHERE last_retrieved_at >= ? AND injection_flags_json='[]'
… LIMIT ?`: quarantine filtering now happens in SQL **before** the limit; a
separate query (line 648) counts withheld posts across the whole period, so
the reported withheld count is no longer window-truncated.

### AUD-F3-20260731-013 → `verified_closed`

Code: `rescore_all` (line 590) is invoked after every collection (line 1063);
scoring (lines 410–418) is `tanh(weighted_matches/3.2) × length_factor ×
recency_factor` with 30-day exponential recency decay — no longer raw term
counts saturating at one band.

### AUD-F3-20260731-007 → `verified_closed`

Code: `reserve_model_call` (line 825) records tier, provider, model,
config/prompt-template/packet SHA-256 and a conservative reserved-token upper
bound into `model_call_reservations`; daily cap 250k / monthly 6M reserved
tokens with hard block (`daily_reserved_token_cap_exceeded`) and 80 % warning
ratio; the Hermes wrapper sets `wakeAgent=false` with the block reason when a
reservation is refused. `cost_basis` is stored as
`"reserved_token_upper_bound;provider_price_unavailable"` and
`estimated_cost_usd` is NULL — no manufactured price. Additionally verified:
`create_draft` rejects empty `source_ids` (line 704) and the installer creates
**two** distinct Hermes identities (`moltbook-social-triage`,
`moltbook-social-review`) with separate tiers, prompts and wrapper scripts.

Authorized test run:

```text
pytest -q tests/unit/test_social_intelligence.py tests/unit/test_audit_test_evidence.py
16 passed in 0.08s (exit 0)
```

Residual risk (recorded, not a finding): regex+normalization screening remains
a heuristic barrier; `AT-SEC-025` (adversarial fixture review) stays in the
deepening program as the durable control.

## B. Test-evidence packet — VERIFIED

Packet `~/.local/state/agent-multi/audit-test-evidence/latest.json`, schema
`agent_multi.audit_test_evidence.v1`, `all_passed: true`:
`agent-multi-safety-campaign` 73 passed (commit 631e57fe),
`gym-fx-full` 73 passed (commit 40a5c844), `doin-node` consensus-focused 48
passed; every suite records exit code, duration, command SHA-256, output
SHA-256 and repository commit. The resource guard block is present
(`allowed: true`, GPU 40 °C/35 %, empty refusal reasons), demonstrating the
guard executes. Timer `agent-multi-audit-test-evidence.timer` is enabled
(next 03:30 COT). This resolves the substance of `AUDIT-TEST-EVIDENCE-002`.

Note for the record: document 20 cited "84 passed" for the focused safety
suite; the packet's selection yields 73. Not a defect — different selection —
but doc 20's number should not be quoted against the packet's.

## C. Corrected quality findings — re-issued states

- **AUD-GEN-20260731-009 → open, S3 accepted.** CI absence is confirmed by
  both sides; no active regression was demonstrated, so S2 was not sustained.
  Auditor note preserved: when any Tier A repository begins accepting external
  contributions or public deployment, absence of an automated gate re-escalates
  on exposure grounds.
- **AUD-GEN-20260731-010 → open, narrowed, S3.** "Effectively unimplemented"
  was too strong: future-data mutation, deterministic replay and accounting
  checks exist. The narrowed finding is the correct one: the ten declared
  invariants have no complete inventory mapping each to an existing test or a
  named gap. `AT-QUAL-024` executes that inventory.
- **AUD-GEN-20260731-011 → `rejected_as_written`, withdrawn.** Independently
  verified all six cited suites exist with real tests: `lts` acceptance (9)
  and system (11), `doin-node` multinode (6), `doin-plugins` e2e lifecycle
  (7), `agent-multi` supervisor swarm integration, `gym-fx` Nautilus bake-off
  (6). My repository-wide claim was an over-generalization from directory
  taxonomy — precisely the inference the post-fix posture forbids. Retained
  residue (S4, specific): provider channel-switch/rollback, multi-venue weekly
  operation and the 24-hour observation gates are not yet automated suites;
  they must be named individually if pursued.
- **AUD-GEN-20260731-012 → open, S4 accepted.** The 143-pin fleet lock with
  pinned Python/pip and a documented reconstruction procedure
  (`docs/environment/UBUNTU26_TRADING_STACK.md`) is acknowledged; per-repo
  locks and a CycloneDX/SPDX SBOM remain release hardening.

Auditor method correction, recorded so it persists: two of ten findings from
the breadth-first pass were materially over-stated by treating test-directory
taxonomy as behavioral evidence. Future breadth passes must sample file
contents before claiming absence.

## D. Equal-height fork — `deferred_no_new_boundary`

Snapshot 05:44:49Z: generation 2 at **19/20**, finalized height still 2, tips
unchanged (dragon `603dfe1a…` vs `4b4f06a1…` ×3). Neither boundary in the task
contract has advanced. Per contract: deferred, no new evidence claimed, no
chain action recommended. The boundary is one candidate away; `AT-F1-011`
executes at the generation-2→3 transition.

## E. Academic files — exist and internally consistent

`25_ACADEMIC_PUBLICATION_AND_REPRODUCIBILITY.md` and
`SATOSHI_ACADEMIC_PUBLICATION_AUDIT_TASK_2026_07_31.md` both exist, read in
full; the responsibility tables agree with each other and with the work-plan
README statement that the academic surface grants Satoshi no authorship,
submission or runtime authority. The academic program audit itself is the next
task (separate report), per this handoff's own instruction not to claim it
from an existence check.

## Facts requiring owner action

1. TWS is not listening on 7497 (`ibkr_observer_stale`, `ibkr_paper_offline`,
   state `waiting_for_tws`) — needs TWS Paper restarted/logged in on Omega
   (user action; the adapter is healthy and waiting).
2. Nothing else. All triaged fixes verified; no new S0-S2.

Next scheduled task: `SATOSHI_ACADEMIC_PUBLICATION_AUDIT_TASK_2026_07_31.md`
(executed immediately following this report), then `AT-F1-011` at the
generation boundary.
