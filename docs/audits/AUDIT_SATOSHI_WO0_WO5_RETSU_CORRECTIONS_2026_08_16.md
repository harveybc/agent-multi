# Audit of Satoshi WO0-WO5 Corrections After the Retsu Review

Date: 2026-08-15/16 America/Bogota-UTC boundary  
Auditor: General Musashi  
Implementer: General Satoshi III  
Independent follow-up verifier: Retsu  
Runtime mutation by this audit: none

## 1. Verdict

**PROCEED WITH CORRECTIONS. Do not merge, deploy or push the delivered
branches yet.**

Satoshi delivered substantial working code. The direct-seat collector, the
same-window comparison machinery and the succession primitives all pass their
focused suites. A disposable integration of WO1+WO2+WO3 on current `lts/main`
also passes all 782 tests. The active corrected P1LR decision remains coherent:
four fresh workers, one identity `cdf30aebf585385b`, all in training, with the
current processes untouched.

The package is not accepted as complete because it has eight reproducible
defects. The highest-priority one is privacy: WO1's locally committed evidence
contains Paper/Demo balances, equity, margin, exact prices, order IDs and MT5
tickets. `harveybc/lts` is public. The branches have not been pushed, so no
public disclosure occurred; they must be sanitized before any push.

## 2. Findings

### AUD-SEC-20260816-255 (S2): public-repository evidence contains private account facts

WO1 calls its evidence redacted, but the committed files include account
balances/equity/margin, exact positions and prices, broker order/ticket IDs and
stable account/server fingerprints. Examples:

- `docs/evidence/seat_truth/mt5_direct_evidence_20260816T014844Z.json:7-13`
- `docs/evidence/seat_truth/seat_truth_inventory_20260816T014844Z.json:200`
- `docs/evidence/seat_truth/seat_truth_inventory_20260816T014844Z.json:1006-1010`
- `docs/evidence/seat_truth/seat_truth_table_20260816T014844Z.txt:21-31`

Direct private evidence belongs under the local 0600 evidence store. Git may
carry a sanitized summary, counts, typed states and a digest of the private
packet, never the packet itself. This finding blocks pushing WO1.

### AUD-GEN-20260816-256 (S2): WO4's claimed deployable identity files are absent

The branch says WO4 is delivered with shipped, byte-pinned seed environments.
All four `seed*.env` files are absent because the repository's `*.env` ignore
rule catches them. The installer nevertheless requires that glob at
`examples/systemd/install_p1lr_v2_identity_supervision.sh:38-40`.

Independent full suite:

```text
1638 passed, 2 failed
test_req1_shipped_env_files_are_byte_pinned_to_the_generator
test_req1_env_files_carry_the_full_v2_identity
```

The failing reads are at `tests/test_wo4_identity_supervision.py:689-711`.
WO4 is not deployable and finding 250 remains open.

### AUD-F2-20260816-257 (S2): succession has no production entry point

`promote_paper_champion()` is only called by the unit-test helper
`tests/unit/test_champion_succession.py:758-774`. The shipped CLI
`tools/succession_preflight.py:1-18` explicitly performs preflight only and
promotes nothing. No service or runner obtains direct venue facts, invokes the
real venue drain executor and calls the promotion function. The primitives are
real; the bridge to a seat is not yet executable.

### AUD-F2-20260816-258 (S2): a crash can split ledger authority from the manifest permanently

`promote_paper_champion()` commits the capability burn and successor session at
`app/champion_succession.py:1485-1519`, then switches the manifest at
`app/champion_succession.py:1521-1522`. Injecting a crash at the switch produced:

```json
{
  "active_sessions": [["challenger-linear-v2", "active"]],
  "capabilities_consumed": 1,
  "manifest_model": "incumbent-linear-v1",
  "rerun_error": "no valid signed promotion capability ... different incumbent model"
}
```

The docstring says the operation can be rerun, but the burned capability and
changed active session make that impossible. The exact target manifest bytes
are not durably recoverable from the promotion row. A resumable promotion saga
is required before any seat promotion.

### AUD-F2-20260816-259 (S2): one due decision can retain contradictory as-of inputs

`as_of_input_bars` uses `input_sha256` inside its uniqueness key
(`app/ibkr_l1_journal.py:347-355`). Therefore changing the input hash changes
the purported decision identity. The independent counterexample inserted two
different bar sets for the same venue/model/timeframe/bar close:

```json
{"first": true, "divergent_same_due": true, "rows": 2}
```

The as-of packet must bind to the already normalized due-decision identity,
including route/account/instrument and decision ID. A changed input hash for
that identity must be an incident, not another valid row.

### AUD-F2-20260816-260 (S3): loss of as-of evidence is not durable or visible in health

Both runners catch every as-of persistence exception and print one transient
JSON line:

- `app/alpaca_model_runner.py:382-400`
- `app/ibkr_model_runner.py:353-371`

The tick continues, but no durable incident or heartbeat field records that the
comparison product is losing its input lineage. This can leave the timer
permanently `NOT_SUBTRACTABLE` while venue health appears normal.

### AUD-GEN-20260816-261 (S3): the v2 lease gate executes mutable checkout code

The worker source is pinned to the immutable runtime worktree, but the second
`ExecStartPre` executes `tools/p1lr_identity_supervision.py` from the mutable
canonical checkout (`20-v2-identity.conf:23-29`). A checkout update can thus
change restart admission without changing the unit's declared experiment
identity. Pin the supervision implementation or its digest as a separately
versioned control artifact.

### AUD-GEN-20260816-262 (S3): the return package has no single pushed integration lineage

The three LTS branches are local-only siblings. They combine cleanly and pass
782 tests, but no pushed integration branch exists. WO4 is a normal descendant
of the accepted agent-multi base, while the WO5 authoring branch
`satoshi/post-outage-209-223@9e4ebc3f` has no common ancestor with the current
Musashi audit lineage and must not be merged wholesale. Transfer its return
document onto a current integration branch and preserve provenance.

## 3. Independent Evidence

Machine-readable summary:
`docs/audits/evidence/repro_runs/MUSASHI_WO0_WO5_ACCEPTANCE_REPRO_2026_08_16.json`.

```text
WO1 full LTS suite:                 729 passed
WO2 full LTS suite:                 720 passed
WO3 full LTS suite:                 735 passed
WO1+WO2+WO3 integration:            782 passed
WO4 focused correction set:         123 passed, 2 failed
WO4 complete agent-multi suite:     1638 passed, 2 failed
P1LR corrected decision runtime:    4/4 fresh workers, identity cdf30aebf585385b
```

The LTS integration cherry-pick order was:

```text
5767aa2 c5f2412 13b2d99 2f3c6b8 e0762af
```

It had zero merge conflicts. This is a verified non-finding: the three code
packages are structurally integrable once the findings above are corrected.

## 4. Existing-Finding Disposition

- **249:** partially corrected. The product and timer now exist; deployment,
  IBKR/Alpaca comparability and findings 259-260 remain.
- **250:** not corrected. WO4's own full suite fails and its install payload is
  incomplete.
- **251:** partially corrected. Compatibility, shadow and promotion primitives
  exist, but no production caller exists and crash recovery is incomplete.
- **252:** corrected pending Retsu verification. A fleet-readable MT5 packet
  exists and carries direct bridge/model/position facts.
- **253:** partially corrected. Direct venue facts clarify the stale IBKR order,
  but current consolidated status still cannot prove a valid artifact SHA-256.
- **254:** WO6 remains a sidecar; this audit did not accept a deployed consensus
  mutation.

No finding is closed by this report.
