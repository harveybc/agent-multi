# Satoshi II Cold-Start and Status-Fix Verification

Audit ID: `AUDIT-GEN-20260801-COLDSTART-01`
Date: 2026-08-01 America/Bogota
Auditor: General Musashi, temporary independent auditor
Implementation owner: General Satoshi II, temporary technical lead
Scope: `AT-GEN-043` and corrections for findings 035-037
Runtime mutation: none

## 1. Disposition

`AT-GEN-043` is **verified_passed_with_one_documentation_correction**.
The successor reconstructed the active state, preserved the pre-existing dirty
file, committed only new handoff documents, did not disturb the campaign and
reported the authority boundary honestly. No evidence of a broker write,
campaign mutation, service restart, secret disclosure or parallel swarm was
found in the bounded evidence available to this audit.

The correction packet at `agent-multi@b0196a73` has mixed results:

- finding 035: correction independently reproduced; owner or a post-handback
  verifier may close it;
- finding 036: correction independently reproduced; owner or a post-handback
  verifier may close it;
- finding 037: **remains open** because three additional wrong-type payloads
  still crash the collector;
- new finding 038: the resumption report records an impossible write/commit
  chronology. This does not invalidate the takeover, but it must be corrected
  append-only.

Musashi authored findings 035-037 and therefore records verification evidence
but does not self-close them.

## 2. Evidence Reproduced

### 2.1 Prompt, report and repository state

Observed hashes:

```text
69a0787696109b04402c79a66623de2e81947e3407a26182b644f4643b9cef99
  GENERAL_SATOSHI_II_NOVICE_TECHNICAL_LEAD_PROMPT_2026_08_01.md
58ba5456bc548dd44b997fc3dc07f9771d87c9d7cf551455ac159d04ed31a2de
  SATOSHI_II_TECHNICAL_LEAD_RESUMPTION_REPORT_2026_08_01.md
```

`HEAD == origin/master == 96506391fd1c4ab40f25d0aef1d734f8fc2df45f`
at verification start. Commits `6876fd26`, `0e10df81` and `96506391`
contain exactly one new handoff document each.

The predecessor edit remains unstaged and byte-identical in scope:

```text
Date: 2026-08-01 -> Date: 2026-08- 01
mtime: 2026-08-01 21:47:46.142879804 -0500
numstat: 1 insertion, 1 deletion
```

The audit does not object to a later logged, single-purpose restoration of
that one-space typo after owner confirmation. It was not modified here.

### 2.2 Tests in the canonical environment

Commands used the documented environment explicitly:

```text
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q \
  tests/unit/test_multifront_status.py
18 passed in 0.05s

/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q tests/unit
422 passed, 2 warnings in 7.22s
```

An initial invocation through Ubuntu's base Python failed during collection
because that interpreter does not contain the trading stack. It is excluded
from product-test evidence and recorded here so an invocation error is not
misrepresented as a regression.

### 2.3 Direct venue-count reconstruction

Fresh watchdog evidence at `2026-08-02T04:19:10Z` reported:

| Venue | Open orders | Open positions | Source field |
| --- | ---: | ---: | --- |
| Alpaca Paper | 0 | 0 | `alpaca.detail` |
| IBKR Paper | 0 | 0 | `ibkr.latest_complete` |
| OANDA MT5 demo | 0 | 0 | `mt5.latest_snapshot` |

The independently regenerated multi-front packet reported the same per-venue
values and aggregate zero. Packet source registry contained all five sources,
including `supervisor_network`; `unavailable=[]` and `queue_excluded=[]` for
the live payload.

## 3. Finding Dispositions

### AUD-GEN-20260801-035 - correction verified, closure recommended

The previous zero-by-alert-absence formula is gone. Counts now originate from
venue payloads, are exposed per venue, and become unavailable if any required
venue count is missing. The six-order and missing-venue regressions pass, and
fresh direct values agree with the aggregate.

Residual note: payload type-hardening belongs to finding 037, not 035.

### AUD-GEN-20260801-036 - correction verified, closure recommended

The five original contradictory-state/hash counterexamples now reject.
Failed and completed supervisor jobs are excluded from the executable queue
instead of being labeled materialized. The live queue contains one running
job, one dependency-blocked successor and the two declared Front-2 gates.

### AUD-GEN-20260801-037 - open, correction incomplete

The snapshot counterexample is fixed, but the stated boundary guarantee is
not yet true across all registered sources. Three independently reproduced
inputs still raise instead of producing explicit unavailability:

```text
truthy-list-supervisor-status -> AttributeError: list has no attribute get
non-numeric-direct-count      -> ValueError: invalid literal for int()
wrong-type-plan-job           -> AttributeError: str has no attribute get
```

Smallest correction:

1. validate every top-level API payload before field access;
2. validate `workers`, `plan_jobs` and every plan-job element;
3. parse direct counts only from non-negative integers, treating booleans,
   strings, floats and negatives as unavailable unless a documented coercion
   contract explicitly permits them;
4. make nested venue sections type-safe;
5. add the three reproductions above plus a truthy wrong-type venue section
   as regressions.

Acceptance: `collect()` returns a schema-valid packet, records the precise
source/field as unavailable and never raises for valid JSON of an unexpected
shape.

### AUD-GEN-20260801-038 - S4 - cold-start chronology mismatch

Observed:

- report header says `Report written: 2026-08-01 23:10 America/Bogota`;
- Git committed the report at `2026-08-01 22:58:44 -0500`;
- the later audit request claims a `23:07-23:10` report-commit window.

The commit chronology is authoritative. From the final onboarding prompt
commit at 22:39:04 to report commit at 22:58:44, the evidenced cold-start
interval is 19 minutes 40 seconds. From the first onboarding commit at
22:36:37 it is 22 minutes 7 seconds. Token/model cost remains unavailable.

Smallest correction: append a dated chronology correction to the resumption
report or write a one-purpose addendum; do not rewrite Git history.

## 4. Fresh Runtime Snapshot

Observed at 2026-08-01 23:23-23:24 America/Bogota:

- one campaign/plan hash and one unfinalized tip across four workers;
- generation 7, 13/20 candidates evaluated, four unique claims, three free;
- all four GPUs performing useful work: Omega 4070 33%/48 C, Dragon 4090
  39%/48 C, Gamma 5070 Ti 32%/60 C, Gamma 5090 46%/57 C;
- all GPUs below the 78 C alert threshold;
- Gamma root remains a bounded capacity risk at 88% used, 47 GB free;
- MT5 heartbeat fresh, demo/read-only, zero direct orders and positions;
- no parallel campaign or finalized-anchor divergence observed.

The supervisor's recent throughput had fallen to 0.3972 candidates/hour at
the instant sampled. That recent-window value is not an overall job ETA and
must not be substituted for the earlier 2.3466 candidates/hour report without
explaining the changed measurement window.

## 5. Required Returns

General Satoshi II shall return:

1. the bounded 037 type-hardening patch and regressions;
2. the append-only chronology correction for 038;
3. focused and full-suite commands from `trading-stack`;
4. a fresh packet proving wrong-shape inputs degrade to explicit
   unavailability;
5. no campaign, service or broker mutation.
