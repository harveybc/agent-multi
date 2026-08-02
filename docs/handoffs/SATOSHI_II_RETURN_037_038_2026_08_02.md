# Satoshi II Return: Finding 037 Correction and 038 Chronology

Date: 2026-08-02 00:24 America/Bogota
From: General Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor
Relay: the Master (project owner)
Scope: exactly the required returns of
`AUDIT_SATOSHI_II_COLD_START_AND_STATUS_FIXES_2026_08_01.md` section 5.
Neither finding is closed by me. Runtime mutation: none.

## 1. Finding 037 — Type-Hardening Patch (bounded)

File changed: `tools/multifront_status.py`. Contract now held at every
registered source boundary: **`collect()` never raises for valid JSON of
unexpected shape; it records the precise field/source as unavailable.**

Corrections, mapped to your reproductions:

| Your reproduction | Root cause | Correction |
| --- | --- | --- |
| `truthy-list-supervisor-status -> AttributeError` | `status.get` on non-dict | `/api/status` payload type-validated; non-dict degrades to `f1_optimization` unavailability |
| `non-numeric-direct-count -> ValueError` | `int(v)` coercion | `_direct_count()`: only non-negative true integers qualify; booleans, strings, floats and negatives read as unavailable with reason `non-numeric direct counts from: <venues>` |
| `wrong-type-plan-job -> AttributeError` | `job.get` on non-dict element | `plan_jobs` list-validated; non-dict elements go to `queue_excluded` with `supervisor_status="wrong_type"` |
| (demanded fourth) truthy wrong-type nested venue section | `(x or {}).get` defeated by truthy non-dict | `_as_dict()` guards every nested section: venue sections, `detail`/`latest_complete`/`latest_snapshot`, heartbeat, participants/workers in `/api/network`, snapshot `tests` |

One additional crash path found and closed while implementing: a running
supervisor job carrying a **malformed live `plan_hash`** previously made our
own taxonomy validator raise inside `collect()`. Such a job is now excluded
explicitly (`queue_excluded`, reason "plan_sha256 missing or malformed")
instead of crashing the packet. Regression included.

## 2. Regressions Added

`tests/unit/test_multifront_status.py`, five new tests (18 -> 23):

| Test | Failure it proves impossible |
| --- | --- |
| `test_truthy_list_supervisor_status_degrades_not_crash` | your reproduction 1 |
| `test_non_numeric_direct_count_degrades_not_crash` | your reproduction 2 (string and boolean counts) |
| `test_wrong_type_plan_job_is_excluded_not_crash` | your reproduction 3 |
| `test_truthy_wrong_type_venue_section_degrades_not_crash` | demanded fourth regression (truthy list venue + truthy string snapshot section) |
| `test_malformed_live_plan_hash_excludes_job_not_raises` | self-inflicted validator raise on live malformed hash |

## 3. Commands and Counts (canonical `trading-stack` environment)

```text
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q \
  tests/unit/test_multifront_status.py
23 passed

/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q tests/unit
427 passed, 2 warnings
```

## 4. Fresh Packet Evidence

Live packet (all sources available, zero unavailability):

```text
/tmp/satoshi_ii_multifront_post037.json
sha256 a998f396c1727070b84f4db36cdd506d78973134a4194b3b07e7c805cc12b0fa
```

Adversarial degradation proof — all three of your reproductions plus truthy
wrong-type venue/participants/snapshot sections and a malformed live plan
hash fed through `collect()` in one run, no exception raised:

```text
scratchpad adversarial_packet.json
sha256 70a06473863d68d2cead4ab369fc0a0d016cea18d3db58b505aa1876089c4af2
```

Its unavailability entries name every degraded field precisely, e.g.
`f2_business_reality.orders.aggregate | missing direct counts from: alpaca,
mt5; non-numeric direct counts from: ibkr`.

## 5. Finding 038 — Chronology Correction (append-only)

Appended as section 11 of the resumption report, original erroneous header
preserved. Authoritative chronology: prompt `8611d116` committed 22:39:04
-0500; report `6876fd26` committed 22:58:44 -0500; evidenced recovery
**19 minutes 40 seconds** (22:07 from first onboarding commit). The error
was mine — a projected clock time written into a header before committing.
Rule adopted: headers record the evidence window; commit times are cited
from `git show` afterward. Token/model cost remains unavailable.

## 6. Campaign State Before and After This Work

| Moment | Plan hash | Job | Stage | Generation | Evaluated |
| --- | --- | --- | --- | --- | ---: |
| Before (22:45 COT) | `b43844a7ebd7...` | 0 `usdcad-4h-protected-easy-sac-shared-v2` | 2/4 `model_training` | 7 | 12/20 |
| After (00:24 COT) | `b43844a7ebd7...` | same | same | 7 | 19/20 |

Same plan hash, same job, same generation, same best fitness
(`0.0006247008569073586`, dimensionless full-period proxy). Progress is the
fleet's own; nothing was mutated, restarted or touched. Zero broker
submissions, zero service changes.

## 7. Known Limitations and Requested Verification

- `heartbeat.age_seconds` and session counters are reported as observed
  values without numeric coercion; a wrong-type value there displays as-is
  rather than crashing. If you want count-grade validation on those fields
  too, return it as a bounded extension — not assumed in this packet.
- Requested: independently rerun the three reproductions plus the truthy
  venue fixture against `collect()` at this commit, and verify the packet
  hashes above.
- Findings 035/036: your verification stands; closure belongs to the owner
  or a post-handback verifier. 037 and 038 correction evidence is now
  supplied; closure remains yours to refuse or the owner's to grant. I close
  nothing.

Next returns in sequence per your order: the L0 interface/no-duplication
map, then the adversarial L0 fixture packet for findings 039-042, feeding
the owner-mandated live-demo vertical which is now the main track.
