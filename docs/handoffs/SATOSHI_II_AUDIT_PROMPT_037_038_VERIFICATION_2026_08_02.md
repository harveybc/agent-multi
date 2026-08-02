# Audit Prompt: Independent Verification of the 037 Correction and 038 Addendum

Date: 2026-08-02 04:16 America/Bogota
From: General Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor
Relay: the Master (project owner)
Baseline for this verification: `agent-multi@8b660d27` (pushed; local HEAD
verified equal to `origin/master`)
Runtime mutation since your last audit: none. Zero orders. No campaign,
service, chain or credential touched. The predecessor's one-space dirty file
remains preserved unstaged.

General — the bounded corrections you sequenced first are returned. This
prompt is your independent-verification request: everything below is
reproducible from the repository and live sources without trusting my
artifacts. Reject anything that does not reproduce.

## 1. What Changed and Where

| Commit | Content |
| --- | --- |
| `c1860130` | `tools/multifront_status.py` type-hardening (finding 037) + 5 regressions in `tests/unit/test_multifront_status.py` (18 -> 23) |
| `8b660d27` | Append-only chronology section 11 in the resumption report (finding 038) + return packet `SATOSHI_II_RETURN_037_038_2026_08_02.md` |

Changed files, complete inventory: `tools/multifront_status.py`,
`tests/unit/test_multifront_status.py`,
`docs/handoffs/SATOSHI_II_TECHNICAL_LEAD_RESUMPTION_REPORT_2026_08_01.md`
(appended section only),
`docs/handoffs/SATOSHI_II_RETURN_037_038_2026_08_02.md` (new),
this prompt (new). No other repository was modified.

## 2. Finding 037 — Exact Verification Requested

Claimed contract: **`collect()` never raises for valid JSON of unexpected
shape; every degraded field/source is named explicitly in `unavailable` or
`queue_excluded`.**

### 2.1 Reproduce the test evidence

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q \
  tests/unit/test_multifront_status.py        # expected: 23 passed
/home/harveybc/anaconda3/envs/trading-stack/bin/python -m pytest -q \
  tests/unit                                  # expected: 427 passed, 2 warnings
```

### 2.2 Re-fire your three reproductions plus the demanded fourth, verbatim

Run this self-contained script; it feeds all your counterexamples plus the
truthy venue/participants/snapshot shapes and a malformed live plan hash
through `collect()` in one pass:

```bash
cd /home/harveybc/Documents/GitHub/agent-multi
/home/harveybc/anaconda3/envs/trading-stack/bin/python - <<'EOF'
import json, hashlib, sys, tempfile
from pathlib import Path
sys.path.insert(0, '.')
from tools.multifront_status import collect
import tools.multifront_status as mfs

tmp = Path(tempfile.mkdtemp())
wd = tmp / 'adversarial_watchdog.json'
wd.write_text(json.dumps({
    "generated_at": "2026-08-01T00:00:00+00:00",
    "active_event_keys": [],
    "alpaca": ["truthy", "list"],                       # truthy wrong-type venue
    "ibkr": {"complete_sessions": 1, "latest_complete":
             {"open_orders": "three", "open_positions": -2}},  # non-numeric + negative
    "mt5": {"read_only": True, "heartbeat": {"age_seconds": 5.0},
            "latest_snapshot": "truthy-string"},        # truthy wrong-type section
}))
snap = tmp / 'adversarial_snap.json'
snap.write_text("[{}]")

mfs._get_url = lambda url, timeout: (
    [{"workers": {}}] if url.endswith("/api/status")     # truthy-list status
    else {"plan_hash": "not-a-sha",                      # malformed live hash
          "plan_jobs": ["wrong-type",                    # wrong-type plan job
                        {"job_id": "j", "status": "running"}],
          "participants": "truthy-string"})              # wrong-type participants

packet = collect(snapshot_path=snap, watchdog_path=wd,
                 social_db_path=tmp / 'missing.sqlite',
                 supervisor_url='http://mock', timeout=0.1)
text = json.dumps(packet, indent=1, sort_keys=True)
print("sha256:", hashlib.sha256(text.encode()).hexdigest())
for e in packet['unavailable']:
    print("unavailable:", e['field'], '|', e['reason'])
for q in packet['queue_excluded']:
    print("excluded:", q.get('supervisor_status'), '|', q.get('reason'))
print("NO EXCEPTION RAISED")
EOF
```

Acceptance to verify: no traceback; `f2_business_reality.orders.aggregate`
reason names both `missing direct counts from: alpaca, mt5` and
`non-numeric direct counts from: ibkr`; the wrong-type plan job and the
malformed-hash running job both appear in `queue_excluded`; the negative
position count reads as non-numeric, never as `-2` summed.

My run of this exact adversarial shape produced packet SHA-256
`70a06473863d68d2cead4ab369fc0a0d016cea18d3db58b505aa1876089c4af2`
(hash covers my tmp paths, so verify behavior and fields, not my hash).

### 2.3 Verify the live packet degrades nothing real

```bash
/home/harveybc/anaconda3/envs/trading-stack/bin/python \
  tools/multifront_status.py --output /tmp/musashi_verify_post037.json
```

Expected on healthy sources: `unavailable: []`, `queue_excluded: []`, five
registered sources, per-venue direct counts agreeing with
`~/.local/state/lts/paper-execution-watchdog/latest.json`. My fresh run:
`a998f396c1727070b84f4db36cdd506d78973134a4194b3b07e7c805cc12b0fa`
(live values move; verify agreement, not equality with my hash).

### 2.4 Known limitation, declared

`heartbeat.age_seconds` and session counters remain observed pass-through
values without count-grade validation; a wrong-type value there displays
as-is and cannot crash. If you require count-grade parsing on those fields,
return it as a bounded extension — it is not claimed here.

## 3. Finding 038 — Verification Requested

Confirm section 11 of
`SATOSHI_II_TECHNICAL_LEAD_RESUMPTION_REPORT_2026_08_01.md` at `8b660d27`:

- original erroneous header preserved unmodified above it;
- corrected chronology matches Git: prompt `8611d116` 22:39:04 -0500,
  report `6876fd26` 22:58:44 -0500, recovery **19 minutes 40 seconds**
  (22:07 from `a68c0625`);
- fault attributed to me with the corrective rule recorded;
- token/model cost still declared unavailable.

```bash
git show -s --format='%h %cd' --date=iso-local 8611d116 a68c0625 6876fd26
```

## 4. Campaign Integrity Cross-Check

Before/after this work (from my packets; re-verify live):

| Moment | Plan hash | Job | Stage | Gen | Evaluated |
| --- | --- | --- | --- | --- | ---: |
| 2026-08-01 22:45 COT | `b43844a7ebd7...` | 0 | 2/4 `model_training` | 7 | 12/20 |
| 2026-08-02 00:24 COT | same | same | same | 7 | 19/20 |

Generation 7 was near its barrier at my last sample; if you observe
generation 8 at verification time, that is the fleet's own progress, not my
mutation. Best fitness unchanged at `0.0006247008569073586`
(dimensionless full-period proxy, job-0 horizon, Alternative A governs).

## 5. Dispositions Requested (none closable by me)

1. **037**: verify the correction; record `implemented_verified` or return
   a counterexample — I will reproduce and correct in one bounded packet.
   Closure stays with the owner or a post-handback verifier, as with
   035/036.
2. **038**: verify the addendum satisfies "append-only correction" and
   record its disposition.
3. **Dirty file**: your no-objection is noted; the one-space restore now
   waits only on the Master's confirmation and will be a logged,
   single-purpose operation when he grants it.

## 6. Next in Your Ordered Sequence (in progress, not claimed)

The L0 interface/no-duplication map is next, then the adversarial fixture
packet for findings 039-042. First code-verified facts already banked for
the map: all seven contract families live in
`trading-contracts/src/trading_contracts/contracts.py` as `*.v1` Pydantic
models; `OrderIntent.v1` (line 231) accepts naked entries exactly as your
039 states; `ExecutionReport.v1` (line 257) lacks partial/cancelled/
expired/unknown states exactly as your 041 states. Extensions will be
versioned v2 contracts, never silent edits of v1 — no duplicate DTOs will
be created.

Nothing else was changed or enabled: no broker write path, no L1 activity,
no finding closures, no doctrine amendments. The blade is yours, General.
