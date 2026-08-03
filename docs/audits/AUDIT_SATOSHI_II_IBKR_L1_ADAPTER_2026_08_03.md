# Audit: Satoshi II IBKR Paper L1 Adapter Packet

Date: 2026-08-03 America/Bogota
Auditor: General Musashi, temporary independent auditor
Implementer: Lieutenant Satoshi II, prior temporary technical lead session
Scope: `lts@614bb7f`, `lts@f0be969`, `agent-multi@2911a918`
Disposition: **reported_changes_required; IBKR Paper L1 remains disabled**
Broker submissions observed or authorized by this audit: **zero**

## 1. Executive Verdict

The packet delivered two useful components:

1. a deterministic translation from a protected `OrderIntentV2` into the
   documented IBKR parent/take-profit/stop-loss transmission ordering; and
2. a fresh read-only TWS preflight proving the current Paper account and
   `EUR.USD` contract are reachable with zero orders and positions.

It did **not** deliver a write-capable adapter. `submit_bracket()` contains no
`placeOrder` call, does not transmit any member of the bracket, does not read
broker acknowledgement and returns `submitted=true` after incrementing local
counters. The module is not referenced by the L0 service, runner, deployment
unit or any non-test code. The canary runner, persistent effects journal,
broker OLAP, alerts and recovery controller are all absent, as the packet
partly acknowledges.

Therefore the proposed activation phrase is inactive and must not be used.
The shortest path remains correction of this implementation, a fresh
zero-submit connected preflight, independent verification and an explicit
owner decision.

## 2. Evidence Reproduced

### 2.1 Positive evidence

- `31 passed` in `tests/unit/test_ibkr_l1_adapter.py`.
- `334 passed` in the complete LTS suite under Python 3.12.13,
  `ib_async==2.1.0`, SQLAlchemy 2.0.51, Pydantic 2.13.4 and pytest 9.1.1.
- Current authenticated read-only TWS preflight passed on port 7497:
  six contracts available and priced, zero open orders, zero positions,
  zero submissions, `protected_execution_eligible=false`.
- The 20,000-unit proposal is about 2.31% gross notional and 0.004% loss at
  the proposed stop against the latest observed Paper equity. It is small
  under the present 10% gross and 0.5% per-order risk caps. This supports it
  as a bounded ceiling for this account, not the packet's unsupported claim
  that it is the venue's practical minimum.
- `build_bracket()` emits the documented `False, False, True` transmit
  sequence and mandatory SL/TP children.

### 2.2 Independent counterexamples

Run:

```bash
/home/harveybc/anaconda3/envs/trading-stack/bin/python \
  docs/audits/evidence/IBKR_L1_ADAPTER_REPRO_2026_08_03.py
```

At the audited commits it proves, without network access:

```text
altered_cancelled_rejected_bracket_marked_protected = true
broker_object_without_place_order_marked_submitted = true
network_submission_counter_without_broker_call = 1
invalid_profile_accepted = negative values, zero budget, arbitrary venue/host
```

## 3. Findings

### AUD-F2-20260803-063 (S2): submission reports success without broker effects

Observed at `lts/app/ibkr_l1_adapter.py:415-464`:

- the non-dry path checks only that `_ib` is non-null;
- it calls no method on `_ib`;
- it increments `submissions` and `network_submissions`; and
- it returns `submitted=true`.

There is no integration reference to `IbkrPaperL1Sink` outside its module and
unit test. The packet and commit descriptions calling this write-capable are
false at the audited revision.

**Required correction:** use the real IBKR contract and three concrete order
objects; obtain broker-assigned/acknowledged state for every leg; return
success only from direct broker evidence. A fake broker must prove exact call
order and payloads before any connected zero-submit preflight. No canary is
authorized by this correction task.

### AUD-F2-20260803-064 (S2): activation is self-asserted and non-persistent

The phrase is stored in the repository profile. Any local process that can
read it can construct `L1Authorization` with an arbitrary unseen string and a
fresh local timestamp. The only ledger implementation is `FakeLedger` inside
the unit test; no durable, atomic issued/consumed capability exists. The
module's statement that an LLM or chat process cannot construct authorization
is unsupported.

**Required correction:** owner issuance must happen through a separate,
manual, non-chat path and create a short-lived capability bound to profile
hash, Paper account fingerprint, instrument, quantity ceiling, entry budget
and expiry. Store only its digest in the durable LTS ledger and atomically
transition `issued -> consumed` with the first effects-journal record. The
executor cannot mint its own capability. Secrets and raw account identifiers
never enter Git or chat.

### AUD-F2-20260803-065 (S2): rejected or altered protection can pass acknowledgement

`verify_bracket_acknowledgement()` checks presence, side, quantity and parent
link only. It does not require an acceptable status and does not compare
order type, SL/TP price, account, contract/instrument or transmit semantics.
The reproducer changes the TP to a cancelled market order and the SL to a
rejected limit order at wrong prices; `protected` remains true.

The returned string `required_action=cancel_flatten_and_global_hold` also
does not perform any of those effects.

**Required correction:** exact leg identity and protection terms must match
direct broker evidence in an explicit lifecycle state machine. Missing,
rejected, cancelled, inactive, wrong-account, wrong-contract, wrong-price or
ambiguous evidence must execute and persist cancel/flatten/hold, then reconcile
to a terminal state. A string recommendation is not an effect.

### AUD-F2-20260803-066 (S2): the accepted L0 risk and lifecycle path is not connected

The L0 service computes account-relative size and atomically reserves risk,
but it invokes only `sink.serialize()`. The new sink is never instantiated by
that service or a deployed runner. Its profile quantity, stop distance, take
profit distance and spread ceiling are unused. No bridge exists from an
accepted L0 decision/outbox record to an L1 side effect and back into the
same lifecycle ledger.

**Required correction:** add one crash-resumable outbox/effects consumer behind
the accepted L0 contracts and ledger. It must consume the already-sized
intent, reject any profile quantity override, bind current capability/quote,
persist every broker fact and support long -> reconciled flat -> short ->
reconciled flat. Do not create a second risk engine or a competing DTO.

### AUD-F2-20260803-067 (S3): profile validation admits invalid execution state

`L1Profile.load()` rejects live environment, the wrong port and budgets above
two, but accepts an arbitrary venue/host, a zero or negative budget, negative
quantity/distances/spread and malformed fingerprints. Several accepted fields
are not consumed anywhere.

**Required correction:** use a strict schema with exact Paper venue, loopback
host, bounded client ID, one or two entries, positive finite numeric limits,
valid fingerprint algorithm/length and exact asset/instrument binding. Reject
unknown fields. Every retained field must be enforced or removed.

### AUD-F2-20260803-068 (S4): fingerprint evidence mixed two algorithms

The reported account discrepancy is explained, not an account switch:

```text
sha256("c0ff137a3cc1a363")[:16] == "86aa086401855219"
```

The new adapter stores `sha256(account_id)[:16]`. The old lab stores
`sha256("|".join(sorted(account_id_hashes)))[:16]`, which hashes the single
account fingerprint a second time. A fresh read-only preflight reproduced the
old value while the direct-account evidence reproduced the new value.

**Required correction:** label and version both concepts separately, for
example `account_id_sha256_16` and `account_set_sha256_16`; migrate evidence
without exposing the account identifier. The present adapter's direct
connect-time comparison is coherent once the profile uses the direct-account
algorithm.

## 4. Acceptance Gate for the Successor

The next technical lead returns only after all of the following exist:

1. exact broker-call tests through a fake IBKR client, including ordering and
   no false submitted result;
2. durable owner-issued capability and crash-atomic effects journal;
3. exact acknowledgement and executable recovery state machine;
4. integration with the accepted L0 decision/risk/lifecycle ledger;
5. lifecycle/race/restart/manual-order fixtures from the original order;
6. one continuous, disabled-by-default L1 runner with deterministic heartbeat,
   OLAP and alerts;
7. full suites green and a fresh connected **zero-submit** preflight; and
8. an owner activation packet that states derived account-relative risk and
   never calls 20,000 units a broker minimum without broker evidence.

The first broker submission remains a separate owner action after independent
verification. This audit grants no activation authority.

## 5. Session-Loss Observation

The attached predecessor transcript records Fable 5 being automatically
switched after its broad safeguards flagged work near a broker execution and
security boundary. It does not prove wrongdoing. The successor prompt narrows
the mission to lawful Paper-account software engineering, forbids live capital
and unrelated sensitive domains, requires milestone commits, and mandates an
immediate clean handoff if a safeguard warning appears. The successor must not
attempt to bypass, argue with or work around provider safeguards.
