# Social-Trading Business-Reality Loop Audit (Invocation 06)

Audit ID: AUDIT-F4-SOCIAL-20260801-01
evidence_observed_at: 2026-08-01 14:30–15:20 America/Bogota
report_written_at: 2026-08-01 15:25 America/Bogota
Auditor: General Satoshi
Frozen implementation: `lts@db80d97` (HEAD == db80d97 verified, clean)
Registry sha256 `4d2c82f70fe2…`; scenario sha256 `9e92fea6a3af…` (both
reproduced identically across two runs)
Method: adversarial in-process attacks on the ledger (my sequences, not the
shipped tests); five official sources opened directly; all reproductions
executed; labels `observed`/`reproduced`/`inferred`/`proposed` throughout.
Scope note: doc 27 was read structurally (planes contract), not line-by-line;
its deep parity audit remains a separate task.

## 1. Findings (severity order)

### AUD-F4-20260801-030 — Document 28's Priority-1 platform plan is impossible per its own registry evidence

- Severity: **S3** | Confidence: high | Status: open
- `Observed`: doc 28 §4 ranks **MQL5 Signals on the OANDA demo as Priority 1**
  with activation "configure free/private experiment after 24-hour MT5
  window." The registry's own entry rules this out:
  `disposition: ineligible_demo_account_live_only_future`, limitation
  "MetaTrader 5 build 4150 removed Signals support from all demo accounts"
  (our terminal: build 6075), `requires_live_capital: true`.
- `Reproduced` from official sources (opened directly): MQL5 rules —
  *"Signals based on demo, contest and cent accounts are not allowed. Such
  signals are deleted automatically"*; providers must be real accounts with
  seller identity verification for paid signals.
- Impact: the top of the platform order directs owner effort at a dead end;
  the provider-side experiment cannot run on ANY demo (cTrader providers must
  also be live — verified) except Darwinex Zero's paid virtual track.
- Smallest correction: fix doc 28 §4 to match the registry (the registry is
  right; the prose table is stale) and adopt the reordering in §6 below.
  Owner ratifies the new order.
- Owner: Musashi (doc); Harvey (order). I do not close it.

### AUD-F4-20260801-031 — In-profit investors can exit fee-free: withdrawal does not crystallize accrued performance fees

- Severity: **S3** (manager-side accounting correctness) | Confidence: high |
  Status: open
- `Reproduced` (my adversarial sequence, not a shipped test): deposit 100 →
  +50 % return → equity 150, HWM 100, eligible profit 50 → `withdraw(150)`
  succeeds → investor exits with the full 150; `manager_fee_balance = 0`
  (expected 10 at 20 %). The ledger permits complete fee avoidance whenever
  the scenario driver does not call `crystallize_performance_fee` first;
  nothing enforces the ordering.
- Real-platform contrast: PAMM-class platforms crystallize on withdrawal or
  rollover precisely to prevent this.
- Smallest correction: inside `withdraw()`, crystallize the accrued
  performance fee on the withdrawn fraction before releasing funds (or a
  constructor flag `crystallize_on_withdrawal=True` as the default), plus a
  regression test that reproduces my sequence and asserts the fee.
- Owner: Musashi. I do not close it.

### AUD-F4-20260801-032 — `round_up_minimum` has no overshoot bound

- Severity: S4 | Confidence: high | Status: open
- `Reproduced`: provider 1.0 lot at 100k equity, investor 100 equity → raw
  0.001, allocated 0.01 = **900 % overshoot**, honestly reported in
  `volume_tracking_error` but uncapped and unguarded by margin.
- Correction: `max_overshoot_ratio` rejection threshold (or explicit margin
  check) + test; document that round-up violates proportionality by up to
  `minimum/raw`.

### AUD-F4-20260801-033 — `minimum_volume` is not validated as step-aligned

- Severity: S4 | Confidence: high | Status: open
- `Reproduced`: minimum 0.015 with step 0.01 emits allocation 0.015 — a
  volume no step-quantized platform would accept. Correction: validate
  `minimum_volume % volume_step == 0` in `from_dict` + test.

### Proposals (invocation A.4/A.7, `proposed`, S4-class)

1. Before any external reconciliation: append-only event hash chain,
   balances-before **and** after per event, and idempotency keys. (The
   invocation hypothesized this; the audit confirms current events carry
   after-state only and no chain.)
2. Money quantization policy at event boundaries: attack 2 left equity
   `139.9999999999999979` vs the exact 140 — harmless at 28-digit context,
   but external platforms reconcile at 2–8 dp; fix the exponent per currency.
3. Copy contract lacks instrument dimension: `contract_size`,
   quote-currency→account-currency conversion, and leverage differential are
   absent; equity-to-equity ratio alone cannot price a copy on cross-currency
   CFDs. Add fields now, before an adapter hardens the gap.

## 2. Verified Non-Findings (all `reproduced`)

1. **No double charge below HWM**: fee charged once (10.00) across
   drop-and-recover; net-HWM update (`max(HWM, net_equity)`) is correct.
2. **Manager protections hold**: manager capital cannot be charged either fee
   (role guards raise); manager role immutable after creation.
3. **Deposit additive / withdrawal proportional HWM** are correct generic
   defaults (verified against gain-then-deposit and loss-then-withdraw
   sequences; aggregate equals per-tranche equalization in both directions).
4. **Registry honesty is exemplary**: all five checked platform claims match
   the official sources opened directly — MQL5 demo-provider ban, cTrader
   SL/TP non-replication (close signal only), cTrader equity-to-equity
   formula (verbatim the lab's), demo-investor-free-strategies-only, provider
   fee caps (40 % performance / 10 % management / $10 per million / 30 %
   cTrader commission from 2026-07-04).
5. **No broker/secret/network capability** in the lab: imports are stdlib
   only (`sqlite3`, `hashlib`, `json`, `uuid`, `decimal`); the single
   network-looking line is an `https://` prefix *validator* for evidence
   URLs. No credential reader, no order path (D.1 proven).
6. **Determinism and safety**: scenario runs twice with identical
   `orders_submitted: 0`, identical registry/scenario hashes, final
   `unit_nav 1.0476`, sequence 9. `lts tests/unit`: **111 passed** at
   `db80d97`; the 9 social-lab tests pass.
7. **Plane separation (C)**: the lab imports nothing from agent-multi's
   fitness path and exposes nothing the optimizer reads; platform
   availability cannot reach alpha ranking through any code path found;
   protected-entry gate defaults to fail-closed
   (`require_protected_entry=True` rejects when neither native replication
   nor local overlay exists — exactly the cTrader observational case).
8. **No DOIN-mutation path (C.5)**: no network surface, no chain imports; the
   feedback contract routes changes through new config/dataset/semantic
   hashes at campaign boundaries (doc 28 §6) — consistent with the
   owner-ratified A-decision discipline.
9. **Hermes boundary (D.6)**: no subscription/publication/allocation/fee/
   order capability exists in any Hermes-reachable surface; the social lab is
   CLI-only, local-DB-only.

## 3. Runtime Sanity (E)

- Campaign (earlier this session, `reproduced`): generation 5 of stage 1,
  new champion `0.0006247`, patience 2/4, tips fully converged at height 11
  (one tip, one finalized anchor, zero alerts). Finding 020 stays S4;
  severity proportionate, untouched by the social work.
- Venues: MT5 heartbeat fresh (≤14 s) with 2,728+ heartbeats; Alpaca 727+
  sessions; IBKR recovering after TWS restart (stale flag clearing).
  Social-intelligence collector timer active.
- Gamma: 50.5 GB disk free (stable), swap free 1.71 GB (mild decline),
  `sock_throttled` 6,688 (+460/day, slow growth) — **trend watch unchanged,
  no escalation** (E.4 answered with proportionate severity).

## 4. Legal Flags (D.4 — explicitly NOT answered technically)

Provider/PAMM activity, cross-border solicitation, investor fee taxation,
and Colombian (SFC) treatment of copy/managed accounts require a qualified
professional before S5/S6. The audit records the questions; it invents no
answers.

## 5. Required Closing (invocation format)

**Three strongest reasons to KEEP the current order:** (1) MQL5 rides the
already-commissioned OANDA terminal — zero new accounts; (2) MQL5 is the only
listed platform whose native copy CAN replicate SL/TP when enabled — uniquely
satisfying the protection rule; (3) zero recurring cost.

**Three strongest reasons to CHANGE it:** (1) finding 030 — demo Signals are
dead ≥ build 4150 and providers must be live: Priority 1 cannot be executed
at all without a live-capital decision nobody has approved; (2) the
provider-track experiment is impossible on every demo platform — only
Darwinex Zero offers virtual provider mechanics; (3) cTrader demo-investor +
Open API is executable *today* (free strategy copy) and its custom-copier
path is the only protection-compliant automated copy route.

**Recommended order (one, explicit):**

| # | Platform | Info value | Owner effort | Recurring cost | Risk |
| - | --- | --- | --- | --- | --- |
| 1 | cTrader Copy demo investor | copy mechanics, sizing, latency (observational) | create cTrader ID (minutes) | 0 | none — no SL/TP copy → measurement only |
| 2 | cTrader Open API demo preflight | the protection-compliant custom-copy path; API reality | same ID; review adapter | 0 | code-review gated |
| 3 | eToro Virtual | investor UX control | free signup | 0 | manual only |
| 4 | Darwinex Zero | the ONLY virtual provider/allocation track | terms review + approval | subscription fee | owner-approved spend |
| 5 | HFM PAMM | real pooled mechanics | legal review first | capital | deferred (S6-class) |
| — | MQL5 Signals | re-enters only with an explicit live-account decision | — | — | live capital |

**Five highest-value corrections before ANY external account connects:**
(1) withdrawal-crystallization guard + regression (031); (2) overshoot cap
and step-alignment validation + tests (032/033); (3) append-only event hash
chain, before/after balances, idempotency keys; (4) instrument dimension in
the copy contract (contract size, FX conversion, leverage); (5) per-currency
money quantization at event boundaries.

**What would falsify each recommendation:** order — an official MQL5 source
restoring demo Signals, or an owner live-capital approval (MQL5 jumps back to
1 for protection reasons); 031 — official evidence that every target platform
crystallizes only on schedule, never at withdrawal (then a scenario-order rule
suffices); cTrader-first — Open API documentation showing demo accounts
cannot read deals/positions; Darwinex slot — terms showing Colombia
ineligibility or a cost the owner rejects; ledger corrections — a
demonstration that current SQLite facts already replay to bit-exact balances
under fault injection (partially plausible; the hash chain is still cheaper
than the argument).

## 6. Change Confirmation

No account was opened, funded or touched; no DOIN, broker, MT5, service or
social-platform state changed; scenario runs wrote only to the session
scratchpad; all repository access read-only; register and recovery updated
per the output contract.
