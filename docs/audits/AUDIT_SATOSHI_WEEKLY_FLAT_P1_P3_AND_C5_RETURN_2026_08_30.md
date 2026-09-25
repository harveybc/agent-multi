# Audit: weekly-flat P1-P3 and C5 return

Date: 2026-08-30

Audited commits: `gym-fx@452cbf1`, `gym-fx@c43de82`,
`agent-multi@2ace0614`, `agent-multi@26fd71a5`

Verdict: **P1-P3 accepted. C5 rejected.** The evidence authority now starts
from raw bytes, rejects duplicate keys, seals parser identity and binds the
complete evidence policy across custody. The real environment wiring contains
critical action and reopening defects.

## P1-P3 acceptance

- original JSON bytes are retained and decoded with duplicate/non-finite
  rejection;
- raw and canonical digests have distinct persisted roles;
- parser source/schema/key identity is recomputed and checked against committed
  sealed identities;
- complete evidence-policy digest is persisted at claim and required at finish;
- prior fabrication and concurrent-transition regressions remain present.

This acceptance is code-integrity authority inside the trusted process; it is
not a claim that plain SHA digests authenticate an untrusted external producer.

## C5 critical findings

### C5-1: action domains are mixed and blocked reversals still execute

The pure overlay expects signed target exposure, but `GymFxEnv` passes the
coerced discrete command (`0=hold`, `1=long`, `2=short`, `3=close`) as though it
were target exposure. When a risk increase is detected while a position exists,
the adapter sets the submitted command back to `before`, executing the command
it says it masked.

Reproduced in the real env: short exposure during `WIND_DOWN`, mapped reversal,
`overlay=masked_risk_increase`, yet `session_action_after_overlay=2`.

The recorded `session_raw_model_action` is also the coerced integer, not the
actual continuous model output computed earlier by `step()`.

### C5-2: reopen blackout never exits

The env always calls `session_state` without `ReopenEvidence`. After a known
closure, that correctly fails closed forever. No spread/gap/volatility,
closed-bar or consecutive-stability evidence is materialized. Reproduced: every
remaining post-reopen bar stays `REOPEN_BLACKOUT`.

### C5-3: simulation tests fabricate bars during closure

The fixture creates regular 4-hour bars through the declared closed interval.
Work plan 42 explicitly prohibits synthesized tradable weekend bars. In real
historical data there should be a timestamp jump and no env step/reward inside
closure. Setting reward to zero on a fabricated step can also hide real account
changes rather than proving no economic event occurred.

### C5-4: pending-order and reconciliation authority are incomplete

Whenever a position exists, `_session_exposure_facts` labels every open order as
protective. A simultaneous pending entry is therefore hidden and not cancelled.
The env uses coercive `or 0` fallbacks forbidden at this authority boundary.
Forced flatten checks a reported position but never executes the shared fresh
zero-position/zero-order reconciliation gate.

### C5-5: termination tests can pass vacuously

The exposure-survival test asserts its contract only inside
`if position != 0`; it does not require the fixture to terminate exposed. It
therefore cannot prove that termination preserves exposure before migration.

## Disposition of reported F-A through F-D

- **F-A critical for data validity:** fix OANDA timestamp lookup for index or
  column and regenerate/parity-check all eleven fields before training.
- **F-B high:** repair observation-space construction; no env observation may
  violate its declared space.
- **F-C high, Nautilus-specific:** block every cross-engine economic comparison
  until bar alignment is exact.
- **F-D high, Nautilus-specific:** mark H1/H4 unsupported and exclude Nautilus
  from H1/H4 authority until aggregation is implemented and tested.

Nautilus liquidation parity is not accepted. Core GymFxEnv itself remains
blocked by C5-1 through C5-5.

## Verification

Focused migration + C5 + parity suites: `65/65` with 12 Nautilus warnings.
The reproduced runtime counterexamples pass because they are absent from the
committed acceptance suite.

