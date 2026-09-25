# Audit: weekly-flat E1-E3 return

Date: 2026-08-29

Audited commits: `gym-fx@aea393b`, `agent-multi@4c4651d0`

Verdict: **E1 accepted; E2 rejected; E3 incomplete because the decisive
split-evidence adversaries are absent.** C5 and deployment remain blocked.

## Accepted

- Terminal transitions now use a process-visible exclusive lock and re-check
  expected state while holding it.
- The claim race uses `Popen` with a barrier rather than sequential runs.
- `prepared` was removed from the declared state machine.
- Final-record symlinks and interrupted empty placeholders are refused.
- Truthy strings no longer substitute directly for an evidence envelope.

## Critical: the payload digest does not bind the interpreted facts

The digest covers `raw_payload`, but authorization consumes separate fields:
`stop_loss_accepted`, `take_profit_accepted`, `positions_total` and
`orders_total`. These values are neither derived from nor checked against the
payload. A valid digest can therefore authenticate one fact while custody acts
on its contradiction.

Reproduced against `gym-fx@aea393b`:

- payload says `sl_accepted=false`, while the separate field says true: claim
  becomes `active`;
- payload says 7 positions and 9 orders, while separate fields say zero:
  custody becomes `completed`.

Re-hashing the payload cannot detect this split because both representations
are independently supplied.

## Critical: evidence chooses its own freshness policy

`max_age_seconds` is part of the evidence object and no policy-side maximum is
checked by `claim()` or `finish()`. Evidence observed one year earlier becomes
fresh by declaring a sufficiently large maximum age. Source is also any
non-empty string rather than an allowlisted, parser-bound authority.

## Test result

Focused custody + session suite reproduced at `59/59`; the contradictory
payload and self-selected freshness cases are absent.

## Disposition

Preserve E1. Replace E2 with parser-derived evidence under a policy-owned
freshness/source contract. E3 must include these cases. No live component is
changed; WP3, WP4, C5 acceptance and long compute remain blocked.

