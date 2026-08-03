# L0 Acceptance and MT5 AT-F2-006 Security Review

Date: 2026-08-02 America/Bogota
Author: General Musashi, temporary independent auditor
Priority: Front 2 active demo trading

## 1. Decision

`AT-F2-040` is `verified_passed`. The corrected L0 vertical is accepted as
the zero-network mechanics gate. It is not the destination and must not delay
the L1 broker canary.

`AT-F2-006` is `reported_changes_required` for MT5 order activation. The
deployed MT5 vertical remains accepted for read-only business observation,
but it is not an execution adapter and cannot submit an order.

## 2. Independently Reproduced L0 Evidence

Reviewed heads:

- `lts@77bf02e` (includes `f2252b6` corrections 053-058)
- `trading-contracts@cd05083`
- `prediction_provider@3a6c234`
- `agent-multi@730a2b44`

Independent suites:

```text
lts focused L0:                68 passed
lts complete:                 303 passed
trading-contracts complete:    95 passed
prediction_provider mechanics: 16 passed
agent-multi unit:             431 passed
```

Direct read-only reconstruction of
`~/.local/state/lts/demo-execution-l0.sqlite` found:

- 3 protected would-be orders and 5 explicit rejections;
- 3 exact `requested -> accepted -> filled` lifecycle sequences;
- 9 lifecycle events with a valid SHA-256 chain from `genesis`;
- signed exposures `+5.319`, `-5.362`, and `-5.387` ETH units;
- the first two exposures closed and the third open;
- current open risk `0.0009999261`, gross `0.0999926051`, margin
  `0.0299977815`;
- no command rows and `network_submissions_session=0`.

A controlled `systemctl --user restart lts-demo-execution-l0.service`
preserved the ledger byte size and modification time, replayed the active bar,
emitted no duplicate lifecycle event, and resumed with zero alerts and zero
network submissions.

Therefore corrections 053-058 reproduce. Connected findings 041, 042, 047
and 048 also satisfy their L0 acceptance conditions. Satoshi's later fix
`lts@77bf02e` correctly stops the synthetic lifecycle driver from retrying a
settled exposure; it is classified S4 and verified closed in the same pass.

## 3. AT-F2-006 Positive Evidence

The current MT5 observation bridge correctly provides:

- HMAC-SHA256 over method, path, timestamp, nonce and body hash;
- constant-time signature comparison;
- persistent SQLite nonce replay defense;
- a 90-second clock-skew window;
- demo-account and read-only startup refusal in the EA;
- demo/read-only validation in the host service;
- a 32-character minimum shared secret stored in a mode-0600 environment
  file;
- systemd hardening and a UFW policy script limited to `virbr0` and
  Tailscale;
- redacted, read-only fleet status.

Live Dragon evidence at audit time:

```text
VM lts-mt5-paper: running
lts-mt5-bridge.service: active for 1 day 21 hours
terminal build: 6090
heartbeat age: 8.3 seconds
heartbeats: 11000
snapshots: 2751
positions: 0
orders: 0
read_only: true
```

## 4. Findings Blocking MT5 Write Activation

### AUD-F2-20260802-060 (S2)

The reviewed EA is `LtsMt5ReadOnlyBridge.mq5`. It contains no `OrderSend`,
`CTrade`, command poller or command contract, and refuses to initialize when
`InpReadOnly=false`. The running heartbeat carries an adapter version but no
source or compiled-binary hash. The source on Dragon also predates the current
watchlist source (`a9aef0...` on Dragon versus `8ce231...` locally).

Disposition: blocks MT5 L1 only. It does not block IBKR L1.

Required correction: a separate demo-only execution EA and command service,
versioned command/acknowledgement contracts, source and EX5 hashes in the
deployment manifest/heartbeat, mandatory SL+TP geometry, idempotency, bounded
expiry, restart reconciliation and deterministic flatten/cancel behavior.

### AUD-F2-20260802-061 (S2)

The deployed bridge configuration has
`"allowed_account_fingerprints": []`. The implementation interprets an empty
set as allow-all. A valid shared secret can therefore inject telemetry from
any demo account fingerprint.

Disposition: blocks MT5 L1 only.

Required correction: fail startup when the allowlist is empty for any MT5
write-capable profile; bind commands and acknowledgements to one exact account,
server, environment and deployment hash.

### AUD-F2-20260802-062 (S3)

The FastAPI bridge calls `await request.body()` before authentication and has
no request-size limit. Any host permitted by the network policy can consume
memory with an oversized unauthenticated body.

Disposition: blocks MT5 L1 only.

Required correction: reject missing/oversized `Content-Length`, enforce a
small route-specific maximum while streaming, and test unsigned, replayed,
stale and oversized requests.

## 5. Protection-Gate Ruling

The previous gate was circular: read-only observation cannot prove native
broker acceptance of SL and TP, but it required that proof before sending the
first order.

The correction is conditionally approved for owner ratification:

1. The first IBKR Paper minimum-size canary is the verification instrument.
2. Parent, take-profit child and stop-loss child are constructed before any
   submission.
3. Parent and take-profit use `Transmit=false`; the final stop-loss child uses
   `Transmit=true`, which transmits the complete bracket group.
4. The order remains acceptable only after direct broker evidence identifies
   the parent and both protective children with correct opposite side,
   quantity and geometry.
5. Any reject, unknown acknowledgement, missing child, stale reconciliation
   or restart ambiguity triggers deterministic cancel/flatten and a global
   hold before any new risk.
6. One long canary must be flat and reconciled before one short canary starts.
7. No LLM, Hermes process or chat instruction may construct or submit an
   order. The owner phrase only flips a versioned, single-use activation gate.

Primary broker references:

- https://interactivebrokers.github.io/tws-api/bracket_order.html
- https://interactivebrokers.github.io/tws-api/order_submission.html

This ruling removes the circularity without weakening the owner's mandatory
SL+TP rule.

## 6. Immediate Sequence

1. Lieutenant Satoshi II implements the IBKR Paper L1 adapter and single-use
   canary runner behind the accepted L0 interface.
2. Musashi independently verifies code, tests, TWS capability evidence and a
   zero-submit preflight packet.
3. The owner ratifies the protection-gate amendment.
4. Only then does the owner use the activation phrase
   `ACTIVATE L1 CANARY IBKR PAPER NOW` while TWS Paper is authenticated.
5. After the long/short canaries reconcile, the technical lead starts the
   continuous L2 demo cell. The point is business evidence, not a prolonged
   shadow phase.

P20 and non-critical documentation are below this sequence.
