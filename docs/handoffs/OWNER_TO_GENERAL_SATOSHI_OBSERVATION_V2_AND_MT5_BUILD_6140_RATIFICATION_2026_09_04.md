# OWNER DECISION: OBSERVATION V2 AND MT5 BUILD 6140

**Date:** 2026-09-04

**Resolves:** items 1 and 2 of `agent-multi@af1ca667`, section 13

**Machine-readable record:** `docs/audits/evidence/OWNER_RATIFICATION_OBSERVATION_V2_AND_MT5_BUILD_6140_2026_09_04.json`

**Record SHA-256:** `399483a14ab4821a49155afd72d153e870e2f9c051945875ca7fdfb5a5726186`

## 1. Observation contract v2: RATIFIED

The owner ratifies the prospective post-P1 observation contract identified by:

- contract file SHA-256 `0ecc3d004b26ef4d913fd06ab585f9ce0885011a4cf4d1cc88d0a743b3e981a7`;
- 83 ordered feature columns, SHA-256 `c4697681c1323245691b8e577905894b96bed81738411b439995e2c2d4b44e4d`;
- `typical_price` excluded and raw price window disabled;
- four agent-state fields: `position`, `equity_norm`, `unrealized_pnl_norm`, `holding_duration_norm`;
- agent-state digest `b5beeb97e2031b8b696fad452cf42d1781d87848ce753855413f0f46eef9f160`;
- window size 32 and flattened observation dimension 2660.

This resolves the owner-ratification blocker for Screen B recovery. It does not rewrite historical P1 artifacts executed under the 84-feature identity. Any semantic or identity drift requires a new ratification.

## 2. MT5 terminal build 6140: ACCEPTED

Build `6140` is accepted as the current expected terminal build for collector preflight and operator-kit validation. It supersedes the stale expected value `6090` in that scope only.

This is not permission to install, update, restart or activate MT5 or its EA. The implementation must reject an observed build that differs from 6140 unless the owner issues another disposition.

## 3. Instructions to Satoshi

1. Consume this record in the current C17-C22 and B4-R0..R3 return; do not infer broader authority from it.
2. Change the observation contract status from pending to owner-ratified without changing its ratified terms or identity.
3. Use this exact observation identity for prospective Screen B audit/materialization only. GPU dispatch remains separately gated.
4. Update the collector preflight and reviewed kit expectation from build 6090 to 6140, with regression coverage that rejects both the old expected build and an arbitrary different build.
5. Keep collector status `COORDINATED_WINDOW_REQUIRED` until every remaining gate below is independently satisfied.

## 4. Gates still pending

- Ed25519 key ceremony in restricted operator storage.
- Real backup manifest, sealed review act and tested rollback kit.
- Fresh direct evidence of zero positions and zero orders at the coordinated activation window.

No GPU run, live deployment, service action, venue command, key generation, weekly-flat activation or checkpoint promotion is authorized by this decision.
