# Evidence: SOTA-C01 observation identity (84 executed vs 83 declared)

Date: 2026-08-25. All commands CPU, read-only against campaign
artifacts; P1 untouched.

## Reproduction (F1.1)

- `seed101_N/normal_report.launch_manifest.json` →
  `len(effective_config.feature_columns) == 84`; ordered digest
  `dd9e05d8e6dffeb4d7f25bf21d9cfea282cfa33061d06924382e759d1b563bd1`;
  first three columns `typical_price, return_1, log_return_1`.
- Declared contract `examples/config/phase_3_eth_sac_dynamics/systems/
  ethusdt_4h_l1_system_v1.json` → `.observation.feature_columns` count
  83, starting `return_1`.
- Set difference: executed − declared = `{typical_price}`; declared −
  executed = ∅. `typical_price` is PREPENDED (index 0), so the entire
  executed ordering is shifted, not merely extended.
- Corroboration: the R06 warmup probe independently reports feature
  denominator 84 in all four roles.

## Trace (F1.2)

`tools/wp4_cpu_smoke.py` (frozen campaign worktree `am-p1-6e7bd128`):

- line 309: `features = [c for c in next(_csv.reader(DATA.open())) if c
  not in _EXCLUDE]` with `_EXCLUDE = {DATE_TIME, OPEN, HIGH, LOW,
  CLOSE, VOLUME}` (line 29). The dataset's second header column is
  `typical_price`; it is not excluded, so it survives at position 0.
- `build_config` (line 103) hard-codes `include_price_window: False`.
- The system observation contract is never read by the driver: the
  executed identity is CSV-header-derived, the declared identity is
  manifest-derived, and nothing bound them.

Second divergence exposed by the same trace: declared
`include_price_window: true` (manifest flattened [2724] = 32×(83+2)+4)
vs executed `false` (flattened 32×84+4 = 2,692). The "documented"
32×83+4 = 2,660 matches NEITHER.

## Correction (F1.3-F1.4)

- `tools/post_p1_screen_contract.check_observation_identity` refuses
  pre-model on count, order, digest, window/price-window/agent-state
  flags; executed-vs-declared real fixture refused with:
  `REFUSED: executed feature count 84 != declared 83
  (extra=['typical_price'], missing=[])`.
- `executed_observation_identity` labels terminal records honestly
  (84 / digest / 2,692 / price_window false). Doc 38 §23.5 binds the
  sealing requirements; artifacts are never rewritten.

## Prospective decision (F1.5) — REQUEST, not decided here

Recommendation to Musashi/owner: post-P1 screens adopt a NEW contract
identity `ethusdt_4h_l1_system_v2` = the declared 83-feature list
(drop `typical_price`: a raw price level (H+L+C)/3, non-stationary,
redundant with the excluded raw OHLC family) with
`include_price_window: false` as executed (flattened 2,660). P1 stays
diagnostic under its executed 84-feature identity. No inheritance
either way without this decision recorded.

## Addendum 2026-08-25 (SOTA-F01)

The digest `dd9e05d8...` above is the LEGACY newline-joined diagnostic,
now labeled as such. Canonical compact-JSON digests
(`pipeline_plugins._observation_contract.feature_columns_sha256`, the
single shared implementation, unity fixture-tested):

- executed 84-feature list: `df2d981dc83367075643d20eb22d484e0543aef512ecdfc12ddb0f321ae3c682`
- declared v1 / proposed v2 83-feature list: `c4697681c1323245...` (see
  `ethusdt_4h_l1_system_v2.json`)
- v2 agent-state fields (position, equity_norm, unrealized_pnl_norm,
  holding_duration_norm): `b5beeb97e2031b8b...`
