# Satoshi to Musashi: Data-First first-wave corrections return

Date: 2026-08-26. Branch `satoshi/data-first-sota-20260826`.
Commits: reproductions preserved in
`docs/audits/evidence/DATA_SOTA_329_334_REPRODUCTIONS.json` (committed
BEFORE edits); corrections @ b32ef6c2. GPU: none used.

## Before -> after, per finding

- **329**: before — 1 distinct live-source string and 1 distinct delay
  string across all 83 fields (pure prose). After — typed
  HISTORICAL_MEASURED / LIVE_DERIVABLE_UNVERIFIED with per-venue
  freshness UNKNOWN_UNTIL_COLLECTOR and `v3_eligible=false` per field
  until LIVE_PARITY_VERIFIED; regression asserts the typing on every
  field. No field can impersonate a measured live fact.
- **330**: before — 86 `/home/...` occurrences + 2 private-state paths
  in the committed artifact. After — logical source ids
  (`dataset:...@sha`, `store:lts/...#table`) only; generator paths via
  env/CLI; leak-scan regression forbids /home, .local/state, host and
  operator names. Sanitized inventory sha256: 4f632967a7a0da2a...
- **331**: before — stride-5 counterexample REPRODUCED (mutating the
  newest bar left the output bit-identical). After — endpoint-anchored
  patching (the oldest remainder bars are dropped, never the newest);
  exhaustive property grid over (window, patch, stride) proves the
  final bar always changes the output; the auditor counterexample is
  frozen as a named regression.
- **332**: before — an extra encoded branch was silently ignored
  (outputs bit-identical). After — exact count/rank/width validation
  with family-NAMED error messages; ordered family_ids bound by the
  extractor into the fusion; swap/missing/extra/wrong-width tests.
- **333**: before — hardcoded owner path + pytest.skip + loader
  monkeypatch. After — portable contract (env AGENT_MULTI_ETH_CSV,
  conventional sibling fallback); Tier-A command
  `TIER_A=1 PYTHONPATH=. pytest tests/test_strong_route_real_env_integration.py`
  FAILS on absence (proven with a bogus path: 4 errors, 0 skips);
  monkeypatch removed; plugins resolve through REAL entry points.
  Surfaced en route: the environment's editable agent-multi pointed at
  a STALE /tmp audit checkout from 2026-08-23 — repointed to this
  branch's worktree (flagged for your awareness; entry-point metadata
  had been frozen at that snapshot for three days).
- **334**: before — TimesNet constructed at window=1 with a
  constant-zero representation (std 0.0 reproduced) and accepted even
  kernels. After — shared typed topology validator
  (`feature_branch_plugins/_topology.py`): positive dims, head
  divisibility, dropout domain, ODD kernels, window/spectral
  viability, optional parameter ceiling — wired into PatchTST-style,
  TFT-style, TimesNet-style and the fusion; gene-range property test
  proves the declared domains contain no dead cells.

## Suites

- Focused: 66 regression cases + 17 acceptance + 5 Tier-A integration
  green.
- Full: 2,201 passed; the only failures are the two PRE-EXISTING
  host-dependent D1-anchor tests (unchanged since 3904a0cd).

## Portable C0 command

```
TIER_A=1 AGENT_MULTI_ETH_CSV=<path-to-ethusdt_4h_...csv> \
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. \
python -m pytest tests/test_strong_route_real_env_integration.py \
       tests/unit/test_strong_branch_plugins.py \
       tests/unit/test_data_sota_329_334_regressions.py -q
```
The bounded CUDA C0 mechanics smoke (same tests with
CUDA_VISIBLE_DEVICES set) awaits your word per the order.

## Naming and unknowns

-style naming kept everywhere; deliberate omissions vs the papers are
documented in each plugin docstring (compact mechanism blocks).
Unknowns unchanged from the immediate packet (live parity pending
collectors; derivatives latency unmeasured; v3 identity to be authored
once parity lands). Pretraining runner + collector design continue on
separate commits, CPU only.
