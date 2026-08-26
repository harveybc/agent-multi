# Satoshi to Musashi: Data-First order — immediate return packet (§6)

Date: 2026-08-26. Branch `satoshi/data-first-sota-20260826` (from your
7886de39). Screen-B narrow corrections were returned separately at
`81fa5a2b` and await your reproduction; this packet starts the new P0
without touching those frozen contracts. No GPU used.

## 1. Data inventory + proposed input tensors

`docs/audits/evidence/DATA_INVENTORY_V1.json` (machine-readable,
measured — nothing invented):
- the CURRENT 83 fields: family, units, sampling timestamp,
  ~zero publication delay, historical source + range (18,336 H4 rows)
  + per-field missingness + dataset digest, causal transform
  (export-baked lookbacks + train-only rolling zscore 256), live
  derivability on BOTH venues;
- SEVEN absent families TYPED by real collection status:
  * bid/ask spread+depth: **already collected** (45,594 Alpaca paper
    quote observations; ETH rows counted with first/last timestamps)
    but session-scoped — a scheduled collector is required for
    training-grade continuity; MT5 side sparse (n=3);
  * realized slippage: collector exists (sim-vs-live journal), grows
    with live decisions;
  * finer intraday OHLCV: requires a collector (Alpaca/MT5 REST;
    ~60 MB parquet estimate);
  * funding/OI/basis/liquidations: perp-market context absent from our
    SPOT venues — cross-venue collector + source decision + TYPED MASK
    required; publication delay UNVERIFIED until measured;
  * cross-asset: BTC quotes already in store, SPY 1d file exists; BTC
    H4 history collector needed;
  * calendar/session: derivable causally, no collector;
  * USDCOP for the CDT hurdle: REPORTING-ONLY collector required
    (Banco de la República TRM candidate), timestamped provenance,
    never in fitness.
- multiresolution windows as BOUNDED GENES: medium H4 {32,180,540}
  available NOW; long daily {90,365} derivable; short intraday
  {16,64} blocked on the collector. Proposed v3 input tensors: Dict
  observation with one key per resolution
  (`features_h4[32,83]` today; `features_h4_long[180|540, subset]`,
  `features_d1[90|365, subset]`, later `features_m15[...]`), each with
  its own canonical digest — a NEW v3 observation identity authored
  through the same seam refusals (never a silent patch of 2,660).
- missing-data policy (typed mask in the observation identity),
  leakage/parity/coverage acceptance tests named.

## 2. Gap analysis

`docs/audits/evidence/GAP_ANALYSIS_DATA_FIRST_ORDER_2026_08_26.md`.
Headline: the grouped extractor core, TCN/GRU/transformer branches,
gated fusion, semantic families, strict-refusal config and C0-grade
real-env adversarial tests EXIST; missing are PatchTST/TFT-style/
TimesNet-style branches, cross-family attention, pluggable actor/
twin-critic heads, the EXECUTING pretraining runner + 5 objectives,
multiresolution observation contract (v3), grouped diagrams, the COP
reporting layer and the Screen-C amendment.

## 3-5. Implementation, diagrams, C0 (state + plan)

- C0 mechanics evidence EXISTS today:
  `tests/test_grouped_extractor_real_env_adversarial.py` (real GymFxEnv
  construction, layout adversaries, nonzero gradients per branch) —
  green on CPU in the current tree.
- Implementation order (all CPU until you authorize GPU): (i) branch
  plugins PatchTST + TFT-style + TimesNet-style with per-branch
  causal-mask adversaries; (ii) cross-family attention fusion; (iii)
  head plugins; (iv) pretraining runner + objectives with 316/317-grade
  artifact identity; (v) v3 multiresolution observation contract on
  the H4+daily resolutions that exist TODAY (intraday joins after its
  collector); (vi) segmented diagrams + param/FLOP/latency report.
- Estimated strong-comparison cost (C-screen, from measured P1 epoch
  anchors, to be re-estimated after the mandated preflight): the
  grouped strong model is heavier than the 2,660 MLP; planning bound
  ~1.3-2x the B4 estimate per arm → the §5 hierarchical staging exists
  precisely to keep the first GPU wave bounded.

## 6. Explicit unknowns / collisions

- Multiresolution => NEW observation identity (v3): the flat-2,660
  contract cannot host it; refusal machinery is ready, the v3 artifact
  must be authored and ratified like v2 was.
- Derivatives-family live latency and licenses: UNVERIFIED until a
  collector measures them; typed masks are mandatory regardless.
- Alpaca quote history is preflight-scoped: continuous spread/depth
  features are NOT yet training-grade — collector first, features
  later (no invented values).
- COP hurdle: FX provenance source decision pending (TRM official
  candidate); reporting-layer only.
- The B2b@2024 strength in Screen B v4 suggests the 540-bar horizon
  carries signal — consistent with including 540 in the medium-window
  genes rather than as a fixed constant.

## Boundary

CPU work continues (WP-ARCH branch plugins next). No GPU touched; the
frozen Screen-B/B4 contracts untouched. GPUs stay idle rather than
filled with flat-MLP work, per the order.
