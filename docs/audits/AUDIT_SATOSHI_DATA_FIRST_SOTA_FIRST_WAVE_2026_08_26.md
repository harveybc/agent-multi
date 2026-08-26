# Audit: Data-First SOTA Multibranch First Wave

Date: 2026-08-26
Auditor: General Musashi
Audited tip: `satoshi/data-first-sota-20260826@d9471da9`
Disposition: **REVISE BEFORE CUDA C0**

## Reproduction

The focused suite independently reproduced: 21 passed. PatchTST, TFT-style,
TimesNet-style and cross-family fusion construct, backpropagate and round-trip
their state on the submitted fixtures. This is useful implementation, not a
rejection of the direction.

## Findings

### DATA-SOTA-329 — S3 — Inventory turns derivability into live evidence

Every one of the 83 fields receives the same prose claim: derivable from Alpaca
and MT5 bars with publication delay approximately zero. No same-bar historical
versus live value is compared, no venue-specific coverage/freshness is measured,
and the transforms are described generically rather than re-derived per field.
The packet itself says parity remains planned. Therefore `live_source` and
`publication_delay` are proposals, not measured facts.

Required: typed states `HISTORICAL_MEASURED`, `LIVE_DERIVABLE_UNVERIFIED`,
`LIVE_PARITY_VERIFIED`; venue-specific freshness/coverage; no eligible v3 field
until actual values reproduce within declared tolerance.

### DATA-SOTA-330 — S3 — Public artifact exposes local filesystem identity

The committed inventory repeats `/home/harveybc/...` for every feature and
publishes paths to private LTS state databases. The generator also hardcodes the
checkout path. A public evidence artifact needs logical source identifiers and
content hashes, not operator/home topology.

Required: sanitize the committed artifact and generator output; accept paths by
CLI/config/env; record logical source IDs plus hashes. Add a repository scan
fixture that rejects absolute home paths, private state paths and host names.

### DATA-SOTA-331 — S3 — PatchTST can silently discard the newest bars

`unfold(patch_len, stride)` starts at zero and the plugin neither pads nor
requires endpoint coverage. For configurations where
`(window_size-patch_len) % stride != 0`, the last observation(s) never enter a
patch. DOIN could select such a topology and train a policy blind to the newest
market information.

Required: endpoint-anchored/padded causal patches or fail closed on uncovered
tails. Mutating the final bar must change the public forward output for every
accepted topology; exhaustive small-window property tests are required.

### DATA-SOTA-332 — S3 — Fusion silently truncates branch lists

`zip(self.proj, encoded)` silently ignores extra encoded branches and then fails
indirectly or behaves ambiguously for missing branches. Family identity is
positional and not bound to names/digests in this plugin surface.

Required: exact branch count, rank and feature-dimension checks before fusion;
bind ordered family IDs into the effective architecture/artifact; adversarial
swap/missing/extra/wrong-width tests.

### DATA-SOTA-333 — S4 — “Real environment” test can silently skip

The test hardcodes the owner's predictor path and calls `pytest.skip` when the
CSV is absent. A green suite on another host can therefore contain zero real-env
integration. The local plugin-loader monkeypatch also bypasses installed entry
point discovery.

Required: a portable fixture/path contract, a strict Tier-A command where
absence is failure, plus a separate explicit optional test if desired. Reinstall
editable metadata and prove discovery through real entry points.

### DATA-SOTA-334 — S4 — Topology domains admit degenerate configurations

TimesNet accepts a one-step window and can produce a constant zero representation;
even kernels are not ruled out despite shape-changing padding. PatchTST/TFT and
fusion validate only a subset of positive/range constraints. These become DOIN
genes, so invalid domains must refuse before model construction.

Required: one shared typed topology validator per plugin, finite probabilities,
positive dimensions, valid head divisibility, odd kernels where same-shape is
claimed, viable spectral bins/periods and resource ceilings. Property-test the
declared gene ranges.

## Architectural wording

Keep the names `PatchTST-style`, `TFT-style` and `TimesNet-style`. These compact
blocks implement selected mechanisms, not full reproductions of the papers.
Promotion must depend on causal/economic evidence, not the names.

## Acceptance

Correct 329-334, preserve the 21 passing cases, add every counterexample above,
run the full suite and return a portable C0 command. CPU work and collector
design continue in parallel. No long GPU campaign is accepted yet.
