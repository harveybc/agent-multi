# L0 Interface and No-Duplication Map

Date: 2026-08-02 04:40 America/Bogota
Author: General Satoshi II, temporary technical lead
For: General Musashi (Required Sequence item 3) and the L0 vertical build
Repos surveyed at: `trading-contracts@534b0349d9a3` (v0.2.0),
`prediction_provider@ac4d9e2fa552`, `lts@11d8958979b7`,
`agent-multi@55d72575`, `gym-fx@62c22050781b`
Method: direct source inspection of contract classes and module surfaces;
depth limitations declared in section 7. Runtime mutation: none.

## 1. Canonical DTO Families — Owners and Versions

All seven demanded families exist, in one module, as strict Pydantic v2
models with `Literal` schema versions:
`trading-contracts/src/trading_contracts/contracts.py`.

| Contract | Version | Line | Notes for L0 |
| --- | --- | ---: | --- |
| `MarketSnapshot` | `market_snapshot.v1` | 110 | causal inputs + manifest hashes |
| `PredictionBundle` | `prediction_bundle.v1` | 143 | model outputs, no decisions |
| `DecisionContext` | `decision_context.v1` | 161 | policy input wrapper |
| `AssetIntent` | `asset_intent.v1` | 194 | has `RiskGeometry` (L177) with `stop_price`/`take_profit_price`/`trailing_distance`; `artifact_hash` required; target-action validator present |
| `PortfolioIntent` | `portfolio_intent.v1` | 221 | `PortfolioConstraintState` (L214): gross/net/cell-weight/turnover limits |
| `OrderIntent` | `order_intent.v1` | 231 | **finding 039 confirmed in code**: `stop_price`/`take_profit_price` optional; `stop_price` ambiguous between stop-entry trigger and protection; naked market entry validates |
| `ExecutionReport` | `execution_report.v1` | 257 | **finding 041 confirmed in code**: states only `requested/accepted/filled/rejected/modified/closed`; no partial/cancel-pending/cancelled/expired/unknown; no bracket parent/child identity or per-leg protection |
| `ComponentManifest` | `component_manifest.v1` | 283 | artifact identity + lineage |
| `DeploymentManifest` | `deployment_manifest.v1` | 308 | channel binding + rollback |

Support surfaces in the same package: `canonical.py`
(`canonical_json`, `content_hash` — the canonical hashing seam),
`compatibility.py` (`evaluate_deployment_compatibility`, major-version
gates), `config.py` (`TradingExperimentConfig`, `CandidateGenomePatch`,
`TradingRuntimeOverlay`).

## 2. The Decisive Integration Fact

Current `trading_contracts` consumers (verified by import grep):

| Repo | Imports contracts? | Where |
| --- | --- | --- |
| agent-multi | yes | `app/canonical_config.py`, `app/runtime_overlay.py` |
| gym-fx | yes | `simulation_engines/bakeoff.py` |
| **lts** | **no** | — |
| **prediction_provider** | **no** | — |

The canonical DTOs stop at the research/simulation boundary. The serving
and execution planes have never consumed them. Therefore the L0 vertical is
primarily **wiring existing contracts across two repos that ignore them**,
not inventing new ones. This is the no-duplication battlefield: the risk is
LTS/provider growing parallel ad-hoc dicts instead of importing the
package.

## 3. prediction_provider Surface (serving plane)

Observed: a plugin service — `plugins_core` (`default_core`, `sync_core`),
`plugins_predictor` (direction/binary/csv/ideal-oracle predictors),
`plugins_feeder`, `plugins_pipeline`, `plugins_endpoints`, plus an `app/`
API layer (client/admin/evaluator endpoints, auth, DB models). Hash usage
exists in core/auth/DB but there is **no `trading_contracts` wiring, no
SB3-artifact hash-verified loader, and no `AssetIntent` production path**
— consistent with doctrine 29's zero-wiring finding.

L0 decision: **reuse the plugin seams, add one new plugin pair** — an
artifact-loading predictor plugin (SB3 `.zip`, SHA-256 verified against a
`ComponentManifest`) and an intent endpoint emitting canonical
`PredictionBundle`/`AssetIntent`. No second serving service; no parallel
DTOs; `mechanics_only_not_alpha_claim` labeling per the owner mandate.

## 4. LTS Surface (execution plane)

| Concern (Musashi's list) | Existing module(s) | L0 disposition |
| --- | --- | --- |
| Risk & sizing | `social_trading_lab.py`: `CopyAllocationContract` (L594) already implements `max_overshoot_ratio`, `margin_buffer_ratio`, leverage/free-margin checks, step-aligned lots, tracking-error rejection | **reuse the discipline, generalize the primitives** — this is finding 040's exact math, already tested, in the social plane; extract into venue-neutral sizing/reservation used by the demo execution service |
| Allocation | `multi_venue_shadow.py` (synthetic NAV, cell marking); `PortfolioIntent` unconsumed | extend shadow marking into intent-consuming allocator; no new NAV concept |
| Execution adapters | per-venue labs: `ibkr_paper_lab.py` (`IbkrTwsPaperClient`, `ContractSelection`, `IbkrPaperOlap`, `IbkrPaperLab`), `alpaca_paper_lab.py`, `mt5_bridge_lab.py`, `capital_demo_lab.py` — all read-only; `oanda_practice_lab.py` holds the only historical canary/order path (REST-v20, unusable for OGM, preserved) | reuse the lab pattern (Config + Client + Olap + Lab dataclass composition, fingerprint redaction, canonical json); add the IBKR **write serialization** path behind the zero-network sink; boundary objects must be `trading_contracts` `OrderIntent`/`ExecutionReport`, not new dataclasses |
| Reconciliation | per-lab session/reconciliation facts (e.g. IBKR `latest_complete` with reconciliation timestamps) | extend to order-lifecycle reconciliation per doc 06 §5.4 |
| OLAP | per-lab SQLite OLAP classes; `social_trading_lab` append-only SHA-256 event chain + idempotency keys | reuse the event-chain/idempotency pattern for `fact_order_lifecycle`; no new persistence idiom |
| Provider client | `prediction_client.py` exists (predates canonical contracts) | reuse/extend to speak `PredictionBundle`/`AssetIntent`; do not write a second client |

## 5. Reuse / Extend / Missing — The Binding Map

**REUSE as-is (no new code):** all seven v1 DTO families for every concept
they already name; `canonical.py` hashing; `compatibility.py` gates;
LTS lab/OLAP/redaction patterns; social-lab sizing mathematics; existing
watchdog packet conventions; `multifront_status` for integration state.

**EXTEND with versioned successors (per findings 039-041; v1 never
silently edited):**

| New version | Owner | Content |
| --- | --- | --- |
| `order_intent.v2` | trading-contracts | explicit `ProtectiveBracket` (`stop_loss_price`, `take_profit_price`) separated from `entry_trigger_price`; both legs mandatory for risk-increasing intents; side/price geometry validated; risk-reducing close/flatten/cancel exempt (039) |
| `execution_report.v2` | trading-contracts | full lifecycle state machine incl. `partially_filled`, `cancel_pending`, `cancelled`, `expired`, `unknown_requires_reconciliation`; bracket parent/child identity; per-leg protection state and covered quantity (041) |
| risk envelope (new fields or `portfolio_intent.v2` extension) | trading-contracts | `risk_fraction_at_stop`, `gross_notional_fraction`, margin cap and daily-loss budget as separate dimensions + atomic reservation identity (040); `PortfolioConstraintState` covers gross/net/weight/turnover only — extend it, do not duplicate it |

**MISSING (create; verified no existing concept covers them):**

| New contract/component | Owner | Purpose |
| --- | --- | --- |
| `BrokerCapabilitySnapshot` DTO | trading-contracts | specified in docs 07/22, absent from `contracts.py`; hash-referenced by `OrderIntent.v2` |
| `OwnerCommand` DTO + deterministic handler | trading-contracts (DTO) / lts (handler) | finding 042: allowlisted identity, nonce, expiry, idempotency; risk-reducing only; zero LLM/Hermes in path |
| `fact_order_lifecycle` store | lts | doc 06 §5.4 exists as spec only; implement with the social-lab event-chain idiom |
| zero-network adapter sink | lts | drives real adapter serialization, proves `submitted_count=0`; the L0/L1 switch point |
| LTS demo execution service | lts | consumes intents, applies risk envelope, emits protected `OrderIntent.v2`, dry-run default |
| SB3 hash-verified loader + intent endpoint | prediction_provider | section 3 plugin pair |

## 6. No-Duplication Rules Binding the L0 Build

1. No second DTO for any concept named in section 1 — extension is a
   versioned successor in `trading-contracts`, nothing else.
2. LTS and prediction_provider take `trading_contracts` as a dependency;
   boundary objects are the canonical models, internal helpers stay
   internal.
3. Schema exports and example fixtures regenerate through the existing
   `scripts/export_schemas.py` / `build_contract_examples.py`, with hashes
   in the evidence packet.
4. `evaluate_deployment_compatibility` is the version gate at load time —
   no hand-rolled version checks in LTS or the provider.
5. Sizing/reservation math has one home; the social-lab implementation is
   generalized, not copied.

## 7. Declared Depth Limitations

- `prediction_provider` internals (`default_core.py`, endpoint auth flow)
  were surface-scanned, not line-audited; the plugin-pair design will be
  re-verified against actual seams before code, and any deviation reported.
- LTS lab internal dataclasses were pattern-sampled (IBKR read in detail;
  Alpaca/MT5/Capital by structure); write-path capability facts will be
  reproduced at implementation time per the owner clarification, not
  assumed from documentation.
- `gym-fx` consumption (`bakeoff.py`) was noted but not mapped for L0; the
  sim/paper shared-contract requirement (doc 14 rule) enters at the
  fixture stage.

Next in sequence: the adversarial L0 fixture packet (findings 039-042)
built against the v2 contracts above, then the live-feed L0 shadow vertical
with the zero-network sink.
