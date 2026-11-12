# Selected71 provider and real-input qualification

Owner instruction: defer Windows Autologon. Do not retry passwords or reboot
the VM for this task. Continue the real demo integration separately from
Musashi's phase2 refinement. No replacement of the existing ETH/SPY runners
is authorized by this inference handoff.

## Exact model, not the old selected70

Use export299, whose source remains unchanged after303. The later head
experiments did not improve its chronological accounts and are not promoted.

| Artifact | Identity |
| --- | --- |
| Model | `69b5254489d224d4b20ac358956e4dd3b9a67a2f214a9aff98045c5def25a536` |
| Bundle manifest | `f52a5c211c8421ccb354561395bcc48339d2f984b3756b48a2a0c981ce193d96` |
| Recipe | `recipe_b5fb7bcdfa1dbec8` |
| Recipe file | `e2cb27de272197ae5bba67e2172569ea89e580fa217b00474d0471d6e35d8ec3` |
| Tensor | 13 observations x71 columns, stride2,28 groups |
| Codec | `NOP_BUY_SELL_CLOSE_OPPOSITE_V1` |

Against your selected70 inventory, add `OPEN@diff` and `RAW_BAR_LOG_RETURN`,
remove `px.range_12`. Do NOT patch column positions by hand: consume the full
ordered recipe and its scaler. No agent-state input, rolling z-score, STL,
invented values or recalibration on live data. Encoding happens exactly once
according to the recorded store/consumer operations. Fixed TRAIN constants
are unchanged. NOP/BUY/SELL indices are0/1/2. While holding a position, the
opposite symbol closes it; it does not also open the opposite position in the
same action. The existing business controller still owns SL/TP/size/calendar.

## Delivered inference adapter

Repository `prediction_provider`, branch `musashi/selected71-provider-20261112`:
published commit `9301cd6`.
`mechanics/src/prediction_provider_mechanics/categorical_sac.py` and
`categorical_service.py`. Eight focused tests cover identity/order/shape,
nonfinite inputs and absence of execution authority. The adapter verifies all
bundle hashes before native loading and rejects native runtime import
collisions. It consumes an already encoded/scaled tensor, not raw MT5 bars.

On the original CPU, the32 saved probe rows reproduce exactly: max error0,
all greedy actions equal. A different CPU measured about1.2e-6 logit error
and was refused by the current exact gate. Do not loosen this gate silently.
Cross-platform inference qualification is a separate explicit test.

The isolated service exposes `/health`, `/v1/model`, `/v1/replay` on loopback.
It is running under a separate restart-on-failure user service.32 single-window
HTTP replays select the same greedy actions as the saved probe. Their logits
versus the saved batch differ by at most1.6689300537109375e-6; this is distinct
from the exact cold batch probe and is not called bit-exact HTTP parity.
It has no broker route and cannot certify freshness from a bare tensor.
The private deployment note gives its host, service, paths and launch config.
`MODEL_LOADED` and a replay response are NOT `SHADOW_OBSERVING` or
`PAPER_DEMO_EXECUTING`. The bundle retains its research qualification.

## Can all real inputs be acquired now?

Not yet demonstrated. An old historical parity receipt is not evidence of
current71 availability. Complete these source tasks, independently of
Windows Autologon, and return machine-readable receipts:

1. Read the existing demo terminal, without orders, for seven actual H1 Bid
   histories: EURUSD,GBPUSD,USDCHF,USDJPY,EURGBP,EURJPY,AUDUSD. Record broker
   symbol mapping, clock identity, bar start/close, gaps, oldest/latest bar and
   retrieval timestamp. Never replace a missing cross with another instrument.
2. Rebuild the original causal price/calendar/SSA/wavelet/Hilbert producers
   and their true warmup. A13xstride2 tensor does not imply only25 raw bars
   suffice for every producer. Require incremental versus batch prefix parity,
   restart/state replay, no change to earlier rows when future bars change.
3. Preserve frozen FX factor full precision and its six-cross parent basket.
   USD-strength wavelet and EUR-strength Hilbert need their real parents.
   Kalman/GMM require original frozen fit state and causal state reconstruction,
   not refitting to today's history or resetting without qualification.
4. Activity uses the full-TRAIN estimator index0 of the recovered artifact,
   SHA `fdc95a0cdabf9a5327c991187f62c8be928476f385551c7e8db3972045167d6b`.
   Do not average its six estimators. Its ordered parents are:
   `cal.utc_dow_cos@0,@1,@120`; `cal.utc_dow_sin@24,@48`;
   `dly.dax.ret_5d@48`; `dly.spx.ret_1d@48`;
   `dow_cos@0,@1,@120`; `dow_sin@48`;
   `ev.tokyo_fix.hours_since@8`; `ev.tokyo_fix.hours_to@8`;
   `px.hours_since_session_open@2`. These are14 columns, with lag in observed
   rows. DAX/SPX remain dependencies even when absent from the actor input list.
5. Qualify the original daily US10Y, DAX/SPX and publication/calendar sources.
   A weekly/daily published observation may remain valid as-of the next
   release by its recorded rule. That is not permission to fabricate an hourly
   update. Record event time AND actual availability time.
6. Resolve COT identity before claiming current parity. Your current evidence
   reports mixed contracts after date-only deduplication:23 EUR and29 other
   contracts in52 compared2025 dates. The actor still consumes
   `cot.eur.net_long`, `cot.eur.net_share`, `cot.eur.net_z156`. Recover the
   original contract-qualified series and explain the historical discrepancy.
   A corrected EUR-only series is a NEW data version, not an interchangeable
   input to this checkpoint. If historical equivalence cannot be established,
   return the affected rows/columns and corrected candidate to Musashi for
   replay/refinement qualification. Do not fake the old contamination in live.

## Integration acceptance

First update your inventory from70 to71 and enumerate ALL transitive parents.
Each cell needs a producer/source identity, transform/state hash, publication
time and admissibility reason. Then compare the71 tensor and greedy logits on
shared historical closed bars with the native builder. Exercise real current
bars only after source qualification. Market decision bars must be closed,
eligible and at most one hour old during the scheduled market observation;
scheduled market closure is distinct from a stale or broken feed.

Missing/nonfinite/stale inputs produce `DATA_UNAVAILABLE`, never a learned NOP.
No zero fill, interpolation, synthetic bars, partial baskets or substitute
macro series. Staleness must reach the existing execution/risk owner; do not
leave an open demo order unmanaged by silently stopping the signal loop.
Broker-side stops and reconciliation of existing positions remain separate
from inference and must be verified before demo execution.

Keep provider -> strategy -> shadow account -> demo executor separate. All
pool candidates observe the same hourly closed bars and recorded costs, with
independent virtual accounts. Preserve the current model until a challenger
passes the agreed forward comparison; do not invent a new promotion metric
or shorten the comparison period in this handoff. Do not deploy EURUSD weights
onto SPY/ETH merely because their services already run.

Return: updated71 availability JSON, source receipt per dependency, COT
identity report, tensor/probe parity and current closed-bar heartbeat. State
exactly which stage is actually running. No password or credential is needed
in these documents. Windows unattended login remains deferred by the owner;
do not claim restart-to-trading recovery while it is unresolved.

## Musashi research continuity

Research stays on303 ->304: diagnose distance to useful FIT decision boundaries
and shared-head constraints before further refinement. Preserved source metrics
are TRAIN16.058224%, VAL18.369946%,745/207 orders, not forward results. No new
feature removal or repeated floating pass is justified by the303 loss-only
change. Once a qualified refinement exists, the return pass includes the full
historical phase2 removal pool, not only the latest outward pass.
