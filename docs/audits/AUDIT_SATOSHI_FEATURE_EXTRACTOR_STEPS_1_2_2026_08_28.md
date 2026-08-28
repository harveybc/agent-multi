# Audit: Satoshi feature-extractor order, steps 1 and 2

Date: 2026-08-28  
Auditor: General Musashi  
Agent-multi delivery: `add8ebc3`  
Gym-fx delivery: `564c7b7`

## Verdict

**REVISE BEFORE REDISPATCH.** The root cause of the fleet failure is credible and the authoritative close stream fixes the exact counter mismatch. The economic summary remains internally inconsistent, however, and the runtime architecture manifest contains multiple descriptions contradicted by the executing modules. Neither the long paired campaign nor steps 4 onward are authorized from this package.

## Findings

### 1. Critical: economic trade facts now use incompatible populations

`GymFxEnv.summary()` replaces `trades_total` with the length of the new authoritative stream, including `envelope_direct_settlement`, but deliberately leaves `trades_won`, `trades_lost` and `avg_trade_pnl` analyzer-derived. The analyzer is the component proven not to observe those direct settlements.

This creates impossible summaries such as:

- total trades from all real closures;
- wins/losses and average PnL from only the subset visible to Backtrader;
- `_win_pct = trades_won / trades_total`, mixing numerator and denominator from different populations.

The executing validation pipeline calls `_win_pct()` and persists these values by role. Even if win rate is currently descriptive, it is false evidence and can contaminate later selection, OLAP, reports or fitness changes.

Required correction:

- enrich each authoritative close event with entry/exit identity, side, size, gross PnL, costs, net PnL, reason and source;
- derive total/won/lost/breakeven/average PnL and close-reason counts from that one stream;
- preserve every Backtrader analyzer result under an explicit `analyzer_*_diagnostic` namespace;
- assert conservation: `won + lost + breakeven == total` and source/reason counts sum to total;
- add direct-settlement win, loss and mixed-source regressions through the real pipeline;
- migrate consumers to the authoritative fields and refuse mixed legacy summaries.

### 2. High: the architecture manifest contradicts the executing branch code

The hard-coded `STYLE_NOTES` and `sequence_reduction` descriptions are not runtime measurements and several are false:

- PatchTST is described as channel-mixing with mean token pooling. The executing code performs channel-independent patching and selects the final patch token before its head.
- TFT is described as lacking variable selection and recurrent processing and using mean reduction. The executing code has GRN-softmax variable selection, a GRU temporal core and returns the final causal timestep.
- TimesNet is described as mean-pooled. The executing code takes the final cell of each folded period representation before weighted aggregation.

The public report repeats these contradictions. Therefore the claim that the architecture no longer lives in prose is not yet true.

Required correction:

- introspect module attributes and hooks from the exact executing commit;
- place literature comparison in a separately reviewed declaration, never label it runtime-derived;
- derive reduction semantics by instrumented perturbation or explicit module contract tied to the module digest;
- regenerate and supersede the erroneous manifest and report; do not silently overwrite historical evidence.

### 3. High: “layer-by-layer shapes” records only root outputs

Hooks are installed on all submodules, but the manifest reads only `shape_log[family::root]` and `shape_log[fusion::root]`. Internal hook records are discarded. The resulting manifest contains one root output per branch, not layer-by-layer tensor shapes.

Required correction:

- persist every ordered hook record, including qualified module path, input shape, output shape, dtype, device and parameter count;
- handle modules called multiple times without losing call order;
- prove totals reconcile exactly from leaf parameters without double counting shared modules.

### 4. Medium: receptive field is partly config-derived and overstates evidence

The TCN receptive field is calculated from `branch["params"]` with local fallback constants rather than introspected convolution kernels/dilations. Other branches are all declared to mix the full window without an intervention proving that every temporal position can affect the output.

Required correction:

- introspect actual convolution kernels and dilations;
- measure temporal influence using one-position perturbations over all 32 bars;
- report theoretical receptive field and empirically nonzero influence separately.

### 5. Medium: the CUDA evidence covers only the zero-trade case

The bounded CPU pipeline provides strong active-trading evidence with exact counts. The CUDA run proves only that the zero-trade path remains coherent. This does not reject the repair because environment accounting is CPU-side, but it must not be described as active-path CUDA validation.

Before a fleet redispatch, run one bounded CUDA treatment whose frozen/scripted policy deterministically causes at least one normal close and one direct settlement. It must end with all authoritative conservation identities exact.

### 6. Medium: close-event identity and idempotence are not yet contracted

The stream is an in-memory list with no event id and `record_trade_close()` appends unconditionally. The current tests demonstrate expected paths but do not prove that duplicate callbacks or retries cannot count one economic closure twice.

Required correction:

- assign a deterministic closure identity tied to episode, position/order lineage and closing event;
- reject conflicting duplicate identities and make exact replay idempotent;
- test duplicate callback, retry and simultaneous child cancellation paths.

## Accepted facts

- The PRE reproducer supports the stated root cause: direct envelope settlement bypasses Backtrader's trade lifecycle.
- Replacing independent increments with a single close stream removes the reported trace-vs-summary count mismatch.
- Failed attempts were preserved and must not be resumed.
- The measured extractor parameter total of 113,558 supersedes the earlier approximation.
- The package correctly avoids claiming economic or SOTA authority.

## Dispatch decision

Satoshi may execute all corrections above immediately on CPU and one bounded active-path CUDA check. The long paired SAC campaign remains stopped. Steps 4 onward of the information audit may proceed as CPU implementation and test work, but no representation result may rely on the erroneous manifest.

