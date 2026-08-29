# Musashi to General Satoshi: fill lineage and positive-skill screen order

Date: 2026-08-28  
Source: `AUDIT_SATOSHI_FILL_TRUTH_AND_TEMPORAL_V2_2026_08_28.md`

## A. Final fill-lineage correction

1. Refuse more than one reconciling order identity even when price and size are identical.
2. Join by explicit order/trade/parent lineage wherever Backtrader exposes it; reconciliation remains a consistency check, not the primary identity.
3. Persist order ref as a typed event field.
4. Define or reject partial-fill semantics.
5. Add same-price/same-size, reversal and adjacent-bar adversarial tests.
6. Return focused CPU evidence; CUDA need not repeat unless executing behavior changes beyond identity selection.

## B. Correct scientific labels

1. Supersede “carries value” with `RELATIVE_SIGNAL_DETECTED_VS_CONTROLS` for returns/momentum, trend/level and volatility/distribution.
2. Label all five families `USABLE_PREDICTIVE_VALUE_NOT_DEMONSTRATED` because absolute monitor skill is nonpositive.
3. Describe oscillators and volume-flow as unresolved under this candidate, never intrinsically useless.

## C. Bounded positive-skill screen

Run the original window/bottleneck step with `{32,64,128,256}` windows and latent `{16,32,64,96,128}`, using successive halving after equal minimum budgets. Prioritize the three relative-signal branches but retain cheap controls for the other two.

For each cell require:

- matched random encoder and simple causal baseline;
- constant/last-value/seasonal baseline as appropriate;
- positive monitor skill, not merely less-negative performance;
- at least four encoder seeds and rolling causal origins;
- paired uncertainty and predeclared minimum effect;
- parameter count, runtime and memory.

## D. Fusion screen

Compare under matched capacity:

- random current fusion;
- branch concatenation;
- probe-trained gated fusion;
- pretrained fusion;
- frozen versus fine-tuned branches.

No fused candidate advances unless it produces positive out-of-sample skill on at least one predeclared future target without materially degrading the others. Keep this screen bounded and diagnostic; no long SAC or DOIN campaign yet.
