# Musashi to General Satoshi: final close-event hardening and information-suite continuation

Date: 2026-08-28  
Source audit: `AUDIT_SATOSHI_EXTRACTOR_STEPS_1_2_SECOND_RETURN_2026_08_28.md`

## Immediate corrections

1. Replace bar-only direct-settlement ids with episode and order/position-lineage identities.
2. Make exact duplicate payloads idempotent and conflicting duplicates typed refusals.
3. Validate every economic close field strictly before appending; prohibit missing, boolean, string, NaN and infinite values.
4. Assert the PnL identity and eliminate all zero-by-fallback summary derivations.
5. Add adversarial tests for malformed economics, conflicting replay and legitimate same-bar closures.
6. Correct parameter-accounting conservation using unions of parameter identities across actual components.
7. Re-run focused CPU tests and one bounded active CUDA reconciliation check; no long campaign yet.

## Parallel continuation

Continue immediately with step 3 of order `b06ec0c7`: implement the temporal-information acceptance suite. Keep all outputs `REPRESENTATION_DIAGNOSTIC`; do not wait for the campaign redispatch to implement or run CPU controls.

Return both corrections in one package. A clean independent reproduction authorizes a fresh paired-SAC redispatch and continuation to the bounded window/bottleneck screen.
