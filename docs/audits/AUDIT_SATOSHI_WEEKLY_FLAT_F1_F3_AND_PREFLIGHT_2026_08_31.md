# Audit: weekly-flat F1-F3 and CPU preflight

Date: 2026-08-31
Verdict: `F1_F3_ACCEPTED_WITH_PREFLIGHT_RESCOPED`

Independent focused suites pass (`250 passed`). The precedence correction, blackout release latch and generation-bound custody claim are accepted for continued non-live development.

The executed preflight proves real SB3 SAC plumbing and bounded CPU execution, but it used `MultiInputPolicy` with a flat `[64,64]` MLP. Its 128 updates/s is not a throughput estimate for the accepted strong grouped feature extractor and must be labeled `SB3_MLP_PLUMBING_ONLY`.

The run also establishes that the section-4 `forced_flatten_hours=4` default is structurally ineligible for H4 under next-bar fills. It cannot remain an admitted H4 experiment/live default.

