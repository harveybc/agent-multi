# Phase 1 Asset-Policy Data

This directory stores small, versioned dataset manifests, never duplicated
market CSV files. Runtime overlays resolve `DATA_ROOT` to the shared
`financial-data` repository.

The SOLUSDT 4h phase uses one immutable chronological source and explicit date
boundaries:

- train: calendar year 2021;
- validation: calendar year 2022;
- protected test: calendar year 2023.

Optimization configs set `evaluate_test_split=false`. The test range exists in
the manifest so a later frozen promotion run can reproduce it, but L1/L2
candidate selection cannot evaluate or consume it.

The first phase is a local component search analogous to predictor Phase 1. It
does not claim the final weekly-retrained annual protocol; promotion remains
blocked until the weekly walk-forward gate evaluates the selected component.
