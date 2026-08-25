# Satoshi to Musashi: final SOTA contract corrections (F1-F4)

Date: 2026-08-25
Order: `MUSASHI_TO_GENERAL_SATOSHI_SOTA_FINAL_CONTRACT_CORRECTION_ORDER_2026_08_24`
after `AUDIT_SATOSHI_SOTA_CORRECTIONS_C1_C5_2026_08_24.md`
(PARTIAL_ACCEPTANCE_REVISE_BEFORE_MATERIALIZATION). All CPU/docs; P1
runtime untouched; no screen materialized.

## F1 — observation identity reconciled (SOTA-C01)

Full evidence:
`docs/audits/evidence/FINDING_SOTA_C01_OBSERVATION_IDENTITY_2026_08_25.md`.
Reproduced 84-vs-83 with `typical_price` PREPENDED at index 0 (whole
ordering shifted); traced to `wp4_cpu_smoke.py` deriving features from
the CSV header minus `_EXCLUDE` and never consuming the system
manifest; exposed a SECOND divergence — executed
`include_price_window=false` vs declared `true` (2,692 vs 2,724 vs the
documented 2,660: three different shapes, none agreeing).
`check_observation_identity` refuses pre-model (real-fixture negative
test); `executed_observation_identity` labels terminal records; doc 38
§23.5 binds sealing (never rewrite as 83).

**Decision request (F1.5)**: recommend post-P1 adopts NEW identity
`ethusdt_4h_l1_system_v2` = declared 83 list, price window false,
flattened 2,660; P1 stays diagnostic at executed 84. Awaiting your and
the owner's ruling; no silent inheritance either way.

## F2 — cadence separated from compute (SOTA-C02)

Doc 40 rev 3: causal Screen R now fixes TOTAL updates at 260,000
gradient steps per scored year allocated across refreshes (weekly
5,000 × 52; daily ≈712 × 365; 12h ≈356 × 730); the fixed-5,000/refresh
design moved to a distinct OPTIONAL operational screen `R-op
(cadence_plus_compute)` compared on value per GPU-hour; all covariate
language removed; no causal claim may cite R-op.

## F3 — temporal and release schemas hardened (SOTA-C03/C04)

`post_p1_screen_contract.py` v3: ISO-parsed dates (malformed REFUSED);
`fit_data_end <= origin.fit_end` enforced (the audited bypass fixture —
trained after fit boundary, before score start — now REFUSED);
declared per-origin `selection_boundary` enforced; `validate_origins`
(ordered, non-overlapping, internally sane); report-only companions
restricted to a typed ALLOWLIST {name, metrics, series_sha256, notes,
decision_authoritative} (authority smuggling via any other key
REFUSED); finalist must carry artifact/config/code/ensemble-rule
digests (missing REFUSED).

## F4 — evidence claims corrected (SOTA-C05/C06)

Validator v3: self-describes `coverage: heuristic_lint` in docstring
and PASS output (the 127 count is "claims detected by the heuristic");
NEW numeric-table coverage — a Markdown table with numeric cells must
carry an inline ref or an immediate Fuente line (file 06's previously
unsourced table caught and bound). Doc 41: block length computed on
the control arm's OWN per-bar net return series ALONE — single series,
no differential, no candidate data (SOTA-C06).

## Tests (all focused, reproduction-before → correction-after)

38 passed: `test_post_p1_screen_contract.py` 21 (fit-boundary bypass,
malformed dates, selection boundary, origin overlap/order, authority
smuggling, missing finalist digests, real 84/83 drift, order drift,
flag drift), `test_sota_validator.py` 11 (incl. numeric table without
source REJECTED, heuristic-coverage label), `test_warmup_context_probe.py` 6.
Registry validator: PASS {coverage heuristic_lint, files 9, sources 31}.

## Explicitly not done

No B/A/R/C materialization; no GPU dispatch; P1 monitored, not
altered (9/12 arms accepted at packet time: 303 all, 404 all, 101 N+EN-W, 202 EN-W; running: 101 EN-F easy phase, 202 EN-F; queued: 202 N). The rule-only baseline
formula unit tests authorized by the verdict are NOT yet written —
queued behind this packet if you want them before your reproduction.
