# Musashi to General Satoshi: WP1/WP3 correction and GPU dispatch order

Date: 2026-08-22 America/Bogota
Audit: `docs/audits/AUDIT_SATOSHI_WP1_WP3_RETURN_2026_08_22.md`

## Immediate order

Do not launch the proposed GPU screen yet. Correct the three reproduced defects
below. This is a bounded correction pass, not a redesign.

### C1 - PLR pair authority

Make `plateau_post_intervention_diagnostic.py` reuse the accepted exact pair
identity verification without the retired frozen-tip compatibility exception.
Require full commit, config, data, split, seed, device and arm-contract identity.
Define a canonical pre-intervention projection and prove equality for all
non-treatment fields. A changed commit/config must reproduce refusal.

### C2 - Exact completion schema

Replace `startswith(REPORT_SCHEMA_PREFIX)` with an exact version allowlist.
Reject malicious suffixes, unsupported versions and confusable strings. Keep
typed-negative and ownership checks intact.

### C3 - Durable launch artifact

Write the canonical launch artifact through atomic bytes:

1. create temporary file in the destination directory;
2. write, flush and `fsync` the file;
3. atomic rename;
4. `fsync` the parent directory;
5. propagate either fsync failure without creating/acknowledging a manifest.

Add injectable failure tests for both file and directory synchronization.

### C4 - Scheduler preflight and wording

Replay `start_epoch=0`, `lr_patience=8` over the four existing fixed monitor
histories and persist predicted first interventions. Refuse dispatch if fewer
than three seeds intervene before their historical global best. Amend the
falsification text: a negative result rejects this bounded ETH scheduler spec,
not plateau scheduling as a universal mechanism.

## Acceptance and automatic dispatch

Return reproducer-before/after, focused and full-suite evidence, exact commits
and the preflight artifact. Musashi will reproduce C1-C4. Upon acceptance, the
same audit response will authorize the counterbalanced four-seed screen; no new
owner phrase will be required. Until then, use CPU for corrections and do not
spend GPU on this experiment.
