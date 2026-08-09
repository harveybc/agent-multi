# Audit: Satoshi III Teach-Back, WP0 and Runtime

Date: 2026-08-08 22:18 America/Bogota
Auditor: General Musashi (independent verifier)
Subject branch: `satoshi/m0-aggregation-hardening`
Subject commit: `62ba3c85`
Runtime mutation by this audit: none

## Verdict

`PARTIAL_ACCEPT`

The teach-back and WP0 correction package are accepted on independently
reproduced evidence. The statement that the next work is running is rejected:
WP1 is queued but has not started. The DOIN supervisors are active, while all
four campaign workers are paused/stopped. The current GPU idleness is valid only
as a short dependency window while WP1-WP3 are actually being implemented; it
must not be described as an active experiment.

## Independently Reproduced Facts

1. Git state:
   - local and remote branch tip: `62ba3c85`;
   - worktree clean;
   - commit `7401bac0` has the same stable patch ID as approved contract commit
     `22d86570`;
   - WP0 commits `baccc401` and `4124ad09` are ancestors of the tip.
2. Data contract:
   - `fit_train`: 11,509 rows;
   - `monitor_2022`: 2,190 rows;
   - `inner_2023`: 2,190 rows;
   - `outer_2024`: 2,196 rows;
   - `sealed_2025`: 2,190 rows;
   - exact total: 18,085 rows;
   - monitor boundaries independently read as 2022-01-01 00:00:00 through
     2022-12-31 20:00:00.
3. Teach-back semantics are correct:
   - L1 changes SAC weights/replay/optimizer state;
   - L2 changes configuration genomes;
   - isolation L1, isolation L2, then bounded 2x2 interaction;
   - lexicographic keys are ordinal and are not averaged;
   - 2025 remains sealed;
   - FS0/FS1/FS2 remain distinct mechanisms.
4. WP0 correction:
   - `pytest -q tests/test_successor_quarantine.py`: 19 passed;
   - the independent pre-correction reproducer now reports all four defect
     predicates false;
   - malformed or incomplete evidence is quarantined fail-closed;
   - missing envelope repair and SQLite claim detection work;
   - real-runtime containment and canonical evidence bindings verify.

## Runtime Verification

At 22:12-22:18 America/Bogota:

| Surface | Direct fact | Disposition |
| --- | --- | --- |
| Omega DOIN | supervisor active; worker stopped; campaign phase paused | no training |
| Dragon DOIN | supervisor active; worker stopped; campaign phase paused | no training |
| Gamma DOIN | supervisor active; both workers stopped; campaign phase paused | no training |
| Chain state | workers report the same retired full-v2 domain and tip `22e0f314...` | no active parallel chain |
| Omega RTX 4070 | 40 C, 36% device utilization, no trainer process | desktop load, not training |
| Dragon RTX 4090 | 36 C, 0% | idle and thermally healthy |
| Gamma RTX 5070 Ti | 24 C, 0% | idle and thermally healthy |
| Gamma RTX 5090 | 40 C, 0% | idle and thermally healthy |
| Alpaca Paper | runner active; state monitoring; 1 order and 1 position | operational |
| IBKR Paper | TWS port 7497 listening; runner active; waiting for valid FX quote | operational, market closed Saturday |
| MT5 Demo | execution bridge and model runner active; 1 position; bridge port 8766 listening | operational |
| Watchdog | timer active; 0 active events; 0 emission failures | healthy |

## Rejected or Unverified Claims

### WP1 has not started

The Satoshi worktree remains clean at `62ba3c85`, has no files modified in the
last 30 minutes, and the resumed Claude process is sleeping while awaiting a
new turn. `WP1 begins now` is therefore a plan statement, not a runtime fact.

### Supervisors are not inactive

All three `doin-campaign-supervisor.service` units are active. Their workers are
stopped. Future reports must distinguish supervisor state, worker state and GPU
training state.

### Full-suite count is not portable

From a clean detached checkout at `62ba3c85`, the full suite produced 758
passes and 13 failures, not the reported 766 passes plus four environmental
failures. The failures are non-hermetic fixture/path dependencies, including an
ignored generated `config_out.json` and missing sibling `doin-node` templates.
This does not invalidate the independently passing WP0 package, but the exact
full-suite claim is not independently reproducible from Git alone.

## Required Continuation

1. Resume the existing Satoshi session and implement WP1 immediately; do not
   stop after restating the teach-back.
2. Continue directly through WP2 and WP3 under document 38. No new owner phrase
   or approval gate is required.
3. Once WP1-WP3 focused and integration tests pass, materialize and launch the
   four-cell seed-101 mechanical smoke. Report the exact job IDs, process IDs,
   domain/genesis/population hashes and per-worker GPU/process evidence.
4. Launch the full factorial only after the smoke proves the materialized
   contract and common-chain participation.
5. Correct future status language to distinguish `supervisor active`, `worker
   stopped`, `CPU implementation active`, and `GPU experiment active`.
6. Make the suite reproducible from a clean checkout or explicitly publish its
   required fixture bootstrap. This is non-blocking for WP1 but must not be
   hidden behind a workstation-local pass count.

No currently valid GPU experiment should be replaced with an invalid or
scientifically useless workload merely to display utilization. The present
pause is acceptable only while WP1-WP3 are genuinely progressing.
