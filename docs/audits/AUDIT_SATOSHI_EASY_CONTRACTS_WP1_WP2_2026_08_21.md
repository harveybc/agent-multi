# Audit: Easy Monitor/Fitness WP1-WP2

Date: 2026-08-21 America/Bogota
Target: `agent-multi@d9888aef`
Verdict: **CORRECTION REQUIRED; WP3 NOT YET RELEASED**

## Accepted Facts

- The accepted 22-epoch smoke cannot be reconstructed completely. Partial
  console notes are auxiliary evidence, not a replacement for missing durable
  history. The four-seed replication is the correct source for the formal rank
  study.
- The smoke report now persists its complete epoch history.
- The two proposed contracts have distinct identities and the isolated
  hierarchy tests pass (`17 passed`).
- The three-epoch CPU artifact demonstrates an actual rank disagreement.

## Findings

### EC-01 (S2): neither contract governs an executing consumer

Outside tests and documentation, both functions are referenced only by their
definition module and the offline rank-study tool. The training checkpoint/
patience path does not call `easy_checkpoint_monitor`; a DOIN materializer,
selector or result publisher does not call `easy_doin_candidate_fitness`.
Launching WP3 would therefore train under the previous executing contract and
calculate the new ranking only afterward.

### EC-02 (S3): the rank tool fabricates missing evidence

`int(vv_tr or 0)` maps a missing count to zero and accepts numeric strings or
fractional values by coercion. Missing returns and drawdowns similarly become
`0.0`. The independent reproducer proves that a missing trade count, string
count and missing validation return all exit successfully. This repeats the
already-corrected pre-coercion defect family.

### EC-03 (S3): no executing-path provenance or OLAP contract exists

There is no call-path proof, persisted contract identity/decomposition in the
executing experiment record, or refusal when the runtime is configured with
the old scalar. The requirement that both contracts govern their respective
real consumers is consequently unimplemented.

### EC-04 (S4): WP1 records are incomplete for the ordered analysis

The CSV omits return, drawdown and economic component columns as first-class
fields and does not preserve source references. JSON embedded in one column is
not sufficient for independently re-deriving every rank from bound evidence.

## Correction And Automatic Dispatch

1. Remove every pre-coercion/default in `rank_disagreement_study.py`; pass raw
   values into strict validators. Missing, boolean, string, fractional,
   negative and non-finite counts must refuse. Missing/non-finite economic
   facts must refuse.
2. Wire `easy_checkpoint_monitor` into the actual checkpoint and patience
   update path. Prove invocation from a bounded real training run.
3. Wire `easy_doin_candidate_fitness` into the actual DOIN candidate
   materialization/selection/publication path. An episodic-easy experiment
   configured with the legacy scalar must refuse before training.
4. Persist both contract identities, components, lexicographic key, source
   references and input hashes in the OLAP/result record.
5. Extend the rank CSV with all raw inputs and decomposed outputs required for
   independent recomputation. Continue to exclude test mechanically.
6. Run this reproducer before and after, focused tests, executing-path tests and
   the full suite.

When EC-01 through EC-04 no longer reproduce, **WP3 is automatically
released**. Dispatch the four seeds immediately without another owner phrase.
Audit runs in parallel and must not create an idle wait after correction.

Reproducer:

```bash
python docs/audits/evidence/EASY_CONTRACTS_WP1_WP2_REPRO_2026_08_21.py
```
