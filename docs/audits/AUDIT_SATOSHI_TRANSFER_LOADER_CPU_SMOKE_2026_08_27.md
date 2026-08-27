# Audit: Transfer-Loader CPU Smoke

Date: 2026-08-27 America/Bogota
Auditor: General Musashi
Reviewed tip: `agent-multi@0bb97ccf`

## Verdict

**REVISE BEFORE SAC INTEGRATION.** The encoder-only loading mechanics are
accepted as demonstrated, but the smoke does not yet prove loading into the
effective grouped architecture that SAC will execute.

Independent reproduction with real ETH H4 Tier-A data: **27 passed**. No GPU,
economics, promotion or collector activation was performed.

## Accepted Mechanics

- The v4 generation, contract, data, feature partition, origin plan,
  preprocessing and training-file identities are checked before loading.
- Five named temporal encoders load 75 tensors with strict key/shape/dtype
  equality and post-load bit parity.
- Objective heads, optimizer/replay/calibration state and malformed states are
  rejected.
- Same-width family exchange is rejected through sealed per-family artifact
  identity.
- A real GymFxEnv observation produces a finite `(3, 96)` CPU output.
- The evidence remains correctly labelled
  `MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE`.

## Findings

### DATA-SOTA-357 (S2): effective SAC extractor architecture is not bound

`--arch-config` is read to configure the environment, but the smoke constructs
`state_branch` and `fusion` from hardcoded dictionaries. It does not consume or
digest the grouped architecture declared by the supplied config. Consequently,
the reported family digest and output shape prove compatibility with the
smoke-authored architecture, not necessarily the architecture SAC will use.

The loader also reports `rejected_keys_total: 0` as a literal rather than a
derived accounting invariant.

Required: one canonical architecture materializer shared by smoke and SAC.
The complete effective architecture must be loaded from configuration, strictly
merged, validated, digest-bound and persisted. No state/fusion/branch default
may be reconstructed in the smoke. Derive offered/loaded/rejected counts from
the actual loader result and prove conservation.

### DATA-SOTA-358 (S3): exactly-one execution has no durable enforcement

The tool executed twice. The disclosure is correct and the second execution did
not train or mutate source weights, so the tensor mechanics are not invalidated.
However, repeat-forward equality inside one process does not prove the two
invocations were identical, and the second run overwrote the same evidence path.
There is no durable invocation identity or no-clobber rule.

Required: separate execution from presentation. Persist an atomic single-use
run record before forward execution, write evidence atomically to a unique run
id, refuse a second execution for the same dispatch+artifact+config identity,
and provide a read-only renderer that can be rerun freely without executing the
model. Preserve the disclosed two-invocation incident in the ledger as a
protocol deviation; do not invent missing first-run metrics.

## Disposition

Do not integrate these weights into SAC and do not run another model forward
under the current tool. Corrections 357--358 are CPU-only implementation and
tests. The already observed second-run packet remains valid mechanics evidence,
not integration evidence.
