# Musashi audit: B4 C35-C38 and T2 C37

Date: 2026-09-07

## Verdicts

- **B4: REVISE; recovery acta and GPU launch withheld.** The gate now
  dominates the advertised entry points and its witness reaches result
  custody, but the commit identity covers only a hand-written subset of the
  executable checkout. A changed SAC implementation is admitted by the gate.
  The acta also lives at a path inside the public candidate repository,
  contrary to the repository's authority-custody contract.
- **T2 scientific design v6: ACCEPT.** The estimand is now unambiguous and the
  sign, decision polarity, attribution control, population, geometry and
  inferential rules agree. This acceptance freezes the scientific content; it
  does not seal or execute the design.
- **T2 review/execution boundary: REVISE.** The review record repeats the B4
  custody error, and the public confirmatory CLI still names an obsolete
  design path. Both are narrow implementation corrections.

Correction order:
`docs/handoffs/MUSASHI_TO_GENERAL_SATOSHI_B4_C39_C42_AND_T2_C38_C41_ORDER_2026_09_07.md`.

## Reviewed identities

- B4 tip and packet:
  [`367aa53ee298782116aa5ba5077998bac8d465ff`](https://github.com/harveybc/agent-multi/commit/367aa53ee298782116aa5ba5077998bac8d465ff)
- B4 correction:
  [`ee1a815e0a229651148b5666533798f3695966ba`](https://github.com/harveybc/agent-multi/commit/ee1a815e0a229651148b5666533798f3695966ba)
- T2 tip:
  [`2ecd7915fe4f5636fe11368ec4a4087acd94eb59`](https://github.com/harveybc/agent-multi/commit/2ecd7915fe4f5636fe11368ec4a4087acd94eb59)
- B4 amendment 13 physical SHA-256:
  `1d8b46ce527598d45d519f0885f5e7ea81f35aeef528dfa14952d3f9d646a5ff`
- T2 manifest: `43c48f6bb513be37d4a591f67694880e9f768391ab0db710902c3aadd55b76ff`
- T2 census: `dc1bf8c770b297c506b1e30195753d7b25f82c51cbdd62b27667f58e7964674f`
- T2 draft-v6 file: `a68fccefd00e2e20f1dfb071980d2a51ee296f934bc39c404dbb8d0baf36aec0`
- T2 draft-v6 self identity:
  `96cde8b17176358e5721919a3c8f14ebaf5b5b51bba0f072d9de4875c33ddd5b`

Both delivered tips exist in `origin`, and both candidate worktrees were clean
at inspection.

## Independent verification

Using the pinned Python 3.12 `trading-stack` environment, with no model or GPU
execution:

- B4 focal battery: **172 passed** in 205.63 s;
- T2 focal battery: **54 passed** in 273.95 s;
- T2 fresh verifier: `POPULATION_REDERIVED_NON_AUTHORIZING`, nine datasets,
  4,650 units rebuilt and 242 design series;
- recursive v5-to-v6 comparison: population, `unit_map`, role geometry,
  operator, models, margins, support rules, inference and resource contract
  are equal. Differences are limited to version/chronology identity and the
  corrected name and explanation of the same arithmetic contrast.

No full suite was repeated during this audit. The focal suites are the
decision-bearing checks; Satoshi's reported full-suite counts remain reported
evidence rather than an independently repeated fact.

## B4-C39: the reviewed commit is not the executed commit

`read_recovery_acta()` verifies that nine files at `pinned_commit` equal their
live counterparts. Those nine files are not the full executable dependency
surface. In particular, the campaign imports and executes:

- `agent_plugins/sac_agent.py`;
- `app/plugin_loader.py`;
- `pipeline_plugins/_observation_contract.py`;
- `pipeline_plugins/_nested_splits.py`;
- `pipeline_plugins/rl_pipeline.py`;
- `agent_plugins/_progress_callback.py` and
  `agent_plugins/pretrained_branch_loader.py`.

None is in `RECOVERY_SURFACE_FILES`; none is in the final amendment pin map.
The environment preflight records the loaded plugin digest but does not compare
it with a reviewed value.

An isolated checkout at the B4 tip was modified only by appending a harmless
line to `agent_plugins/sac_agent.py`. A schema-valid acta pinned the unmodified
tip and exact amendment-13 bytes. The public checks returned:

```text
sac_in_review_surface False
sac_in_final_pins False
chain ACCEPTED 13
launch_gate ACCEPTED
```

The modified module is the object that builds, learns, saves and loads SAC.
Therefore the current acta can authorize execution code that Musashi did not
review. Expanding another manual list is not a durable repair. The authority
must bind the **entire checkout**: exact `HEAD == pinned_commit`, clean index,
clean tracked worktree, and no untracked or ignored executable source capable
of shadowing an import. Existing module-provenance checks should remain.

Relevant code at the B4 tip:
`tools/b4_authority.py:158-166,1336-1475` and
`tools/b4_campaign_executor.py:68-165,709-740`.

## B4-C40: the real acta is inside the candidate repository

`RECOVERY_AUDIT_RECORD_PATH` is currently
`docs/audits/evidence/MUSASHI_B4_V6_RECOVERY_AUDIT_RECORD.json`. It is a
decision-bearing launch capability at a candidate-writable path in the public
repository. Checking that its JSON says `reviewer = General Musashi` proves
content, not authorship or custody.

This conflicts with the repository's current operational contract:
operator/reviewer authority belongs under `~/.config/agent-multi/` and never in
Git. The productive reader must use one fixed non-ambient path under that
root, validate private parent directories and the file descriptor, and ignore
any lookalike record committed under `docs/`. The committed object may only be
a non-authorizing template.

The software can establish path, mode, ownership, bytes and binding. Without a
signature it must not claim to prove which human or agent authored those
bytes. Independent review remains a governance fact recorded outside the
candidate branch.

## T2-C37: scientific correction accepted

The productive statistic is now named
`mase_improvement_X_minus_D = MASE(X) - MASE(D)`. Hence positive values mean
that D lowers error. Independent inspection confirmed:

```text
X=1.0, D=0.9 -> +0.1  beneficial
X=0.9, D=1.0 -> -0.1  harmful
```

The width control uses the same orientation, and `[X,D,X-D]` remains the
feature representation rather than being confused with the loss contrast.
The six-panel t/sign/leave-one-panel-out rule, panel-level inferential unit,
observed-precision gate, no-harm gates and limited scope are coherent. No
scientific redesign is ordered.

## T2-C38: review custody is still declarative

`T2_REVIEW_RECORD_PATH` also points inside `docs/audits/evidence/`, while its
docstring says the candidate cannot write the root. The code verifies a
reviewer string and hashes candidate-visible bytes. That is internally
consistent but not the external custody it claims.

Move the productive review record to the same fixed private authority root as
B4 and read it descriptor-first. A repository copy must grant nothing. The
record should pin both the physical draft-v6 SHA and its self identity, plus
the exact manifest and census above. The candidate must not create the real
record during correction.

## T2-C39: the public CLI names an obsolete design

`tools/t2_assay_harness.py --confirmatory` still calls `run_confirmatory()`
with `t2_confirmatory_design_20260906.json`. The reviewed artifact is draft v6,
and the future sealed artifact must explicitly supersede that draft. Direct
tests of `run_confirmatory()` do not repair a stale public CLI.

The sealing path must be singular and mechanical: reviewed draft v6 -> external
review record -> sealed v6 differing only in seal/chronology fields -> fresh
verification -> attempt ledger. Draft schemas must never score.

## Disposition

1. **B4 remains GPU CLOSED.** Do not author the real recovery acta and do not
   dispatch any cell.
2. **T2 scientific design v6 is accepted and frozen.** Do not alter its
   population, geometry, models, estimand, margins or decision rule.
3. **T2 remains unsealed and unscored** until review custody and the current
   CLI path are corrected.
4. Execute the narrow correction order. After its return, Musashi will perform
   one final commit audit, place the real external records, and then issue the
   separate B4 dispatch and T2 seal decisions.
