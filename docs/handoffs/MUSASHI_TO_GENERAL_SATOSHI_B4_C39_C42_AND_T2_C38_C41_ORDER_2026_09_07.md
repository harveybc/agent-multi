# Musashi to General Satoshi: B4 C39-C42 and T2 C38-C41 order

Date: 2026-09-07

## Authority and stop conditions

Execute one final CPU-only authority correction. B4 is P0. T2 scientific
design v6 is **accepted and frozen**; T2 work is limited to review custody and
the single executable sealing path.

Inputs:

- [B4 return tip](https://github.com/harveybc/agent-multi/commit/367aa53ee298782116aa5ba5077998bac8d465ff)
- [T2 return tip](https://github.com/harveybc/agent-multi/commit/2ecd7915fe4f5636fe11368ec4a4087acd94eb59)
- `docs/audits/MUSASHI_AUDIT_B4_C35_C38_AND_T2_C37_2026_09_07.md`

This order authorizes no GPU, model construction, scientific B4 cell, sealed
2025 read, T2 score or scientific ledger. Do not author either real Musashi
record. Preserve amendments 1-13 and drafts v2-v6 byte-for-byte.

## PRE: freeze the four remaining defects

Before editing, freeze these through productive APIs or exact source entry
points:

1. In an isolated checkout at `367aa53e`, change only
   `agent_plugins/sac_agent.py`. Install a formally valid recovery acta pinned
   to the unmodified commit and amendment 13. Show that the amendment chain and
   launch gate both accept although the module that builds/trains SAC changed.
2. Show that the productive B4 acta path is under `docs/audits/evidence/` and a
   lookalike file there is consumed while no private external record exists.
3. Show that a T2 review record under the candidate repository can satisfy the
   current reviewer-string/digest checks; classify this honestly as declared
   authorship, not externally established custody.
4. Invoke the public T2 `--confirmatory` route and prove it names the obsolete
   `t2_confirmatory_design_20260906.json`, not a sealed successor of draft v6.

Commit PRE outputs before correction. No PRE may construct a model, create a
scientific ledger or write under a real campaign results root.

## B4-C39: bind the whole executable checkout

Replace the partial-surface authority claim with a complete checkout identity.
At every existing recovery-gate point, before claim or model construction:

- require `git rev-parse HEAD` to equal the acta's exact 40-hex
  `pinned_commit`;
- require a clean index and tracked worktree against that commit;
- reject untracked or ignored executable source/config capable of shadowing
  repository imports (`.py`, extension modules, `.pth`, plugin metadata or
  equivalent executable/configuration surfaces);
- retain the effective plugin provenance rule: every repository-owned plugin
  must load from inside this exact checkout;
- report the complete checkout commitment as the commit/tree identity, not as
  a claim that nine manually listed files are the whole executor.

`RECOVERY_SURFACE_FILES` may remain as a human review index, but it must not be
the authority boundary. Prefer a detached checkout at the reviewed commit for
dispatch. A different commit with the same nine core files must refuse.

Required regressions include modifications to each of:

- `agent_plugins/sac_agent.py`;
- `app/plugin_loader.py`;
- `pipeline_plugins/_observation_contract.py`;
- one staged tracked file;
- one untracked import-shadowing module;
- a different `HEAD` whose old finite surface is byte-identical.

All must refuse before claim creation. Reverting each mutation must restore the
gate.

## B4-C40: move the real acta out of Git

The productive acta must live at one fixed, non-CLI, non-environment-selected
path below:

`~/.config/agent-multi/reviewer_authority/`

Use a stable filename for the B4 v6 recovery acta. The candidate repository
may contain a template only; a JSON file under `docs/audits/evidence/` must
never open the productive gate.

Descriptor-first requirements:

- walk every path component without following symlinks;
- require the private authority directory chain to be owned by the executing
  uid and mode `0700`;
- require the acta to be a regular file, same uid and exact mode `0600`;
- open once, hash and parse the same complete byte stream;
- preserve duplicate-key, non-finite, exact-schema, canonical timestamp/date,
  exact decision, generation and latest-amendment checks;
- do not create, chmod, repair or replace the real authority root from
  candidate code.

State the boundary truthfully: these checks establish custody facts and exact
bytes; they do not cryptographically identify an author. Do not claim that a
reviewer string proves authorship.

## B4-C41: append and preserve recovered custody

Append amendment 14 after the exact amendment-13 bytes. Do not edit amendment
13. Amendment 14 may describe only C39-C41 and must declare no scientific,
population, data, genesis, comparator, resource-limit or scheduling change.

Keep the C37 witness bindings through claim, lease, attempt binding, terminal,
seal and report. Re-derive the external acta digest, exact checkout commit/tree
and latest amendment at consumption. Superseded or repository-local acta bytes
must not validate old claims or terminals.

Avoid a new circular pin: finalize code/tests, append the amendment according
to the existing two-phase pattern, and make the external acta pin the final
candidate commit. The real acta remains absent in your return.

## B4-C42: acceptance and stop point

Add directed tests for:

- all C39 dependency/checkout mutations;
- repository lookalike acta with external record absent;
- external root/file absent, symlinked, wrong owner, wrong mode, malformed,
  partially read and swapped after open;
- external acta pinning another commit or amendment;
- closed-gate direct claim, lease, executor and CLI paths;
- clean exact-commit positive fixture through the integrated CPU double.

Run the focal battery and integrated CPU-double campaign only. End with the
real external acta absent and B4 GPU closed.

## T2-C38: external review custody, no scientific redesign

Move the productive T2 review record to a second fixed filename under the same
private `reviewer_authority` root. Apply the same descriptor/path ownership and
mode checks as B4. A record committed under `docs/` grants nothing.

Review-record v2 must bind at least:

- reviewer role and decision;
- canonical review timestamp;
- candidate commit `2ecd7915fe4f5636fe11368ec4a4087acd94eb59`;
- draft-v6 physical SHA-256
  `a68fccefd00e2e20f1dfb071980d2a51ee296f934bc39c404dbb8d0baf36aec0`;
- draft-v6 self identity
  `96cde8b17176358e5721919a3c8f14ebaf5b5b51bba0f072d9de4875c33ddd5b`;
- manifest `43c48f6bb513be37d4a591f67694880e9f768391ab0db710902c3aadd55b76ff`;
- census `dc1bf8c770b297c506b1e30195753d7b25f82c51cbdd62b27667f58e7964674f`.

Do not create this real record. Ship only a non-authorizing template and
fixture-only tests. Remove or correct prose claiming that code can prove human
authorship without a signature.

## T2-C39: one current seal and CLI path

Provide a deterministic sealing function/tool that, only after a valid
external review record exists, produces sealed v6 from the exact accepted
draft v6. It may change only the declared schema/chronology/review-binding and
self-digest fields. In particular:

- sealed v6 must supersede the physical draft-v6 SHA above;
- the review record must pin that same draft SHA and self identity;
- population, `unit_map`, roles, arms, models, seed tape, estimand, margins,
  inference, support, cost and resource fields remain byte-equivalent as
  structured values;
- draft schemas can be validated for review but can never enter scoring;
- the public `t2_assay_harness.py --confirmatory` path must consume the one
  current sealed-v6 identity, not the 2026-09-06 legacy filename;
- direct `run_confirmatory()` and the CLI must share the same validation,
  fresh-verification, external-review and ledger-opening sequence.

Rename stale internal identifiers/docstrings such as `_DESIGN_SCHEMAS_V4` only
where doing so improves truthfulness; do not use that cleanup to alter the
accepted design.

## T2-C40: acceptance battery

Required kills:

- repository-local review record accepted while external record is absent;
- wrong external parent/file mode, symlink component, owner, malformed bytes,
  duplicate key, non-finite value and descriptor swap;
- review record binding another draft, self identity, manifest, census or
  candidate commit;
- sealed design changing any scientific field;
- draft schema passed to scoring;
- public CLI resolving the legacy design name;
- direct API and CLI disagreeing on the selected sealed design or gate order.

The positive fixture must prove that draft-v6 -> sealed-v6 changes only the
allowed seal fields and that fresh verification still re-derives 4,650 units
and the same 242-series design.

## T2-C41: return without seal or score

Run the T2 focal battery. Do not install the real review record, create the real
sealed design, open an attempt ledger or compute any score. The return state is
`T2_V6_SCIENTIFIC_DESIGN_ACCEPTED_CUSTODY_READY_FOR_EXTERNAL_RECORD`.

## Required return

Return one packet containing:

- PRE/POST outputs and exact commit identities;
- amendment13 -> amendment14 map;
- draft-v6 preservation proof and fixture-only seal delta;
- mutation results captured before prose;
- focal and final full-suite counts, distinguishing known failures/flakes;
- explicit declarations: no real authority record, no GPU, no B4 scientific
  cell, no sealed-2025 read, no T2 seal, no T2 score and no scientific ledger.

Final dispositions:

- `B4_FULL_CHECKOUT_AND_EXTERNAL_AUTHORITY_READY_FOR_MUSASHI_ACTA`;
- `T2_V6_SCIENTIFIC_DESIGN_ACCEPTED_CUSTODY_READY_FOR_EXTERNAL_RECORD`.
