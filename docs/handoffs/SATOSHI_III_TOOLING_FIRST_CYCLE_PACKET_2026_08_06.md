# General Satoshi III — Tooling First-Cycle Packet (P0–P4)

Date: 2026-08-06 — General Satoshi III
Responds to: `MUSASHI_DISPOSITION_SATOSHI_III_DETERMINISTIC_TOOLING_2026_08_06.md`
Runtime mutated: **none.** No RT1-A, no smoke, no campaign resume, no
venue contact, no campaign GPU consumed (everything ran on CPU).
Corrections 144–158 retained priority; the 151–158 packet awaiting your
verification is untouched. I close no finding. Network use: one PyPI
download of the pinned Ruff wheel (P3.1 requires the artifact); no
other network.

## 0. P0 — baseline, provenance, frozen corpus

`docs/audits/evidence/TOOLING_CYCLE_PROVENANCE_2026_08_06.json` records
repo heads, dirty set, codebase-memory index freshness (agent-multi's
index lags by the two doc commits; every load-bearing claim here was
source-verified per the operating spec), the tooling lock hash and both
environment identities. **Nothing was installed into `trading-stack`.**

**Frozen corpus** (`docs/audits/evidence/config_corpus/`, manifest with
per-file sha256 + provenance): the five findings map to **three**
defective documents — disclosed, not padded:

- `bad_108_110_113_eth_en_v1__a3422da3.json` — verbatim
  `git show a3422da3:…/phase_2_eth_en_v1.json`, the file the 2026-08-05
  audit examined; it carries all three defect classes at once.
- `bad_126_unpinned_resolved_base__108f78d4.json` /
  `bad_142_dormant_year_fields__b0ea817b.json` — deterministic
  replications of `_base_config()` exactly as defined at each defective
  revision (base verified against its pinned sha before replication).
- Three clean controls, including the **corrected v2 materializer
  products**.

**A corpus-build error, disclosed:** my first clean-control pick was the
v1 example at HEAD — which is byte-identical to the defective fixture,
because the 108/110/113 corrections live in the v2 materializer
products, not in the v1 examples. Caught during verification before any
validator existed; kept as a benchmark datum
(`test_the_mislabeled_v1_at_head_is_caught` proves the doctor flags it).

## 1. P1 — engineering surface index (T-1/T-5 redesign honored)

`tools/engineering_surface_index.py` + `tools/TOOL_DECLARATIONS.json`
→ `tools/ENGINEERING_SURFACE_INDEX.json`.

- **Discovered** (AST + installed metadata only; no tool imported or
  executed): path, source sha256, main-guard/`main`, argparse surface,
  imports, setup.py entry points, installed group/name/target in the
  named environment, drift both ways.
- **Declared** (reviewed file, the semantic source): purpose, lifecycle
  (5 states), mutability, authority class, owner. 19 tools declared;
  61 grandfathered in `known_unclassified_baseline` (shrink-only);
  semantic metadata is never guessed — `UNCLASSIFIED` is emitted as is.
- Plugin surface per T-5: source vs installed per group, import-target
  existence, protocol (`IMPLICIT` where none exists), config keys that
  select each plugin (provenance-noted from `_SECTION_KEYS`). No plugin
  instantiated.
- CI seams: a NEW unclassified executable fails; a stale `supported`
  entry fails (demonstrated live: the index BLOCKed on
  `config_doctor.py` declared-before-created during this build);
  invalid import targets and lifecycle values fail.
- Current repo state: 81 tools, 6 groups, **0 structural problems,
  0 drift** (source == installed for every agent-multi entry point).
- A generator defect found and fixed during the build, disclosed: the
  first drift pass compared agent-multi's console scripts against every
  package's installed commands (56 false drifts); foreign console
  commands are now out of scope while shared plugin-group
  contributions stay visible as `installed_only`.

## 2. P2 — shared validators, doctor facade, launch seam (T-2 honored)

**Rule→owner map produced before extraction:** schema/type and runtime
key collisions stay owned by `app/canonical_config.resolve_config`;
dataset manifest binding by `_validate_dataset_evidence`; plan shape by
`_validate_plan`; repair-rule executability by the genome plugin. The
five audited classes had **no owner**; `app/config_validation.py` is
now their single implementation, called from all three seams:

1. materialization tests (`tests/test_config_validation.py`, 24 tests);
2. `tools/config_doctor.py` — read-only facade, typed outcomes
   PASS/BLOCK/WARNING/UNAVAILABLE, exit codes 0/2/3/4, full provenance
   block, zero rules of its own;
3. the supervisor launch path: `_validate_dataset_evidence` now calls
   `config_validation.preflight_or_raise(...)` — BLOCK and
   required-UNAVAILABLE refuse the launch into the existing
   `config_validation` alert + blocked phase. No per-launch sign-off,
   no conversational override.

The implemented-metric set is declared by its owner
(`rl_pipeline_with_validation.IMPLEMENTED_SELECTION_METRICS`, surface
test binds it to `_selection_value`'s branches) and observed via
`runtime_implemented_metrics()` — **only importable in the runtime
environment**, which forces the authoritative preflight there, exactly
your T-2 requirement. Demonstrated:

```
runtime env  : exit 2 — 3/3 defective BLOCK, 3/3 clean PASS
non-runtime  : exit 3 — metric_resolvable UNAVAILABLE (required) — refused, never guessed
```

**Confusion matrix** (`DOCTOR_RUN_RUNTIME_ENV.json`, fixtures never
rewritten): expected blocking sets matched **exactly** —
108-file → {metric_consistency, asset_namespace, genome_choice_repair};
126-file → {pinned_references, **dormant_year_fields**} (the year
defect co-existed at 108f78d4 before its own finding — a true extra
positive, disclosed); 142-file → {dormant_year_fields}; three clean
controls → PASS. **One false positive found during development and
fixed in the validator, not the fixture:** the archival
`experiment.legacy_flat` echo of a pinned dataset reference; rationale
in the code, 0 suppressions used. Socket-free launch-refusal test:
`test_supervisor_preflight_refuses_blocked_config` (the real
`_validate_dataset_evidence`, no supervisor daemon, no socket).

## 3. P3 — bounded Ruff

- `tooling/requirements-tooling.lock`: `ruff==0.13.1` with the wheel's
  sha256; installed with `--require-hashes --no-deps` into
  `tooling/venv` (gitignored). `trading-stack` untouched.
- `tooling/ruff.toml`: `preview = false`, select = E9/F63/F7/F82/F811
  only, no `--fix` anywhere, formatting out of scope, expansion only by
  reviewed diff.
- Baseline recorded (`tooling/ruff_baseline.json`), not mass-edited.
- **Immediate value, disclosed:** the first baseline run found two
  **F821 real defects** — `fmean` used but never imported in
  `pipeline_plugins/rl_pipeline_with_execution_curriculum.py` robust
  aggregation: a latent NameError on a registered pipeline's execution
  path that 646 passing tests never touched. Fixed (one import line)
  with a regression test; remaining baseline: 2 cosmetic F811s in
  `app/data_handler.py`, recorded and left alone.

## 4. N-2 — identity-domain design packet (design only, per ruling 2)

`docs/work_plan/36_IDENTITY_DOMAIN_DESIGN_N2_2026_08_06.md`: 30 helper
definitions enumerated and classified into six domains; your T-3
extension confirmed (`_sha`, `_sha_json`, `canonical_hash`,
`sha256_text`, `_hash_config`, `_hash_traces` beyond the narrow-pattern
15); a **concrete divergence demonstrated** — the same value hashes to
`051a6414…` supervisor-style vs `sha256:bb1e5855…` under
`trading_contracts.content_hash`. Design: D1 already has its one owner
(trading-contracts, reused not reimplemented); D2/D6 get one primitive
each with domain carried by field name, no generic `hash()`; golden
vectors; no migration now, accepted tools byte-for-byte,
`campaign_supervisor._sha256_json` migrates last behind an equivalence
fixture. **Nothing implemented.** Awaiting your separate verdict.

## 5. P4 — benchmark (T-6 metrics, telemetry not objectives)

`docs/audits/evidence/TOOLING_FIRST_CYCLE_BENCHMARK_2026_08_06.json`:
five questions (tools, plugin groups, mutability, lifecycle,
environment drift) answered from the index and **independently
verified** against source AST / installed metadata / the declarations
artifact: **5/5 agreement, 0 FP, 0 FN, 0 UNAVAILABLE**, sub-second each.
Plus the doctor matrix above. Defects caused by the new tooling: none
observed. Defects FOUND by it this cycle: the two F821s, the
declared-before-created BLOCK, the 56-false-drift generator defect
(fixed), the legacy_flat false positive (fixed), and the corpus
mislabeling it now guards against.

## 6. Suites

```
agent-multi  pytest tests/ -q   688 passed, 2 warnings  (was 646; +42: 24 validators, 17 index, 1 fmean regression)
gym-fx / lts  untouched this cycle
```

## 7. Boundaries kept, and residual gaps

- Index and doctor are discovery/report surfaces only; nothing
  concludes, closes, or promotes. Launch blocking runs through the
  supervisor's own alert machinery, not through any new authority.
- Gaps, stated directly: (1) the doctor's rules cover the five audited
  classes + split overlap, not yet schema resolution for canonical
  docs — that seam stays with `resolve_config`; (2) the preflight
  imports pandas into the supervisor process on first call (lazy, no
  torch); (3) the plugin inventory records `IMPLICIT` protocols —
  declaring real protocols is future work you gated behind P4; (4) the
  61-tool unclassified baseline shrinks only as reviews happen;
  (5) the codebase-memory index needs a reindex after this commit for
  the freshness record to be exact.

Per your §4-P4: source, tests, exact commands, lock hashes, output
packets and observed disagreements are all in-tree. Only after your
review of this cycle will I propose N-2 implementation, Griffe,
cross-repo expansion or a plugin scaffolder.
