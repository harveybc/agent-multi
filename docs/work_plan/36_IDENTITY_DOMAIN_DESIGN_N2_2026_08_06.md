# 36. Identity-Domain Design (N-2) — design packet, no implementation

Status: DESIGN ONLY — v1.0.0, 2026-08-06. Implementation awaits a
separate Musashi verdict per ruling 2 of
`MUSASHI_DISPOSITION_SATOSHI_III_DETERMINISTIC_TOOLING_2026_08_06`.
Author: General Satoshi III.

## 1. The measured problem

30 identity-helper definitions exist across `app/` and `tools/`
(enumerated by `grep -rnE "def (_?sha...|_?hash_...|canonical_hash|
content_hash|source_tree_digest)\(" tools/*.py app/*.py`; the three
name-collision false matches `_sharpe`, `_shared_validator`,
`_shared_population_seed` excluded). Musashi's T-3 observation is
reproduced and extended: beyond the 15 files matching the narrow
pattern, helpers also hide under `_sha`, `_sha_json`, `sha256_text`,
`canonical_hash`, `_hash_config`, `_hash_traces`, `source_tree_digest`.

**Concrete divergence, demonstrated:** for the same value
`{"b": 1, "a": [1.0, 2]}`,

- `app/campaign_supervisor._sha256_json`-style (default separators)
  → `051a6414c441795e…`
- `trading_contracts.content_hash` (compact separators, normalized,
  `sha256:` prefix) → `sha256:bb1e5855a74b2579…`

Two "canonical JSON hashes" of identical semantic content disagree.
This is the exact mechanism behind the finding-130/141/149/151 family.

## 2. Domain classification of every existing helper

| Domain | Byte contract (today, observed) | Existing members |
|---|---|---|
| **D1 canonical structured-content** | normalize → compact sorted ASCII JSON → sha256, `sha256:`-prefixed hex | `trading_contracts.content_hash` (the declared owner); DIVERGENT re-implementations: `campaign_supervisor._sha256_json`, `rolling_origin_adaptation._sha_json`, `audit_test_evidence.sha256_json`, `project3_evidence_pool._sha256_json`, `swarm_telegram_watchdog.canonical_hash`, `audit_snapshot_collector.canonical_hash`, `update_registry._hash_config` |
| **D2 raw file-byte** | sha256 of file bytes, bare hex | `_sha256_file` ×4 (`campaign_supervisor`, `weekly_promotion`, `multifront_status`, `project3_stageb_run_plan`), `_sha256` ×7, `_sha` ×2, `sha256_file` ×2, `sha256` ×1 — 16 definitions, all byte-identical semantics, 16 copies |
| **D3 ordered-collection** | file-path→hash map over a sorted tree walk | `eth_curriculum_decision_experiment._hash_traces` |
| **D4 source-tree** | HEAD + tracked diff digest + untracked content digest (`--untracked-files=all`, post-155) | `rolling_origin_adaptation.source_tree_digest` (sole owner, correct home) |
| **D5 artifact-manifest** | schema-versioned JSON binding artifact hash + contracts + evidence | anchor manifest (finding 158), `ARCHIVE_MANIFEST.json`, corpus manifest — emitted ad hoc, no shared emitter |
| **D6 text/content fragments** | sha256 of UTF-8 text, bare hex | `social_intelligence.sha256_text`, `project3_*._sha256_bytes` ×3 |

## 3. Design

1. **D1 has exactly one implementation and it already exists:**
   `trading_contracts.content_hash`. Every D1 re-implementation above
   is a migration target, none is a template. The `sha256:` prefix is
   part of the D1 contract and marks the domain in stored records.
2. **D2/D6 get one shared primitive each** (`file_sha256`,
   `bytes_sha256`) in `trading_contracts` (proposed home — it is the
   only shared package), bare-hex by historical compatibility: every
   existing artifact record stores bare hex, and re-prefixing would
   orphan the archives. The domain is carried by the FIELD NAME
   (`*_sha256` = D2/D6, `*_content_hash` = D1), never inferred from
   the value shape.
3. **No generic `hash(value)` is ever exposed.** Each primitive takes
   the type its domain declares (`Path`, `bytes`, JSON-compatible
   value); mixing domains becomes a type error at the call site.
4. **D3/D4/D5 stay where their semantics live** (runner/manifest
   emitters), each importing D1/D2 primitives instead of hashlib.
   D5 gains a tiny shared `emit_manifest(schema, payload)` /
   `verify_manifest(path)` pair only after D1/D2 land.
5. **Golden test vectors:** a `tests/vectors/identity_domains.json`
   with input → digest pairs per domain, frozen at implementation
   time; any change to a primitive that alters a vector is a
   compatibility break and needs a schema-version bump plus a
   migration note.
6. **Compatibility rule:** new records carry a `hash_domain` or the
   domain-marking field name; readers accept old bare records
   unchanged; nothing rewrites stored digests.

## 4. Migration boundary (per ruling 2 — restated, binding)

- **No migration now.** Accepted-packet tools stay byte-for-byte.
- New code MUST import the shared primitives from day one of an
  approved implementation.
- Existing tools migrate only at a designated freeze point, each with
  a before/after identity-equivalence proof (same inputs → same
  digests, or a documented schema bump).
- `campaign_supervisor._sha256_json` is the highest-risk migration
  (its digests live in persisted campaign state); it migrates LAST,
  behind its own equivalence fixture over recorded state files.

## 5. What implementation approval would authorize (and nothing else)

Two functions in `trading-contracts` (D2/D6 primitives), the golden
vector file, and lint enforcement ("no new `def _sha*`/`hashlib`
call outside the primitives") for NEW files only. Estimated diff:
under 150 lines plus tests. No behavior change to any existing
digest anywhere.
