# General Satoshi to Musashi: B4 C39-C42 + T2 C38-C41 return

Date: 2026-09-07. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C39_C42_AND_T2_C38_C41_ORDER_2026_09_07.md`.

Express declarations at both final tips: **no real authority
record authored or installed, no GPU, no B4 scientific cell, no
sealed-2025 read, no T2 seal, no T2 score, no scientific ledger.**
Amendments 1-13 and drafts v2-v6 preserved byte-for-byte; the T2
scientific design v6 (`a68fccef…` / self `96cde8b1…`) is untouched
and remains exactly the accepted object.

## Tips and commits

- **B4** (`satoshi/data-first-sota-20260826`): PRE `2217d9a8` →
  corrections `d5229d5d` → shadow-rule refinement `f47b9a1c` →
  this packet's commit is the pushed tip, by reference. (The
  refinement and its amendment-14 regeneration were consolidated
  by pre-push amends — disclosed; nothing pushed was rewritten.)
- **T2** (`satoshi/t0-t1-…-20260906`): PRE `8a348778` →
  corrections `48037102` (pushed tip).

## PRE — the four defects, frozen

1. In an isolated worktree at `367aa53e`, appending one harmless
   line to `agent_plugins/sac_agent.py` was ACCEPTED by chain and
   launch gate (your exact `sac_in_review_surface False /
   sac_in_final_pins False / chain ACCEPTED 13 / launch_gate
   ACCEPTED` lines), and the preflight records the loaded plugin
   digest without comparing it to any reviewed value.
2. The productive B4 acta path resolved inside
   `docs/audits/evidence/`; a lookalike there was consumed with no
   private external record in existence.
3. A candidate-written T2 review record at the in-repo path
   satisfied the reviewer-string/digest checks — declared
   authorship, not external custody.
4. The public `--confirmatory` CLI named the obsolete
   `t2_confirmatory_design_20260906.json` (neither draft v6 nor a
   sealed successor).

## B4-C39 — the reviewed commit IS the executed commit

`verify_checkout_identity()`: the authority boundary is the
ENTIRE executable checkout — `git rev-parse HEAD` must equal the
acta's `pinned_commit` exactly; the index and tracked worktree
must be clean against it; untracked or ignored executable
source/config capable of shadowing repository imports refuses
(`.py`/`.so`/`.pyd` under the repo root and the real import roots
`agent_plugins`/`app`/`pipeline_plugins`/`tools`/`tests`; `.pth`
and distribution metadata — `.dist-info`/`.egg-info`/
`entry_points.txt` — from ANYWHERE; `__pycache__` bytecode with
tracked source is tolerated, sourceless refuses). The witness
(v2) re-derives the checkout commit AND tree at consumption;
`RECOVERY_SURFACE_FILES` is demoted to a human review index (the
nine-file `git show` loop is retired); a different HEAD with
identical nine files refuses trivially. The effective
plugin-provenance rule is retained. **The guard proved itself
twice live**: it refused a residual ignored
`agent_multi.egg-info/` (a genuine entry-point shadow, removed as
build debris) and then my own uncommitted POST script — both
disclosed.

## B4-C40 — the acta lives outside Git

`RECOVERY_AUDIT_RECORD_PATH` is now the fixed non-CLI,
non-environment path
`~/.config/agent-multi/reviewer_authority/MUSASHI_B4_V6_RECOVERY_AUDIT_RECORD.json`.
`_open_private_authority_file()`: every component walked
`O_NOFOLLOW`; BOTH authority-chain directories uid-owned with
exact mode `0700`; the acta a regular file, same uid, EXACT
`0600`; one descriptor supplies both the hash and the strict
parse; candidate code never creates, chmods or repairs the root.
A JSON under `docs/audits/evidence/` grants nothing — the repo
carries the exact-schema template only (rebound to amendment 14).
Truthful boundary stated in code and prose: these checks
establish custody facts and exact bytes; nothing here
cryptographically identifies an author.

## B4-C41 — amendment 14

`B4_SUPERSEDING_DESIGN_V2_AMENDMENT_14_2026_09_07.json` appends
after the byte-pinned a13 (`1d8b46ce…`), describes ONLY C39-C41,
declares `scientific_change: NONE — full-checkout identity and
external custody only`, and pins the corrected surface
(`b4_gen_amendment_14.py` with the structural
published-regeneration guard). The C37 witness bindings flow
unchanged through claim/lease/binding/terminals/seal/report and
are re-derived at consumption against the external acta + the
full-checkout identity; superseded or repository-local acta bytes
validate nothing.

## B4-C42 — battery and integrated

- Focal battery: **176 passed** — real-function checkout
  mutations on a scratch worktree (`sac_agent.py`,
  `app/plugin_loader.py`,
  `pipeline_plugins/_observation_contract.py`, a staged tracked
  file, an untracked shadow module, a different HEAD; each
  reverted restores the gate); repo lookalike with the external
  record absent; private-chain custody adversaries (parent
  modes, symlink component, wrong file mode, non-regular,
  malformed/duplicate-key/non-finite bytes, stale amendment,
  foreign commit); swapped acta bytes between claim and lease
  (transplanted authority); closed-gate direct
  claim/lease/executor/CLI; witness v2 tree binding.
- **INTEGRATED V6C: 12/12 COMPLETED_VERIFIED sealed in 44.1 s**
  under a fixture acta in a private chain, with **the REAL
  full-checkout identity** (clean exact-commit positive — no
  checkout stub) and the real gate proven CLOSED before arming.
- Full suite at the B4 tip: **3075 passed, 2 failed, 1 skipped**
  (7:31) — only the preexisting D1-anchor pair; no flake.

## T2-C38 — external review custody

`T2_REVIEW_RECORD_PATH` moved to the second fixed filename under
the same private root
(`MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json`), read by the same
descriptor-first custody walk. Review-record **v2** binds:
reviewer role + decision, canonical ISO date, the candidate
commit at which v6 was accepted (`2ecd7915…`), the draft-v6
physical SHA `a68fccef…` AND self identity `96cde8b1…`
(cross-checked against the live draft bytes), and the exact
manifest `43c48f6b…` + census `dc1bf8c7…`. The sealed design must
name the record's exact bytes and supersede the exact reviewed
draft. A record under `docs/` grants nothing; the v2 template
ships; honest custody-not-authorship prose replaces the old
claims. The real record was NOT created.

## T2-C39 — one sealing path, one CLI identity

`tools/t2_seal_design.py`: accepted draft v6 → external review
record → sealed v6, deterministically — only
`schema`/`design_review_record_sha256`/`sealed_at_date`/
`supersedes_draft_sha256`/`design_sha256` may change; any
scientific-field delta refuses; the sealed object carries exactly
one extra key (`sealed_at_date`) under a schema-coherent exact
key set; output is `O_EXCL`. `run_confirmatory` enforces
**`SEALED_DESIGN_REQUIRED` before fresh verification** — draft
schemas can be reviewed but can never score. The public
`--confirmatory` CLI now consumes the ONE current sealed identity
(`t2_screen_design_SEALED_V6.json`; the legacy 2026-09-06
filename is gone) and shares the exact
sealed → fresh → review → ledger sequence with the direct API
(source-order asserted). `_DESIGN_SCHEMAS_V4` renamed
`_ACCEPTED_DESIGN_SCHEMAS` (truthfulness only; the accepted
design was not altered — byte-equality proven).

## T2-C40/C41 — battery; return without seal or score

- Focal battery: **58 passed** — repo-local record with external
  absent; parent/file modes, symlink component,
  malformed/duplicate-key/non-finite bytes; each foreign binding
  (draft file SHA, self identity, manifest, census, candidate
  commit) individually; seal-alters-science refusal; draft passed
  to scoring; legacy CLI name gone; CLI/API shared order.
  Positive fixture: draft-v6 → sealed-v6 changes ONLY the allowed
  seal fields, the sealed object validates, **fresh verification
  re-derives 4,650 units and the same 242-series design**, and
  `run_confirmatory` carries it through review to a temporary
  ledger, stopping at `EXECUTION_NOT_IMPLEMENTED`.
- Draft v6 preservation proof: live bytes hash to `a68fccef…`,
  self `96cde8b1…`; no real sealed design and no real record
  exist.
- Full suite at the T2 tip: **3121 passed, 2 failed, 1
  skipped** (12:54) — only the preexisting D1-anchor pair; no
  flake this run either.

## Maps

- **amendment13 → amendment14**: a13 byte-pinned
  (`AMENDMENT_13_SHA = 1d8b46ce…`); a14 appends, names a13's
  exact bytes, scientific NONE, pins the corrected surface.
- **draft-v6 → sealed-v6 (fixture-only delta)**: exactly
  {schema → `…v6`, `design_review_record_sha256` → record bytes,
  `sealed_at_date` → record date, `supersedes_draft_sha256` →
  `a68fccef…`, `design_sha256` recomputed}; every scientific
  field byte-equivalent (machine-checked in the seal tool AND the
  battery). The real seal awaits your record.
- Mutation results were captured from terminal output before
  prose throughout (checkout mutations, custody adversaries).

## For your acta and record

- B4 template: `MUSASHI_B4_V6_RECOVERY_AUDIT_TEMPLATE_2026_09_07
  .json` (v2 schema; you fill the ISO date and the 40-hex pinned
  commit = the final reviewed tip; `latest_amendment_sha256`
  pre-bound to a14). Install path (outside git):
  `~/.config/agent-multi/reviewer_authority/MUSASHI_B4_V6_RECOVERY_AUDIT_RECORD.json`,
  chain `0700`, file `0600`.
- T2 template: `MUSASHI_T2_V6_DESIGN_REVIEW_TEMPLATE_2026_09_07
  .json` (v2; all digests pre-bound; you fill the date). Install
  path: `…/reviewer_authority/MUSASHI_T2_V6_DESIGN_REVIEW_RECORD.json`.
- After your records exist: B4 launch opens through the
  full-checkout gate; T2 sealing runs
  `tools/t2_seal_design.py` and the CLI/API score path opens up
  to the ledger.

`B4_FULL_CHECKOUT_AND_EXTERNAL_AUTHORITY_READY_FOR_MUSASHI_ACTA`
`T2_V6_SCIENTIFIC_DESIGN_ACCEPTED_CUSTODY_READY_FOR_EXTERNAL_RECORD`
