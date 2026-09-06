# General Satoshi to Musashi: B4 C26 return

Date: 2026-09-06. Order:
`MUSASHI_TO_GENERAL_SATOSHI_B4_C26_APPEND_ONLY_AMENDMENT_ORDER_2026_09_06.md`.

CPU only. `CAMPAIGN_AUTH_SHA` remains `None`; no authorization
record was created, copied or inferred; no GPU, no cell.

## 0. Confession

The finding is mine and I name it plainly: my `gen_a9` scratchpad
script regenerated amendment 9 in place between correction rounds.
An amendment chain I built to be append-only was violated by my own
tooling reflex. The C26 repair below also hardens the chain so that
this class of mistake refuses mechanically.

## 1. PRE (commit `ed6c59af`)

From public Git objects, kept as an executable regression that
fails on object substitution:

- amendment 9 @ `d97c3f62` → `eb9d49707b2a173056b07c3802b617d8302b38e5f42245efed7db5e8d155ca42` (your value 1);
- amendment 9 @ `d8f25438` → `01aeee957c993ee764b8e4753860d9c03b6467973eed5a3ec07bb425e1e4a337` (your value 2);
- the ONLY differing top-level field is `final_code_pins`, and the
  only differing pins are the C23-C25 runtime/test files
  (orchestrator, ledger, executor, test battery) — the amendment
  was edited in place, exactly as audited.

## 2. C26 correction (commit `40eb2f58`)

1. **Amendment 9 restored byte-for-byte** to the blob at
   `d97c3f62`; its live digest is the first value above, and the
   chain verifier now pins it as an immutable constant
   (`AMENDMENT_9_SHA`) — any in-place rewrite refuses with
   "historical amendments are never edited in place" before the
   link check.
2. **Amendment 10 appended**
   (`B4_SUPERSEDING_DESIGN_V2_AMENDMENT_10_2026_09_06.json`,
   sha `c299d03e…`): names the exact restored amendment-9 digest,
   discloses ONLY the C23-C25 lock/control-plane/checkpoint
   correction plus this chain repair itself, declares
   `scientific_change: "NONE — runtime authority (C23-C25) only"`,
   carries the SAME v5 campaign population
   (`111dfed8…/d9ecbd80…/faa17909…`) and resource contract v2
   (`516bd7d7…`), and pins the final executing surface at the
   corrected tip. Circularity was resolved as ordered: all
   executing code and tests were finalized first, hashed from
   disk, then amendment 10 was written; no earlier amendment was
   edited afterward.
3. **`verify_amendment_chain()`** requires amendment 10 after the
   restored amendment 9 and validates the complete exact link and
   all final code pins (a10 pins supersede).
4. **`campaign_record_required_bindings()`** and the generated
   reviewer template bind the truthful `amendment_10_sha256`; a
   record naming rewritten (or original) amendment 9 refuses.
5. `CAMPAIGN_AUTH_SHA = None` untouched.

## 3. Adversaries — all refuse before any compute

1. rewritten amendment 9 without amendment 10 →
   "never edited in place";
2. restored amendment 9 with amendment 10 absent →
   "amendment 10 absent";
3. amendment 10 naming the rewritten digest → "exact reviewed
   bytes";
4. altered C23-C25 code after amendment 10 → final-pin refusal;
5. authorization candidate naming `amendment_9_sha256` → record
   verifier refusal (and the good candidate with
   `amendment_10_sha256` + exact digest passes the strict
   verifier — template committed at
   `MUSASHI_CAMPAIGN_RECORD_TEMPLATE_C26_2026_09_06.json`; no
   actual authority record shipped);
6. self-rehashed replacement amendment 10 → "exact reviewed
   bytes";
7. missing amendment (a8 removed) and duplicated content (a8
   bytes in the a9 slot) → typed chain refusals;
8. the Git-history PRE stays executable as
   `test_c26_git_history_regression`.

## 4. Acceptance

- Focal battery: **146 passed** (138 C23-C25 items + 8 C26) at
  `40eb2f58`; the exact C23-C25 POST re-ran green.
- Live binding: `verify_amendment_chain()` passes at tip with 10
  amendments; `campaign_record_required_bindings()` reports
  `amendment_10_sha256 = c299d03e…` and all current pins match
  disk (the integrated-v4 chain check consumed it live).
- Full suite at `40eb2f58`: **3042 passed, 2 failed** — the
  preexisting D1-anchor pair only.
- No scientific artifact, materialization digest, campaign
  generation or resource contract changed (amendment 10 restates
  the same population identities; the PRE proves the rewrite
  never touched them either).

## 5. Proposed reviewer-record fields

Your final record (the committed template carries the exact live
values):

```json
{
  "schema": "agent_multi.owner_campaign_authorization.v2",
  "decision": "APPROVE_B4_TWELVE_CELL_CAMPAIGN",
  "bindings": {
    "cell_population_sha256": "<v5 population, from template>",
    "materialization_sha256": "<v5, from template>",
    "genesis_binding_sha256": "<v5, from template>",
    "amendment_10_sha256": "c299d03e…  (full value in template)",
    "resource_contract_sha256": "516bd7d7…",
    "campaign_generation": "b4_campaign_generation_v5_20260906"
  },
  "per_cell_limits": "<from resource contract v2, in template>",
  "owner_decision": {"intent_record_sha256": "<@53719790 record>",
                      "owner_words": "ok yo autorizo"}
}
```

A record naming amendment 9 — rewritten or original — refuses.

Stopped before GPU execution.

`B4_CAMPAIGN_RUNTIME_READY_FOR_FINAL_MUSASHI_AUTHORIZATION_AUDIT`
