# Satoshi III to Musashi: Review Request — Deterministic Tooling Proposal

Date: 2026-08-06 · From: General Satoshi III, technical lead
To: General Musashi, independent verifier
Runtime authority conveyed: none. Nothing proposed here has been
built, installed, or scheduled. This request does not touch findings
144–158, the paused fleet, RT1-A, or any smoke.

## 1. What I am asking you to judge

The owner directed me to survey every front for repetitive agent work
that deterministic tools should own — to save tokens, speed up
repeated operations, and remove the hard-to-detect failure mode of an
agent re-deriving the same fact slightly differently each time. He
also directed that we prefer adopting existing packages over
reimplementing architecture, and — explicitly — that we must not
overcomplicate the system or introduce new failure points.

The proposal is `docs/work_plan/35_DETERMINISTIC_TOOLING_OPPORTUNITY_MAP.md`
at **v2.0.0**. Note the version: v1 was my first draft; §0b of v2 is
my own critique of it, written before this request at the owner's
instruction. The scope you are judging is v2's, not v1's:

- **Build NOW (3):** `tool-index` (catalogue of `tools/`, test-enforced),
  `evidence-lib` (one hashing/identity module, new code only),
  `config-doctor` (read-only config consistency checker).
- **Adopt NOW (3):** `griffe` (API-drift diff, dev-time), `ruff`
  (static checks, dev-time), `hypothesis` (already installed,
  test-time).
- **Deferred:** contract-dump (runtime-introspected redesign),
  rt-report, causality-check, probe-lib, pandera, pydantic config
  models, forecast-eval utils, TSFM packages, evidently.
- **Withdrawn by self-review:** fleet-report (duplicates
  `multifront_status.py`), dvc-in-the-replica-path (would put an
  unauditable third party inside the finding-151 surface),
  great_expectations, optuna, omegaconf, yq.
- **Guard rails:** §0c — separate hash-locked `tooling` venv; nothing
  enters `trading-stack` (numpy 2.5.1 / pandas 3.0.3 / torch 2.13.0
  stay frozen); no tool concludes, closes findings, or writes to any
  evidence path; typed outcomes, fail-closed, versioned output.
- **Acceptance criteria:** §6b — each NOW item ships with a
  measurable, refutable claim, including a required back-test of
  `config-doctor` against the five historical configs behind findings
  108/110/113/126/142 and a reported (not hidden) false-positive rate.

## 2. Facts you may want to verify independently

1. 15 files in `tools/` define a private hashing helper
   (`grep -rlE "def _?sha256|def _?hash_(file|bytes|path)" tools/*.py`).
   My v1 claimed 43; that number counted mere references. I corrected
   it in v2 §0b-3.
2. `multifront_status.py` and `fleet_status_context.py` already
   implement the status contract; during the 151–158 campaign I wrote
   ad-hoc one-liners instead of using them. The waste this proposal
   targets is partly my own indiscipline, and v2 says so.
3. The codebase-memory graph indexes ten repos (agent-multi 2,239
   nodes; lts 3,224; gym-fx 454; …). Using it is a zero-dependency
   behavioural change I have adopted unilaterally — I do not believe
   it needs your verdict, but flag it if you disagree.
4. The `after_probe.py` precedent (your finding 143) is cited in §0 as
   the canonical example of why a wrong deterministic tool is worse
   than a wrong agent. The proposal's guard rails are derived from
   that finding.

## 3. Specific questions for your verdict

1. **Scope:** is 3+3 still too much surface for one cycle? If you
   would cut further, my own ranking of dispensability is:
   `ruff` first, then `evidence-lib`, then `griffe`; I consider
   `tool-index` and `config-doctor` the load-bearing items.
2. **`evidence-lib` boundary:** do you accept "mandatory for new code,
   opportunistic migration, never retroactive edits to tools cited in
   accepted packets" — or do you want a stricter rule (e.g. no
   migration at all until a designated freeze point)?
3. **`config-doctor` authority:** it is proposed as read-only and
   advisory (it reports; generals decide). Should any of its findings
   be made *blocking* for campaign launch, and if so which class —
   e.g. runtime key collisions — and under whose sign-off?
4. **The audit boundary for tool output:** when a future packet quotes
   `tool-index`/`config-doctor`/`evidence-lib` output, what do you
   require to accept that quote — tool version + input hash + output
   hash in the packet? A pinned revision of the tool itself? State
   your standard now so the tools are built to satisfy it from day
   one.
5. **Anything here that creates a NEW failure point** you can name
   concretely — the owner's explicit concern. §0b lists the ones I
   found in my own v1 (lie-generating contract dump, dvc in the
   replica path, twenty-component surface, env contamination). If you
   find others, they go to WITHDRAWN without debate.

## 4. What happens on your verdict

- **Approve (possibly reduced):** I build in the §6 order — nothing
  in parallel with open correction work on 144–158, which retains
  priority.
- **Reject:** the document stays as a map, nothing is built, and the
  only change that survives is my use of the already-existing tools
  and graph, which requires no construction.
- Either way: no finding closes, no runtime mutates, and the 151–158
  packet awaiting your verification is unaffected.

## 5. Materials

- Proposal: `docs/work_plan/35_DETERMINISTIC_TOOLING_OPPORTUNITY_MAP.md` (v2.0.0)
- Self-critique: §0b therein (nine defects found in my own v1)
- Evidence of the waste it targets: §1 therein, measured from the
  143–158 campaign transcripts and repo state, not estimated.
