# Evidence Packet: Three-Venue Runtime Audit (P0) and K0 Review

Date: 2026-08-03 America/Bogota (samples extend past 2026-08-04T02:28Z UTC)
Verifier: Satoshi III (Mujuro Utsutsu), independent verifier for this packet
Order: `MUSASHI_TO_SATOSHI_III_RUNTIME_AUDIT_AND_OKF_GBRAIN_PILOT_2026_08_03.md`
Broker actions by this audit: none — no order placed, cancelled or altered;
no broker session opened; all ledger reads used SQLite `mode=ro`
This packet closes nothing. K1-K4 are NOT claimed (see §5).

## 1. P0 — Findings 079-085: reproduction and verification

Head audited: `lts@6daf85e` (clean, synced). Reproduction used an isolated
worktree at the common parent `8b67235` with the head fixtures copied in
(worktree removed afterward):

- `test_mt5_execution_bridge.py` at parent: **6 failed / 6 passed** —
  including `test_execution_status_reconciles_protected_model_position`
  and `test_execution_status_refuses_altered_or_foreign_position`
  (findings 080/081/082/083 bridge surface absent or wrong at parent).
- `test_alpaca_l1.py` at parent: 084's exact fixture
  `test_terminal_broker_effect_reconciles_l0_and_unblocks_next_signal`
  **fails**; 7 pre-existing pass.
- `test_paper_execution_watchdog.py` at parent: **ImportError**
  (`read_execution_runtime` does not exist) — the 083/085 watchdog
  correction surface is provably absent at the parent.
- 079 verified textually: `git show ebdfec5` shows the missing `+`
  string-concatenation operator before the closing JSON brace in
  `BarJson()`, now repaired, with a source regression assertion.

At head: focused suites (bridge, watchdog, alpaca_l1, mt5_model_runner)
**40 passed**; complete LTS suite **544 passed** — matching 085's claim.

## 2. P0 — Direct runtime resample (read-only)

Sampled 2026-08-04T02:21-02:28Z UTC:

- Services: `lts-ibkr-model-runner` **active**, `lts-alpaca-model-runner`
  **active**; the consolidated watchdog is timer-driven —
  `lts-paper-execution-watchdog.timer` fired at 02:26:39Z with the next
  run scheduled ~5 min later (`inactive` between oneshot runs is its
  normal state, not a defect).
- Heartbeats: IBKR and Alpaca both `monitoring`, seconds-fresh, venue- and
  model-labeled (`usdcad-4h-linear-live-v1`).
- **Open exposure reconciliation (order item 4):** exactly one open
  position exists — IBKR Paper short **25,000 USD.CAD**
  (`exp-oi2-rsv-f4993c2dda8cdc2a`), account fingerprint matching the
  profile, model-bound idempotency key
  (`usdcad-4h-linear-live-v1:2026-08-03T20:00Z:2026-08-04T00:00Z`),
  cumulative fill 25,000 applied once, `position_reconciled: true`, with
  **146 periodic position-reconciliation facts** and passing ack verdicts
  in the journal — protection is being re-verified continuously, not
  assumed. Alpaca is flat with a clean `terminal_flat` lifecycle.
- The journal also retains the complete first-canary history (recovery
  hold/cancel/flatten/reconciled facts and one
  `operator_reconciliation_after_completed_order_fix` fact), append-only.
- Restart/replay non-duplication (item 5): the live heartbeat's `resumed`
  block shows the open effect re-classified `acknowledged` from durable
  facts with 3/3 call attempts/results and **zero new broker calls**; the
  replay/restart property suites in the 544 pass at head.
- Adversarial paths (item 6) and state distinctions (item 7): covered by
  the head fixtures (wrong account/instrument/protection/count mismatch in
  the bridge, watchdog and L1 suites; effect and lifecycle state machines
  refuse illegal promotion).

### Explicit unknowns (P0)

- **MetaEditor zero-error compile (079)** happens inside the Windows VM; I
  could not execute it from Omega. The running v2 execution bridge implies
  a successful compile, but per doctrine implication is not proof — the
  next Dragon-side sample should capture the compile/version handshake
  directly.
- **No direct Dragon/MT5 sample was taken by this session**; MT5 evidence
  here derives from the corrected fixtures and the consolidated watchdog
  freshness, not from a first-hand bridge OLAP read.

## 3. P0 — Owner-disposition table (recommendations only; I close nothing)

| Findings | State | Recommendation |
| --- | --- | --- |
| 069-074 | independently verified (Musashi, A-E/F0 audits) | eligible for owner closure |
| 075-078 | independently verified (my packet `AUDIT_SATOSHI_III_MULTI_VENUE_CONTINUITY_VERIFICATION_2026_08_03.md`) | eligible for owner closure |
| 079 | fixtures + source verified; VM compile = explicit unknown | owner closure after one direct MT5 version/compile sample |
| 080-085 | reproduced on parent, verified at head, runtime resample consistent | eligible for owner closure on Musashi's concurrence |

## 4. K0 — Pins, collisions, threats, verdict

**Pins (exact revisions and licenses):**

- OKF: `GoogleCloudPlatform/knowledge-catalog` @ `main`
  `3fcbb9f828c2f23d109c855ee403c3a4c81f3a96`, Apache-2.0. SPEC v0.2
  confirmed; only `type` is mandatory frontmatter — therefore OKF
  provenance/verification/freshness semantics are CONVENTIONS our bundle
  must enforce itself; generic Markdown import cannot be claimed as
  compatibility (doc 31's suspicion confirmed). A deterministic
  frontmatter adapter + validator is required in K1 regardless of GBrain.
- GBrain: `garrytan/gbrain` @ `master`
  `82fe0216ff04e4b1e898a1062d3abe6487fa8383`, MIT, TypeScript on Bun,
  PGLite embedded option (fits the local-engine requirement), 30+ MCP
  tools over stdio and HTTP, dream cycle + autonomous enrichment present
  and MUST be disabled per the order.
- Hermes installed on Omega: **v0.12.0 (2026.4.30)** at
  `~/.local/bin/hermes`.
- Issue `NousResearch/hermes-agent#23997`: **closed** via PR #50117. Cron
  `enabled_toolsets` silently rejects `gbrain`/`mcp:gbrain`; the internal
  alias is `mcp-gbrain`. K4 must test whether v0.12.0 contains the fix or
  needs the `jobs.json` alias workaround; a silent-rejection failure mode
  is exactly the "cron tool disappearance" threat the order names.

**Install-path inspection:** `INSTALL_FOR_AGENTS.md` instructs an agent to
retrieve and follow remote instructions from a raw URL — a fetch-and-obey
agent supply-chain injection surface. It will NOT be executed, per the
order and per my own judgment. K2, if it proceeds, installs from the
pinned revision with a frozen lockfile, with postinstall scripts inspected
and no autonomous cycle enabled.

**Collision matrix (compact):** Git/work-plan Markdown stays the ONLY
canonical narrative truth (GBrain is derived and rebuildable, never
cited as evidence); `codebase-memory-mcp` answers code-structure
questions, GBrain answers approved-knowledge questions — disjoint scopes,
both discovery-only; OLAP/DOIN blockchain/direct broker evidence are
never ingested; Hermes consumes read-only with write/admin tools
disabled. No replacement of any existing truth source.

**Threat model → mitigations:** memory poisoning (import only the
validated OKF bundle; no autonomous enrichment; dream cycle off), stale
synthesis (freshness metadata + K3 adversarial stale/contradictory
corpus where the required behavior is reporting the conflict), secret
ingestion (K1 validator scans prohibited secret/account patterns before
any import), unauthorized canonical writes (no GBrain write path to Git;
MCP write tools disabled for Hermes), exposed endpoints (stdio/loopback
only, HTTP listener disabled), dependency supply chain (pinned revision,
frozen lockfile, postinstall inspection — REMAINING GATE, done at
install time), cron tool disappearance (issue 23997 check in K4),
single-index failure (K4 stop-and-fallback drill; canonical files always
sufficient).

**Verdict: PROCEED-WITH-REVISIONS.** K1 (OKF bundle + deterministic
validator) has no external dependency and may start immediately in the
bounded lane. K2 is conditionally approved: pinned `82fe0216`, PGLite,
stdio-only, all autonomous features disabled, frontmatter adapter tested,
and the lockfile/postinstall inspection passing at install time — any
surprise there stops the install and reports, per §3.6 of the order.

## 5. Not claimed

K1-K4 are not implemented in this packet. They begin in the bounded
CPU/local lane per doc 31, only in windows that cannot delay broker
reconciliation, the DOIN campaign or machine health.

*Ritsurei.* The live blade is clean: one authorized short, continuously
reconciled, and every correction the General forged holds under
independent steel. The knowledge lane may open — with the revisions named.

— Satoshi III (Mujuro Utsutsu)
