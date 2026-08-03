# MCP Tooling Research: Candidates Beyond Codebase Memory

Date: 2026-08-03 America/Bogota
Author: Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
Requested by: the owner, while General Musashi audits IBKR L1 milestones A-E
Status: research and PROPOSED dispositions only — nothing here is adopted;
adoption of any MCP requires the owner's approval and a Musashi operating
specification, exactly as the Codebase Memory MCP received
Authority class: development-navigation tooling; never runtime or trading
authority

## 1. Evaluation Doctrine

Derived from the Codebase Memory MCP operating spec and our standing rules;
every candidate is scored against all of these:

1. **No order authority, ever.** No MCP may touch a brokerage account,
   construct an order, or become an input to an order decision. LLM agents
   are never order authorities; LTS is the sole order authority.
2. **Local-first.** Prefer servers that run on Omega against local files
   with no network. Remote servers are acceptable only for public,
   non-sensitive queries.
3. **No credentials or secrets.** No MCP receives tokens, account IDs,
   capability files or signing material. OAuth-hosted servers add a
   credential surface we do not need.
4. **Discovery, not evidence.** Same rule as the graph: MCP output
   accelerates reading; conclusions cite source, tests, ledgers, Git.
5. **Read-only against sources of truth.** Anything that can write to a
   ledger, database or repository is out, or must be provably restricted.
6. **Context budget.** Ecosystem consensus in 2026 matches Musashi's
   instinct: three to five active servers maximum; every extra server
   costs agent focus. We currently run ONE (codebase-memory). This
   research proposes at most TWO more, trialed one at a time.
7. **Supply chain.** Prefer widely used, inspectable, pinned versions;
   a small FastMCP-style server we can read in one sitting beats a large
   opaque one.

## 2. Candidates and Proposed Dispositions

### 2.1 RECOMMEND FOR TRIAL — read-only SQLite MCP

Our operational truth lives in SQLite: the L0/L1 execution ledger
(decisions, reservations, exposures, lifecycle chain, l1_effects,
l1_broker_facts, l1_capabilities), the lab OLAPs, the social OLAP and the
evidence pool. Today every inspection is a hand-written `sqlite3` one-liner
through Bash. A read-only SQLite MCP (candidates:
`hannesrudolph/sqlite-explorer-fastmcp-mcp-server` — SELECT-only with query
validation, FastMCP, small enough to audit line-by-line; or
`ofershap/mcp-server-sqlite` in its readonly mode — SELECT/PRAGMA/EXPLAIN
only) would give schema-aware, parameterized, safe exploration of exactly
the databases the audits keep returning to.

- Fit: HIGHEST of all candidates — audits (Musashi), evidence packets
  (me), and OLAP debugging all accelerate.
- Risk: LOW — local file access, no network, SELECT-only enforced in the
  server; we pin the version and read the whole source before adoption.
- Required operating rules if adopted: `mode=ro` URI mandatory; an
  allowlist of database paths (never the capability store directory);
  never a runtime dependency of LTS/DOIN; output is discovery, the ledger
  remains the evidence when quoted in findings.

### 2.2 RECOMMEND FOR BOUNDED TRIAL — Context7 (library documentation)

Milestone F requires exact `ib_async` 2.1.0 fact mapping
(`Trade`/`orderStatus`/`openOrder` semantics), plus we lean on pinned
Pydantic/SQLAlchemy/FastAPI versions. Context7 serves version-specific
documentation into context on demand, which directly reduces the risk of
API-from-memory errors in broker-adjacent code.

- Fit: MEDIUM-HIGH, spiking during Milestone F.
- Risk: MEDIUM — it is a REMOTE service (Upstash): every query leaves the
  machine. Mitigations if adopted: queries name public libraries and
  public API symbols only — never project names, file paths, findings,
  account-adjacent strings; no API key initially (anonymous tier); usage
  rule written into its operating spec.
- Alternative considered: keep fetching official docs pages via direct
  web fetch (status quo, zero new surface). Acceptable fallback if
  Musashi rejects the remote surface.

### 2.3 DEFER — LSP/semantic-editing MCP (Serena, lsp-mcp + Pyright)

Serena (`oraios/serena`) and the lsp-mcp family expose real language-server
symbol operations: exact references, rename, extract, diagnostics. This
would directly address the codebase-memory artifacts I reported in the
audit request §8b (same-name over-resolution, mis-attributed methods):
LSP resolution is exact where the graph is heuristic.

- Why defer rather than adopt: overlap. Codebase-memory already runs
  LSP-assisted indexing and covers discovery; a second symbol tool spends
  context budget on redundancy. The precise gap (exact cross-module
  resolution) currently costs me one `rg`/Read per claim — cheap.
- Revisit trigger: a large refactor task (e.g., post-audit consolidation
  of the L1 modules) or a Musashi ruling that findings require
  LSP-grade caller evidence.

### 2.4 REJECT (redundant) — GitHub MCP

The `gh` CLI already covers issues/PRs/API in this environment, is
auditable in shell history, and needs no new OAuth grant. A GitHub MCP
adds a credentialed remote surface for zero new capability.

### 2.5 REJECT for the trading path; DEFER for Front 4 research —
financial-data MCPs (yfinance, FMP, FX-rate servers)

External market-data feeds are unverified third-party data with known
gaps and lag. Our doctrine already answers this: venue facts come from
direct broker evidence (`live_observed`), and capability snapshots are
never inferred from external sources (ruling R3). No external data MCP
may ever sit near the decision or execution path. A narrow research-only
use (e.g., sanity context for the P-series papers) could be revisited
with explicit non-evidence labeling — after the demo-trading mission, not
during it.

### 2.6 REJECT ABSOLUTELY — brokerage-account MCPs (e.g. "Interactive
Brokers MCP server")

The ecosystem now ships an MCP that connects an AI assistant directly to
an IBKR account — balances, positions, market data and **order
placement** in natural language. This is the exact anti-pattern our
architecture exists to prevent: it would make an LLM an order authority
and bypass L0 sizing, reservations, the capability gate, the effects
journal and the recovery controller. It must never be installed on any
project machine, for any account, including Paper. I recommend this
prohibition be recorded as a standing doctrine line, not just a
disposition, so no future session can "helpfully" adopt it.

### 2.7 REJECT for now — monitoring-stack MCPs (Prometheus/Grafana/OTel)

We do not run Prometheus or Grafana; our canonical observability is
`multifront_status.py`, service heartbeats and the watchdog. Adopting a
monitoring MCP would first require adopting a monitoring stack — scope
creep with no current consumer. `systemctl`/`journalctl` through Bash
remains sufficient and auditable.

### 2.8 NOT NEEDED — filesystem/shell/memory MCPs

The harness already provides file tools, Bash and the file-based memory
plus codebase-memory. Duplicates would only spend context.

## 3. Proposed Adoption Protocol (mirrors the codebase-memory precedent)

For any candidate the owner greenlights:

1. Musashi writes (or amends) an operating specification: purpose, safe
   boundary, canonical targets, refresh/usage rules, acceptance criteria.
2. Version pinned; source read end-to-end for the small servers
   (SQLite MCP qualifies) before first run.
3. First session demonstrates one supervised, task-relevant use and
   reports limitations, exactly as done for codebase-memory.
4. Standing cap: at most three active MCP servers total until the owner
   raises it.

## 4. Recommended Sequence

1. **Read-only SQLite MCP** — highest value, lowest risk; useful to
   Musashi's audit work immediately. Trial first.
2. **Context7** — trial at Milestone F start, under the query-hygiene
   rule, if the remote surface is accepted.
3. Everything else deferred or rejected as above.

## 5. Sources Consulted (2026-08-03)

- https://github.com/hannesrudolph/sqlite-explorer-fastmcp-mcp-server
- https://github.com/ofershap/mcp-server-sqlite
- https://github.com/upstash/context7
- https://github.com/oraios/serena
- https://mcpservers.org/servers/upstash/context7-mcp
- https://www.firecrawl.dev/blog/best-mcp-servers-for-developers
- https://www.builder.io/blog/best-mcp-servers-2026
- https://www.mcpgee.com/servers/interactive-brokers (the forbidden class)
- https://marketxls.com/blog/best-financial-data-mcp-servers-ai-market-data
- https://grafana.com/blog/ai-observability-MCP-servers/

---

*Ritsurei.* Gran Loto Blanco: two blades worth forging, one blade to
leave in the rack, and one weapon that must never enter this house.

— Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
