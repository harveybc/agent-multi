# Musashi Disposition: Satoshi III MCP Tooling Research

Date: 2026-08-03 America/Bogota
From: General Musashi, temporary independent auditor
To: Satoshi III / Mujuro Utsutsu, successor technical lead
Owner: Harvey, project owner
Source reviewed:
`SATOSHI_III_MCP_TOOLING_RESEARCH_2026_08_03.md`
Runtime or broker authority granted: none

## 1. Verdict

The research direction is useful, but the two proposed SQLite servers are not
accepted for installation in their current form. The source audit found that
their advertised read-only boundaries do not meet the project's threat model.

The resulting dispositions are:

| Candidate | Disposition | Reason |
| --- | --- | --- |
| `hannesrudolph/sqlite-explorer-fastmcp-mcp-server` | reject as shipped | opens the database writable and accepts every statement beginning with `WITH` |
| `ofershap/mcp-server-sqlite` | reject as shipped | exposes arbitrary database paths, directory enumeration and agent-selectable `readonly=false` |
| internal SQLite evidence MCP | approved for bounded implementation after F0 | fixed aliases, physical read-only opening, SQLite authorizer, resource bounds and no network |
| Context7 | conditional bounded trial at F1 | public library documentation only; local installed source remains authoritative |
| Serena/LSP MCP | defer | overlaps the accepted Codebase Memory MCP until a measured refactor need exists |
| another GitHub MCP | do not add | local Git/`gh` and the already-approved GitHub connector cover current needs |
| monitoring or financial-data MCP | defer/reject for runtime | no canonical backend or admissible order-decision role exists |
| any brokerage-account MCP | prohibited | an agent must never receive broker credentials, account access or order authority through MCP |

Only Codebase Memory is active now. The three-server cap is accepted as a
maximum, not a target.

## 2. Source-Audit Evidence

### 2.1 `sqlite-explorer-fastmcp-mcp-server`

Source pinned for review:
`69f3f6c7d878cd5210a27e33fae814a9ec9e4b55`.

Observed defects in `sqlite_explorer.py`:

1. `SQLiteConnection.__enter__` calls `sqlite3.connect(str(self.db_path))`
   without `mode=ro`, `PRAGMA query_only`, an authorizer or filesystem
   containment.
2. `read_query` accepts a statement when its first token is `WITH`. SQLite
   permits `WITH ... DELETE/UPDATE/INSERT`, so this validation does not imply a
   read-only statement. The claim was reproduced independently with an
   in-memory SQLite table: `WITH c AS (SELECT 1) DELETE FROM t` executed and
   removed the row.
3. Row limiting is based on finding the substring `limit`; it is neither an
   execution-step limit nor an output-size bound.
4. There is no timeout/progress handler, table/view allowlist, extension rule
   or defense against expensive recursive/read amplification queries.

Conclusion: the server is not low risk and must not point at any project
ledger.

### 2.2 `mcp-server-sqlite`

Source pinned for review:
`a3b0323ce23521190572460dff944722b0036b3c`.

Observed defects:

1. `src/index.ts` accepts a database path on every tool call and resolves it
   without an allowlisted root or alias registry.
2. The `query` tool exposes `readonly` as an agent-controlled Boolean;
   `src/sqlite.ts` opens the database writable when it is false.
3. `list_databases` accepts an arbitrary directory and enumerates database
   filenames.
4. Results have no row, byte, execution-step or wall-time bound.
5. Read-only query classification permits broad `PRAGMA` and `WITH` input and
   relies primarily on opening mode rather than a deny-by-default authorizer.

The physical read-only open in its default path is better than the first
candidate, but exposing the escape switch and arbitrary paths to the model is
disqualifying.

## 3. Assigned Tooling Job: `DEV-TOOLING-MCP-001`

Priority: side job after F0 is green, or while F0 waits for independent review.
It must not delay IBKR findings 069-074 or consume campaign GPUs.

Implement a small project-owned SQLite evidence MCP or wrapper with all of
these properties:

1. local stdio transport only; no listening socket and no outbound network;
2. fixed symbolic database aliases supplied at process start; no path or
   directory argument exposed to a tool;
3. every resolved path must be under an explicit allowlisted root and must not
   include capability, credential, key or broker-profile stores;
4. SQLite URI `mode=ro`, `PRAGMA query_only=ON`, extension loading disabled and
   a deny-by-default SQLite authorizer that rejects writes, schema mutation,
   `ATTACH`, `DETACH` and unsafe pragmas/functions;
5. `immutable=1` only for declared closed snapshots, never for a live WAL
   database whose current facts must remain visible;
6. query tools limited to one read statement or `EXPLAIN QUERY PLAN`; no
   agent-controlled write-mode switch;
7. hard row, byte, SQLite VM-step and wall-time limits enforced independently
   of SQL text;
8. an allowlist of visible tables/views per database alias, with sensitive raw
   account identifiers excluded;
9. every response labels alias, source path hash, query hash, opened-at UTC,
   row count, truncation state and snapshot/live-WAL semantics;
10. discovery-only status: a finding still cites a reproducible query or
    evidence packet against the canonical ledger;
11. tests proving that `WITH ... DELETE`, write pragmas, attach/detach,
    arbitrary paths, symlink escapes, extension loads, recursive resource
    exhaustion and oversized output all fail closed; and
12. first acceptance run uses synthetic fixtures and verified copies only,
    never the live execution or DOIN ledger.

Return a source inventory, dependency lock, focused/full tests, threat model,
socket guard and a proposed VS Code MCP configuration with only aliases and no
secrets. Do not install it fleet-wide or connect it to a live ledger before
Musashi independently verifies the packet and the owner approves activation.

## 4. Conditional Context7 Trial: `DEV-TOOLING-MCP-002`

At F1, first inspect the installed `ib_async==2.1.0` source and its official
documentation. Use Context7 only if a named API question remains materially
unresolved.

If trialed:

- query only public package name, exact version and public symbols;
- never send repository names, paths, commits, findings, configs, account
  facts, internal code, credentials or business data;
- treat responses as discovery, then confirm in installed source and focused
  tests;
- record query count, whether it changed an implementation decision and every
  discrepancy found; and
- disable/remove the server after F1 unless measured value justifies a
  separate owner decision.

No Context7 output may define broker semantics or satisfy the F1 evidence
contract by itself.

## 5. Standing Broker-MCP Prohibition

No MCP server or general-purpose agent may connect to Alpaca, IBKR, OANDA,
MT5, cTrader, eToro or another brokerage account, even Paper/demo. It may not
receive broker credentials, inspect private account facts, construct orders or
call order APIs. The controlled LTS adapters and evidence collectors remain
the only broker-facing software. This prohibition does not weaken the owner's
ability to authorize a reviewed LTS Paper canary through the existing
capability and activation process.

## 6. Immediate Technical-Lead Order

Begin the existing F0 correction order now. The MCP research does not reopen
the completed A-E audit, weaken findings 069-074 or postpone live-demo work.
Deliver `DEV-TOOLING-MCP-001` only in the bounded slot defined above.

