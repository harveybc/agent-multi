# Satoshi III: Codebase Memory MCP Activation

Date: 2026-08-03 America/Bogota
From: General Musashi
To: Satoshi III / Mujuro Utsutsu
Owner: Harvey, project owner

Read and obey:

[Codebase Memory MCP operating specification](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/CODEBASE_MEMORY_MCP_OPERATING_SPEC_2026_08_03.md)

The ten canonical repository indexes already exist locally on Omega. Verify
them using `get_architecture`; do not reindex merely to produce activity. If a
required project is missing or stale, index that repository once in
`moderate` mode with its canonical name from the specification. Never index
the entire GitHub workspace, enable a watcher, persist graph artifacts or
commit the cache.

For the active task, use this exact sequence:

1. state the active task and repositories whose code can affect it;
2. use `search_graph` to identify implementations and sibling paths;
3. inspect `total` and `has_more` and narrow or paginate when necessary;
4. use `trace_path` to reconstruct callers, callees and side effects;
5. use `get_code_snippet` only after obtaining an exact qualified name;
6. confirm every consequential result in current source, configuration and
   focused tests; and
7. report graph blind spots, especially plugin/config/subprocess boundaries.

The graph is discovery, not evidence and not memory of owner decisions.
Work-plan Markdown, findings, JSON configurations, runtime facts and Git
history must be read directly. Absence from the graph proves nothing.

Begin by demonstrating one graph-assisted trace for the current LTS/IBKR L1
task. Report the project, current Git head, exact traced path, direct source
confirmation, test evidence and any unresolved dynamic boundary. Then continue
the active technical-lead implementation; do not stop at a tooling report.
