# Codex Response to Audit Delta 2026-08-01

Date: 2026-08-01 America/Bogota
Respondent: Codex / Musashi, experimental and technical lead
Responds to: `MUSASHI_RESPONSE_INVOCATION_2026_08_01_04.md`
Governing audit: `AUDIT_DELTA_2026_08_01.md`

## 1. AUD-F2-20260801-029

Finding accepted. Dragon-local evidence was valid, but Omega's consolidated
watchdog still attempted to read a Dragon-local SQLite path and therefore
raised `mt5_bridge_missing`. A permanently false venue alarm was not an
acceptable fleet state.

Correction:

- `lts@a5fe0d971a24cb075bb107fc16e111e60d42a9c4` adds the read-only
  `/v1/status` operational evidence route;
- the route exposes connection, freshness, terminal diagnostics and aggregate
  heartbeat/snapshot/event and position/order/symbol counts;
- the route excludes account/server fingerprints, balances, equity, margin,
  instrument names, tickets and the bridge HMAC secret;
- Dragon's firewall remains restricted to the libvirt NAT and Tailscale
  sources;
- Omega's consolidated watchdog now loads `LTS_MT5_STATUS_URL` from an
  untracked mode-0600 environment file and treats Dragon's remote status as
  authoritative when configured.

Runtime verification at `2026-08-01T07:52:35.924524+00:00`:

```text
Dragon LTS commit: a5fe0d971a24cb075bb107fc16e111e60d42a9c4
Omega LTS commit:  a5fe0d971a24cb075bb107fc16e111e60d42a9c4
Dragon bridge:     active/running, service result success
Omega watchdog:    service result success, timer active
MT5 available:     true
MT5 connected:     true
Heartbeat age:     10.0 seconds
MT5 positions:     0
MT5 orders:        0
Observed symbols:  6
MT5 alert state:   mt5_bridge_missing active=false
```

Omega's remaining active event keys were `ibkr_observer_stale` and
`ibkr_paper_offline`, both expected while TWS Paper was closed. No MT5 event
remained active.

Reproduction:

```text
lts tests/unit: 101 passed, 1 upstream Starlette deprecation warning
```

Relevant file SHA-256 identities:

```text
app/mt5_bridge_lab.py
da43ce306f68bed6b8c869ce8bd924318d0d479805c044d6b569d12d17e85d3d

tools/paper_execution_watchdog.py
a4135433dea8f93de39aa3475009eb6ae28758c653e0452f4b99aff61172639e

tests/unit/test_mt5_bridge_lab.py
aaf155bafbc7ef7a93e041552efcef39d0ab7d002b7af0f33b15768fa6491075

tests/unit/test_paper_execution_watchdog.py
874cba7f28cbc01e64fa94624b7411bd0cb05ab39f878f64c05f4ccd78e2011a
```

The auditor retains closure authority for finding 029.

## 2. Job-1 Selection-Metric Rider

Implemented at
`agent-multi@06de651f06caa8c59fd6ac5916ec6248260b4d11`.

`test_materialized_job_1_resolves_to_robust_weekly_l1_fitness` exercises the
actual path:

```text
protected curriculum template
  -> materialize job 1
  -> canonical resolve_config
  -> runtime selection_metric
  -> _selection_value
```

The test preserves the conflicting source facts deliberately, then proves:

```text
objectives.selection_metric = train_validation_l1_score
training.selection_metric   = robust_weekly_rap_fitness
runtime.selection_metric    = robust_weekly_rap_fitness
selected value              = robust_weekly_rap_fitness
```

It supplies a conflicting `total_return=0.99` and
`robust_weekly_rap_fitness=0.0025`; the selected value is asserted to be
`0.0025`. This prevents a passing test that only confirms both keys contain
strings.

Reproduction:

```text
execution curriculum pipeline + materializer: 12 passed
```

Test file SHA-256:

```text
28f3961722125490370f8fcbfa317f77af735f54df2c7c254903a035758def21
```

## 3. Runtime and Decision Boundary

- No active job-0 config, chain, objective or worker was changed.
- The A/B/C objective decision remains reserved for Harvey.
- Job 0 remains initialization evidence under its declared full-period
  objective unless Harvey decides otherwise.
- Job 1 is now mechanically guarded to use weekly robust fitness after
  materialization.
- At job-0 archive time, the weekly top-2 elite-preservation fact must be
  re-verified against the final chain; this response does not claim it in
  advance.

## 4. Access Clarification

The failed independent SSH check used Dragon's ordinary SSH endpoint. Dragon's
fleet SSH contract uses port `22022`; refusal on port 22 is expected. This does
not invalidate finding 029, because watchdog evidence should not depend on
interactive SSH. The new status path supplies the required read-only fleet
evidence directly.
