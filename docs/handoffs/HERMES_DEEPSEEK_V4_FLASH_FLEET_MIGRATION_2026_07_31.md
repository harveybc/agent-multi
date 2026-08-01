# Hermes DeepSeek V4 Flash Fleet Migration

Date: 2026-07-31 19:20 America/Bogota

## Objective

Route every active Hermes role on Omega, Dragon and Gamma through the current
OpenCode Go `deepseek-v4-flash` alias, remove Pro and fallback routing, and
prove the route with real inference rather than configuration inspection
alone.

## Provider Identity

The official OpenCode Go model identifier is `deepseek-v4-flash` and the
OpenAI-compatible endpoint is:

```text
https://opencode.ai/zen/go/v1
```

The provider does not expose a dated `deepseek-v4-flash-0731` identifier in
its model catalog. Successful responses identify the model as
`deepseek-v4-flash` and return no revision fingerprint. Before owner opt-in,
the provider rejected the alias with a region error stating that its latest
version was available only through the China-hosted path. After explicit
opt-in, the same credential and alias succeeded. This is the strongest
available operational evidence for the latest provider-served revision; no
more specific build claim is made.

Primary references:

- <https://opencode.ai/docs/go/>
- <https://api-docs.deepseek.com/guides/coding_agents/>

## Effective Topology

| Host | Default provider/model | Hermes version | Source commit | Telegram gateway |
|---|---|---|---|---|
| Omega | `opencode-go/deepseek-v4-flash` | `0.12.0` | `05c63259` | active, sole poller |
| Dragon | `opencode-go/deepseek-v4-flash` | `0.12.0` | `75e1339d` | inactive by design |
| Gamma | `opencode-go/deepseek-v4-flash` | `0.12.0` | `05c63259` | inactive by design |

All hosts have `fallback_providers: []`. Omega's triage and supervisory-review
cron jobs explicitly select Flash; the business-review job inherits the same
Flash default. No active configuration or cron job selects Pro.

Only Omega runs the Telegram gateway because the fleet currently shares one
Telegram bot identity. Activating multiple pollers for one bot would create a
message-consumption race. Dragon and Gamma remain independently callable
Hermes workers and passed direct Hermes inference.

## Acceptance Evidence

1. A direct OpenCode chat-completions request returned the expected sentinel
   and identified its model as `deepseek-v4-flash`.
2. Hermes one-shot requests on Omega, Dragon and Gamma each returned the exact
   `HERMES_FLASH_OK` sentinel.
3. Omega's gateway restarted and remained active.
4. A real Telegram interaction was served by
   `opencode-go/deepseek-v4-flash` after restart.
5. Forced `moltbook-social-triage` and `moltbook-social-review` jobs completed
   with `last_status=ok`, null execution errors and null delivery errors. The
   review completed at `2026-07-31T19:20:38-05:00`.

## Backup And Rollback

Pre-migration files are stored locally on each host under:

```text
~/.hermes/backups/20260801T001142Z/
```

The backup includes the prior config and locally relevant cron/auth/env files
where present, with mode `0600`. Secret values are intentionally absent from
Git. Rollback requires restoring the host-local files, validating ownership
and mode, and restarting only Omega's gateway.

## Remaining Operational Debt

The Hermes package version matches, but source commits do not. A fleet update
must be performed as a separate controlled change with backup, three-host
one-shot inference, cron acceptance and Telegram acceptance. It is not a
blocker for the verified Flash route and was deliberately excluded from this
migration to avoid changing agent code and provider routing simultaneously.
