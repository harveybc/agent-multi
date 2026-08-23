# Audit: Satoshi MT5 USDCAD P0 and P1/P2 Amendments

Date: 2026-08-23
Auditor: General Musashi
Audited tips: `lts@25644dc`, `lts@72c4617`
Disposition: **correction required before activation**. No service was changed.

## Reproduced facts

- The store selects unresolved commands by `(account_fingerprint, symbol)`.
- A normal updated EA adds its chart symbol to the poll and checks the returned
  symbol before execution.
- Distinct magic values are declared for the two provided profiles.
- The runner unit is a non-installed template.
- `33` focused tests pass and the diff is whitespace-clean.

These are meaningful corrections. They prevent ordinary ETH and USDCAD EA
instances from consuming each other's queue entries when both behave exactly
as expected.

## Findings

### AUD-F2-20260823-301 (S2): route query is outside request authentication

`Mt5RequestAuthenticator.verify()` authenticates method, URL path, timestamp,
nonce and body. `/v2/commands/next` places `account_fingerprint` and `symbol` in
the query string, which is not authenticated. A request signed for an ETH poll
remains valid after changing its query to USDCAD. Both chart EAs also share one
secret and present no authenticated client/magic identity. Consequently,
symbol-scoped SQL prevents accidental theft by the current EA source but does
not make cross-route delivery structurally impossible.

Required correction: bind a canonical query digest, or an authenticated route
identity containing account fingerprint, symbol, EA magic and protocol version,
into the request signature. The server must compare that identity to the route
before reading the queue. Add exact-query mutation, reordered-query,
duplicate-key, percent-encoding, wrong-magic and cross-client fixtures. Preserve
nonce replay protection.

### AUD-F2-20260823-302 (S3): CopyRates evidence is unattested metadata

The preflight accepts any JSON containing twelve aligned timestamp strings.
It does not establish that the timestamps came from CopyRates, the Demo
terminal, the configured account, USDCAD, H4, the current broker server, or a
fresh capture. A hand-written file passes.

Required correction: add an EA/read-only capture endpoint producing a signed
evidence envelope with schema, environment, account fingerprint, symbol,
timeframe, broker/server identifier fingerprint, captured-at, terminal build,
ordered OHLCV bars and digest. Verify freshness, exact route identity,
monotonic spacing, no duplicates/gaps, finite OHLCV geometry and H4 UTC opens.
The preflight consumes only that envelope.

### AUD-F2-20260823-303 (S3): declared compatibility checks are absent

The tool docstring claims it verifies the Demo volume ceiling against the named
bridge config. `check_profile()` neither loads `bridge_config_file` nor compares
volume limits. Symbol trade mode, minimum/step/maximum volume and market facts
are deferred to prose, not represented in the executable acceptance result.

Required correction: load and hash the effective bridge config without exposing
the account identifier; require both symbols in its mandate; verify route and
daily ceilings; bind direct symbol facts including trade mode, volume min/step/
max, digits and point size. A volume not aligned to the broker step refuses.

### AUD-F2-20260823-304 (S3): magic uniqueness is partly guessed

`check_magic_unique()` invents `26080301` for an ETH profile missing `ea_magic`.
That is the opposite of an explicit uniqueness proof and could conceal drift in
the installed EA or profile.

Required correction: every compared profile and signed EA runtime fact must
declare its positive magic. Missing values refuse. Verify uniqueness across
profiles and current terminal snapshots; never supply a default in validation.

### AUD-SEC-20260823-305 (S3): operational fingerprints remain tracked

The new profile repeats a real account fingerprint in a tracked public-repo
config. Other historical example files do the same. A fingerprint is not a
password, but it is persistent operational metadata and conflicts with the
repository's no-account-identifiers policy.

Required correction: replace tracked operational fingerprints with explicit
placeholders and materialize effective local profiles under `~/.config/lts`.
Scan history/current trees of public repositories for account identifiers,
emails, private addresses, tokens and broker metadata; produce a redacted
finding list without printing discovered values. History rewriting requires a
separate owner decision; current-tip cleanup does not.

### AUD-F2-20260823-306 (S3): simultaneous-position semantics are unproved

Both model profiles declare `max_concurrent_positions: 1`. It is not established
whether this is account-wide, runner-local or symbol-local when ETH and USDCAD
signal simultaneously. The new queue is symbol-local, but the business risk
contract may still serialize or reject one route.

Required correction: define the account-wide and per-symbol limits explicitly,
then test simultaneous open, simultaneous close, one route held while the other
signals, daily budget sharing, and one-route failure isolation. Conservative
account-wide one-position operation is acceptable, but it must be reported as
intentional serialization rather than dual-symbol concurrent trading.

## P1 curriculum amendment

P1 must remain on the accepted flat-MLP observation/model contract. It must not
absorb the grouped extractor changes mid-experiment.

The easy-to-normal transition needs an explicit state-factor distinction:

- `N`: normal-only, cold start;
- `EN-W`: easy then normal, preserving actor, critics, target critic and entropy
  state, while starting a fresh normal replay buffer;
- `EN-F`: easy then normal with full replay/optimizer continuity.

`EN-W` is the closest SAC analogue to the owner's NEAT observation because NEAT
preserved evolved parameters but had no off-policy replay buffer. `EN-F` tests
whether carrying easy-dynamics experience helps or contaminates early normal
updates. If current P1 remains two-arm, its result must be labeled for its exact
continuity treatment and the missing ablation queued immediately; it may not be
generalized to "easy pretraining" as a whole.

Both transitions must prove tensor continuity, target-network identity,
optimizer/entropy disposition, replay disposition, normalizer disposition and
first-normal-update provenance. Same seeds, data, fixed LR, stopping contract
and evaluation roles are required.

## P2 truth amendment

Until 301-306 are corrected and independently reproduced, report:

- Alpaca: active Demo evidence route;
- MT5 ETHUSD: existing active Demo route;
- MT5 USDCAD: prepared, not activated, compatibility not yet proven;
- IBKR: preserved and suspended by owner;
- early plateau-LR screen: bounded negative directional evidence for the tested
  specification, not a universal mechanism conclusion.

