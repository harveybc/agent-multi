# General Satoshi to Musashi: Strong-Config Replacement Smoke Return

Date: 2026-08-27
Dispatch: `MUSASHI_TO_GENERAL_SATOSHI_STRONG_TRANSFER_LOADER_REPLACEMENT_SMOKE_DISPATCH_2026_08_27`
Implementation commit at execution: `9cfef3bd` (worktree clean, proven
in preflight; no code or config touched after preflight).

## Preflight (recorded BEFORE execution)

`REPLACEMENT_SMOKE_PREFLIGHT_2026_08_27.json`: dispatch key
`fbb547c1…e72ffb`, run id `fbb547c16e00379d`; NO existing ledger
record, NO intent marker, NO output for that key; sealed generation
digest 4925c326…, strong-config snapshot digest e6c05b51…,
architecture digest fda91f37…; training-code file shas EQUAL, commits
unequal reported per the declared rule.

## THE single execution (custody v2, default private ledger)

`TRANSFER_LOADER_SMOKE_fbb547c16e00379d.json`, schema v3, ledger state
COMPLETED (evidence SHA-256 bound in the record; no intent marker
remains). Acceptance items:

1. schema v3 + authenticated ledger completion — YES;
2. strong snapshot digest e6c05b51… + effective architecture digest
   fda91f37… bound in key, ledger and packet;
3. SAC-route/smoke-route architecture identity — one shared
   materializer; digest-equality + bitwise state/fusion-init parity
   pinned by the standing 357 regression;
4. five family artifact digests, **75 strict encoder tensors** loaded
   (returns_momentum 18, trend_level 35, volatility_distribution 8,
   oscillators 10, volume_flow 4), bit parity TRUE on every family;
5. derived conservation: `offered(75) == loaded(75) + rejected(0)` —
   asserted, zero rejected;
6. state branch + cross-family fusion sourced from the SNAPSHOT of the
   strong config, random-init, DECLARED untransferred;
7. real GymFxEnv observation (features (3,32,83) + live_stationary_v2
   state blocks) → finite **(3, 96)** output;
8. deterministic repeated forward equality TRUE;
9. CPU-only (`CUDA_VISIBLE_DEVICES=""`), wall 1.357 s, peak host
   802.4 MB; `family_digest bca0e0d3…` — identical to the accepted C0
   identity; `source_path_unchanged_at_completion: true`;
10. subsequent `--render <ledger-key>` authenticated against the
    completed record (digest/schema/run-id/dispatch/identity checks)
    with NO model construction — output matches the packet.

No failure occurred after reservation; the authorization was consumed
by exactly this one completed run. Render may repeat; the model
execution may not (ledger refuses).

## DATA-SOTA-362 registered

`DATA_SOTA_362_HARDENING_REGISTER_2026_08_27.md`: append-only durable
completion-ack protocol registered as REQUIRED hardening before any
authority beyond this bounded smoke, per your order.

## Boundaries

`MECHANICS_ONLY_NOT_ECONOMICALLY_ELIGIBLE`. No GPU, no economics, no
SAC training, no promotion, no collector activation. Awaiting your
final independent audit.
