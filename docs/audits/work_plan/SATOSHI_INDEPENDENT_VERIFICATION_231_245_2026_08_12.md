# Independent verification 231-245 — Satoshi return (append-only)

From: General Satoshi III · To: General Musashi · Date: 2026-08-12 ·
Runtime rule honored: verification ran in parallel; no compute was
stopped, no broker order submitted or cancelled, no IBKR hold touched.
**No finding is closed here — verdicts and facts only.**

## P0 RUNTIME EVENT (discovered during this verification)

**Gamma is OFFLINE.** Tailscale reports `gamma … offline, last seen
26m ago, tx 312 rx 0`; `gamma.lan:22` and `192.0.2.12:22022` both
fail from omega AND from dragon (so it is not an omega-side route
problem). Consequence: `p1lr-decision@303` and `p1lr-decision@404`
are DOWN — two of the four decision workers and both gamma GPUs
(5070 Ti, 5090) are not producing work.

- Not remotely recoverable from here: no reachable interface, and no
  wake-on-LAN tooling (`wakeonlan`/`etherwake`/`wol`) is installed on
  dragon, which shares gamma's LAN segment.
- **Owner physical item:** gamma needs power/network attention.
- A watcher is armed; the moment gamma answers ssh I verify whether
  `p1lr-decision@303/404` auto-resumed from their enabled units and
  report the resumption facts.
- Nothing was restarted, re-materialized or re-pinned in response.
- The consolidated status correctly reports **2 of 4** fresh workers
  rather than fabricating 4/4 — the finding-233/245 corrections
  behaving exactly as intended under a real partial outage.

## Item 9 — live decision observation (two fresh heartbeat intervals)

Identity `8cc6ca5e45e4f993`, mode `decision`, units `p1lr-decision@`.

| interval (UTC) | worker | cell | stage | GPU util | GPU temp | restarts | cgroup memory |
|---|---|---|---|---|---|---|---|
| 20:29:35 | omega/101 | P1N_LR1E4 | training | 39 % | 48 °C | 0 | 2.502 GB |
| 20:29:35 | dragon/202 | P1N_LR3E5 | training | 23 % | 39 °C | 0 | 2.485 GB |
| 20:30:09 | omega/101 | P1N_LR1E4 | training | 39 % | 49 °C | 0 | 2.502 GB |
| 20:30:09 | dragon/202 | P1N_LR3E5 | training | 26 % | 38 °C | 0 | 2.485 GB |

Heartbeat ages 14.3-41.2 s across the two intervals (fresh). Memory
sits at ~2.5 GB against `MemoryHigh=5G` / `MemoryMax=6G` — the
finding-237 ceilings are not being approached. `NRestarts=0` on both
live units. **Records landed: 0/16** at observation (expected for a
2,000-checkpoint decision run at this elapsed time; the first landed
record will be reported when it exists). **gamma/303 and gamma/404
are absent — see the P0 above.**

## Item 10 — runtime pin (finding 244)

| host | pinned worktree | HEAD | tree | preflight decision identity | training_used | refusals |
|---|---|---|---|---|---|---|
| omega | `~/Documents/GitHub/.runtime/agent-multi-p1lr-182bac7e191c9143` | `182bac7e191c9143…` | clean | **`8cc6ca5e45e4f993`** | false | [] |
| dragon | same path | `182bac7e191c9143…` | clean | **`8cc6ca5e45e4f993`** | false | [] |
| gamma | not verifiable — host offline | — | — | — | — | — |

The canonical checkout has independently advanced to `1e1745ad`
(three commits past the pinned revision) while both live units keep
`WorkingDirectory=` the pinned worktree — exactly the property
finding 244 required. Unit facts read read-only; **no worker was
restarted to prove the pin.** The unit also carries
`MemoryHigh=5G`, `MemoryMax=6G`, `MemorySwapMax=1G` and an
`ExecStartPre` gate check.

**Verdict: `verified_corrected` on the two reachable hosts; gamma
unverified (host offline).**

## Item 11 — sequential-reader latency (finding 245)

Mechanism read at `1e1745ad`: each fact is aged at its own
observation time (`observed_now = (now_fn or utcnow)()` per worker),
with `hb_age = max(0, raw)` and `hb_clock_ahead = max(0, -raw)`
published separately as `heartbeat_clock_ahead_seconds`.

Injected latency fixture (collection starts at t0; a heartbeat is
written at t0+8 s during the sequential remote reads):

| case | raw delta | displayed age | reported clock lead |
|---|---|---|---|
| read AFTER the write (t0+12) | +4.0 s | **4.0** | 0.0 |
| read BEFORE the write (t0+4), genuine lead | −4.0 s | **0.0** | **4.0** |
| OLD single-`now` behavior (t0 vs t0+8) | −8.0 s | **−8.0 (negative)** | n/a |

Displayed age is never negative; genuine lead is surfaced separately
and not folded into age. `now_fn` is injectable, so the fixture is
deterministic. Focused suite `tests/unit/test_multifront_l1_factorial.py`:
**38 passed**.

**Verdict: `verified_corrected`.**

## Live broker table (read-only evidence, identifiers redacted)

| venue | state | position | orders | model artifact sha (prefix) | manifest sha (prefix) |
|---|---|---|---|---|---|
| Alpaca Paper | monitoring | 1 (SPY) | 1 | `b0ab77e0` | `f916696f` |
| IBKR Paper | monitoring | **−25,000 USD/CAD** | 2 (protection children) | `dc95edcb` | `03b6a794` |
| MT5 OANDA demo | (dragon-side; reported in the lts verification item) | ETHUSD short 0.01 with native SL/TP | — | — | — |

**Fact that differs from the audit snapshot:** the audit records IBKR
as "connected and flat". At my observation the runner reports a
**live short of 25,000 USD/CAD with 2 protection orders**, from fill
effect `l1e-201af4f2f1f4b448` at the 20:00 UTC bar, with
`position_reconciled: true`, `flattens: []` and a `resumed` entry
classified `acknowledged` (owner-signed capability path). This is a
legitimate protected entry taken after the audit's snapshot, not a
contradiction of it — but it means IBKR is **not flat now**, and any
statement conditioned on "flat" must be re-derived before use. I
touched nothing: no order, no cancel, no hold change.

Both live seats publish artifact/config/manifest and
`input_feature_sha256` hashes (finding 241's requirement) — the
independent manifest join is reported in the lts verification item.

## Items 1-8 — delegated independent verification

Three independent verification streams are running in parallel
(reproducer + 234/235/236 attacks; 237/242 profile and warm-start;
lts 238-241 plus CI 243). Their reproduced facts, counterexamples and
counts are appended to this packet as they land. Nothing from those
streams is asserted here in advance.

## Doubts and observations so far

1. **Gamma outage is the dominant open fact.** Until it returns, the
   decision campaign is running 2/4 workers and seeds 303/404 have
   produced nothing. No result may be aggregated on a partial fleet.
2. My earlier packet declared the 2,191-vs-2,190 transition and the
   manifest-cache staleness as residual doubts; you converted both
   into corrected findings (235, 236). I record that as my miss
   turned into your correction, not as my finding.
3. **I missed finding 237 in plain sight.** The
   `replay buffer 21.80GB > 10.59GB` warning appeared in the logs I
   was reading during the screen launches and I treated it as benign
   framework noise instead of the OOM precondition it was. That is my
   error, not a tooling gap.
4. The verification of gamma's pin, units and preflight identity is
   impossible while the host is offline; I will not infer it from the
   other two hosts.

## Item 5 — finding 237 (bounded replay execution profile)

**Verdict: `verified_corrected` for the MECHANISM; the finding's
quantitative premise is FALSE (new defect, below).**

- One execution profile, sha `7606ac12dc6b0f88…`, byte-identical in
  canonical and in the pinned runtime; all 16 decision cells carry
  that single sha and one replay declaration
  (`buffer_size=40000, optimize_memory_usage=false,
  uniform_across_seeds=true`); 16 distinct cell identities.
- Identity binding proven three ways: hand-recomputed payload
  reproduces the tool's identity; dropping only the profile field
  yields `6d78d1e6b900aca9` ≠ live; a counterfactual profile
  (only `$doc` text changed) moves the decision identity to
  `cca0ead1325f1d21` and leaves the screen identity untouched.
- Decision 40000 vs screen 200000 across all 16 cells — screen
  deliberately unchanged, as claimed. Arithmetic from materialized
  values: `2 × 20000 = 40000`, remainder 0; corroborated live by the
  journal's `buf=0→20000`, `20000→40000`, then steady 40000.
- Shipped and installed units byte-identical; live cgroup
  `MemoryHigh=5 GiB`, `MemoryMax=6 GiB`, `MemorySwapMax=1 GiB`,
  accounting on, current ≈2.48 GiB.

## Item 6 — finding 242 (warm start must not load the archived buffer)

**Verdict: `reproduced` (old paths) and `verified_corrected` (both
current paths).** Real SB3 2.9.0 SAC on CPU, replay allocation
instrumented at `ReplayBuffer.__init__` (nothing mocked), memory by
`VmPeak` (`ru_maxrss` under-reports calloc'd zero pages — stated
method).

| path | allocated capacities | max single alloc | VmPeak Δ |
|---|---|---|---|
| old `load_for_training` | `[200000]` | 784.302 MiB | +1031.2 MiB |
| old equal-dim expansion | `[200000]` | 784.302 MiB | +1103.2 MiB |
| corrected, target 40000 | `[1, 40000]` | 156.86 MiB | +404.0 MiB |
| corrected, target 40 | `[1, 40]` | 0.157 MiB | +247.1 MiB |

Source transfer capacity 1; target keeps its exact requested
capacity; replay empty (`size=0, pos=0, full=False`); optimizer state
fresh (`state == {}`, max step 0.0, target lr) against an archive
carrying step 100; transferred policy tensors byte-identical
(`c5663097…`, component L1 distances 0.0); capacity 200000 never
appears in any corrected path. The genuine 512→520 expansion
correctly differs and records 8 added dimensions.

Focused suites: **209 passed**. (Six extended files fail only inside
a scratch worktree because they resolve a sibling `doin-node` by
`REPO.parent`; the same six pass **10/10** in the canonical checkout —
a worktree-portability defect in the tests, not in the code.)

## D7 raised and RESOLVED — which screen verdict gates the live run

I raised an alarm that the live decision run might be gated by the
superseded screen verdict, because
`systemctl --user show p1lr-decision@101 -p Environment` renders
`P1LR_SCREEN_GATE=…/p1lr_collection_cd823e2b_20260812/screen_verdict.json`
— my ORIGINAL pre-correction screen, whose contract sha is
`8405b70b…` against the current `4a4e0f16…`.

**Resolved against evidence: the run is correctly gated.** The
per-instance `~/.config/agent-multi/p1lr-decision@101.env` overrides
that variable with the CORRECTED verdict
`…/p1lr_collection_886b776e022d0d7c_20260812/screen_verdict.json`,
and the unit journal proves `ExecStartPre` verified exactly that file
at start (`verified: true`, `refusals: []`, 14:05:25 and 14:15:46).
Independently: the gate script exits **4** against the old verdict
with `REFUSED_SCREEN_GATE_FOREIGN` naming both shas, and **0** against
the corrected one — so a start on the stale gate was impossible.

**Residual operator hazard (real, S4):** `systemctl show -p
Environment` displays only the static `Environment=` line and not the
`EnvironmentFile` override that actually wins, so the standard
operator probe reports the WRONG gate. The stale literal should be
removed from the unit template, or the template should carry no
default gate at all so the per-instance file is the only source.

## Delegated stream findings I am forwarding (not mine to close)

D1 (S2) — 237's memory numbers are wrong by ~5×, and mis-attributed:
the real observation space is `Box(2724,) float32`, i.e. 21,804 B per
transition, so 200000 → 4.36 GB (not 21.8 GiB) and 40000 → 0.87 GB.
The observed `21.80GB` warning came from the ANCHOR archives, which
declare SB3's default `buffer_size=1000000` — that is finding 242's
defect, not 237's. These wrong figures are content-addressed into the
live decision identity `8cc6ca5e45e4f993`, so correcting the profile
changes the identity and cannot be done casually mid-run.

D2 (S3) — 242's asserted consequence ("all four workers would fail
rather than train") was not evidenced: the archived buffer is
calloc-backed, so untouched pages are never charged to the cgroup.
The defect is real; the stated failure mode is unproven.

D3 (S3) — `agent_plugins/sac_agent.py` `Plugin.load()` is still an
unbounded `SAC.load(path, env=env)` reached from inference, the
best-checkpoint reload for final evaluation (while the training
model's buffer is still referenced), a curriculum fallback and the
P1LR runner. Bounded today only because the reloaded decision
checkpoints declare 40000.

D4 (S4) — `load_with_observation_expansion` records no source/target
policy-tensor hashes or `replay_size_at_boundary`, unlike
`load_for_training`.

D5 (S4) — the screen unit template carries no memory ceiling while
screen mode still materializes 200000. No screen unit is loaded now.

D6 — three decision identities have output under the live decision
root (`1434685bfdf52911`, `7b55ef7eac30ae6a`, live `8cc6ca5e45e4f993`);
the audit names `2f5054dc59785e2a` as superseded but no such directory
exists. The stale roots need explicit disposition so aggregation can
never see them.

## Items 7-8 — lts 238-241 and CI 243

| Finding | Verdict |
|---|---|
| 238 | `reproduced` (pre-fix) + `verified_corrected` at `lts@ea239a4` |
| 239 | `reproduced` + `verified_corrected` at `lts@ea239a4` |
| 240 | `reproduced` as a **live exploit** at `lts@3b94569` + `verified_corrected` at `lts@49c79af` |
| 241 | `verified_corrected` at `lts@4fcec85` — independent join agrees |
| 243 | `verified_corrected` — pytest 9.0.3, 38/38, 0 open Dependabot alerts |

- **lts suite: 686 passed, three consecutive deterministic runs.** The
  686-vs-685 discrepancy is explained, not flaky: `ea239a4` yields 685
  and `4fcec85` deletes one test and adds two.
- **238 counterfactual on one fixture** (filled parent evicted from the
  open-order view, position +20,000 intact, both children valid):
  pre-fix `protected=False, cancels=[1001,1002], flattens=1,
  halt=hold, positions=[]`; post-fix `protected=True, cancels=[],
  flattens=0, halt=none, positions=[+20000]`. Fail-closed still holds
  for zero position, foreign account, inverted sign, partial fill,
  conId mismatch, cancelled/missing child, no execution evidence and
  a forged proof source. 16/16.
- **239** driven through a REAL `Mt5ExecutionStore` on tmp sqlite:
  durable pending/delivered/failed reported truthfully and stably
  across idempotent re-evaluation; flat only on a newer snapshot with
  0/0; unknown states, NULL/naive timestamps and empty ids raise
  fail-closed; a foreign account never adopts state. 20/20.
- **240 exploit proven and closed:** restoring honest bytes right
  after the parser's read made an UNSIGNED attacker capability with a
  different `resume_of_effect_id` classify as VALID at `3b94569`
  (`EXPLOITED=True`); at `4fcec85` the same attack yields
  `kind=unsigned, parsed_effect=None`. One immutable snapshot proven
  three ways (single `read_bytes`; swaps during and after
  `ssh-keygen -Y verify` inert; recorded sha is the snapshot's).
  Symlinks, FIFOs and directories named `*.json` are ineligible and
  never deny the one valid capability. 18/18. Live owner store
  asserted unmodified by an autouse fixture.
- **241 independent join** (standalone script importing no lts module,
  MT5 side executed on dragon): all three seats publish artifact,
  config, manifest, `input_feature_sha256` and `preprocessing_sha256`;
  every hash joins its local manifest; on-disk artifact and config
  re-hash to the manifest values; zero blocking reasons on all three,
  repeated ~50 minutes after the auditor's capture. Musashi's evidence
  file sha256 `25746b3cbf7e1a65…` verified by me and matches.
- **243** isolated venv from the hash-locked lock: exactly
  `pytest 9.0.3` (trading-stack's 9.1.1 untouched), the six Tier-A
  files from `.github/workflows/tier-a.yml` pass **38/38**, and
  Dependabot reports **1 alert, 0 open** (pytest
  `GHSA-6w46-j5rx-g56g` state `fixed` at 2026-08-12T19:23:41Z).

### New defects forwarded from this stream (not mine to close)

1. **(S3) Coverage regression at `lts@4fcec85`:** it deleted the only
   test of `write_runner_heartbeat`; `grep -rn write_runner_heartbeat
   tests/` now returns nothing repo-wide, so parent-dir creation and
   the atomic `.tmp`→replace have zero regression guard. Behaviour
   hand-verified correct today, unguarded tomorrow.
2. **(S3, live-trading robustness) 241 identity computation is
   unguarded inside the degraded-error path:** `write_heartbeat` calls
   `linear_model_identity(...)` unconditionally, including from inside
   the runner loop's `except` block, and `json.dumps(...,
   allow_nan=False)` means a non-finite mean/scale would raise INSIDE
   the exception handler and escape the runner loop.
3. **(S4) 239 freshness is non-strict** (`snapshot_at < completed`),
   so a snapshot stamped at the exact completion instant counts as
   "newer" and can yield `command_succeeded_flat`.
4. **(S4) 239's "zero positions and zero orders" is route-symbol
   scoped**, not account-wide; the audit prose omits the qualifier.
5. **(S4) 241 `input_feature_sha256` cannot distinguish routes** — it
   is identical on all three seats because it hashes the contract
   string plus ordered feature NAMES, the same 11 everywhere. It
   proves ordering, not that the SPY seat is fed SPY features.
6. **(S4) 241 config/artifact bytes are not continuously re-verified**
   — `refresh(force=False)` short-circuits on unchanged manifest
   bytes, so an on-disk swap without a manifest change is invisible to
   the runner.
7. **(S4) heartbeat key naming is inconsistent across seats** — IBKR
   publishes `position` (signed float), Alpaca and MT5 publish
   `positions` (count).
8. **(S4) live store hygiene:** the owner capability is 0600 but its
   detached `.sig` is 0664; classification checks the capability's
   mode only. Contained by the 0700 store directory, so not a bypass,
   but a writable `.sig` could downgrade a valid capability to
   `unsigned` and DENY the owner's resume.
9. **(S4) hard links are eligible** (same inode, same signed bytes —
   not a bypass, but it deserves an explicit decision record).
10. **OWNER ITEM (observed, out of scope):**
    `usdcad_4h_linear_v1/manifest.json` declares
    `live_inference_eligible: false` and
    `live_execution_eligible: false`, yet the live IBKR **paper** seat
    runs it, admitted through the `demo_research_canary` tier which
    only requires `research_validated: true`. The owner should
    confirm that is intended for a paper-execution route.

## Finding 230 collateral: the historical replica stalled with gamma

Destination on dragon now holds **40,703,148,144 bytes / 170,304
files** against a source of 234,207,109,018 bytes / 249,434 files
(17.4 % by bytes, 68.3 % by file count — the large artifacts trail
the many small ones). The newest destination file is **~50 minutes
old**, matching gamma's disappearance: the transfer stalled because
its SOURCE is gamma, not because of any fault on dragon.

`rsync --partial` makes it restartable; it resumes when gamma returns.
Byte and file counts only — no rsync percentage is reported.
Finding 230 stays open; the dual-side sorted SHA-256 manifest
comparison cannot run until the transfer is terminal.

## Items 1-4 — reproducer, 234, 235, 236

### Item 1 — the unchanged reproducer CANNOT RUN while gamma is down

The reproducer aborts before emitting anything:
`subprocess.CalledProcessError … ssh gamma 'du -sb …' exit 255`
(`_command(check=True)` raises). It was re-run through a wrapper that
imports the file VERBATIM and stubs only the `ssh gamma` calls; the
deviation is recorded inside the emitted JSON and gamma-derived values
are explicitly NOT evidence. **Any future "all flags false" claim must
state whether gamma was reachable.**

All emitted booleans: `nested_context_execution.finding_reproduced`
**false** (wrapper_declared / wrapper_applied_by_internal_split_factory
/ rollout_filters_is_context_prefix all true);
`inactive_decision_publication.finding_reproduced` **false**
(`decision_runner_unconditionally_requires_best` false);
`decision_observability.finding_reproduced` **false**
(`idle_guard_reads_top_level_output_root` false,
`idle_guard_unit_template_is_screen` false,
`status_reads_top_level_output_root` still true);
`historical_replica.terminal_digest_proof_available` false.
`internal_roles_with_context_rows = {train_monitor: 256,
inner_validation: 256}`; contract `4a4e0f16…`.

**I do NOT report 233 as independently verified from this reproducer.**
Three of its flags are source-string heuristics, not behaviour: the
"requires best" flag flips partly because the literal is now
line-wrapped; `idle_guard_unit_template_is_screen` is false because the
source reads `UNIT_TEMPLATE = P1LR_UNIT_TEMPLATES["screen"]` — that
module constant STILL resolves to the screen template, and only the
per-call `P1LR_UNIT_TEMPLATES[mode]` is mode-aware. And
`decision_observability.finding_reproduced=false` is reached for the
WRONG REASON: the reproducer hardcodes the superseded identity
`1434685bfdf52911`, so status returns `state="refused"` with
workers/records null — the conjunct is false because the values are
null, not because the mode moved to decision. (233's semantics ARE
correct; I verified them directly in items 9-11. The reproducer is
simply weak evidence for it.)

### Item 2 — finding 234: `verified_corrected` for (a)(b)(c), **`still_open` for (d)**

Byte-identical fixture (`model.terminal.zip` and `model.best.zip`,
same sha `6724c5ea…`, distinct paths), driven through the REAL
`assert_cell_record_custody` / `record_is_complete`, with the pre-fix
module extracted at `04414417^`:

| case | current | pre-fix |
|---|---|---|
| ACTIVE, byte-identical best/terminal | **succeeds** | **refused → 234 reproduced** |
| ACTIVE missing best path / missing best hash | **refused** | succeeded (the fix adds a real requirement) |
| INACTIVE labeling terminal as best — top-level, nested in `selection`, nested in `outer_validation_final`, or `promotion_eligible=true` | **all refused** | — |

**232 is not weakened.** But:

| case | expected | observed |
|---|---|---|
| ACTIVE, best FILE MISSING | refuse | **SUCCEEDED** |
| ACTIVE, best file hash ≠ record | refuse | **SUCCEEDED** |
| ACTIVE, `best_model_path == terminal_model_path` | refuse | **SUCCEEDED at custody** (only the production-time outer-eval role check refuses it) |

`record_is_complete` returns **true** for both broken-best cases —
only the TERMINAL is re-hashed. During a live run the damage is caught
earlier (`classify_cell_activity`, and the outer-eval hash binding),
but **after publication, tampering with or deleting an ACTIVE cell's
best artifact is invisible**: the record can be reused on restart and
can win selection with a corrupted or absent best `.zip`. Recommended:
re-hash `best_model_sha256` in `record_is_complete`, symmetric with the
terminal check.

### Item 3 — finding 235: `verified_corrected`

Mechanism confirmed from the pre-fix source: `next()` published the
last real bar as NON-terminal and blocked, so Backtrader reached
`stop()` (`data_end`) only after the env supplied one more action
purely to release the thread — and that release was counted.

Real gym-fx, 40-row CPU fixture: pre-fix `b71429a` emits **41**
transitions with a duplicated final `bar_index` for BOTH an active and
an inactive policy; current `634c3fd` emits exactly **40**, no
duplicate.

Real nested contract, real env, real `_eval_on_split` and real
`_outer_validation_final_eval`:

| role | manifest scored | csv rows | env steps | scored_steps (active / inactive) |
|---|---|---|---|---|
| train_monitor | 2,190 | 2,446 | 2,446 | **2,190 / 2,190** |
| inner_validation | 2,190 | 2,446 | 2,446 | **2,190 / 2,190** |
| outer_validation | 2,196 | 2,452 | 2,452 | **2,196 / 2,196** |

With the pre-fix bridge the literal off-by-one reproduces and BOTH new
guards fire: `rollout scored 2191 steps but the verified manifest
declares 2190 … refusing` and `outer_validation replay scored 2197 …
refusing`. My exact-2,190 residual doubt from the previous packet is
now settled against the real environment.

### Item 4 — finding 236: `verified_corrected` for mtime/size mutations, **`still_open` for a same-size same-mtime swap**

| mutation of the role CSV (manifest untouched) | re-verified? | outcome |
|---|---|---|
| in-place rewrite, natural new mtime | yes | REFUSED (hash drift) |
| append + `os.utime` restore | yes | REFUSED (hash drift) |
| role file removed | yes | REFUSED (missing) |
| **content swap, same size, `st_mtime_ns` restored** | **no (0 verifications)** | **ACCEPTED — NOT CAUGHT** |
| **replacement by a new inode, same size, mtime restored** | **no** | **ACCEPTED — NOT CAUGHT** |

The custody tuple is `(role, resolved path, st_mtime_ns, st_size)` —
content is not in it, so a length-preserving writer that restores mtime
is trusted for the life of the process cache (a restart re-hashes and
refuses). The manifest's own cache key has the same shape. Also
`_nested_role_file_stats` catches only `OSError`, so a manifest entry
missing its `csv` key raises `KeyError` out of the cache-hit path
instead of failing closed.

### Suites

`agent-multi` full suite in the temp worktree with the sibling
`doin-node` symlinked: **1,303 passed, 0 failed**. `gym-fx`: **84
passed**. Focused: 148 + 75 passed. New verification tests: 3 + 7.
(Without the sibling checkout beside a scratch worktree, four tests
fail on missing `doin-node` templates — a test worktree-portability
defect, not a regression.)

### Additional observation

Worker RSS on omega/dragon now reads ~2.48 GiB against the audit's
first observation of 1.37-1.80 GiB. Still far under `MemoryHigh=5G`,
but it is GROWING and deserves watching against findings 237/242.

## Consolidated verdicts (no closures — the owner disposes)

| Finding | My verdict |
|---|---|
| 231 | verified_corrected |
| 232 | verified_corrected (not weakened by the 234 fix) |
| 233 | verified_corrected **by direct observation** (items 9-11); NOT independently established by the reproducer's string heuristics |
| 234 | verified_corrected for byte-identical-active, missing-best-fields and inactive protection; **still_open** for missing/mismatched ACTIVE best artifact after publication, and for `best_path == terminal_path` at custody |
| 235 | verified_corrected |
| 236 | verified_corrected for every mtime/size mutation; **still_open** for same-size same-mtime content swap and same-size inode replacement |
| 237 | mechanism verified_corrected; **quantitative premise false** (D1) and now content-addressed into the live identity |
| 238 | reproduced + verified_corrected |
| 239 | reproduced + verified_corrected (non-strict freshness noted) |
| 240 | reproduced as a live exploit + verified_corrected |
| 241 | verified_corrected; independent join agrees |
| 242 | reproduced + verified_corrected (asserted failure mode unproven, D2) |
| 243 | verified_corrected |
| 244 | verified_corrected on omega and dragon; gamma unverifiable (offline) |
| 245 | verified_corrected |
| 230 | still_open — transfer stalled with gamma at 40.70 GB / 170,304 files; no dual digest yet |

## P0 RESOLVED — gamma returned, and the guard chain worked in production

Gamma came back after ~55 minutes offline. The boot chronology is
itself the best live evidence of the 233/237 machinery:

1. **t+3 min:** `p1lr-decision@303` and `@404` both `failed`,
   `status=4/NOPERMISSION`. The reason is exactly right:
   `REFUSED_GPU_UNBOUND`, `classification: GPU_UUID_MISMATCH`,
   `expected ['GPU-b77fc3ad…','GPU-a9f35631…'] observed
   ['GPU-b77fc3ad…']`. **The launch gate refused BEFORE the framework
   imported**, so nothing fell back to CPU and nothing trained on the
   wrong device.
2. **Diagnosis:** the RTX 5090 was present at PCI (`0a:00.0`, GB202,
   `Kernel driver in use: nvidia`) and registered in
   `/proc/driver/nvidia/gpus/0000:0a:00.0` with the exact expected
   UUID — it had simply not finished enumerating for
   `nvidia-smi --query-gpu` at 3 minutes post-boot. Kernel
   `7.0.0-29-generic`, module `580.173.02`, `modinfo` OK: no
   repeat of the post-outage module-missing failure.
3. **t+~12 min:** with both GPUs enumerated, the gate returns
   `GPU_READY, blocking: []`, and `p1lr-idle-guard.service`
   (finding 233's bounded recovery) ran at 16:08:12 and brought both
   seeds up.

Fleet now: **4/4 decision workers active** — omega/101 and dragon/202
`NRestarts=0` and never interrupted, gamma/303 and gamma/404 running
at 2.97 GB and 2.80 GB. Both gamma GPUs at 40 % / 44 %. The
historical replica rsync resumed on its own.

**No manual intervention was required or performed:** I diagnosed
read-only and the shipped guard chain did the recovery. The one thing
that WAS required was the owner's physical attention to the machine.

**Nuance worth recording for the register:** the gate refuses on the
HOST's full expected UUID set, so a single missing GPU blocks BOTH
seeds on that host — including the seed whose own assigned UUID is
present. During the ~9-minute enumeration window that cost the healthy
5070 Ti. Whether to scope the refusal per-assigned-UUID instead of
per-host-set is a design decision with a safety trade-off, and it
cannot be changed casually: the gate lives inside the pinned runtime
worktree, so editing it would change the dirty digest and therefore
the decision identity (finding 244).
