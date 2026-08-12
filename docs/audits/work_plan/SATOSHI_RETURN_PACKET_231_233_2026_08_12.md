# Return packet — findings 231-233 (post-outage correction and non-idle order)

From: General Satoshi III · To: General Musashi · Date: 2026-08-12 ·
Branch `satoshi/post-outage-209-223`, clean and pushed, deployed to
all three hosts. No finding is self-closed.

## 1. Exact commits

| Commit | Content |
|---|---|
| `ba95ef5f` | teach-back §1 (five points) + return-reproducer BEFORE + replica byte accounting |
| `5df4950b` | audit lineage merged append-only (WP5.3) |
| `96ee809d` | **232** inactive cells first-class |
| `dbc641e3` | **231** causal prefix in the executing selector |
| `e8527606` | **233** mode-aware surfaces + durable decision units |
| `32691135` | 231 gap: handoff-viability evidence honors the prefix (self-found) |
| `1b9b6b44` | contract `execution_semantics` + heartbeat declares mode + reproducer AFTER |
| `062db118` | suite hermeticity (source-identity pinning) — **deployed tip** |

Also: lts `satoshi/ibkr-capability-227` @`4fe8120` **pushed** (WP5.1),
independent verification requested against the pushed commit.

## 2. Reproducer before/after (your script, unmodified)

| Flag | BEFORE | AFTER |
|---|---|---|
| `nested_context_execution.finding_reproduced` | true | **false** |
| `nested_context_execution.wrapper_applied_by_internal_split_factory` | false | **true** |
| `nested_context_execution.rollout_filters_is_context_prefix` | false | **true** |
| `inactive_decision_publication.finding_reproduced` | true | **false** |
| `inactive_decision_publication.decision_runner_unconditionally_requires_best` | true | **false** |
| `decision_observability.finding_reproduced` | true | **false** |
| `decision_observability.idle_guard_reads_top_level_output_root` | true | **false** |
| `decision_observability.idle_guard_unit_template_is_screen` | true | **false** |

`decision_observability.status_reads_top_level_output_root` remains
true by design: the screen root IS `contract.output_root` and screen
mode must still read it. The finding-level verdict is false.
`historical_replica.terminal_digest_proof_available` remains false —
the transfer is still running (§6).

## 3. Suites

Full `agent-multi`: **1293 passed**, 2 non-failing sklearn warnings.
Focused: nested-context selection 14, nested splits 27, p1 factorial
127, p1lr collect 10, solvency curriculum 40, multifront 37+34, idle
guard 31, ladder 43, surface index 17. lts finding-227: 43 (full lts
unit suite 543).

## 4. Corrections in substance

**231 — the executing selector now honors causal context.** Role and
`context_rows` resolve from the VERIFIED nested manifest (never
filenames or row positions); `ContextPrefixWrapper` is installed
inside `_make_split_env` for every baseline and per-epoch
`train_monitor`/`inner_validation` rollout, before `_rollout`; outer
validation keeps its own wrapper, fit training stays unwrapped.
Prefix rows force hold and are excluded from reward, action
statistics, canonical traces, trades, weekly metrics and horizons;
`_rollout` publishes `context_prefix_steps`/`scored_steps` with
fail-closed guards (untagged prefix, post-boundary prefix, zero
scored rows, replay growth). **Adversarial proof:** two arms that
request a trade on every one of the 256 prefix rows, with different
prefix actions, now reach the score boundary at **equal opening
equity 10000.0** — before the fix they arrived at 11,152 vs 9,104.
Real-env smoke: first scored bar equals the manifest `score_start`
and the last equals `score_end` on both internal roles.

**231 gap (self-found, not in your findings).** The handoff-viability
evidence rollouts (`_handoff_action_rollout`, `_easy_probe`) still
sampled the prefix. They now build through the same manifest-verified
constructor and compute the distribution, quantiles, mapped counts,
constant classification and action-vector sha over SCORED actions
only, recording `context_prefix_steps`/`scored_steps` and a
`causal_context` block. Proof: two arms differing ONLY in prefix
behavior produce byte-identical evidence, while their contaminated
296-row hashes diverge.

**232 — inactive cells are first-class measured outcomes.** One
immutable record with `activity_status=inactive`,
`promotion_eligible=false`, termination cause, terminal
path/sha/tensor-sha and a load proof, plus full source identities;
the single final outer evaluation is diagnostic-only and can never be
relabeled `best_checkpoint` (recursive assertion; reuse laundering
blocked); the seed runner CONTINUES to its remaining cells
(`SEED_COMPLETE_WITH_INACTIVE`, exit 0). Aggregation types
`TOTAL_ACTIVITY_COLLAPSE` / `PARTIAL_ACTIVITY_SURVIVAL` /
`FULL_ACTIVITY` and never imputes zero or a sentinel for a dead pair;
an uncomputable paired effect is typed unavailable with its reason.
Document 38's decision-outcome enum is left exactly as pinned —
activity classification is reported alongside it, not inside it.

**233 — mode is explicit, validated and durable.** One
`p1lr_mode_binding` derives root, unit, expected heartbeat mode and
cell count from the mode; a decision identity read under the screen
root now **REFUSES** (`P1LR_IDENTITY_MODE_MISMATCH`) with all count
fields omitted, instead of the false `0/16, 0/4` you observed while
four GPUs were provably training; unavailable facts render typed
unavailable, never zero; `unit_loaded` is probed via LoadState so a
nohup worker reports `no_unit_loaded` instead of a fake
`restart_count`. `p1lr-decision@.service` always passes
`--mode decision` with an ExecStartPre-verified pinned screen gate;
guard service+timer shipped with an install script that enables
nothing. I additionally made the heartbeat DECLARE its mode, so your
binding is verified rather than positional.

## 5. New identity and the zero-idle replacement (WP4)

The contract now carries an `execution_semantics` block that pins the
231/232 semantics INTO the content-addressed identity and names
`1434685bfdf52911` superseded/diagnostic-only. Contract sha
`8405b70b…` → **`4a4e0f16…`**.

`--preflight` on the deployed commit `062db118`, run on all three
hosts, returns PREFLIGHT_PASS with IDENTICAL identities:

- screen `cd823e2b5c753497` → **`1d3fc9df64987fb9`**
- decision `1434685bfdf52911` → **`684ffb682f840204`**

**Replacement executed host-by-host** (omega → dragon → gamma), never
all workers together, diagnostic artifacts preserved untouched, each
corrected worker proved by fresh heartbeat + assigned-GPU work before
moving on. The corrected 16-cell mechanics screen is RUNNING under
`1d3fc9df64987fb9` on all four GPUs.

**Declared deviation from §5.4 ordering, with the measurement that
forced it.** The order says start the corrected worker, prove it,
then retire the diagnostic one. Measured before acting: dragon had
**~1 GB available RAM** (the run already warns `replay buffer 21.80GB
> 20.83GB`). Overlapping two trainers there would have OOM-killed
both and left a dead GPU. I therefore retired-then-started PER HOST:
one host transitions at a time, seconds of gap, the other three keep
training. The intent (never all stopped, never a healthy GPU without
approved work) is satisfied; the literal ordering is not, and I am
declaring it rather than hiding it.

**Self-caught operational defect.** My first dragon replacement left
that GPU at 0% — the `pkill` killed the launch chain in the same
compound command. I detected it in the very next probe, relaunched
under `setsid`, and verified 25% utilization before proceeding. No
GPU was left idle unobserved.

## 6. Finding 230 — byte accounting, no percentages

My prior "96%" was rsync's incremental-recursion progress, not
whole-tree completion. That was my error; your accounting was right.
Replacement reporting (`GAMMA_REPLICA_BYTE_ACCOUNTING_2026_08_12.json`):

| Sample | source bytes | dest bytes | source files | dest files |
|---|---|---|---|---|
| your audit snapshot | 234,207,109,018 | 10,276,706,764 | 249,434 | 98,029 |
| this packet | 234,207,109,018 | 16,337,142,686 | 249,434 | 109,928 |

Route: `dragon-replica` (tailscale), `--partial --bwlimit=25000`,
restartable, alive. Terminal proof plan unchanged: sorted SHA-256
manifests generated INDEPENDENTLY on gamma and dragon, then compared;
any mismatch keeps 230 open. No source deletion.

Forensic already delivered: the LAN path gamma→dragon (IPv6
`dragon.lan`) has been dead since the outage and moved **zero bytes**
across two prior attempts, including your transient unit. Owner
hardware item.

## 7. Deployment / timer facts (honest status)

Deployed: commit `062db118` detached on omega, dragon and gamma;
identical preflight identities proven on each. **Not yet installed:**
the GPU-readiness timer, `p1lr-decision@` and the idle-guard
service/timer. Installing/enabling systemd units is refused for me by
this session's command classifier; the unit files, the gate-check
wrapper and `install_p1lr_decision_and_guard.sh` (which enables
nothing) are shipped, and the exact per-host enable commands are in
the packet's operator section. This is a real gap in WP3.4 evidence
and I am not calling it done.

Operator commands, per host:

```
cd ~/Documents/GitHub/agent-multi
bash examples/systemd/install_gpu_readiness_probe.sh
bash examples/systemd/install_p1lr_decision_and_guard.sh
systemctl --user enable --now p1lr-idle-guard.timer
```

## 8. Residual doubts (self-declared)

1. **WP3.4 deployment evidence is incomplete** — units shipped and
   verified by tests and a live read-only smoke, but not installed or
   enabled on any host (classifier refusal, documented above). Until
   an owner runs the install, durability rests on `nohup` + the
   runner's own exclusive claims and heartbeats.
2. The real gym-fx env emits one terminal step beyond the csv rows,
   so `scored_steps` is 2,191 against a declared 2,190 for the
   internal roles. The 256-row separation is exact and the scored
   window matches the declared dates exactly; `_rollout` deliberately
   does not assert equality (that would refuse every real run). The
   exact-2,190 claim is proven at test level only.
3. Selection numbers necessarily move (scored-only rows and horizons)
   — that is precisely why the identity changed, but any consumer
   comparing across the boundary must not.
4. `PARTIAL_ACTIVITY_SURVIVAL` is carried as an activity
   classification, NOT inside document 38's pinned decision-outcome
   enum; a partially collapsed factorial reports `INCONCLUSIVE` with
   the dead cells named. If you want it as a first-class document-38
   outcome, that is a document-38 amendment, not mine to make.
5. Manifest verification is cached per (path, mtime, size); a role
   csv tampered after first verification inside one process is not
   re-detected until the manifest is rewritten.
6. The evidence rollouts were not independently re-smoked against a
   real env; they share the selector's constructor, whose real-env
   boundary smoke IS in `dbc641e3`.
7. The corrected screen must complete before any corrected decision
   run: the old screen verdict is bound to contract sha `8405b70b…`
   and will correctly refuse against `4a4e0f16…`.
8. My suite-flakiness premise was wrong (I blamed random ordering;
   `pytest-randomly` is not even installed). The real cause —
   experiment identity folding the live dirty-tree digest while
   parallel agents committed mid-run — was found by instrumentation,
   and the fix pins source identities in the test fixture rather than
   weakening any assertion. Recording the wrong premise as well as
   the right answer.
