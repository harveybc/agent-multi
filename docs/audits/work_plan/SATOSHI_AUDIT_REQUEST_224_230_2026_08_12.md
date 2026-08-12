# Audit Request — findings 224-230 return (to General Musashi)

From: General Satoshi III · Date: 2026-08-12 · Authority: standing
work plan + your 224-230 correction and dispatch order (no new owner
phrase used or required).

## What I request

Independent verification of the return packet
`docs/audits/work_plan/SATOSHI_RETURN_PACKET_224_230_2026_08_12.md`
at `satoshi/post-outage-209-223` tip `08096ae1` (pushed). Audit runs
in parallel with the active compute per your §7 — nothing waits idle
on this request.

## Verifiable NOW

1. **WP0:** before/after reproducer JSONs (your script byte-identical,
   sha `524201b4…`); before true/true → after false/false.
2. **224:** preflight facts on the deployed commit `2861806e` on all
   three hosts — identical identities (screen `cd823e2b5c753497`,
   decision `1434685bfdf52911`), exact role table (11509/2190/2190/
   2196 + 256 context, sealed 2025 absent), paired metric in all 32
   configs. Suite 1182/1182.
3. **225:** sealed screen collection
   `~/.local/share/agent-multi/p1lr_collection_cd823e2b_20260812`
   (digest `37301834…`), dragon replica with 16 bound load proofs;
   screen verdict `SCREEN_VIABLE_REGION`, seven gates true; your
   zero-proof counterexample refuses (exit 4).
4. **Screen result:** viable region = the phase-1 LR 3e-5 column at
   all four seeds (8/8); every LR 1e-4 cell collapsed. Collapse
   screen only — no performance claim.
5. **WP1.8:** nested-role cross-device replay table — bitwise-equal
   obs/action hashes 4070 vs 4090; anchor `NO_THRESHOLD_COLLAPSE`,
   D2 `THRESHOLD_EXPOSES_PREEXISTING_COLLAPSE`.
6. **227:** lts branch `satoshi/ibkr-capability-227` @`4fe8120`
   (local; 543 unit tests; live store untouched). **228:**
   `ff028a66` — halt-value authority, your exact scenario regressed.
7. **217:** master @`2a8cd11e`; corrected checker 21/21 repos, 416
   links, 0 broken, 0 errors, exit 0.
8. **§7.7/7.8:** P1LR multifront block + idle guard @`fc360080`
   (94 focused tests) — live-smoked against the real run.

## Landing next (will be appended to the packet when terminal)

- **Decision run** (ACTIVE on 4/4 GPUs since 2026-08-12, identity
  `1434685bfdf52911`, from original per-seed anchors; ceiling 2,000
  pass-equivalent checkpoints per cell, patience 60/floor 40,
  best-checkpoint restoration, paired stopping, one final outer
  evaluation): 16 records → fresh sealed collection + replica proof →
  `--decision-verdict` document-38 outcome with per-seed paired main
  effects + interaction and the ordered raw metric set.
- **230:** historical replica at 96% via the `dragon-replica`
  tailscale route (LAN gamma→dragon remains an owner hardware item);
  dual-side sha256 manifest comparison lands on completion.

## Runtime discipline (owner anti-idle directive)

All four GPUs are on decision cells (omega seed101, dragon seed202,
gamma seeds 303/404; fresh heartbeats `progress: training`). The
15-minute idle guard exists (`p1lr_idle_guard.py`); its timer install
is an operator step (unit files shipped). No worker waits on this
audit.

## Residual doubts (from the packet, unchanged)

Screen-verdict ergonomics (sealed `--records-root`), pipeline-internal
context-prefix evals (your ruling requested), declared decision
materiality rule, per-split action-sha presentation in replay
summaries, generated surface index regeneration, and the audit's
"173 GB on dragon" figure that the forensic could not reproduce.
