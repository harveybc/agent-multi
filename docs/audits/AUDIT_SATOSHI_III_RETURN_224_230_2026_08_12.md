# Audit: Satoshi III Return 224-230

Date: 2026-08-12 America/Bogota  
Auditor: General Musashi, independent verifier  
Subject branch: `satoshi/post-outage-209-223`  
Return tip: `08096ae1`  
Audit-request tip: `465a5238`  
Verdict: **screen accepted; decision run is diagnostic pending corrections**

## 1. Findings First

### AUD-F1-20260812-231 (S2): internal L1 selection scores causal-prefix rows

The nested contract correctly materializes 256 context rows before both
`train_monitor` and `inner_validation`, and the reusable
`ContextPrefixWrapper` correctly forces hold and detects account mutation.
The executing internal selection path never installs that wrapper:

- `pipeline_plugins/rl_pipeline_with_validation.py:691-699` creates and wraps
  the environment only through the agent plugin;
- `pipeline_plugins/rl_pipeline_with_validation.py:701-737` sends that raw
  environment into the baseline and per-epoch split rollout; and
- `pipeline_plugins/rl_pipeline_with_validation.py:789-844` predicts actions,
  updates action statistics, accumulates rewards and emits return-trace rows
  without checking `info["is_context_prefix"]`.

The final outer-validation helper does install `ContextPrefixWrapper`, but
that occurs after checkpoint selection. Pairing does not cure this: different
arms may take different context actions and therefore contaminate the very
train-monitor/inner-validation values used to choose the checkpoint.

Required correction: apply the typed role context to every internal split
evaluation, force hold before the scored interval, and exclude prefix rows
from rewards, traces, action statistics, trade counts and metric horizons.
Prove zero account/replay mutation and equal scored rows before and after the
fix. Generate a new decision identity; the active identity may be retained
only as diagnostic evidence.

### AUD-F1-20260812-232 (S2): an inactive decision cell cannot publish its typed result

The pipeline deliberately returns an inactive cell as a measured outcome with
`best_model_path=None` and a hashed terminal artifact
(`rl_pipeline_with_validation.py:1480-1538`). The decision runner then
unconditionally requires a best checkpoint and raises before publishing the
record (`tools/p1_difficulty_lr_factorial.py:1292-1307`). This contradicts the
declared 16-record factorial and makes `TOTAL_ACTIVITY_COLLAPSE` unreachable.
It also terminates a seed runner before its remaining three cells run.

This is not hypothetical: at the audit snapshot seed 101 and seed 303 still
had `best=-inf`, while all four current cells had failed their trade gate.

Required correction: inactive cells land an explicit non-promotable record,
bind and load-prove the terminal artifact, receive one final outer evaluation
only as diagnostic truth, and never masquerade as a best checkpoint. A seed
must continue after an inactive cell. Aggregation must distinguish all-cell
collapse, partial activity survival and complete paired performance without
inventing a score for missing active pairs.

### AUD-GEN-20260812-233 (S3): decision monitoring and recovery observe the screen

The active decision run writes below the distinct `decision_run.output_root`,
but both operator surfaces read the top-level screen root:

- `tools/multifront_status.py:1435-1454` reads `contract.output_root`;
- `tools/p1lr_idle_guard.py:202-209` does the same;
- `tools/p1lr_idle_guard.py:48` hard-codes `p1lr-screen@{seed}.service`; and
- the only shipped P1LR unit is
  `examples/systemd/p1lr-screen@.service`, which defaults to screen mode.

The independent live invocation with decision identity `1434685bfdf52911`
reported mode `screen`, 0/16 records and 0/4 workers, while direct process and
GPU facts proved one decision process per seed and all four GPUs active. No
idle-guard service/timer is shipped or installed; the run currently depends on
`nohup` processes. Finding 229 is therefore only partially corrected.

Required correction: mode-aware root/unit selection, a durable decision unit,
an actual guard service/timer and deployment evidence from all three hosts.
The status acceptance fixture must prove 4/4 for the live decision root and
must fail if pointed at the screen root.

## 2. Existing-Finding Dispositions

| Finding | Independent disposition |
| --- | --- |
| 217 | Verified corrected, pending owner closure. `origin/master` contains `2a8cd11e`; the committed remote-default check records 21 repositories, 416 relative links and zero broken links/errors. |
| 220 | Remains open. The successor screen is valid, but the decision-bearing successor cannot be promoted under findings 231-232. |
| 221 | Component evidence verified; integrated promotion remains blocked by 231-232. |
| 222 | Corrected replay wording and nested-role cross-device replay are accepted as diagnostic evidence. |
| 223 | Remains independently verified corrected, pending owner closure. |
| 224 | Structural role counts, hashes, paired metric and sealed-test absence reproduce exactly; integrated acceptance remains blocked by finding 231 because the executing selector does not honor the declared context semantics. |
| 225 | Independently verified corrected, pending owner closure. The screen refuses malformed custody and the live sealed proof has 16/16 load-proven terminals on Dragon. |
| 226 | Decision mode exists and is running, but is not accepted until 231-232 are corrected and a new identity passes smoke. |
| 227 | Code correction and 43 focused tests independently pass at local LTS commit `4fe8120`; remains open because the branch is unpushed, unmerged and undeployed. |
| 228 | Independently verified corrected, pending owner closure. Current durable halt value is authoritative. |
| 229 | Partially corrected: 4/4 workers are dispatched, but monitoring and restart durability do not cover decision mode; finding 233 carries the remaining defect. |
| 230 | Running, incomplete. The reported 95-96% is rsync incremental-recursion progress, not whole-source completion. Snapshot: 234,207,109,018 source bytes versus 10,276,706,764 destination bytes (4.39%), 249,434 versus 98,029 files; no dual digest exists yet. |

## 3. Accepted Evidence

- Independent original reproducer: both prior 224/225 defect flags are false.
- Nested roles: 11,509 fit rows; 2,190 monitor; 2,190 inner; 2,196
  outer; 256 causal context rows on each evaluation role; 2025 absent.
- Screen verdict independently rerun: `SCREEN_VIABLE_REGION`; all seven gates
  true; `3e-5` is viable under both easy and normal at all four seeds; `1e-4`
  collapses at all four seeds.
- Dragon terminal custody independently rehashed: 16/16 file hashes match.
- Focused correction suite: 199 passed.
- Full merged audit worktree: 1,205 passed, two non-failing sklearn convergence
  warnings.
- LTS finding-227 focused suite: 43 passed.
- Active runtime: four direct decision processes, one per seed; GPU snapshot
  showed 22-44% utilization and 38-57 C. No worker was stopped for this audit.

Canonical independent reproducer and result:

- `evidence/repro_runs/MUSASHI_224_230_RETURN_REPRO_2026_08_12.py`
- `evidence/repro_runs/MUSASHI_224_230_RETURN_REPRO_2026_08_12.json`

## 4. Scientific Ruling

The screen answers a real question: phase-1 LR `3e-5` prevents the immediate
one-pass collapse observed at `1e-4`, across both difficulty arms and all four
seeds. It does not yet answer whether easy dynamics improves the long-run
normal policy.

The current decision identity continues only to preserve operational
continuity and useful diagnostics while corrections are built. Its results
must not enter the decision verdict, a warm start, DOIN genes, champion
succession or publication. The corrected decision run starts from the same
original per-seed anchors under a new content-addressed identity.

Report raw per-seed paired effects and effect magnitudes. Do not call a merely
nonzero, sign-consistent delta "material" unless a practical threshold is
predeclared; otherwise label it directional and preserve the raw values.

## 5. Owner Action

No owner phrase is required. Existing standing authorization covers the
correction, smoke and corrected decision run. Satoshi must execute the
companion order while the present diagnostic run keeps the GPUs occupied.

