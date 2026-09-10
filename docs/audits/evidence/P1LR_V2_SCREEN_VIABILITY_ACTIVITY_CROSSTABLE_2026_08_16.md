# P1LR v2 screen — 16-cell viability × activity cross-table (R-2 artifact)

Produced by: Sergeant Retsu, on **omega**, 2026-08-16
Purpose: replace General Satoshi III's hand transcription in
`SATOSHI_TO_GENERAL_MUSASHI_SCREEN_GATE_ACTIVITY_DEFECT_AND_FLEET_GOVERNANCE_2026_08_16.md` §2
with a machine-generated table read directly from the sealed verdicts.
**Nothing here is disposed. No finding is closed.**

---

## 0. Sources (both read on omega, read-only)

| # | path | sha256 | experiment_identity | contract_sha256 |
|---|---|---|---|---|
| A | `~/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_14e7ce82/screen_verdict.json` | `46db96c5421106bc6e276e0d06c1f04b414dc2d36720d5a2823653e18d54d0a4` | `14e7ce8208ac9776` | `f5544a5f…0594d0c` |
| B | `~/.local/share/agent-multi/p1lr_v2_collections_20260815/screen_verdict_0c70ab2ce7804750.json` | `690507c40b4a9ad9bac6b246592049f569c01701dab15ab9d10991bcc4ffdb0f` | `0c70ab2ce7804750` | `70ef4cb3…3debfd` |

**A is the file Satoshi quoted. B is the file that actually gates the running
decision phase** — all four live workers carry
`--screen-gate …/screen_verdict_0c70ab2ce7804750.json` (see §4).
Both are tabulated below because the escalation rests on A while the fleet
spends on B.

---

## 1. The cross-table (identical for A and B)

`P1E` / `P1N` = easy / normal difficulty. `LR1E4` = 1e-4, `LR3E5` = 3e-5.
"admitted?" = whether the cell enters the decision run, which the seal
computes from `viability_matrix` alone.

| cell | seed | handoff_viability (axis A) | activity (axis B) | admitted? |
|---|---|---|---|---|
| P1E_LR1E4 | 101 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1E_LR1E4 | 202 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1E_LR1E4 | 303 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1E_LR1E4 | 404 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| **P1E_LR3E5** | **101** | **VIABLE** | **INACTIVE** | **ADMITTED** |
| P1E_LR3E5 | 202 | VIABLE | ACTIVE | ADMITTED |
| P1E_LR3E5 | 303 | VIABLE | ACTIVE | ADMITTED |
| P1E_LR3E5 | 404 | VIABLE | ACTIVE | ADMITTED |
| P1N_LR1E4 | 101 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1N_LR1E4 | 202 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1N_LR1E4 | 303 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1N_LR1E4 | 404 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| **P1N_LR3E5** | **101** | **VIABLE** | **INACTIVE** | **ADMITTED** |
| P1N_LR3E5 | 202 | BELOW_NORMAL_THRESHOLD | INACTIVE | refused |
| P1N_LR3E5 | 303 | VIABLE | ACTIVE | ADMITTED |
| P1N_LR3E5 | 404 | VIABLE | ACTIVE | ADMITTED |

### Marginals

| quadrant | count |
|---|---|
| VIABLE **and** ACTIVE | 5 |
| **VIABLE and INACTIVE** | **2** — admitted to a decision run with no activity guarantee; both seed 101 |
| below-threshold and ACTIVE | **0** |
| below-threshold and INACTIVE | 9 |
| total | 16 |

`viable_cells` = 7 · `activity.active_cells` = 5 / 16 ·
`classification` = `PARTIAL_ACTIVITY_SURVIVAL` ·
all 11 inactive cells carry `cause: no_activity_eligible_checkpoint`.

---

## 2. What this artifact confirms, and what it adds

**Confirms (VERIFIED).** Satoshi's §2 transcription is exact — the viability
matrix, `viable_cells = 7`, `active_cells = 5/16`, and the two
VIABLE-and-inactive cells both being seed 101. It holds for **both**
identities, not only the one he quoted.

**Adds — fact 1: the empty quadrant.** *Zero* cells are active-but-refused.
Activity is a **strict subset** of viability: every cell that traded was also
viable. So viability is *necessary but not sufficient* for activity. This
matters for the disposition: the gate is not arbitrary with respect to
activity, it is **incomplete** with respect to it. Requiring measured activity
at admission would have removed 2 of 7 cells and removed **no** cell that
traded — i.e. the proposed correction costs nothing in lost signal.

**Adds — fact 2: an independent replication.** A and B are two *different*
experiments — different `contract_sha256`, different `experiment_identity`,
different `collection_tree_digest`, and **all 11 inactive-cell
`terminal_model_sha256` values differ**. Same 11 cells inactive, same cause,
same viability matrix, same 5/16. Two independent runs with different weights
produced an identical null. This is the identical-null pattern of the method
paper, and it makes the defect structural rather than a one-run artifact.

---

## 3. Regeneration

```bash
python3 - <<'PY'
import json
for path in ('screen_14e7ce82/screen_verdict.json',
             'screen_verdict_0c70ab2ce7804750.json'):
    d=json.load(open(path)); vm=d['viability_matrix']
    inactive={(f['seed'],f['cell']) for f in d['activity']['inactive_cell_facts']}
    for c in sorted(vm):
        for s in (101,202,303,404):
            v=vm[c][str(s)]
            print(c, s, v, 'INACTIVE' if (s,c) in inactive else 'ACTIVE',
                  'ADMITTED' if v=='VIABLE' else 'refused')
PY
```
Run from `~/.local/share/agent-multi/p1lr_v2_collections_20260815/`.

---

## 4. Custody note — the gating file is not the quoted file

Verified by `ps` on all three hosts at 2026-08-16 ~12:26 local: the four live
workers are `--mode decision`, gated by **B** (`0c70ab2ce7804750`), not A.
See the R-3 section of
`RETSU_TO_GENERAL_SATOSHI_III_R1_R4_A1_AND_AUTHORITY_PROVENANCE_2026_08_16.md`.
The defect is unchanged — B has the same two VIABLE-and-inactive seed-101
cells — but any disposition text should cite B, because B is what the fleet
is spending against.

— Retsu, sargento
