# Independent Audit: L1 Round-3 Acceptance

Date: 2026-08-10 America/Bogota  
Auditor: General Musashi (Codex), independent auditor during the role swap  
Subject: `agent-multi@fe6224aa`  
Executable fleet revision: `agent-multi@f5e18696`  
Subject branch: `satoshi/m0-aggregation-hardening`

## 1. Verdict

**ACCEPT. Findings 196-200 are independently verified corrected. Decision
identity `2de49ea9225e2baf` is authorized and was launched on all four assigned
GPUs under the owner's standing execution authority.**

The corrected smoke has one immutable source digest, an independently rehashed
equal Dragon replica, sixteen loadable terminal models and no path from smoke
records into decision evidence. Both public aggregation entry points now use
the collection envelope. The deployed environments bind each seed to its exact
GPU UUID, and the physical smoke records show one CUDA device visible inside
each process. Financing is explicit and phase-1 training epochs no longer count
the baseline evaluation.

The smoke aggregation remains correctly `INCONCLUSIVE`: a mechanics smoke is
not scientific evidence. This does not block the decision run; it is the
expected refusal.

## 2. Independent Results

| Check | Result |
| --- | --- |
| source/replica seal | exact digest `6da390b5afe837af52f4b5574b017dc574d308a82bacd08ab799120262fd318f` |
| aggregation placement | outside the sealed tree; independent post-write rehash unchanged |
| aggregation authority | both CLIs route through `load_collection_envelope` |
| terminal artifacts | 16/16 load on the Dragon replica |
| GPU binding | all 16 records bind assigned UUID = `CUDA_VISIBLE_DEVICES`; CUDA device count = 1 |
| financing | explicit `charged=false`, mechanism and reason in 16/16 records |
| epoch accounting | requested 1, trained 1, baseline evaluations 1 in 16/16 records |
| decision identity | independently recomputed as `2de49ea9225e2baf` |
| pre-existing decision state | absent on Omega, Dragon and Gamma before launch |
| focused tests | 109 passed |
| complete suite | 898 passed; two declared sklearn convergence warnings |

Evidence:

- `docs/audits/evidence/repro_runs/MUSASHI_L1_ROUND3_ACCEPTANCE_REPRO_2026_08_10.py`
- `docs/audits/evidence/repro_runs/MUSASHI_L1_ROUND3_ACCEPTANCE_REPRO_2026_08_10.json`
- `docs/audits/evidence/eth_sac_inner_curriculum/SATOSHI_III_L1_ROUND3_RETURN_PACKET_2026_08_10.md`

## 3. Runtime Launch

The four durable workers are active on one decision identity:

| Seed | Host | GPU | Start evidence |
| --- | --- | --- | --- |
| 101 | omega | RTX 4070 Laptop | active, `L1_N_M10`, GPU 39%, 51 C |
| 202 | dragon | RTX 4090 Laptop | active, `L1_N_M10`, GPU 37%, 42 C |
| 303 | gamma | RTX 5070 Ti Laptop | active, `L1_N_M10`, GPU 39%, 42 C |
| 404 | gamma | RTX 5090 | active, `L1_N_M10`, GPU 45%, 51 C |

During launch, remote seed units were initially invoked on Omega by operator
error. The launcher's host/GPU contract refused all three before model
construction or training. Those refusal-only heartbeats were removed, the
local units were stopped/reset, and the correct Dragon/Gamma units were then
started. No decision cell, artifact, lock or result was produced by the
refused invocations. This is direct evidence that finding 198's fail-closed
binding works under a real operator mistake.

## 4. Scientific Meaning of This Run

This is a matched `2 x 2` factorial:

- phase-1 difficulty: normal (`N`) versus easy (`E`);
- phase-2 normal learning rate: baseline multiplier `1.0` versus `0.3`;
- baseline learning rate: `1e-4`, fixed for phase 1 in all arms;
- four paired seeds: 101, 202, 303 and 404.

It therefore estimates three distinct quantities:

1. the main effect of adding the easy phase while holding normal LR constant;
2. the main effect of normal-phase LR while averaging over difficulty; and
3. the interaction: whether the effect of easy changes with normal LR.

It does **not** estimate the best easy-phase learning rate. Adding that factor
now would destroy the clean attribution this decision run was built to obtain.
The independent `LR_easy x LR_normal` question is scheduled conditionally in
document 38 after this run reports its typed result.

## 5. Non-Blocking Improvement

The aggregation function performs the mandatory post-publication rehash and
returns `sealed_digest_after_write`, but writes the JSON artifact immediately
before adding that field to the in-memory result. The seal is independently
reproducible and unchanged, so this is not an acceptance blocker. Persist the
post-write digest in the next compatible evidence-schema revision so the
artifact is self-contained as well as independently verifiable.

