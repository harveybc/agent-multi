# P0 — Honest Report on the Active Zero-Activity Campaign

Date: 2026-08-06 — General Satoshi III
Runtime mutated by this report: **NONE** (read-only fleet queries only).
Prepared, NOT executed: the pause/archive packet in §4.

## 1. Direct fleet snapshot (read-only, ~14:57 America/Bogota)

One plan, one domain, one genesis, one population, one tip across all
four workers — coordination is healthy:

| Worker | tip | height | generation | claimed candidate | GPU |
|---|---|---|---|---|---|
| omega | `22e0f31417…` | 2 | 0 | 5/20 | 47 °C / 34 % |
| dragon | `22e0f31417…` | 2 | 0 | 2/20 | 50 °C / 35 % |
| gamma-5070ti | `22e0f31417…` | 2 | 0 | 1/20 | 49 °C / 42 % |
| gamma-5090 | `22e0f31417…` | 2 | 0 | 4/20 | 55 °C / 5 % |

Zero supervisor alerts. Component revisions on every worker:

```
agent-multi 5437a31 | doin-core e05a332 | doin-node b70ea03
doin-plugins 8c959a6 | gym-fx 9a084ac | trading-contracts cd05083
```

## 2. What the candidates are actually doing

Direct log evidence from omega's active candidate:

```
[epoch 1089/2000] L1 step-warmup<2001  trade_gate=FAIL
  composite=-1000000.0000  (checkpoint ineligible)
  actor|w|=8786.72 Δa=+0.0000  critic|w|=20496.63 Δc=-0.0024
  ent=0.0000  steps=8704000->8712000  buf=100000->100000
        TRAIN trades=0  profit=+0.00%  bal=10000.00
        VAL   trades=0  profit=+0.00%  bal=10000.00
```

Reading it plainly:

- **8.7 million environment steps, zero trades** on train and validation;
- **entropy coefficient is 0.0** and the **actor weight delta is exactly
  0.0** — the policy has stopped changing; it is a frozen no-trade
  policy being re-simulated 8,000 steps at a time;
- `trade_gate=FAIL` on every epoch, so no checkpoint can ever become
  eligible;
- the epoch budget is 2,000, so each of the four candidates can burn
  roughly twice what it has already burned before it stops on its own.

Musashi's characterisation is correct and I confirm it: **this is a
zero-activity compute sink, not progress.** GPU utilisation is not
evidence of useful work.

## 3. Why the running code cannot stop it

The workers run `agent-multi@5437a31`, which predates the bounded
activity budget (`5aca0450`, finding 127). On that revision an
activity-ineligible epoch never consumes patience, so the candidate
runs to the hard epoch cap. The corrected code terminates such a
candidate after a bounded no-activity streak and rejects it — but it is
not the code in flight, and hot-patching a running campaign under
changed evaluation semantics is explicitly forbidden.

## 4. Prepared pause/archive packet (NOT executed)

Requires the owner's explicit word. Tooling ready:

1. `tools/pause_doin_fleet.py --profile <phase-2 anchored profile>` —
   verified pause: stops every worker process group, verifies process,
   API port and GPU-owner absence, records a per-node binding.
2. Archive label: **`ZERO_ACTIVITY_INELIGIBLE_RUNTIME_5437a31`**.
   The chain, state dirs and all artifacts are preserved byte-for-byte
   under that label and are marked **never resumable as
   decision-bearing evidence** — they are diagnostic lineage of a
   code revision that could not enforce the activity budget.
3. `tools/replicate_decision_evidence.py` produces the canonical
   manifest of the archive before anything is moved.
4. No hot-patch, no resume under changed semantics: a corrected
   campaign gets a fresh plan/domain/genesis, as doc 33 requires.

## 5. The decision that belongs to the owner

Three options, stated without spin:

| Option | What it buys | What it costs |
|---|---|---|
| **A. Pause and archive now** | frees four GPUs immediately; ends the sink; keeps the chain as diagnostic evidence | loses nothing of scientific value — no candidate can produce an eligible checkpoint |
| **B. Let the four candidates finish** | a complete generation-zero record under the old semantics | roughly as much GPU time again, for candidates that are already proven ineligible |
| **C. Pause after the current generation** | tidier lineage boundary | same cost as B for the four in flight |

My recommendation, as technical lead: **Option A.** The four active
candidates cannot become eligible under their own selection contract;
continuing spends the fleet on a result whose value is already known to
be `rejected`. I will not execute it without the owner's word, and
Musashi must verify the pause packet afterwards.
