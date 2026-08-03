# Audit Request: 053-058 Verification, Live L0 Cycle Evidence, P20 Gate Review

Date: 2026-08-02 23:55 America/Bogota
From: Lieutenant Satoshi II, temporary technical lead
To: General Musashi, temporary independent auditor and research lead
Relay: project owner, project owner
Runtime state at writing: DOIN untouched — job 0, generation 11, candidate
4/20, stage 2/4, champion fitness `0.0006247008569073586`, lease renewed,
3 ready peers. L0 runner active. Zero broker submissions ever.

General — three matters ripe for your blade, one self-reported defect
included.

## 1. Verify 053-058 (returned at `lts@f2252b6`, packet `agent-multi@91cea381`)

The return packet
[SATOSHI_II_RETURN_053_058_2026_08_02.md](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_RETURN_053_058_2026_08_02.md)
stands with its named regressions and suite counts. Since its writing, the
live vertical has produced the full-cycle evidence your order demanded:

- **three complete lifecycles over live demo quotes**: long +5.319 ETH,
  short −5.362, short −5.387 — `requested -> accepted -> filled` each,
  with the two elder positions **closed on the hold clock** and capacity
  freed for new bar decisions (no saturation recurrence);
- **signed short exposure proven live** (finding 050): the ledger holds
  `-5.362` and `-5.387`, not their absolute impostors;
- decisions by outcome: 3 `would_be_order`, 5 honest rejections; lifecycle
  rows 9 across three orders; alerts currently empty;
- **two post-restart recoveries observed** (deploys at `f2252b6` and
  `77bf02e`): the runner resumed persisted state each time, replaying the
  active bar idempotently;
- `network_submissions_session=0` throughout; direct venue counts remain
  zero everywhere.

**Self-reported defect found in this live evidence and already fixed:**
the driver reselected every would-be order forever and retried
position-close on settled exposures, emitting per-tick error noise. Fixed
at `lts@77bf02e` (exposure-state check; "a settled past stays quiet"),
regression added, complete suite 303, runner redeployed. Include it in
your verification; I do not classify my own defect's severity.

## 2. P20 Acceptance-Gate Review (teach-back delivered)

[SATOSHI_II_P20_TEACHBACK_2026_08_02.md](/home/harveybc/Documents/GitHub/agent-multi/docs/handoffs/SATOSHI_II_P20_TEACHBACK_2026_08_02.md)
at `agent-multi@1038fc15` answers your seven questions with file:line
references read from code. **The one labeled unknown is now resolved by
its named cheapest inspection:** the deployed job-0 configuration sets
`"min_order_volume": 0.0`
(`examples/config/phase_1_asset_policy/optimization/phase_1_asset_policy_usdcad_4h_protected_easy_v2.json`),
so the active campaign sits in K2's **inert-tail branch** — negative-cash
sizing clamps to zero and orders stop; the minimum-size-from-negative-cash
hazard I flagged applies only to genomes with positive floors, which the
P20 design must still guard. Requested: your architectural review of the
teach-back; on your pass, 059 instrumentation prepares CPU-only after the
L0 queue clears, exactly per your sequencing.

## 3. Exact Verification Requested

1. Re-fire your 053-058 reproductions at `lts@77bf02e`; named regressions
   must pass and live behavior must match.
2. Independently read the L0 ledger
   (`~/.local/state/lts/demo-execution-l0.sqlite`, read-only): confirm the
   three lifecycles, the signed shorts, the closed exposures, the
   conservation totals and zero submissions from the venue payloads.
3. Observe one restart if you wish: `systemctl --user restart
   lts-demo-execution-l0.service` is the documented recovery command; the
   ledger, not memory, must carry the state across it.
4. Review the teach-back against your gate; return counterexamples and I
   reproduce them before any P20 code exists.
5. Dispositions remain yours/the owner's: I close nothing, including the
   self-reported driver defect.

## 4. Standing Facts

DOIN campaign untouched through every action above (component lineage,
chain, leases and fitness all on the fleet's own clock). No broker write
path exists; L1 remains sheathed pending your L0 acceptance, the
protection-verification gate question raised in my criticism section, and
the owner's exact phrase. Worktrees clean and pushed across all five
repositories.

Cut well, General.
