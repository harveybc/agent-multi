# Satoshi Amendments to the Role-Swap Protocol

Date: 2026-08-01
Author: General Satoshi, responding to the owner's order to review and improve
before activation
State: proposed; owner agreement activates the amended protocol
Base document: `TEMPORARY_MUSASHI_SATOSHI_ROLE_SWAP_PROTOCOL_2026_08_01.md`
(accepted in full; these amend, none replace)

## A1. Tacit-State Handover Packet (the largest gap)

The cold-start drill correctly tests repository-only reconstruction — but the
*warm* handover should first capture what repositories do not hold. Known
concrete example: the transient five-minute lease guard (doc 15 §3.5) that
"must remain active until the current campaign stops" — an undocumented-
runtime-dependency class that a new technical lead could break innocently.

Amendment: before `ROLE_SWAP_ACTIVE`, Musashi produces
`TECH_LEAD_TACIT_STATE_2026_08_01.md`: in-flight work and intentions; live
cron/systemd/timer inventory per host; fragile spots and workarounds
(PyTorch GPU ordering, linger/SSH bridge, port 62024, lease guard, anything
similar); environmental quirks; "what I was about to do next." This also
converts tacit knowledge to versioned knowledge permanently — a resilience
gain independent of the drill.

## A2. Cross-Epoch Closure Rule

Protocol §7.7 says findings are never closed by their implementer. Add: **nor
by their author, in either role epoch.** Concrete case: finding 034 was
authored by Satoshi and fixed by Musashi (`lts@11d8958`). After the swap,
Satoshi is a party as author and Musashi as implementer — neither may close
it. Findings with both agents as parties escalate to owner closure or wait
for handback.

## A3. Default Exit Checkpoint

§5 state 4 leaves the end open. Add a default review event so the drill
cannot drift: at **job-0 completion + verified champion/elite archive** (a
natural high-stakes test of the new technical lead), the owner decides: end,
extend, or adopt. Emergency clause: any S0/S1 during the swap entitles either
agent to recommend immediate handback; the owner decides.

## A4. Auditor Queue Transfers Explicitly

The auditor's open task queue (AT-F1-013 archive verification, AT-ACADEMIC-031
ledger verification, AT-SEC-020/021/022 deep passes, fork/idle re-measurement,
034 verification subject to A2) transfers to Musashi as a **binding backlog**
(`docs/audits/work_plan/01`), not as ambient context. His prompt must name it.

## A5. Clean-Worktree Activation

Baseline §1 records Satoshi's register/recovery edits as uncommitted.
Amendment: the swap activates only from clean worktrees; the **outgoing**
technical lead commits pending counterpart artifacts as his final act (as he
has done throughout), so the incoming lead's first commit is never his own
prior-role state. Boundary hygiene on day zero.

## A6. First Technical-Lead Deliverable Pre-Declared

Protocol §8 asks the new lead to replace manually reconstructed status with a
machine-readable multi-front status contract. Satoshi pre-commits to it as
the **first implementation deliverable**: extend the existing audit-snapshot
collector with Front-2/3/4 sections and an executable-queue-versus-plan
section, so the improvement is measurable, not aspirational.

## A7. Staggered Cold-Starts

§9 launches new conversations for each role. Amendment: stagger them — never
both agents cold simultaneously. During each single cold-start, the other
agent remains warm as incident cover. The fleet must always have one
context-bearing agent while a drill runs.

## Acceptance

With A1-A7 accepted (or amended by the owner), Satoshi will execute the
invocation completely: state preservation, independent takeover verification
(including the Omega finalized-anchor lag, Gamma disk at 88 %, IBKR window,
claim uniqueness, and finding 034 evidence), the complete Musashi auditor
cold-start prompt, and the takeover report with `TAKEOVER_ACCEPTED`.
