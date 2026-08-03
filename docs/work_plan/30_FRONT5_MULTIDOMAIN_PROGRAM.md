# 30. Front 5: Multi-Domain Program

Status timestamp: 2026-08-03 America/Bogota
Version: 1.0.0
Author: Satoshi III (Mujuro Utsutsu), successor technical lead in bootstrap
Authority: owner decision of 2026-08-03 — the program is DOCUMENTED now,
its resourced execution is GATED (§5); the flagship domain, portfolio
algorithmic trading, holds absolute priority at all times
Verification: General Musashi audits this program under the normal protocol

## 1. Purpose

Prove and preserve the multi-domain generality of the DOIN ecosystem —
every repository is plugin-based, including the main pipeline — without
diverting one candidate, one GPU-hour or one audit cycle from the
flagship mission while Fronts 1-2 are in their critical phase.

Three benefits justify documenting Front 5 today rather than later:

1. **Architectural insurance.** Interfaces only stay generic if a second
   domain is held against them while they evolve (§6).
2. **Evidence for DOIN's central claim.** A domain-agnostic optimization
   result is direct support for the P-series publication portfolio.
3. **Cold-start elimination.** When the gates open, work starts from a
   ranked, criteria-scored portfolio instead of a blank page.

## 2. Tier Structure

- **Tier A — authorized now, near-zero cost:** this document; the
  conformance discipline (§6); the Domain Zero probe design (§4.1) and
  the F5-SCOUT Hermes integration (§7), each as bounded side jobs under
  §8 rules and Musashi disposition.
- **Tier B — first resourced pilot domain:** opens only when every gate
  in §5 is green and the owner explicitly activates it.
- **Tier C — portfolio expansion:** subsequent domains, each with its own
  owner decision, after Tier B produces an audited result.

## 3. Selection Criteria

Each candidate is scored on: (a) adjacency to the existing stack
(time-series → decision under constraints → capped execution);
(b) open data availability; (c) synthetic generator or simulator
availability; (d) evaluation-loop speed; (e) regulatory/ethical burden;
(f) publication value; (g) compute demand relative to a CPU-only budget.

## 4. Domain Portfolio (ranked)

### 4.1 Domain Zero — optimization benchmark suites (generality probe)

HPOBench / NAS-Bench-201-class suites with precomputed evaluations: zero
data engineering, instant evaluation, CPU-only, directly publishable as a
domain-agnosticity result for the DOIN optimizer. This is the first Front
5 artifact and the only one admissible before the §5 gates, because it
consumes no scarce resource and touches no live system.

### 4.2 Pilot ranking

| Rank | Domain | Why here | Data / synthetic source |
| --- | --- | --- | --- |
| 1 | **Energy optimization** (battery arbitrage, load/price forecasting, dispatch) | Near-isomorphic to trading: forecast → constrained decision → capped execution; LTS risk discipline and gym environment transfer almost directly; "paper dispatch" against historical prices preserves our zero-harm staging | Grid operators / ENTSO-E-class open load & price series; historical-replay simulators |
| 2 | **Materials discovery** | Best pure fit for the evolutionary half of DOIN (composition/structure search over surrogates); fast surrogate loops; high publication value | Materials Project, OQMD; surrogate evaluators |
| 3 | **Precision agriculture** | Open crop simulators make it gym-compatible; simulation removes the seasonal ground-truth delay | DSSAT, APSIM simulators; satellite/weather open data |
| 4 | **Climate prediction** | Treated as a FEATURE SOURCE for the energy domain (weather → load/price), not a standalone front: compute-hungry and crowded with big-lab baselines at our scale | ERA5-class open reanalysis |
| 5 | **Medical diagnosis model optimization** | Deliberately last: highest regulatory, privacy and validation burden; a domain where wrong claims cause real harm deserves the mature, audited version of our method — the credibility flagship of a later phase | MIMIC/ISIC-class licensed datasets, when admitted |

### 4.3 Reserve candidates (documented, unranked)

- **Logistics / vehicle routing** — synthetic instance generators are
  trivial, evaluation is instant; the cleanest combinatorial-GA showcase.
- **Protein-ligand / antibody affinity surrogate optimization** —
  adjacent to materials; open affinity datasets and surrogate models.
- **Epidemic forecasting** — time-series adjacency; open surveillance
  data; high social value, moderate crowding.
- **Water-resource management** — reservoir/irrigation dispatch under
  constraints; structurally similar to energy.
- **Industrial predictive maintenance** — open run-to-failure datasets
  (C-MAPSS-class); forecast-plus-decision structure.
- **Spectrum / network resource allocation** — constrained combinatorial
  optimization with simulators.

## 5. Gates for Tier B (all must hold; Musashi-verifiable)

- **G1 — Front 2 proven:** IBKR L1 accepted through the complete audit
  chain and the owner-ratified canary executed: one long and one short
  protected round-trip, directly reconciled.
- **G2 — Front 1 self-driving:** job-0 champion archived and
  independently verified; job 1 selecting on robust weekly RAP without
  intervention.
- **G3 — statistical demo evidence** (replaces any single-day profit
  notion, which a coin flip can pass): at least 14 consecutive
  demo-trading days with zero S0/S1 incidents AND at least 30 closed
  protected trades whose weekly-RAP bootstrap confidence interval
  excludes zero.
- **G4 — capacity:** the pilot is CPU-only, never touches campaign GPUs,
  and the owner judges that Hermes, the technical lead and the auditor
  have demonstrable slack.

Gate evidence is collected from the existing status/OLAP tooling; no
parallel measurement system is created for gating.

## 6. Engineering Discipline (active immediately)

Every review of a NEW plugin contract or cross-repository interface must
answer, in the review record: *"Would this interface survive a
grid-dispatch or materials-discovery domain without modification?"* A
"no" does not block the merge — flagship first — but it must be recorded
as a named generality debt so Tier B starts with an honest inventory
instead of an archaeology project.

## 7. F5-SCOUT: Front 4/Hermes Integration (owner-directed 2026-08-03)

Hermes agents perform bounded internet research for cross-domain use
cases and draft "how our systems could address X" material for Moltbook
or other channels. Governance is inherited UNCHANGED from Front 3 — this
creates zero new authority:

1. runs only inside the existing Front 3 collection/triage slots and
   models; flagship cadence is never displaced;
2. deterministic collection into the existing social OLAP — no parallel
   idea store; the idea register is a labeled view of that OLAP;
3. every outbound draft is HUMAN-APPROVAL-GATED; Hermes never publishes;
4. drafts describe exploratory ideas only: no fabricated benchmarks, no
   claimed results we do not have, no non-public system details, no
   credentials, no account facts, no findings under audit;
5. Hermes remains a no-authority observer: it may propose, never decide;
   and
6. scout output that suggests a new domain feeds §4.3 through normal
   review, not directly into any work plan.

Activation of F5-SCOUT requires a Musashi disposition on its collection
scope and prompt set, same as every other Hermes job.

## 8. Resource Rules

- Flagship absolute priority: any Front 5 activity yields immediately to
  Fronts 1-2 work, including audit-response work.
- Tier A jobs run only in review-wait windows, exactly like
  `DEV-TOOLING-MCP-001`.
- CPU-only; no campaign GPUs; no new services on Omega without owner
  approval; no new external accounts or credentials.
- Every Front 5 artifact is versioned, committed and auditable; no
  acceptance claim exists only in chat.

## 9. Review Cadence

The portfolio and gates are re-examined at every gate state change, at
Tier B activation, and otherwise quarterly. Domain rankings are
falsifiable: new evidence (data access, simulator quality, regulatory
shifts) may reorder them through a versioned update to this document.
