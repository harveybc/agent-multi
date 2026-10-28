# Musashi: audit of the expert-guided mechanism proposal

Date: 2026-10-28 (America/Bogota). Reviewer: Musashi.
Audited identity: heuristic-strategy `b4b19ed3dc3ed69da8e1e388043fb375c700c6cc`.
Proposal and request: `docs/short_horizon/clean_round/` files
`PROPUESTA_MECANISMO_CARRIL_GUIADO_ZIGZAG_2026_10_28.md` and
`SOLICITUD_AUDITORIA_MUSASHI_MECANISMO_GUIADO_2026_10_28.md`.

## Verdict and immediate handoff

**REVISE before G2 or learned-policy claims. G0/G1 mathematical and contract
fixtures may proceed.** The mechanism is a useful design candidate, not rejected
as a research direction. Preserve the independent game identity and the full
trajectory difference reward. Fix the verifier and endogenous target contracts
before optimizing the mechanism against their verdicts.

| Dimension | Verdict |
|---|---|
| Mathematical model | ACCEPT_WITH_CHANGES: potential identity valid; Shapley learning is not the same game |
| Business/execution | REVISE: updated owner contract, causal query anchor, barrier/cost sufficiency |
| Test sufficiency | REVISE: reproduced false-effective-PASS paths and unsupported expectations |
| Labels/learning | REVISE: profile-dependent labels, horizon censoring, learning loss and uncertainty semantics |
| Feasibility | ACCEPT_WITH_CHANGES: exact G1 first, measured rollout budget, no claimed free 32-coalition learning |
| Independent identity | ACCEPT: guided labels and producer cannot be imported as pure-RL evidence |

**Owner clarification supersedes both old configurations:** leverage 50,
rel_volume 0.1, one simultaneous position, **no strategy-imposed daily entry
cap** (three or more openings permitted), wait 12h after weekly reopening,
forced flat 12h before weekly close, no entry at/after that cutoff. Read
[BUSINESS_CONTRACT.json](../work_plan/mechanism_synthesis/BUSINESS_CONTRACT.json)
from agent-multi commit `ba2af54a`. Do not rewrite old experiments. Read back
identical bytes/digest for new paired runs, plus compatible broker/calendar/cost
adapter identities. This is not a legal ruling about PDT.

## Evidence actually examined

Read both design documents and pinned source for `strategy_behaviour_matrix.py`
(validity and witness revisions), `strategy_qualification.py` (controls, seeded
positions and confirmation), `strategy_output_simulation.py` (BCHead,
head_core, advantage_full, path_advantage), and `strategy_zz.py` (filter/close).
Read QUALIFICATION_SPEC.json and HEAD_ERROR_PHI.json. No full matrix, DEAP,
GPU training or broker run was repeated; reported 73/96%, 4/12% and 12x variance
figures remain **AUTHOR_REPORTED_NOT_INDEPENDENTLY_REPLAYED** in this audit.

Executed the actual pinned `validity` function, extracted by AST without loading
training modules, on three synthetic dictionaries. Also evaluated algebraic
counterexamples. Reproducer: `tools/audit_guided_mechanism_design.py --source-repo
<strategy-checkout>`. Outputs and source SHA:
[GUIDED_COUNTEREXAMPLES.json](evidence/mechanism_design_20261028/GUIDED_COUNTEREXAMPLES.json).
These witnesses are bounded correctness tests, not financial results.

## Findings and required changes

| ID | Severity / evidence | Finding, correction and acceptance test |
|---|---|---|
| GM01 | BLOCKER / EXECUTED | `validity` at line 1768 returns PASS on an empty witness set; zero-event controls can select a corrected passing rule; one failing and one passing control still selects DEFECT and the corrected rule. Require expected witness IDs, non-vacuous applicability and separate DEFECT/UNVALIDATED/CONFLICT states that cannot confer acceptance. Reproduce the three retained input/output witnesses. |
| GM02 | MAJOR / SOURCE_VERIFIED | W4 v3-prime permits up to 5% destructive closes at barrier-reach edges after observing failures; source cites costs up to 1.79 sigma. Small price distance to a barrier does not bound economic regret. Retain as exploratory revision, not proof of original requirement. Use an explicit economic-error bound, independent oracle checks and a new blind panel under one rule per defined scenario class. |
| GM03 | MAJOR / EXECUTED | `fitness >= .9 * optimum` rejects the optimum itself when best=-10 (threshold=-9), and changes under an additive shift of utility. Replace by declared absolute/normalized degradation from a named control, or a Pareto epsilon constraint with a positive fixed scale. All safety witnesses still apply. The 90% rule is not a settled portable contract. |
| GM04 | BLOCKER for equilibrium-learning claims / EXECUTED | Shapley can reward a unilateral change that lowers Phi. Retained table gives Phi 4->3, player Shapley 2->6.5. Keep difference reward for the solved game; Shapley is diagnostic unless a separately specified learner objective is analyzed. Recompute deviation certificates for the actual trained objective. |
| GM05 | MAJOR / DERIVED | Per-level ideal opener bid depends on the risk profile and exit rule being optimized. Changing SL/TP or close semantics changes its labels and `p_act*`. Either regenerate labels per profile with identity or condition a declared producer on the profile. Frozen labels from another profile cannot govern G2/G4. Test a profile change that reverses the sign of net payoff. |
| GM06 | MAJOR / DERIVED | Retrospective label semantics do not make p=0.9 calibrated; ECE/Brier are meaningful for binary events, not arbitrary continuous bids/curves. Report binary reliability separately from regression error, interval coverage and proper distribution scores. The event for a close must be advantage against a specified continuation, not positive total PnL: closing at -1 can be right if holding yields -10. |
| GM07 | MAJOR / SOURCE_VERIFIED boundary | `advantage_full` uses `env.o[b+1]` to query remaining barriers; `head_core` derives lead h from the true future path and uses it in noise and uncertainty. Valid as declared ideal-output construction, not evidence of a causal production interface. Freeze reports at decision time; forbid queries using unobserved next-open prices or true future lead. Define how fill-time anchoring works without revising an already selected action. |
| GM08 | MAJOR / DERIVED | Exact market path curves can be sufficient under stated fills/costs; pointwise mean excursions need not be sufficient for expected first passage, correlated barriers or stopping. Define joint/order-aware outputs, horizon, censor mask and terminal continuation; do not extrapolate a 1h curve into a 24h optimum. Add equal marginal summaries with different event order, bid/ask, financing, gap and simultaneous-barrier witnesses. |
| GM09 | MAJOR / DERIVED | Temporal persistence/filtering can reduce false closes but delay a true urgent exit; a 5x5 cross-scale penalty can learn a disguised veto. Keep immediate/unfiltered/zero-penalty controls, fast genuine reversal and false reversal pairs, reset state by position. Scores and uncertainty must share declared units across scales. No matrix entry may bypass mandatory protection. |
| GM10 | MAJOR / DERIVED | MD-2 is not individual rationality: nonpositive historical group Shapley does not imply exclusion is optimal in a new state, and Shapley need not match U_i. Remove the claimed theorem and hard exclusion; evaluate it as a calibration-frozen pruning hypothesis with complementarity and regime-shift controls. |
| GM11 | MAJOR / DERIVED | Relative-risk bids are not common account utility: 2 units of return on risk 1 yields less money than 1 unit on risk 10, with different durations and blocked opportunities. Declare economic conversion/reference and evaluate full trajectories. Hedge regret bounds for fixed bounded losses do not apply to position-dependent histories. Keep Hedge only as an empirical ablation; no guaranteed MD-3 threshold without its assumptions. |
| GM12 | MAJOR / DERIVED | Mandatory quality-monotonic performance is false for a fixed controller. Noise can accidentally improve a biased decision, and a worse closer can improve account opportunity turnover. Separate information-value claims for an optimal controller under nested observations from empirical ladders for fixed policies. Nonmonotonicity is a diagnostic, not automatically a defect. |
| GM13 | MAJOR / SOURCE_VERIFIED | QUALIFICATION_SPEC and `cmd_controls` still require swapped net <= -0.5 baseline and monotone delayed Sharpe; this conflicts with the proposal's claimed replacement. Permutation need not lose exactly costs in trending markets. Use effect-specific falsifiable witnesses, retain old outcomes and version the changed contract; do not call all these negative controls universal laws. |
| GM14 | MAJOR / EXECUTED | Planned stop risk is not a hard realized-loss bound: entry100/SL99/gap95 loses5 versus planned1 before fees. Compounding, spread and currency conversion also matter. The claimed n_trades*risk bound and complementary win rate are not valid generic replacements for mirror-loss expectations. Mirror complete worlds/quotes/costs for symmetry tests; channel swaps are a different intervention. |
| GM15 | MAJOR / DERIVED | G-2 >=80% matching-scale wins, G-3 monotonically larger stops and G-4 every level retains pure-world behavior are not implied by ideal signals. Overlap changes incentives; distinct experts may be equivalent. Use oracle optimal action sets and regret, with event support. A smaller-scale expert winning is not by itself an error. |
| GM16 | MAJOR / DERIVED | G0 may map two requested durations to the same ZigZag level; maximum segment length is not necessarily full label support including pivot confirmation. Require unique-assignment rule or declared unavailable scales, event-time maturity including confirmation lag, boundary censoring, calibration-only level choice, and effective episode counts, not correlated row counts. |
| GM17 | MAJOR / DERIVED | Negative regret cannot be used literally as a weighted squared-error coefficient: weight=-1 makes loss unbounded below. Specify complete positive-weight AWR/imitation loss, policy log-likelihood, NOP/HOLD counterfactual labels and normalization. Do not assert generic rho-squared shrinkage is a conditional mean outside its assumed Gaussian model. |
| GM18 | MAJOR / SOURCE_VERIFIED + limit | `cmd_confirm` compares only the intersection of cell IDs and drops nonfinite differences; it does not enforce exact expected seeds/config identities. `cmd_seeded` omits empty strata and permits recovery n=0 to pass. Require manifest completeness, explicit NOT_EVALUABLE and paired identity checks before statistics. No claim that these code paths actually caused the quoted campaign results was established. |
| GM19 | MAJOR / AUTHOR_REPORTED limit | An incorrectly specified Bayesian reference is not an upper bound, and even a correct filter is not automatically an optimal stopping controller. Reclassify D7 as named comparator; repair and test process/observation models and stopping rule separately. H6 residual phi=0.587 is a sensitivity point, not measured 24h closer noise. Add horizon/regime/cross-expert errors and bias. |
| GM20 | MINOR / DERIVED | Sum(k_i-1)+1 payoff evaluations certifies a given profile on a fixed manifest; joint equilibrium enumeration is exponential. A subset certificate remains subset-scoped even if final. Publish rollout cost before G3; 32 coalition profiles per five-player role are manageable fixtures, not free per-tick Shapley training. |

## Responses to the eight requested priorities

1. Witness revisions: correcting real specification errors is legitimate, but a
   failing/non-vacuous oracle is not a proof the requirement is defective. A
   control can be wrong or target a different objective. W7 global-to-allowed
   choice changes the benchmark unless admissibility is independently justified;
   W13 formula matching proves execution consistency, not economic optimality;
   W8 same-position replay improves comparability but cannot justify per-outcome
   switching rules. Audit every old/new witness and use fresh confirmation.
2. Curves: promising position-independent representation, not a demonstrated
   universal sufficient statistic. GM07/08 define required causal and nonlinear
   boundary checks. Do not silently replace the owner's probability interface
   with a high-dimensional prediction task without a versioned design decision.
3. Noise: AR(1) phi values are experimental axes. The supplied H6 JSON does not
   verify closer-error distributions for every level or new selected features.
4. Bayes comparator: reported 12x variance mismatch invalidates its use as an
   optimality bound, not the existence of measured comparisons against that
   specific implementation. A repaired comparator is a new identity.
5. Hierarchy error model: useful stress model; not evidence of real neural error
   processes. Report sensitivity to correlated directional/time errors and bias.
6. Fitness: preserve net/Sharpe/drawdown frontier, eligibility and holding/trade
   metrics. A net-floor scalar arm is reasonable if frozen against an independent
   control, not an unrequested decision changing an existing campaign's objective.
7. Learnability: exact duplicate-input aliasing is insufficient for continuous
   data. Use designed indistinguishable pairs, bounded-observation tasks, causal
   planted positive controls, per-level holdouts and rare-event/barrier metrics.
   Passing synthetic learning does not establish real-market predictability.
8. Hedge: optional ablation, not a theorem-backed allocator in this stateful
   game. Frozen calibration weights are not online Hedge regret minimization.

## Smallest useful next implementation

G1: exact coordination games, exhaustive unilateral certificates, redundancy
and Shapley incentive counterexamples; reject empty/vacuous acceptance; account
contract digest; two-position causal sufficiency fixtures; explicit profile-bound
label schemas. Do not repair tests by observing candidate failures.

Before G2: define the decision/fill timeline and full closer-curve contract,
freeze per-witness applicability independently of candidates, fix risk units,
define profile-dependent labels and a complete learning loss. Then pair ideal,
imperfect and no-signal reports on new synthetic seeds. Keep ideal-output
execution acceptance distinct from information-limited account optimization.

Recommended retention rule instead of raw 90%: define a positive fixed scale
S from units or a named calibration control, require `J >= J_best - delta*S`,
keep all hard witnesses and risk constraints, then minimize a specified risk
measure. For multiobjective selection use an explicit epsilon constraint on
each preserved objective. Neither delta nor 'most prudent' is selected secretly
after seeing confirmation. The owner has not approved a new retention formula.

## Reuse and boundaries

| Component | Disposition |
|---|---|
| New owner business contract | SHARED_UNCHANGED; broker/calendar bindings must also agree |
| Exact game arithmetic and analytical fixtures | SHARED_UNCHANGED |
| Counterfactual accounting and immutable reporting | SHARED after independent adapter tests |
| Probabilities/uncertainty metadata | SHARED schema, producer-specific calibration |
| Closer curves and position sufficiency | NEEDS_SEPARATE_TEST and versioned signal decision |
| Persistence/Kalman/cross-scale matrix | Experimental mechanisms, not required winning designs |
| ZigZag labels, maturity, multi-resolution bids | EXPERT_GUIDED_ONLY |
| Pure reward-based policy training | RL_ONLY |
| Shapley | Diagnostic shared arithmetic; training objective needs separate justification |

Strong points: explicit labels, independent game identity, full-trajectory
counterfactuals. Main risks: adaptive verifier acceptance, changed learning
incentives, and hindsight information in a purported production signal.
Keep the mechanism exploration; remove unsupported guarantees and close these
contract defects before expensive experiments. No running Satoshi job was
stopped, changed or qualified by this review.
