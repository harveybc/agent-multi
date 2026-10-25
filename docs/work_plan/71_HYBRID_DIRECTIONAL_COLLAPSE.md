# Directional collapse: suspended financial queue

2026-10-25 14:23 UTC. The owner challenged the universal long-only behavior.
The prior progression through reward treatments without conditional-policy
qualification was insufficient. No complete-strategy acceptance is supported.

## Executed checks

The native engine maps OPEN_SHORT to a negative position. Both close-direction
regressions pass, including exhausted entry allowance. The three synthetic
constant-action controls learn NOP, long and short respectively. Together these
five existing tests pass (36.55 s CPU). They do not test discrimination between
different observed situations and should not have been treated as sufficient
qualification for a financial reward sweep.

Read-only probe of CAT_FIXED_s7, CAT_FIXED_s19 and HYBRID_DD400_s7 authenticated
each checkpoint/result and sampled the same 256 eligible flat observations
from INNER_A (2205 eligible origins while holding flat). SHORT was enabled in
all of them. Each checkpoint's deterministic choice was LONG in 256/256 cases.
Mean probabilities [NOP, LONG, SHORT]:

| Checkpoint | NOP | LONG | SHORT |
| --- | ---: | ---: | ---: |
| CAT_FIXED_s7 | .36059 | .54268 | .09674 |
| CAT_FIXED_s19 | .31595 | .58768 | .09637 |
| HYBRID_DD400_s7 | .26055 | .68018 | .05927 |

Inputs and latent outputs vary. This is not an exact constant-input witness
and not a disabled-short mask. It demonstrates an effectively constant
deterministic directional choice on this sample, not its causal origin.
Probe: tools/recipe_hybrid_direction_probe.py. No fit, no TEST; CPU probabilities
are not asserted bitwise identical to GPU. Probe result SHA256:
f30b4bcba9f461b7ece74fe5e284dc4a12cc03f7fa3f129df841787428fd7a77.

New observable, balanced conditional-signal test uses the same policy/optimizer
with the tiny recipe fixture: negative signal -> SHORT, zero -> NOP, positive
-> LONG, sampled with a fixed seed. At 16384 decisions it FAILED: negative
signal selected LONG, expected SHORT. The failure is retained as a regression,
not marked passing or erased. This establishes failure at the declared fixture
and budget, not impossibility of learning or proof that 102400 decisions with
the full financial recipe must fail for the same reason.

## Operational disposition

The campaign worker was stopped and its restart timer disabled. MainPID=0.
19 units completed; HYBRID_DSR_s19 was interrupted and remaining units were
not dispatched. Root status records SUSPENDED_DIRECTIONAL_QUALIFICATION.
Completed checkpoints/reports remain unchanged. The collector remains enabled
to preserve and publish evidence. Prior running snapshots and ETAs are obsolete.
The tested deployed source and spec were not patched in place.

Financial metrics remain measurements of these deficient policies, not evidence
that the business strategy works and not proof that profit accounting is wrong.
No model from this batch is promoted. Do not impose a 50/50 direction quota or
randomly force shorts to hide the defect.

## Next bounded diagnosis

1. Separate feature/gradient transmission, representation capacity and shared
   value/policy optimization effects using the conditional synthetic task.
   Keep a simple observable positive control and identical seeds/budgets.
2. Qualify context-dependent NOP/long/short and paired adverse/favorable
   close-versus-hold behavior, including long/short symmetry and native costs.
3. Only then define a new identified financial control. Changes to evaluation
   sampling, initialization, representation or optimizer are explicit treatments,
   never retrospective repairs of the stopped campaign.
