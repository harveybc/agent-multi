# Critic-only dropout contrast: unattended execution

Owner authorized execution and sequential automation on 2026-10-25.
Parent: completed COMMANDS5 order study, spec79c4b87e, frozen diagnosis return64.

## Fixed experiment

Two fresh sequential fits: CRITIC_BASE then CRITIC_NODROP. Both use COMMANDS5,
seed7,100000 decisions, checkpoints25k/50k/75k/100k. The actor retains dropout0.3;
weight decay0.01, architecture, action mapping, initialization, input recipe,
TRAIN/INNER_A/2024 support, costs, calendar and optimizer settings are unchanged.
No feedback, new labels, imputation, rolling scaling, early stopping, refit or TEST.

Only the critic feature-extractor dropout probability changes:0.3 to0.0.
The same override is stored for the target critic (normally in evaluation mode).
Do not edit the shared recipe: this is an explicit consumer-model override,
serialized in the policy constructor and recorded in effective receipts.

The base's final policy tensor digest must match the authenticated parent
COMMANDS5 final digest. If that fails, do not dispatch the dependent treatment.
No intermediate profit determines dispatch or checkpoint selection.

## Tests before deployment

| ID | Acceptance evidence |
| --- | --- |
| CD01 | Initial tensors exactly equal legacy/base/treatment; actor remains0.3 |
| CD02 | Legacy and base produce exact tensors after identical synthetic CPU updates |
| CD03 | Treatment's critic and target dropout remain0.0 after save/load; prediction parity |
| CD04 | Parent/config/recipe identities bound; changed settings rejected |
| CD05 | Failed/missing base cannot dispatch treatment; completed children resume authenticated |
| CD06 | Both arms admitted with real recipe shape and synthetic GPU updates before fit |
| CD07 | Actor/Q probes preserve RNG/weights/modes; only TRAIN/INNER_A, not extra2024 selection |
| CD08 | Collector accepts new arm identities, binds checkpoint/native values, normalizes units |

## Automation and output

Reuse recipe_sac_queue and recipe_sac_collect, not a second trainer or warehouse.
One heavy process on the primary GPU. Memory admission, finite timeout, two
bounded fresh attempts per arm, durable status, reboot/resource-wait timer,
authenticated result adoption, second-machine checksum copy and warehouse
readback. Failures remain explicit; never silently alter scientific settings.

Each checkpoint retains policy weights, effective regularization, native metrics,
trade ledger, order-command counts and frozen actor/Q diagnostics on deterministic
evenly-spaced TRAIN/INNER_A observations. Final comparison includes both arms
and all three periods. Diagnostic arrays are separate from realized trading scores.

Scope ends at comparison/publication/verified copy. Entropy changes, action masks,
position routing, other algorithms and larger searches are not implicit successors.

## Predeployment evidence

48 focused CPU tests passed with the pinned shared engine configured, including
16 engine scenarios. The new native SAC tests failed before the override existed;
after implementation they prove exact legacy/base initial and trained tensors,
independent critic dropout and reload preservation. The real-engine integration
also verifies that adding frozen probes leaves all native scores and RNG unchanged.

## Deployment evidence

At 2026-10-25T07:37:59Z the primary GPU was training CRITIC_BASE:
1500 decisions,499 gradient updates, fresh heartbeat. Producer afb3bc37;
spec1c63c1a4f42630d87b8f35a0fd97dbde451badc4731d76fd5dd9f227fc908fd6.
Both synthetic GPU admissions passed with14 updates per arm. Effective model
records actor/critic/target dropout0.3 and weight decay0.01 for the control.

Worker and collector timers are enabled; worker user lingering is enabled.
Worker limits: memory high5GiB/max6GiB, no swap, one heavy GPU process.
Collector first run verified the copied spec and pumped an empty outbox:
zero reports expected, zero pending or mismatches, no completed arms yet.
The independent artifact destination is on the writable archival volume;
queue and outbox remain on native local storage. No new final metric exists
at this start check. Provisional total duration80-90min comes from the prior
approximately38min/arm, not a measured duration of this new treatment.
