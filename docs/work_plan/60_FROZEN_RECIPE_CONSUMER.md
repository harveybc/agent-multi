# Frozen Shared Recipe Consumer

Design 2026-10-24. The shared candidate is recipe_5a8f2155318ba7c7,
derived from selected C, not the retired rolling-z observation path.

## Acceptance Tests Before Implementation

| ID | Requirement | Evidence to execute |
| --- | --- | --- |
| R01 | Immutable named recipe, store and manifest | Reject changed pins, duplicate columns, unknown operations, inconsistent groups |
| R02 | Fixed constants, no clipping or refit | Exact independently computed float32 windows, extreme value retained |
| R03 | Completed current bar included, no future | Perturb suffix; earlier observation unchanged; closures occupy no rows |
| R04 | No policy feedback or expert labels | Observation contains features only; loader never invokes expert generation |
| R05 | Selected grouped representation reaches SAC | Actor and twin critics build from recipe; save/reload; input hook verifies tensor |
| R06 | Real input parity | Native SAC input adapter matches peer's 23-row reference and all period window-valid masks |

## Executed Result

R01-R05: 17 focused CPU tests pass (input adapter, native SAC build/reload,
and existing temporal encoder regressions). R06: real probe PASS, shape
23x13x121, tensor SHA256
f087d1a77bbb36999e4aa01192db28435b32b68c807f291057feaf2da233f60f.
Window-valid rows: TRAIN19514, INNER_A2589, OBS_2024 5600. Those are not
entry masks or retrospective expert-label eligibility. Actor and twin critic
inputs are exact; C topology with identical test weights yields exact trunk
outputs; native SB3 policy save/reload is exact. Zero financial fits or engine
steps. Evidence lives in predictor's shared_recipe_rereview_20261024 packet.

`observation_encoder=shared_recipe` activates only the explicit new path. The
legacy defaults are untouched. `shared_recipe` carries the frozen document;
the deployment spec must pin its file digest. This adapter uses C's measured
group topology and branch dropout. The SAC optimizer/weight decay and heads
must be explicitly pinned in the next finite design; a no-fit input test does
not certify those training settings or apply proxy AdamW settings silently.

Only observation preparation and CPU model construction are authorized by this
implementation step. It does not replace the trading engine. Before a campaign,
bind its timestamp-to-action mapping and real business accounting to these
observations, verify no extra normalization wrapper, then pin a finite SAC run
spec and publication/recovery configuration. Do not claim a training launch
from a probe. Shared feature acceptance and consumer deployment are separate.

No new fitted transform, padding of missing market data, rolling scaler,
clipping, agent feedback, demonstration pretraining, or TEST read. The selected
representation is an E3-M proxy; SAC must measure its own benefit. Dropout and
weight decay from proxy T2 are not proof of the best SAC regularization.
