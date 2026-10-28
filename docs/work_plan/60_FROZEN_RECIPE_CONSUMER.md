# Frozen Shared Recipe Consumer

Current execution, 2026-10-24 21:42 UTC: native SAC is RUNNING. See
61_RECIPE_NATIVE_SAC_QUEUE.md. The remaining integration described below is
closed via direct reuse of the shared BarEnv engine, not by claiming the
Backtrader cost adapter is equivalent. The old bridge evidence is historical.

Current input revision 3 (2026-10-24): recipe_a58261c0fdc0a59d removes only
px.er_1. There are 120 columns; all survivor constants and common valid tensors
are exact. New valid counts TRAIN/INNER_A/OBS_2024: 21221/2805/6208. Both native
consumers agree and the eight-step real engine witness passes. No interpolation
or new financial training. The previous 121-column results below refer to the
parent. Remaining no-observation handling and economic parity still apply.

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
| R07 | Engine observation binding | Exact completed-bar timestamp, no positional join or old preprocessor, only market features exposed; refuse invalid windows before reset |
| R08 | Engine transition preservation | Real backtrader CPU reset/step and SAC replay retain next-open transitions, finite rewards and daily quota while input tensors remain exact |

R07-R08 implementation is an observation bridge, not certification that the old
engine's cost/calendar configuration equals the shared recipe. Before any new
scientific fit, compare every economic configuration field and execute the
business accounting witnesses. In particular, the legacy annual financing and
constant commission defaults must not silently replace the recipe's side-specific
swap and per-lot costs. No expert label design is a dependency of this SAC work.

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

## Engine Bridge Executed

R07-R08: 27 focused CPU tests pass, including the earlier input tests and legacy
temporal bridge regressions. Seven new guard/clock tests fail before the bridge
exists and pass afterwards. The real backtrader integration performs four
synthetic SAC training transitions, checks every replay observation and next
observation exactly, and completes an episode with the daily-entry cap intact.
This is engineering training on synthetic prices, not a financial experiment.

The real-data witness in predictor `tools/verify_sac_recipe_engine.py` passes
eight HOLD transitions on the first eligible TRAIN prefix, with exact market
tensor and actor input, matching action timestamps, zero orders, zero financial
fits and no TEST read. The replaced preprocessor is never called. Engine
session/quota/account fields remain internal diagnostics, not policy inputs.

The bridge currently refuses any invalid interior window before reset. It does
not silently remove price bars, forward-fill observations or invent zeros.
Real support is TRAIN 19514/21809, INNER_A 2589/3086 and OBS_2024 5600/6221
valid windows/bars. In OBS_2024, px.er_1 has 53 nonfinite source cells and
px.er_2 has one; windows propagate the missingness. These match the already
verified recipe mask, not new corruption by the bridge. A full rollout needs
an explicit causal no-observation transition policy or a separately identified
upstream semantic correction. Preserving every price bar and every protection,
cost and forced close is mandatory. This is not authority to interpolate.

Next implementation: account for unavailable-observation intervals without
silently changing SAC's transition duration/discount, and reconcile the recipe's
cost, sizing and calendar semantics with the engine. Then pin the finite run
and deploy the sequential queue. No GPU campaign is started by these witnesses.
