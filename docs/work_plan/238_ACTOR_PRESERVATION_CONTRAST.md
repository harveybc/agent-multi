# Actor Preservation Contrast

Owner authorization:2026-11-09, launch the proposed bounded comparison.
Question: can the actor improve without losing its preserved account behavior,
given the already measured paired-greedy critic correction?

## Frozen Design

Reuse authenticated r19/r43 paired_greedy/CALIBRATED.pt from236. Both contain
the unchanged source70 actor and their own corrected critics. No recalibration,
new inputs, state feedback, rolling normalization, changed windows or strategy.
Only actor head parameters train. Encoders, online critics and alpha stay exact.

Three arms per seed: native actor update; same update with learning rate/3;
native actor objective plus beta * KL(parent || current), beta=1.0. Beta is a
prespecified diagnostic coefficient in reward units, not an optimized setting.
The reference logits are detached and come from the frozen parent at the same
TRAIN replay observations. No validation gradients or validation-driven beta.
This is a SAC diagnostic regularizer, not TRPO or a performance guarantee.

All arms use the same full TRAIN stochastic7 replay, batch128 and paired
schedule seeds19/43. Points:0,1,8,32,64,128,256. Reference no-update accounts
are evaluated once per seed; source epoch-zero parity is mandatory. Native
arm checkpoints1/8/32 must reproduce236 exactly before accepting extended
results. No early stopping or selection of favorable checkpoints in this
bounded mechanism contrast. Fixed endpoint256 plus the entire curve reported.

## Tests Before Launch

| ID | Requirement | Evidence |
|---|---|---|
| A01 | KL zero at equal distributions; detached parent; finite/nonnegative within numerical precision | unit fixtures |
| A02 | beta0 follows byte-exact native update; positive beta changes actor gradients after drift | native model unit fixture |
| A03 | Optimizer uses actual lr/3 and declared beta | effective config and unit fixture |
| A04 | Only actor heads move; parent files untouched | frozen parameter and source digest checks |
| A05 | All data/support and producer identities authenticated | runtime preflight and receipts |
| A06 | Native points1/8/32 reproduce both existing controls | full account parity gate |
| A07 | Both seeds/all arms reported even if inactive or negative | automatic combined CSV/PNG and terminal receipt |

Full-period diagnostics include action changes, KL to parent, return, Sharpe,
drawdown, orders, win rate and native early-close quality diagnostics. Parameter
distance is not itself trading performance. Sequential RTX5090 workers, followed
by automatic collection; no source or live-policy replacement. Resume accepts
only completed, authenticated arms from the identical effective config.
