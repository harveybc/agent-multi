# Causal history controller experiment

Identity: `452931b1fe97840255ac60abe87911a97724053ffc7d6c1249318c5c5026d36e`.
Reporter: median. Smoke: False. Runtime-test search gate: True.
Synthetic prefix-only reports; NOT financial validation, a learned RL model or final R6 qualification.
Size fixed at 1; rel_volume=0.1 once; native costs and calendar. No hindsight fitness.

| Policy | Return % | Worst DD % | Mean native Sharpe | Orders/week | Buy / sell | Win % | Early closes | Harmful closes | Regret / risk | Close behavior |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| neither | -0.6207 | 4.9836 | -1.6080 | 5.50 | 292 / 302 | 29.6296 | 246 | 90 | 0.5705 | EARLY_CLOSE_OBSERVED_NOT_QUALIFIED |
| entry | -0.2639 | 3.4772 | -1.8054 | 2.58 | 132 / 147 | 30.1075 | 43 | 29 | 0.5922 | EARLY_CLOSE_OBSERVED_NOT_QUALIFIED |
| close | -0.6086 | 4.3892 | -1.0408 | 4.33 | 226 / 242 | 26.2821 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| both | -0.1815 | 2.6290 | -0.8924 | 2.45 | 122 / 143 | 30.9434 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| no_close | -0.6086 | 4.3892 | -1.0408 | 4.33 | 226 / 242 | 26.2821 | 0 | 0 | - | EXPLICITLY_DISABLED |
| nop | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0 / 0 | - | 0 | 0 | - | EXPLICITLY_DISABLED |
| c3f252db58272a8c2d86777c744da5ff3c7b99ff942c1ca9d565f143690e7915 | -0.2398 | 2.1584 | -3.1464 | 1.93 | 102 / 106 | 28.3654 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| e16e479902782cca7d48f4ea9713d0b3fa9ef45708f6b9626eed9c9c3b6efcef | -0.2122 | 1.8812 | -3.2019 | 1.89 | 100 / 104 | 26.9608 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| 4496afdb24b1b119d288de7109d3c4001c493102617f7c4de2f1f8cd7706f4eb | -0.2214 | 1.9005 | -3.3297 | 1.89 | 101 / 103 | 26.4706 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| a29b2bb28c2a8fd705c01f8b3af1dde747a2d6ac8c5dfe0b93069e75de969b22 | -0.1868 | 1.8812 | -2.8485 | 1.87 | 98 / 104 | 28.7129 | 0 | 0 | - | NO_EARLY_CLOSE_OBSERVED |
| c3f252db58272a8c2d86777c744da5ff3c7b99ff942c1ca9d565f143690e7915_no_close | -0.2398 | 2.1584 | -3.1464 | 1.93 | 102 / 106 | 28.3654 | 0 | 0 | - | EXPLICITLY_DISABLED |
| e16e479902782cca7d48f4ea9713d0b3fa9ef45708f6b9626eed9c9c3b6efcef_no_close | -0.2122 | 1.8812 | -3.2019 | 1.89 | 100 / 104 | 26.9608 | 0 | 0 | - | EXPLICITLY_DISABLED |
| 4496afdb24b1b119d288de7109d3c4001c493102617f7c4de2f1f8cd7706f4eb_no_close | -0.2214 | 1.9005 | -3.3297 | 1.89 | 101 / 103 | 26.4706 | 0 | 0 | - | EXPLICITLY_DISABLED |
| a29b2bb28c2a8fd705c01f8b3af1dde747a2d6ac8c5dfe0b93069e75de969b22_no_close | -0.1868 | 1.8812 | -2.8485 | 1.87 | 98 / 104 | 28.7129 | 0 | 0 | - | EXPLICITLY_DISABLED |

Mean native Sharpe is an average of defined account Sharpes, not a pooled portfolio Sharpe.
Paired draw-cluster contrasts are descriptive, not independent tests for each transformed account.
The error penalty is not a calibrated joint-path band. It can suppress beneficial closes.
Activity eligibility is mean orders/week >2. Zero executed closes do not qualify a closer, even if return improves.
All confirmed policies have native ledgers, costs, directions and fixed-position close alternatives.
No confirmation result selects or modifies a policy. No synthetic result opens TEST.
Full PC1--PC6 learned-interface qualification and final R6 remain outside this engineering experiment.
No automatic RL training, floating selection or strategy promotion follows this report.
