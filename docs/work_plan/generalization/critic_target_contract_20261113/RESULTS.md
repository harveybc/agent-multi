# Critic target contract results

All16 fixed endpoints; same55 previously inspected TRAIN anchors. No actor promotion.

| Seed | Sampling | Loss | Output | Held accuracy | Held regret | Opening regret | Q RMSE |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| 19 | uniform | smooth_l1 | shared_mean3 | 0.476923 | 0.264142 | 0.447773 | 0.615099 |
| 19 | uniform | smooth_l1 | routed | 0.392308 | 0.294779 | 0.477138 | 0.746271 |
| 19 | uniform | mse | shared_mean3 | 0.453846 | 0.270397 | 0.449345 | 0.595421 |
| 19 | uniform | mse | routed | 0.407692 | 0.285775 | 0.460631 | 0.725373 |
| 19 | visitation | smooth_l1 | shared_mean3 | 0.423077 | 0.283552 | 0.461758 | 0.557228 |
| 19 | visitation | smooth_l1 | routed | 0.438462 | 0.297693 | 0.460631 | 0.653820 |
| 19 | visitation | mse | shared_mean3 | 0.453846 | 0.269713 | 0.436388 | 0.553994 |
| 19 | visitation | mse | routed | 0.438462 | 0.297693 | 0.460631 | 0.648579 |
| 43 | uniform | smooth_l1 | shared_mean3 | 0.469231 | 0.250186 | 0.422121 | 0.603589 |
| 43 | uniform | smooth_l1 | routed | 0.446154 | 0.258120 | 0.412876 | 0.695814 |
| 43 | uniform | mse | shared_mean3 | 0.453846 | 0.255661 | 0.432158 | 0.586902 |
| 43 | uniform | mse | routed | 0.453846 | 0.245167 | 0.389129 | 0.682687 |
| 43 | visitation | smooth_l1 | shared_mean3 | 0.415385 | 0.281786 | 0.470936 | 0.554465 |
| 43 | visitation | smooth_l1 | routed | 0.476923 | 0.244839 | 0.406447 | 0.649307 |
| 43 | visitation | mse | shared_mean3 | 0.438462 | 0.258548 | 0.428333 | 0.550798 |
| 43 | visitation | mse | routed | 0.476923 | 0.240492 | 0.398477 | 0.647897 |
