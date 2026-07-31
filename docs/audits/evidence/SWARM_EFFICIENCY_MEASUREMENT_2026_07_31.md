# Swarm Efficiency Measurement

Generated: `2026-07-31T03:27:23.804803-05:00`
Campaign filter: `trading-asset-policy-usdcad-4h-protected-easy-v2`

## Result

- Complete generations measured: **3**.
- Tail-barrier idle: **8.42%** of measured fleet wall-clock capacity.
- Total non-evaluation gap: **28.13%**. This includes barrier waits, scheduling, communication, restarts, and any unlogged work; it is not attributed to one cause.
- Peer-tip adoptions observed: **7**.
- Median announcement-to-convergence latency: `7.0` seconds.

## Generation Detail

| Generation | Candidates | Complete | Tail idle | Non-evaluation gap |
| ---: | ---: | :---: | ---: | ---: |
| 0 | 20/20 | yes | 10.79% | 55.27% |
| 1 | 20/20 | yes | 1.37% | 1.42% |
| 2 | 20/20 | yes | 12.05% | 12.10% |

## Interpretation Boundary

The tail metric starts at each worker's last completed candidate and ends at the generation's final candidate. It directly measures generational straggler waiting. The broader gap is descriptive only and must not be called barrier loss without additional instrumentation.

Input log hashes are embedded in the JSON evidence packet.
