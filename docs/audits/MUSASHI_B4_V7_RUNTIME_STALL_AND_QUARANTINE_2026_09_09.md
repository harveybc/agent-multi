# Musashi: B4 v7 runtime stall and quarantine record

Date: 2026-09-09 (America/Bogota)

## Decision

The owner authorized the following action in the active session:

> ok estas autorizado para detener y poner encuarentena la tercera celda B4 atascada

This record covers only the stop and quarantine of the active B4 v7 runtime.
It does not authorize a retry, a new generation, another GPU cell, promotion,
or interpretation of partial scientific outcomes.

## Runtime identity

- Service: `b4-v7-campaign-20260907.service`.
- Checkout: `repo:.runtime/b4-v7-282c5771`.
- Pinned execution commit: `282c57712f27e94874cbd58b3c58089ed1442f07`.
- Campaign generation: `b4_campaign_generation_v7_20260907`.
- Population digest: `111dfed8acd281b269fc9916b8d6b5a39898d71b3a894f3f5dbb9b5f55792b05`.
- Campaign ledger digest: `e569e511def2dac83938d757b0aa6444dbc921cb6589b06755a1116eb6c54067`.
- Results-root logical identity: `b4_campaign_results_v7_20260907`.
- Active cell: `o2022_seed303`.
- Attempt: `attempt_b459e72dc47a4d72`.
- Holder PID: `307630`.

## Observed incident

The service remained active with `NRestarts=0`, and the process consumed one
CPU core and the RTX 4070 at approximately 100 percent utilization, 82 C and
70 W. Nevertheless:

- the last durable epoch status was epoch 111 at
  `2026-09-08T08:56:11-05:00`;
- the last progress artifact was written at
  `2026-09-08T08:57:36-05:00`, at 2,231,000 environment steps;
- no campaign artifact, progress record or journal line advanced for roughly
  29 hours before intervention;
- `/proc/307630/io` showed no write-counter movement during a five-second
  observation at approximately `2026-09-09T14:07:53-05:00`;
- the cell had held its claim since `2026-09-08T03:34:53-05:00`, exceeding
  the effective per-cell hard wall of 43,200 seconds;
- the in-process stop and wall callback therefore did not regain control.

This is `RUNTIME_STALL_OUTSIDE_CALLBACK_CONTROL`. It is not training progress
and it is not a scientific result.

## Stop sequence and final custody

At `2026-09-09T14:18:07-05:00`, two mode-0600 stop signals were installed:

- campaign stop: `CAMPAIGN_STOP`;
- cell stop: `o2022_seed303/STOP`.

For more than one minute the process consumed neither signal and wrote no
terminal. The service was then stopped through systemd. It reached
`inactive/dead`, PID 307630 disappeared, and no B4 compute process remained on
CUDA. Systemd recorded the stop at `2026-09-09T14:19:28-05:00` after 1 day,
20 hours, 39 minutes and 18.812 seconds of service wall time. Systemd's
`Result=success` describes the operator stop only; it is not a campaign result.

The candidate cell retains its original claim and lease but has no
`B4_CELL_TERMINAL.json` and no seal. The production dry-run then derived:

- `o2022_seed101`: `COMPLETED_VERIFIED`;
- `o2022_seed202`: `COMPLETED_VERIFIED`;
- `o2022_seed303`: `AMBIGUOUS_CLAIM`;
- remaining nine cells: `PENDING`;
- zero dry-run writes;
- 44.69 GPU-hours reported spent and 51.30 GPU-hours reported remaining at
  the time of the post-stop dry-run.

The open interval must not continue accruing after the observed operator-stop
time. Its exact charge must be re-derived and bound by the recovery design; the
rounded dry-run figures above are evidence summaries, not accounting inputs.

## Preserved completed evidence

- `o2022_seed101/B4_CELL_TERMINAL.json`:
  `3a836df218702adfa8eb8d436b81064e3f517e59f16ab9a750b58304a8c921d5`.
- `o2022_seed101/SEAL_COMPLETE_attempt_f16bff0d0a624d63.json`:
  `aea56c68cfbe12d1ed10e5c9dbb6751ec6ffddb4d6c1d677aae69c9b65b6624e`.
- `o2022_seed202/B4_CELL_TERMINAL.json`:
  `cd3e61e7e33772663e808745a1135a1d4cebf6ddd7c4b67f54c134f0c3c5fd7c`.
- `o2022_seed202/SEAL_COMPLETE_attempt_1015641321af481b.json`:
  `ec763fd82b9958cf1e4789e2c9d178b8fc9f174c669e25172027f742c7118526`.

## Binding disposition

`B4_V7_THIRD_CELL_QUARANTINED_RUNTIME_STALL`. The entire v7 results root is
read-only historical evidence for the recovery design. In particular:

1. do not remove either stop signal;
2. do not mint or backfill a terminal for `o2022_seed303`;
3. do not resume from its model, replay buffer, RNG or checkpoint artifacts;
4. do not remove, overwrite or reinterpret its claim or lease;
5. do not launch cell four from the v7 root;
6. preserve the two completed and sealed cells byte-for-byte;
7. require a separately reviewed generation and dispatch record before any
   further GPU work.
