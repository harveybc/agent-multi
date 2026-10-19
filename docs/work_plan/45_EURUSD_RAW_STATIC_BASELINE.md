# EURUSD Raw Static Baseline

Owner continuation 2026-10-19. This is the independent price-only control,
not a substitute for rebuilding final financial feature selection.

## Frozen Design

`examples/config/rl_raw_eurusd_static.v1.json`: TRAIN2020-2023 (24894 rows),
VALIDATION2024 (6221 rows), all required months. One documented initial raw
return warmup is excluded. Evaluation also declares the strict 24-observed-bar
context warmup; do not report it as scoring every source row. No TEST fitting.
Inputs are H1 completed-bar availability, observed open-market bars only,
13 observations at stride two, no smoothing/interpolation. The raw policy
contains OHLC plus causal bar/range/observed-close-return and elapsed-time
features; it is explicitly not a selected inventory.

One seed (7), SAC flat and small per-feature
causal Conv1D (four branch filters, eight core filters, kernel three), heads
[32,32]. Each SAC has a fixed 100000-transition budget, 1000 learning starts,
batch256, one gradient step per transition. No genetic search, pretraining,
retraining, hidden early stopping or claimed optimal capacity. Reward is net
equity change divided by initial equity, scaled by the fixed factor1000 for
learning; reported profit remains unscaled USD. Initial equity USD10000.

One concurrent position and one entry/day, NY quota, forced flat12h before
closure and no entry until at least12h after reopening plus causal stability
evidence. The conservative operating calendar is identity-bound to admission;
it is not represented as universal broker certification. All arms share it.
Commission0.00005 and slippage fraction0.00005 per fill; analytical annual
notional financing0.05. The constant one-pip ASSUMED_SPREAD column is a
non-observed stability scenario, not fake Ask data or the cost-debit authority.
These costs are assumptions, not verified broker charges.

Operational correction 2026-10-19: prioritize both SAC arms, followed by the
four mechanical controls. The earlier controls-first queue left the GPU idle
while replaying NO_TRADE. It had not trained a SAC or completed/published any
terminal. Its partial control replay is explicitly cancelled for scheduling,
not presented as a result. A new contract/root changes only dispatch order and
operational identities; model, data, seed, costs, splits and compute budgets
are unchanged. No code in the training/evaluation engine changes.

## Execution And Recovery

The materializer validates both split receipts and pins runtime sources before
creating the contract. The runner uses registered components, accepts only the
assigned CUDA GPU (no CPU fallback), and isolates each attempt's files/custody.
One heavy process on the5090; no parallel5070Ti load. CPU controls are sequential.
Operational SQLite, locks and active artifacts stay on ext4. Interrupted
incomplete work has at most two attempts; accepted fits/checkpoints are reused,
not retrained to retry publication. An integrity refusal is not retried blindly.

Persistent user services own continuation after SSH/agent exit or reboot.
A separate bounded publisher timer retries only the existing outbox, never
training, and stops doing work when CLOSE.json exists. Close requires all12
task terminals, exact warehouse readback and a closed SQLite backup. Undefined
risk statistics remain null. Each result retains effective configuration,
input/calendar identity, code/runtime pin, seed, weights and return trace.

```bash
python -m tools.materialize_raw_rl_campaign --help
python -m tools.run_raw_rl_campaign run --root <private-campaign-root>
python -m tools.run_raw_rl_campaign status --root <private-campaign-root>
python -m tools.run_raw_rl_campaign results --root <private-campaign-root>
python -m tools.run_raw_rl_campaign publish --root <private-campaign-root>
```

Status reads the journal without polling workers. Current-fit ETA is projected
only after5000 measured steps with a fresh heartbeat. It is not an asserted ETA
for a different encoder or the entire experiment. Training diagnostics and
frozen validation are always separate. No automatic real-money deployment.

Closed evidence collection is a separate one-shot scheduled task:

```bash
python -m tools.collect_closed_raw_rl \
  --source-host <worker> --source-root <private-campaign-root> \
  --archive-root <bulk-closed-artifact-root> --state-root <private-ext4-state-root>
```

It returns PENDING after one read while computation/publication is incomplete.
After closure it transports only closed artifacts, not the live SQLite outbox,
WAL, SHM or locks; verifies every checkpoint/config/result digest, the complete
12-task population and the physical SQLite backup; then writes its completion
receipt. A timer conditional on absence of that receipt stops further work.
This keeps the active operational database off NTFS while placing closed
evidence on the large bulk volume with a second-machine physical copy.
