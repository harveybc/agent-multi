# Audit: Satoshi P1 Materialization and P2 Truth

Date: 2026-08-23
Auditor: General Musashi
Audited commit: `agent-multi@7290c4db`
Disposition: P2 accepted; P1 mechanics partially accepted, correction required
before GPU dispatch.

## Independently reproduced

- `23` focused tests pass.
- The driver exposes distinct `N`, `EN-W` and `EN-F` arms.
- The grouped extractor is absent from this experiment path.
- Easy dynamics are scoped to the fit environment; evaluation remains normal.
- Replay loading is explicit and SHA-checked; `EN-W` reports an empty replay
  and `EN-F` reports a non-empty replay in the CPU smoke.
- The directional rule separates `EN-W` from `EN-F`.
- P2 truth correctly reports Alpaca and MT5 ETH active, MT5 USDCAD prepared but
  inactive, IBKR owner-suspended and the plateau result as bounded evidence.

## Findings

### AUD-F1-20260823-307 (S2): handoff trace is not the selected checkpoint trace

`verify_handoff()` reads the single mutable
`traces/validation_epoch_return_trace.csv`. `_eval_on_split()` overwrites that
path each epoch. The selected checkpoint can be from an earlier epoch, while
the crossings gate is calculated from the terminal epoch. Therefore a terminal
policy can authorize handoff of a selected policy that never demonstrated the
required crossings.

Correction: when a checkpoint improves, atomically snapshot and hash its train
monitor/inner-validation traces beside the model artifact. Bind epoch, policy
tensor digest, config/data/observation hashes and trace digests in one selected
checkpoint manifest. Handoff consumes only that manifest.

### AUD-F1-20260823-308 (S2): EN-F is a mixed-time state, not full continuity

The model warm start is the selected easy checkpoint, but the saved replay is
captured from the terminal easy model after all later epochs. The resulting
normal initialization combines checkpoint-time networks/optimizers with a
terminal-time replay generated under later policies. Calling this
`full_continuity` is false and creates an uncontrolled treatment.

Correction: snapshot replay at every checkpoint improvement and load the replay
paired with the selected checkpoint, or define a separate terminal-continuity
arm that loads both terminal model and terminal replay. Never mix selected and
terminal state. The primary `EN-F` should use a checkpoint-coherent state bundle.

### AUD-F1-20260823-309 (S3): scalar L1 equality is not bit-exact continuity

The implementation compares one scalar: the sum of absolute actor parameters,
with tolerance `1e-6`. Different tensors can have the same L1 norm. Critic,
target critic, entropy state and optimizer state are not inspected in the
loaded runtime; hashes of members in the source zip prove artifact custody, not
that the loaded model contains those exact states. The packet's “exact at the
bit” statement is unsupported.

Correction: hash each named tensor of actor, critic, target critic, entropy and
each optimizer state before save and immediately after load using deterministic
dtype/shape/name/bytes framing. Require exact map equality. Record replay
position, capacity, size, observation/action spaces and RNG state separately.

### AUD-F1-20260823-310 (S2): the proposed endpoint reuses checkpoint selection

`normal_best_monitor` is actually the maximum L1 `composite`; that composite is
already used to select the checkpoint and combines train-tail, validation,
economic/risk and gap terms. Using it as the primary treatment endpoint rewards
selection on the same data rather than independent generalization. Its name is
also misleading.

Correction: retain the composite only for checkpoint selection and early
stopping. Evaluate each selected checkpoint once on `outer_validation` 2024.
The primary paired endpoint is the predeclared outer economic/risk score with
activity eligibility reported separately. Keep raw return, max drawdown,
Sharpe, trade count, exposure and weekly distribution visible; do not reduce
the report to the composite alone.

### AUD-F1-20260823-311 (S2): proposed day splits abandon the accepted data contract

`1460/240/240` and `120/40/40` are ad hoc sequential slices from the beginning
of the source. The former provides only about eight months for each evaluation
role, and neither uses the accepted nested manifest. The project already owns
the exact contract: fit 11,509 rows through 2022; monitor 2,190 rows in 2022;
inner 2,190 rows in 2023; outer 2,196 rows in 2024; sealed 2,190 rows in 2025.

Correction: consume the verified nested manifest and exact role hashes. Fit on
`fit_train`; checkpoint/stop on `train_monitor + inner_validation`; compare
treatments on `outer_validation`; never materialize or evaluate `sealed_test`
in this screen. Context-prefix rows initialize observations but cannot update
or score.

### AUD-F1-20260823-312 (S3): arm identity is prose, not enforced comparison identity

The record says the arms differ only in state factors, but the driver does not
materialize and compare canonical effective configs across all three arms.
Output paths and treatment fields should differ; learning rate, data, feature
order, network topology, reward, action mapping, stopping, seed and budgets
must be mechanically identical. Counterbalanced order is documented but not
bound to a fleet launch manifest.

Correction: emit canonical `pair_contract`, `arm_contract` and
`transition_state_contract`; verify exact equality with a factor allowlist
before launch and aggregation. Bind seed-to-host/GPU and arm order in a durable
pre-launch manifest.

## Dispatch ruling

Do not dispatch the 12 long GPU arms from `7290c4db`. This is not a request to
leave compute idle: MT5 corrections 301-306, grouped-extractor completion and
bounded correction smokes may proceed in parallel. P1 dispatch becomes
automatic after independent reproduction of 307-312.

