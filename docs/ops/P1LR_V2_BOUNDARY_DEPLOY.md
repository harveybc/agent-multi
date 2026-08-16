# P1LR v2 Identity Supervision — Boundary Deploy and Evidence

Corrects findings **AUD-GEN-20260816-256** (the shipped identity files did
not exist) and **AUD-GEN-20260816-261** (the lease gate executed mutable
canonical-checkout code). Nothing in this document may be executed while a
seed's matching v2 PID is alive, except the install step, which enables
nothing and starts nothing.

## 1. What is shipped

| Path | Role |
| --- | --- |
| `examples/config/phase_3_eth_sac_dynamics/p1lr_env_v2/seed{101,202,303,404}.env.conf` | per-seed identity: seed, host, CUDA UUID, mode, contract path + sha256, chain id, experiment, output root |
| `examples/systemd/p1lr-control/CONTROL_MANIFEST.sha256` | sha256 of every module the restart admission gate needs |
| `examples/systemd/p1lr-decision@.service.d/20-v2-identity.conf` | the drop-in; pins the runtime worktree, the v2 screen gate, the identity files and the control-bundle digest |
| `examples/systemd/install_p1lr_v2_identity_supervision.sh` | reviewed operator install; enables nothing, starts nothing |

All of the above are **generated** by
`tools/p1lr_identity_supervision.py materialize` and byte-pinned to that
generator by `tests/test_wo4_identity_supervision.py`. Regenerate, never
hand-edit.

### Why `.env.conf` and not `.env`

The repository ignores `*.env`, `.env` and `.env.*` under *"Credentials and
machine-local authority"*. That rule silently swallowed the earlier
`seed*.env` payload: the branch claimed four deployable environments that
git never contained, while the installer required their glob. Force-adding
files under a credential rule would have fixed *these four* files and left
the trap armed for the next seed. The suffix `.env.conf` is outside the
rule, so a newly generated identity is visible to `git status` instead of
vanishing. systemd's `EnvironmentFile=` parses by content, never by
extension, so the deployed semantics are identical. The files are
non-secret by construction — seed, hostname, GPU UUID, mode, contract path
and sha256, chain id, experiment, output root — and a test asserts they
carry no credential-shaped key.

### Why the lease gate is content-addressed

The lease gate decides **whether a worker may start**. Running it from
`~/Documents/GitHub/agent-multi` meant a routine `git pull` could change
restart admission without changing the unit's declared identity.

The gate is now installed read-only (0444 files in a 0555 directory) under

```text
~/.local/lib/agent-multi/p1lr-control/<sha256(CONTROL_MANIFEST.sha256)>/
```

and the drop-in carries that digest **literally**, twice: once as the
directory it executes from, once as the value it re-verifies. The
verification is an `ExecStartPre` of its own, before the gate:

```ini
ExecStartPre=/bin/sh -c 'echo "<D>  %h/.local/lib/agent-multi/p1lr-control/<D>/CONTROL_MANIFEST.sha256" | /usr/bin/sha256sum --status -c - && cd %h/.local/lib/agent-multi/p1lr-control/<D> && /usr/bin/sha256sum --status -c CONTROL_MANIFEST.sha256 || exit 4'
```

Exit 4 is the `RestartPreventExitStatus` class of the base template, so a
drifted, tampered or absent control bundle **fails closed and is never
retried**. The runtime worktree was not used for this: it is pinned at
`924910fe`, which predates the supervision module, and adding files to it
would break the WP0 source-isolation identity of the live workers.

Changing what may start therefore requires regenerating the manifest and
the drop-in — a reviewed, versioned edit — not a checkout update.

## 2. Install (safe at any time; enables nothing, starts nothing)

Per host, from the canonical checkout on the integration branch:

```bash
bash examples/systemd/install_p1lr_v2_identity_supervision.sh
```

It refuses (exit 2) if the drop-in's pinned digest and the shipped manifest
disagree, or if the installed bundle does not verify.

Read-only verification afterwards:

```bash
~/anaconda3/envs/trading-stack/bin/python \
  ~/Documents/GitHub/agent-multi/tools/p1lr_identity_supervision.py verify-control
~/anaconda3/envs/trading-stack/bin/python \
  ~/Documents/GitHub/agent-multi/tools/p1lr_identity_supervision.py plan --with-deploy-commands
```

## 3. Arm reboot reconstruction (safe beside a live PID)

`enable` without `--now` writes a symlink and starts nothing:

```bash
systemctl --user enable p1lr-decision@101.service   # omega
systemctl --user enable p1lr-decision@202.service   # dragon
systemctl --user enable p1lr-decision@303.service   # gamma
systemctl --user enable p1lr-decision@404.service   # gamma
```

The template is already `enabled` on omega from the v1 era, which is
precisely why the drop-in repoints it **wholesale**: after install, a
reboot reconstructs the v2 identity instead of the legacy one.

## 4. Start — ONLY at the seed's own next process boundary

Never beside a live matching PID. This command refuses instead of racing:

```bash
SEED=101
if pgrep -f "python\s+\S*p1_difficulty_lr_factorial\.py --seed ${SEED} --mode decision" >/dev/null; then
  echo "REFUSED: seed ${SEED} v2 PID $(pgrep -f "python\s+\S*p1_difficulty_lr_factorial\.py --seed ${SEED} --mode decision") is alive — no start"
else
  systemctl --user start p1lr-decision@${SEED}.service
fi
```

Guard timer (touches no worker, safe at any time):

```bash
systemctl --user enable --now p1lr-idle-guard.timer
```

## 5. Evidence capture (read-only; run before and after each seed)

```bash
SEED=101 PHASE=before bash docs/ops/p1lr_v2_boundary_evidence.sh \
  | tee ~/.local/state/agent-multi/p1lr-v2-boundary/seed101.before.json
```

The emitted record carries the fields the order requires — PID, chain,
contract hash, output root and CUDA UUID — plus the launcher, the unit
state and the control-bundle verdict:

```json
{
 "schema": "agent_multi.p1lr_v2_boundary_evidence.v1",
 "phase": "before", "host": "omega", "seed": 101,
 "pid": 731019, "launcher": "nohup",
 "proc_cwd": "~/Documents/GitHub/.runtime/agent-multi-p1lr-v2-924910fe",
 "unit_active_state": "inactive", "unit_main_pid": "0",
 "chain_id": "cdf30aebf585385b",
 "contract_sha256": "f5544a5f57d0a6a5d658e1fde4bcc941331346e5c7cc67dbc6740703c0594d0c",
 "output_root": "~/.local/share/agent-multi/p1_difficulty_lr_factorial_20260815_v2_decision",
 "cuda_uuid_declared": "GPU-612d1e0c-...", "cuda_uuid_process": "GPU-612d1e0c-...",
 "control_bundle": {"verdict": "PASS"}
}
```

The `after` capture must show the same `chain_id`, `contract_sha256`,
`output_root` and `cuda_uuid_process`, a **different** `pid`, and
`launcher` flipped from `nohup` to `systemd`. Any other combination is a
failed boundary: stop and report.
