# Auto-orchestrator quick reference

Long-running daemon on Omega that polls all 3 hosts every 2 min and dispatches
work from `tools/work_queue.json`.

## How to queue new work

Edit `tools/work_queue.json` and append a task object to the `tasks` array:

```json
{
  "id": "eth-sac-iter9-s0-2",
  "config": "examples/config/sac_eth_1h_twelve_atr.json",
  "seeds": [0, 1, 2],
  "run_tag": "p4sacfix2",
  "preferred_hosts": ["dragon", "omega"],
  "requires_gpu": true,
  "priority": 10,
  "status": "pending",
  "notes": "second SAC iteration after p5 rank"
}
```

The daemon will pick it up on the next poll and launch it on the first idle
compatible host, then update `status` to `running` and later `done`.

## Control commands

- **Status**: `tail -f ~/p4_launch/orchestrator.log`
- **Pause** (finishes in-flight, stops dispatching new):
  `touch /home/harveybc/Documents/GitHub/agent-multi/tools/.orchestrator_pause`
- **Resume**: `rm /home/harveybc/Documents/GitHub/agent-multi/tools/.orchestrator_pause`
- **Stop daemon** (in-flight jobs keep running): `pkill -f auto_orchestrator`
- **Restart**:
  ```
  nohup /home/harveybc/anaconda3/envs/tensorflow/bin/python \
      /home/harveybc/Documents/GitHub/agent-multi/tools/auto_orchestrator.py \
      > /home/harveybc/p4_launch/orchestrator.log 2>&1 < /dev/null & disown
  ```

## Finalize step

When the queue is empty AND all hosts are idle, the daemon runs:
1. `tools/p5_eval_holdout.py --skip-if-exists` (hold-out eval on d6 for any
   new policies).
2. `tools/p5_rank.py --top 3 --min_seeds 3` → writes `logs/partIII/p5_rank.{csv,md}`.
3. Exits cleanly.

To restart it after it exits, just relaunch with the same nohup command.

## Hosts

| Host   | Kind   | Concurrency | GPU            |
|--------|--------|-------------|----------------|
| dragon | remote | 2           | RTX 4090 16 GB |
| gamma  | remote | 1           | RTX 5070 Ti 12 GB |
| omega  | local  | 1           | RTX 4070 Laptop 8 GB |

Omega launches bypass `bash -ic` and use the absolute python path directly
to avoid SIGTTOU from the VS Code pty. Remote launches use `nohup bash -ic`
so conda activates via `.bashrc` on ssh.
