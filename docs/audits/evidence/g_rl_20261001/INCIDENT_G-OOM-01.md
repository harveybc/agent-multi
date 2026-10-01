# INCIDENT G-OOM-01: lane G test run OOM-killed on the coordinator (2026-09-30 23:22:28 local)

Scope: crispdm-g-rl-green2-1790828298-1027440 (pytest tests/rl_temporal, full suite), declared cap 1G (crispdm-run -m 1G).
What exceeded the cap: the pytest parent (anon 774,608 kB + file 749,552 kB) plus the RL05 fresh-process child (anon 261,064 kB + file 263,900 kB) inside one scope: about 2.0 GB resident against a 1 GB MemoryMax. The 1G cap was declared from the earlier 11-test SAC baseline, not measured for a suite that imports torch+SB3+backtrader and spawns a child; that is the error.
Consequence: both python processes killed by the memory cgroup; the owner desktop was under pressure at the time (PSI avg60 19.6, swap half used, gnome-settings-daemon crashed) as reported by the coordinator lane. My scope was inside the batch slice, so the kill stayed inside my scope, but the pressure was mine to avoid.
Orders received and applied: no further runs of any size on the coordinator; all test/parity/fixture runs on worker_a or worker_b under crispdm-run with a cap declared from this record (3G for the suite: 2.0 GB measured + child + margin); scratch under ~/.local/state/scratch/g-rl/; the Keras-side parity export in the pinned env on a worker.
Live coordinator jobs at the time of the order: none (the g-rl-keras lease had already exited with the test failure; systemd listed no crispdm-g-rl-* unit; nothing to stop).

## Other lane G scopes on the coordinator in the same window (coordinator readings)

- g-rl-rest (suite minus the Keras test, 1G): self-stopped 23:28:08 local at 1.064 GB of 1 GiB.
- g-rl-test_rl05_checkpoint (RL05 alone, 1G): stopped 23:30:38 local at 1.034 GB of 1 GiB (the fresh-process child inside the same scope).
- g-rl-keras (Keras fidelity test alone, 1G): exited by itself with a test failure (config-identity mismatch, since fixed); not killed, no stop needed.

Caps re-declared from these measurements: 3G for the whole suite (parent + child, measured peak 1.17 GB RSS on worker_a plus the coordinator's 2.0 GB combined reading and margin); 3G for the pinned-env Keras export (TF import). Nothing on the coordinator since the order.

## Kernel journal lines (host name redacted)

```
Sep 30 23:22:28 <coordinator> kernel: python invoked oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0, oom_score_adj=100
Sep 30 23:22:28 <coordinator> kernel: Memory cgroup stats for /user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope:
Sep 30 23:22:28 <coordinator> kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,mems_allowed=0,oom_memcg=/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope,task_memcg=/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope,task=python,pid=1027457,uid=1000
Sep 30 23:22:28 <coordinator> kernel: Memory cgroup out of memory: Killed process 1027457 (python) total-vm:10898456kB, anon-rss:774608kB, file-rss:749552kB, shmem-rss:0kB, UID:1000 pgtables:5700kB oom_score_adj:100
Sep 30 23:22:28 <coordinator> kernel: python invoked oom-killer: gfp_mask=0xcc0(GFP_KERNEL), order=0, oom_score_adj=100
Sep 30 23:22:28 <coordinator> kernel: Memory cgroup stats for /user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope:
Sep 30 23:22:28 <coordinator> kernel: oom-kill:constraint=CONSTRAINT_MEMCG,nodemask=(null),cpuset=user.slice,mems_allowed=0,oom_memcg=/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope,task_memcg=/user.slice/user-1000.slice/user@1000.service/crispdm.slice/crispdm-batch.slice/crispdm-g-rl-green2-1790828298-1027440.scope,task=python,pid=1027654,uid=1000
Sep 30 23:22:28 <coordinator> kernel: Memory cgroup out of memory: Killed process 1027654 (python) total-vm:6402000kB, anon-rss:261064kB, file-rss:263900kB, shmem-rss:0kB, UID:1000 pgtables:2228kB oom_score_adj:100
```
