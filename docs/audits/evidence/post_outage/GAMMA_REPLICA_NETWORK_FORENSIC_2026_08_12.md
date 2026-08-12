# Gamma→Dragon Replica: Network Forensic + Repair (finding 230)

Date: 2026-08-12 early morning.

## Forensic facts

1. **Neither prior transfer ever moved a byte.** My 2026-08-11 rsync
   (`replica_rsync.log`) and the auditor's 19:35 transient unit
   (`gamma-history-replica.service`) both targeted `dragon:` from
   gamma and both died with `ssh: connect to host dragon port 22:
   Connection timed out … (0 bytes received)`. The unit was flapping
   in a restart loop every ~3 minutes until stopped.
2. **Root cause:** gamma resolves `dragon.lan` to IPv6 addresses only
   (ULA `fdd3:…` + global `2803:…`) and gamma's IPv6 path to dragon
   is broken since the outage; omega→dragon works, gamma→dragon:22
   times out. Gamma's `~/.ssh/config` has only two host blocks
   (`omega`, `dragon-replica`); bare `dragon` uses the dead LAN path.
3. **Dragon had no replica data at all** — `/home/harveybc/replicas/`
   did not exist; dragon free space unchanged at 542 GB. The audit's
   "173 GB at the intended path" could not have come from these
   transfers; it should be re-derived (residual doubt for Musashi).
4. **Working route:** the owner-provisioned `dragon-replica` ssh
   alias (tailscale 192.0.2.11 port 22022) — verified to be the
   real dragon (hostname + `tailscale ip` match) with 542 GB free.

## Repair executed

- Stopped the flapping unit (`systemctl --user stop
  gamma-history-replica.service`) — it could never succeed.
- Created `/home/harveybc/replicas/` on dragon (rsync does not create
  parents; observed exit 11 without it).
- Relaunched from gamma via the WORKING route:
  `rsync -a --partial --info=progress2 --bwlimit=25000
  /home/harveybc/Documents/GitHub/_pre_trading_stack_20260713/
  dragon-replica:/home/harveybc/replicas/_pre_trading_stack_20260713_gamma_replica/`
  under nice/ionice; log `replica_rsync_v2.log` on gamma. Transfer
  confirmed flowing (destination populated). ETA ≈ 2.5 h at the
  25 MB/s ceiling.
- After completion: dual-side deterministic digest manifests
  (sha256 per file, sorted) on source and destination, compared;
  OLAP/chain DBs, manifests, configs, metrics and model artifacts
  explicitly covered. No source deletion — owner review required.

## Owner item (network)

gamma→dragon LAN (IPv6) is broken post-outage: bare `dragon` is
unreachable from gamma while omega→dragon works. Until the
router/firewall path is repaired, gamma-originated transfers must use
`dragon-replica` (tailscale). Worth an owner look at the LAN.
