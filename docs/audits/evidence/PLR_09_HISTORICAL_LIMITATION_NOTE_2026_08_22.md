# PLR-09 — historical-evidence limitation (recorded, not repaired)

AUD-F1-20260822-PLR-09 (S4): the eight migrated reports of the closed
bounded screen (`plateau_screen_20260821/`) carry no explicit
`pair_contract`/`arm_contract` and no materialization-time
config-minus-treatment hash. The strict diagnostic proves their
reported identity fields and a 33-field trajectory prefix — adequate
for the zero-authority exploratory artifact — but this is NOT
cryptographic proof of complete historical config identity, and must
never be described as such.

Per the dispatch order, the old reports are NOT rewritten to
manufacture contracts they never emitted. Every report generated from
this point on persists, BEFORE training, a launch manifest with the
full effective config, canonical pair/arm contracts,
`pair_config_sha256` (config minus the treatment key, hashed at
materialization time), commit, dataset hash and exact argv; the
aggregator refuses any arm lacking them.
