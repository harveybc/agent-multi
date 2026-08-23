# Finding 315 — execution evidence (non-destructive, mid-campaign)

- Active N training child NEVER signaled: wrapper bash SIGSTOPped
  (state Ts) while driver (S) and training grandchild (Rl) continued;
  process table captured at action time.
- Immutable detached worktree `am-p1-6e7bd128` created on omega,
  clean at full commit `6e7bd128f422939cba21d55b3eaed66739ef515f` —
  matching the Dragon/Gamma execution model.
- Executable identity verified (item 6): `6e7bd128 -> 08e0f724` diff
  is ONE docs-only file (+48 lines); all seven governed
  executable/contract hashes IDENTICAL at both commits
  (a8ff6e34… e4dc76ab… a078331d… 2f4fce18… 8083f761… 8c081166…
  2b31b777…); dragon/gamma worktrees clean at 6e7bd128. All twelve
  arms therefore share one executable identity; 08e0f724 is reported
  as provenance only.
- External launch-identity manifest written
  (`seed101_launch_identity_manifest.json`, sha256 cdbc6540d2cba31e…):
  full commit, clean-tree proof, per-file hashes, effective command
  template, arms governed EN-W/EN-F.
- Guarded wrapper `run_seed101_rest.sh`: before EACH remaining arm it
  re-verifies HEAD, tree cleanliness and every file hash against the
  manifest (refusal token ARM_<A>_REFUSED_IDENTITY_DRIFT), echoes the
  manifest digest, and writes a per-arm launch-identity sidecar that
  the return packet binds to the arm report.
- Automated handover unit `p1-seed101-handover`: waits for the N
  terminal report AND driver exit, verifies the stopped wrapper's
  cmdline before killing ONLY it, then starts `p1-seed101-rest` from
  the immutable worktree. No training process receives any signal.
- Item 7 regression: `tools/guarded_sequential_launcher.py` +
  4 tests (clean pass, HEAD drift, executable-hash drift, dirty tree)
  on branch `satoshi/315-correction-20260823` — the campaign
  worktrees stay untouched.
