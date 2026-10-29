# Commands executed (placeholders: <am-audit-worktree> = detached agent-multi @82862801; <hs-engine-pin-worktree> = detached heuristic-strategy @b936d01; <python> = conda env trading-stack python; <scratch> = this directory)

Environment for every python step: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

```
git -C <agent-multi> worktree add --detach <am-audit-worktree> 82862801
git -C <heuristic-strategy> worktree add --detach <hs-engine-pin-worktree> b936d01
sha256sum <hs-engine-pin-worktree>/app/short_horizon/execution.py   # 05179b24...
sha256sum <hs-engine-pin-worktree>/app/short_horizon/zigzag.py      # c06592f6...

# A
cd <am-audit-worktree>
crispdm-run -q -W 1800 -m 2400M -t 3600s -n sat-audit-A -- <python> -m pytest -q -p no:cacheprovider \
  tests/unit/test_zigzag_opportunities.py tests/unit/test_zigzag_risk_profiles.py tests/unit/test_horizon_risk_rules.py \
  tests/unit/test_zigzag_extension.py tests/unit/test_mechanism_games.py tests/unit/test_mechanism_m0_runner.py
  -> 105 passed (A_pytest.log); per-file counts via pytest --co (A_counts.txt)

# B
crispdm-run -q -W 1800 -m 2400M -t 3600s -n sat-audit-B -- <python> tools/verify_horizon_risk_rules.py \
  --profiles docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json \
  --profiles-sha aa2cb1f8f524db4091f8d3cd3b86b7b0f31af6f2ff71bb5a974f49e9d33d9d96 \
  --engine-root <hs-engine-pin-worktree> \
  --engine-sha 05179b244b9c095e4b541f5337ee9157b071aea78c8fe002c045f7625c39ff07 \
  --config docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json --output <scratch>/B_verify_out
sha256sum <scratch>/B_verify_out/*.json            (B_digests.txt)
cmp <scratch>/B_verify_out/CONFIG.json docs/audits/evidence/shared_risk_rules_20261028/CONFIG.json
cmp <scratch>/B_verify_out/REPORT.json docs/audits/evidence/shared_risk_rules_20261028/REPORT.json
zcat docs/audits/evidence/shared_risk_rules_20261028/CASES.json.gz | cmp - <scratch>/B_verify_out/CASES.json
<python> -I - < (inline script)  -> B_cases_independent_check.log

# C
<python> -I <scratch>/C_spotcheck.py docs/audits/evidence/zigzag_risk_profiles_20261028/PROFILES.json  -> C_spotcheck.log
<python> -I - < (inline script)  -> C_profiles_audit_fields.log

# D
<python> <scratch>/D_adversarial.py <am-audit-worktree> <hs-engine-pin-worktree>  -> D_adversarial.log

# E
<python> <scratch>/E_deviation_probe.py <hs-engine-pin-worktree>  -> E_deviation_probe.log
git -C <heuristic-strategy> diff a3130bac 28ad98e8 -- app/short_horizon/zigzag.py   (fc24c6f2 -> c06592f6)

# F
grep -n -i -E "population|generations|seeds|K ?=|leverage|rel_volume|3/6/12/24|12 ?h" handoff + docs 113-128  -> F_constants_grep.txt
<python> -I - (RESULTS.json metadata dumps)  -> F_k3_results_meta.log, F_k3_results_rows.log
git cat-file -t ba2af54a 3326f898 660511a0

# G
<python> tools/mechanism_m0.py > G_m0_run1.json ; again > G_m0_run2.json ; cmp ; json-equality vs M0_EXACT_REPORT.json
git show satoshi/audit-musashi-mechanism-synthesis-20261028:docs/audits/SATOSHI_AUDIT_VERDICT_MUSASHI_MECHANISM_SYNTHESIS_2026_10_28.md > G_satoshi_verdict.md

# H
sha256sum of deliverable modules  -> H_file_digests.txt
grep in <satoshi expert-guided worktree, read-only>: expert_guided/env.py _levels (lines 201-216), execution.py check_order (198-221)

# hygiene
git -C <am-audit-worktree> status --short      -> empty
git -C <hs-engine-pin-worktree> status --short -> empty
```
