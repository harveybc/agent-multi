. <scratch>/env.sh
cd $AM
echo "== archive sha256 BEFORE"; sha256sum docs/audits/evidence/dynamic_account_pareto_20261029/* 
echo "== archive vs predecessor commit 3644ece9 (git diff --stat)"; git diff --stat 3644ece9 2318b0e3 -- docs/audits/evidence/dynamic_account_pareto_20261029/ ; echo "(empty = byte-identical)"
/usr/bin/time -v python tools/verify_external_audit_repairs.py --engine-root $HS --test-report $S/tests.xml --output $S/B_repair 2>&1 | grep -vE '^\s+(Page|Minor|Major|Volun|Invol|Swaps|File|Socket|Signals|Average|Exit|Percent|System|User|Elapsed \(wall|Command)' 
echo "== archive sha256 AFTER"; sha256sum docs/audits/evidence/dynamic_account_pareto_20261029/*
git status --porcelain
