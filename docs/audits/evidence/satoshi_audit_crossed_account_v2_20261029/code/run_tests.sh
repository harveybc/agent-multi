#!/bin/bash
source $(dirname $0)/env.sh
cd $AM
T="test_external_audit_repairs test_crossed_account_panel test_dynamic_account_arbiter test_dynamic_account_panel test_dynamic_account_queue test_close_hold_controller test_close_hold_reports test_close_hold_trace_audit test_early_close_metrics test_absolute_exit_corpus test_horizon_risk_rules test_zigzag_risk_profiles test_protection_reference test_mechanism_games test_mechanism_m0_runner test_crossed_account_queue test_crossed_account_analysis"
ARGS=""; for t in $T; do ARGS="$ARGS tests/unit/$t.py"; done
echo "== focused suite (16 preflight files + analysis)"
$PY -m pytest -q -p no:cacheprovider $ARGS --junitxml=$S/out/focused_suite.xml -rs 2>&1 | tail -15
echo "== two files named in request"
$PY -m pytest -q -p no:cacheprovider tests/unit/test_crossed_account_analysis.py tests/unit/test_crossed_account_queue.py -rs 2>&1 | tail -5
