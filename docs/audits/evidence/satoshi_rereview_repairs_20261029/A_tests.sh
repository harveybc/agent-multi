. <scratch>/env.sh
cd $AM
python -m pytest -p no:cacheprovider tests/unit/test_external_audit_repairs.py \
  tests/unit/test_dynamic_account_arbiter.py tests/unit/test_dynamic_account_panel.py \
  tests/unit/test_dynamic_account_queue.py tests/unit/test_close_hold_controller.py \
  tests/unit/test_close_hold_reports.py tests/unit/test_close_hold_trace_audit.py \
  tests/unit/test_early_close_metrics.py tests/unit/test_absolute_exit_corpus.py \
  tests/unit/test_horizon_risk_rules.py tests/unit/test_zigzag_risk_profiles.py \
  tests/unit/test_protection_reference.py tests/unit/test_mechanism_games.py \
  tests/unit/test_mechanism_m0_runner.py --junitxml=$S/tests.xml -q -rs 2>&1 | tail -15
