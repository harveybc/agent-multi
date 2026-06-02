from __future__ import annotations

from tools.project3_no_trade_preflight import diagnose, forced_action


def test_forced_action_uses_continuous_extremes():
    cfg = {"action_space_mode": "continuous"}
    assert forced_action(cfg, 0, 16) == [1.0]
    assert forced_action(cfg, 16, 16) == [-1.0]


def test_forced_action_uses_discrete_long_short():
    cfg = {"action_space_mode": "discrete"}
    assert forced_action(cfg, 0, 16) == 1
    assert forced_action(cfg, 16, 16) == 2


def test_diagnose_execution_path_closes_trades():
    payload = {
        "closed_trades": 1,
        "min_trades": 1,
        "entry_orders_submitted": 1,
        "entry_actions_seen": 1,
        "blocked_atr_warmup": 0,
        "blocked_session_filter": 0,
        "blocked_non_positive_atr": 0,
        "blocked_non_positive_size": 0,
        "blocked_non_positive_price": 0,
    }
    assert diagnose(payload) == "execution_path_closes_trades"


def test_diagnose_execution_guard_block():
    payload = {
        "closed_trades": 0,
        "min_trades": 1,
        "entry_orders_submitted": 0,
        "entry_actions_seen": 10,
        "blocked_atr_warmup": 10,
        "blocked_session_filter": 0,
        "blocked_non_positive_atr": 0,
        "blocked_non_positive_size": 0,
        "blocked_non_positive_price": 0,
    }
    assert diagnose(payload) == "execution_guard_blocked_all_forced_actions:atr_warmup"
