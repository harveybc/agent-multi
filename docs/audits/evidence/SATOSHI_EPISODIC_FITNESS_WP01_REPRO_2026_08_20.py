#!/usr/bin/env python3
"""Independent adversarial reproducer for Satoshi commit 342f4a84."""
from pipeline_plugins._episodic_activity_fitness import evaluate_episode


def score(trades, total_return, *, drawdown=0.0, config=None):
    return evaluate_episode(
        total_return=total_return,
        max_drawdown_fraction=drawdown,
        sharpe=None,
        closed_trades=trades,
        scored_rows=2190,
        config=config,
    )["selection_value"]


one_trade = score(1, -0.00001)
active_learner = score(40, -0.0021)
deep_losses = [score(50, value) for value in (-1.0, -10.0, -100.0)]
invalid_config_loss = score(
    50, -0.2, config={"loss_activity_relief": 3.0})
invalid_config_gain = score(
    50, 0.1, drawdown=1.0, config={"gain_drawdown_share": 2.0})

print({
    "one_trade_score": one_trade,
    "active_learner_score": active_learner,
    "one_trade_still_wins": one_trade > active_learner,
    "deep_loss_scores": deep_losses,
    "deep_losses_alias": len(set(deep_losses)) == 1,
    "invalid_config_turns_loss_positive": invalid_config_loss > 0.0,
    "invalid_config_turns_gain_negative": invalid_config_gain < 0.0,
})

for bars_per_year in (0, -2190, True, 1.5):
    try:
        result = evaluate_episode(
            total_return=0.1,
            max_drawdown_fraction=0.0,
            sharpe=None,
            closed_trades=5,
            scored_rows=2190,
            bars_per_year=bars_per_year,
        )
        print({
            "bars_per_year": bars_per_year,
            "accepted": True,
            "scored_years": result["scored_years"],
            "annualized_trade_rate": result["annualized_trade_rate"],
        })
    except Exception as exc:  # evidence records crashes too
        print({
            "bars_per_year": bars_per_year,
            "accepted": False,
            "exception": type(exc).__name__,
        })
