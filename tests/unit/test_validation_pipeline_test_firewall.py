from __future__ import annotations

from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin


def test_final_evaluation_does_not_open_protected_test_when_disabled(monkeypatch) -> None:
    pipeline = PipelinePlugin()
    observed_splits = []

    def fake_eval(_env_name, _config, _path, _agent, _model, _seed, split):
        observed_splits.append(split)
        return {
            "total_return": 0.02,
            "max_drawdown_fraction": 0.01,
            "trades_total": 2,
        }

    monkeypatch.setattr(pipeline, "_eval_on_split", fake_eval)
    result = pipeline._final_eval(
        agent_plugin=None,
        model=None,
        train_env=None,
        env_plugin_name="unused",
        paths={"train": "train.csv", "train_tail": "tail.csv", "val": "val.csv", "test": "test.csv"},
        config={
            "evaluate_test_split": False,
            "selection_metric": "risk_adjusted_return",
            "risk_penalty_lambda": 1.0,
            "l1_generalization_gap_penalty_beta": 0.25,
            "eval_seed": 7,
            "save_model": None,
        },
        agent_plugin_for_wrap=None,
    )

    assert observed_splits == ["train", "train_tail", "validation"]
    assert result["splits"]["test"] == {
        "evaluation_skipped": True,
        "skip_reason": "protected_test_disabled_for_optimization",
    }
