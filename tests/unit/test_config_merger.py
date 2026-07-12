from app.config_merger import process_unknown_args


def test_unknown_cli_values_are_typed_before_canonical_resolution() -> None:
    assert process_unknown_args(
        [
            "--optimization_resume", "true",
            "--evaluate_test_split", "false",
            "--ga_population", "20",
            "--risk_penalty_lambda", "1.5",
            "--load_model", "null",
        ]
    ) == {
        "optimization_resume": True,
        "evaluate_test_split": False,
        "ga_population": 20,
        "risk_penalty_lambda": 1.5,
        "load_model": None,
    }
