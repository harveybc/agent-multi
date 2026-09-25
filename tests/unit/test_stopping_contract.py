"""AUD-P1LR-20260815-234: a run executes the stopping rule it declares.

The regressions below are anchored on the TERMINAL P1LR decision run
(identity ``c0e53cf18b7d60dd``): sixteen cells declared a 2000
pass-equivalent ceiling with patience 60 / floor 40 and every one of them
ended at epoch 80 because the activity-ineligible patience (40, start 40)
fired.  The counterexamples are the observed numbers, not invented ones.
"""
from __future__ import annotations

import pytest

from app import stopping_contract as sc
from pipeline_plugins.rl_pipeline_with_validation import PipelinePlugin


# The realized P1LR decision cell config (the four stopping fields the
# runner wrote) and the block the contract declared.
P1LR_CONFIG = {
    "max_epochs": 1996,
    "l1_patience": 60,
    "l1_patience_start_epoch": 40,
    "l1_activity_patience": 40,
    "l1_activity_patience_start_epoch": 40,
}
P1LR_DECLARED = {
    "max_global_pass_equivalent_checkpoints": 2000,
    "patience": 60,
    "patience_floor": 40,
    "budget_knobs": {"epoch_timesteps": 20000, "phase1_epochs": 4,
                     "phase2_max_epochs": 1996},
    "stopping": "train-monitor plus inner-validation paired stopping",
    "stopping_knobs": {
        "l1_patience": 60,
        "l1_patience_start_epoch": 40,
        "l1_activity_patience": 40,
        "l1_activity_patience_start_epoch": 40,
        "total_max_passes": 2000,
        "phase1_max_fraction": 0.3,
        "normal_phase_min_passes": 10,
    },
}


def _corrected(**overrides):
    declared = {k: (dict(v) if isinstance(v, dict) else v)
                for k, v in P1LR_DECLARED.items()}
    declared["effective_stopping_rules"] = {
        "terminators": ["l1_early_stop", "activity_stop",
                        "max_epochs_budget"],
        "earliest_stop_epoch": 80,
    }
    declared.update(overrides)
    return declared


class TestSurfaceHeldTogether:
    def test_defaults_mirror_the_pipeline_plugin(self):
        params = PipelinePlugin.plugin_params
        for key, value in sc.PIPELINE_STOPPING_DEFAULTS.items():
            assert key in params, (
                f"{key} is a stopping default this module mirrors but "
                "the pipeline plugin does not declare it")
            assert params[key] == value, key

    def test_every_declared_terminator_is_a_real_stop_reason(self):
        import inspect

        source = inspect.getsource(PipelinePlugin.run_pipeline)
        assert 'stop_reason = "max_epochs_budget"' in source
        assert 'stop_reason = "l1_early_stop"' in source
        # The activity terminator is assigned through the disposition
        # helper, which owns both of its labels.
        assert "_activity_stop_disposition(" in source
        assert set(sc.TERMINATORS) == {
            "l1_early_stop", "activity_stop", "max_epochs_budget"}

    def test_activity_patience_is_visible_in_the_debug_surface(self):
        assert "l1_activity_patience" in PipelinePlugin.plugin_debug_vars
        assert "l1_activity_patience_start_epoch" in \
            PipelinePlugin.plugin_debug_vars


class TestEffectiveRules:
    def test_p1lr_earliest_stop_is_the_observed_epoch_80(self):
        rules = sc.effective_stopping_rules(P1LR_CONFIG)
        assert rules["earliest_effective_stop_epoch"] == 80
        assert rules["rules"]["activity_stop"]["earliest_stop_epoch"] == 80
        assert rules["rules"]["l1_early_stop"]["earliest_stop_epoch"] == 100
        assert rules["rules"]["max_epochs_budget"][
            "ceiling_epochs"] == 1996

    def test_improvement_patience_cannot_fire_against_an_inactive_policy(
            self):
        rules = sc.effective_stopping_rules(P1LR_CONFIG)
        assert rules["rules"]["l1_early_stop"][
            "requires_activity_eligible_epochs"] is True
        assert rules["rules"]["activity_stop"][
            "requires_activity_eligible_epochs"] is False

    def test_hidden_default_terminator_is_derived_from_silence(self):
        # A config that declares NOTHING still stops at epoch 80: the
        # activity budget defaults to 40 and inherits floor 40.
        rules = sc.effective_stopping_rules({})
        assert rules["earliest_effective_stop_epoch"] == 80
        assert "activity_stop" in rules["enabled_terminators"]

    def test_disabled_activity_patience_leaves_only_the_paired_rule(self):
        rules = sc.effective_stopping_rules(
            {**P1LR_CONFIG, "l1_activity_patience": 0})
        assert rules["enabled_terminators"] == [
            "l1_early_stop", "max_epochs_budget"]
        assert rules["earliest_effective_stop_epoch"] == 100

    def test_activity_start_inherits_the_improvement_floor(self):
        rules = sc.effective_stopping_rules({
            "max_epochs": 500, "l1_patience": 60,
            "l1_patience_start_epoch": 25, "l1_activity_patience": 10})
        assert rules["rules"]["activity_stop"]["start_epoch"] == 25
        assert rules["earliest_effective_stop_epoch"] == 35

    def test_describe_names_every_enabled_terminator(self):
        text = sc.describe(sc.effective_stopping_rules(P1LR_CONFIG))
        assert "activity_stop@80" in text
        assert "l1_early_stop@100" in text


class TestRefusals:
    def test_the_terminal_p1lr_contract_is_refused(self):
        # The whole point: the shipped contract's declaration never
        # states that the run can end at 4% of its budget.
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=P1LR_DECLARED, config=P1LR_CONFIG)
        assert excinfo.value.code == sc.UNDECLARED_PREEMPTION
        assert "epoch 80" in str(excinfo.value)
        assert "1996-epoch ceiling" in str(excinfo.value)

    def test_the_corrected_declaration_passes(self):
        evidence = sc.assert_declared_stopping_contract(
            declared=_corrected(), config=P1LR_CONFIG)
        assert evidence["verdict"] == "STOPPING_CONTRACT_DECLARED"
        assert evidence["earliest_effective_stop_epoch"] == 80
        assert evidence["preempts_declared_ceiling"] is True
        assert set(evidence["enabled_terminators"]) == set(sc.TERMINATORS)

    def test_undeclared_terminator_is_refused(self):
        declared = _corrected()
        declared["stopping_knobs"].pop("l1_activity_patience")
        declared["effective_stopping_rules"] = {
            "terminators": ["l1_early_stop", "max_epochs_budget"],
            "earliest_stop_epoch": 80}
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=declared, config=P1LR_CONFIG)
        assert excinfo.value.code == sc.UNDECLARED_TERMINATOR
        assert "activity_stop" in str(excinfo.value)

    def test_decorative_knob_that_never_reaches_the_config_is_refused(
            self):
        # The silent-substitution case: the contract says 400, the
        # runtime applies 40.
        declared = _corrected()
        declared["stopping_knobs"]["l1_activity_patience"] = 400
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=declared, config=P1LR_CONFIG)
        assert excinfo.value.code == sc.KNOB_NOT_PROPAGATED
        assert "decorative" in str(excinfo.value)

    def test_paired_patience_substitution_is_refused(self):
        declared = _corrected()
        declared["stopping_knobs"]["l1_patience"] = 60
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=declared,
                config={**P1LR_CONFIG, "l1_patience": 5})
        assert excinfo.value.code == sc.KNOB_NOT_PROPAGATED

    def test_ceiling_mismatch_is_refused(self):
        declared = _corrected()
        declared["budget_knobs"] = {"epoch_timesteps": 20000,
                                    "phase1_epochs": 4,
                                    "phase2_max_epochs": 1996}
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=declared,
                config={**P1LR_CONFIG, "max_epochs": 80})
        assert excinfo.value.code == sc.CEILING_MISMATCH

    def test_declared_earliest_that_disagrees_is_refused(self):
        declared = _corrected()
        declared["effective_stopping_rules"]["earliest_stop_epoch"] = 2000
        with pytest.raises(sc.StoppingContractViolation) as excinfo:
            sc.assert_declared_stopping_contract(
                declared=declared, config=P1LR_CONFIG)
        assert excinfo.value.code == sc.UNDECLARED_PREEMPTION
        assert "declares earliest_stop_epoch=2000" in str(excinfo.value)

    def test_a_contract_without_preemption_needs_no_acknowledgement(self):
        # Activity budget wide enough that only the ceiling can fire.
        config = {"max_epochs": 50, "l1_patience": 60,
                  "l1_patience_start_epoch": 40,
                  "l1_activity_patience": 40,
                  "l1_activity_patience_start_epoch": 40}
        declared = {"stopping_knobs": {
            "l1_patience": 60, "l1_patience_start_epoch": 40,
            "l1_activity_patience": 40,
            "l1_activity_patience_start_epoch": 40}}
        evidence = sc.assert_declared_stopping_contract(
            declared=declared, config=config)
        assert evidence["preempts_declared_ceiling"] is False


class TestNonRaisingWrapper:
    def test_refusal_is_typed_for_a_launcher(self):
        evidence, refusal = sc.stopping_contract_refusal(
            declared=P1LR_DECLARED, config=P1LR_CONFIG)
        assert evidence is None
        assert refusal["outcome"] == \
            "REFUSED_STOPPING_CONTRACT_UNDECLARED"
        assert refusal["code"] == sc.UNDECLARED_PREEMPTION

    def test_pass_returns_evidence_and_no_refusal(self):
        evidence, refusal = sc.stopping_contract_refusal(
            declared=_corrected(), config=P1LR_CONFIG)
        assert refusal is None
        assert evidence["verdict"] == "STOPPING_CONTRACT_DECLARED"
