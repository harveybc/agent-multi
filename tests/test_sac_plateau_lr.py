"""Contract tests for the epoch-level SAC plateau-LR controller.

Order: MUSASHI_TO_GENERAL_SATOSHI_SAC_PLATEAU_LR_AND_LONG_HORIZON_ORDER_2026_08_21 §3/§5.
"""
import inspect
import math

import pytest

from pipeline_plugins import _sac_plateau_lr as pl


def _controller(**over):
    base = dict(factor=0.5, lr_patience=2, min_lr=1e-6, threshold=0.0,
                cooldown=0, start_epoch=0, initial_lr=3e-4)
    base.update(over)
    return pl.SacPlateauLrController(**base)


def _noop_apply(lr):
    return {"applied": lr}


class TestTypedRefusals:
    @pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -0.5, True, "0.5",
                                     float("nan")])
    def test_factor_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(factor=bad)

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, "20", None])
    def test_lr_patience_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(lr_patience=bad)

    @pytest.mark.parametrize("bad", [0.0, -1e-6, float("inf"), True, "x"])
    def test_min_lr_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(min_lr=bad)

    @pytest.mark.parametrize("bad", [-0.1, float("nan"), True, "0"])
    def test_threshold_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(threshold=bad)

    @pytest.mark.parametrize("bad", [-1, True, 1.5, "0"])
    def test_cooldown_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(cooldown=bad)

    def test_initial_lr_below_min_refused(self):
        with pytest.raises(pl.SacPlateauLrError):
            _controller(initial_lr=1e-7, min_lr=1e-6)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), None,
                                     "0.5", True])
    def test_monitor_value_refused(self, bad):
        c = _controller()
        with pytest.raises(pl.SacPlateauLrError):
            c.observe(epoch=1, monitor_value=bad, apply_fn=_noop_apply)

    def test_epoch_must_advance(self):
        c = _controller()
        c.observe(epoch=3, monitor_value=1.0, apply_fn=_noop_apply)
        with pytest.raises(pl.SacPlateauLrError):
            c.observe(epoch=3, monitor_value=1.0, apply_fn=_noop_apply)
        with pytest.raises(pl.SacPlateauLrError):
            c.observe(epoch=2, monitor_value=1.0, apply_fn=_noop_apply)


class TestPlateauSemantics:
    def test_reduction_then_renewed_improvement(self):
        """Acceptance §5: plateau reduction followed by renewed improvement."""
        c = _controller(lr_patience=2)
        curve = [(1, 1.0), (2, 0.9), (3, 0.8), (4, 2.0)]
        records = [c.observe(epoch=e, monitor_value=v, apply_fn=_noop_apply)
                   for e, v in curve]
        assert records[0]["reason"] == "improved"
        assert records[2]["reduced"] is True
        assert records[2]["old_lr"] == pytest.approx(3e-4)
        assert records[2]["new_lr"] == pytest.approx(1.5e-4)
        assert records[3]["monitor_improved"] is True
        assert records[3]["best_value"] == pytest.approx(2.0)
        assert c.reductions_total == 1

    def test_reduction_never_masquerades_as_improvement(self):
        c = _controller(lr_patience=2)
        c.observe(epoch=1, monitor_value=1.0, apply_fn=_noop_apply)
        c.observe(epoch=2, monitor_value=0.5, apply_fn=_noop_apply)
        r = c.observe(epoch=3, monitor_value=0.5, apply_fn=_noop_apply)
        assert r["reduced"] is True
        assert r["monitor_improved"] is False
        assert r["best_value"] == pytest.approx(1.0)

    def test_flat_curve_reaches_min_lr_without_infinite_loop(self):
        """Acceptance §5: early stop path has no infinite reset loop."""
        c = _controller(factor=0.5, lr_patience=2, min_lr=1e-4,
                        initial_lr=3e-4)
        reasons = []
        for e in range(1, 30):
            r = c.observe(epoch=e, monitor_value=0.0, apply_fn=_noop_apply)
            reasons.append(r["reason"])
        assert c.current_lr == pytest.approx(1e-4)
        assert reasons.count("plateau_reduction") == 2
        assert "at_min_lr" in reasons
        # LR patience bookkeeping is separate: nothing here reset an
        # early-stop counter, which the caller owns exclusively.

    def test_two_reductions_before_early_stop_contract(self):
        """Order §3: >=2 reductions can occur before early stopping with
        the initial experimental contract (patience 60, lr_patience 20,
        start epoch 40)."""
        c = _controller(factor=0.5, lr_patience=20, min_lr=1e-6,
                        threshold=0.0, cooldown=0, start_epoch=40,
                        initial_lr=3e-4)
        early_patience = 60
        no_improve = 0
        stop_epoch = None
        reduction_epochs = []
        for e in range(1, 2001):
            r = c.observe(epoch=e, monitor_value=0.0, apply_fn=_noop_apply)
            if e == 1:
                continue  # first observation sets best
            if e > 40:
                no_improve += 1
            if r["reduced"]:
                reduction_epochs.append(e)
            if no_improve >= early_patience:
                stop_epoch = e
                break
        assert stop_epoch == 100
        # A third reduction coincides with the stopping epoch itself;
        # the ordered property is that AT LEAST TWO occur strictly
        # before early stop, and they do: epochs 60 and 80 < 100.
        assert reduction_epochs == [60, 80, 100]
        assert len([e for e in reduction_epochs if e < stop_epoch]) >= 2

    def test_warmup_accumulates_no_bad_epochs(self):
        c = _controller(lr_patience=2, start_epoch=10)
        for e in range(1, 11):
            r = c.observe(epoch=e, monitor_value=-float(e),
                          apply_fn=_noop_apply)
        assert r["reason"] in ("warmup", "improved")
        assert c.num_bad_epochs == 0
        assert c.reductions_total == 0

    def test_cooldown_suppresses_bad_epoch_accumulation(self):
        c = _controller(lr_patience=2, cooldown=3)
        c.observe(epoch=1, monitor_value=1.0, apply_fn=_noop_apply)
        c.observe(epoch=2, monitor_value=0.0, apply_fn=_noop_apply)
        r = c.observe(epoch=3, monitor_value=0.0, apply_fn=_noop_apply)
        assert r["reduced"] is True
        for e in (4, 5, 6):
            r = c.observe(epoch=e, monitor_value=0.0, apply_fn=_noop_apply)
            assert r["reason"] == "cooldown"
            assert c.num_bad_epochs == 0

    def test_threshold_is_min_delta(self):
        c = _controller(threshold=0.5, lr_patience=2)
        c.observe(epoch=1, monitor_value=1.0, apply_fn=_noop_apply)
        r = c.observe(epoch=2, monitor_value=1.4, apply_fn=_noop_apply)
        assert r["monitor_improved"] is False
        r = c.observe(epoch=3, monitor_value=1.6, apply_fn=_noop_apply)
        assert r["monitor_improved"] is True


class TestStateRoundTrip:
    def test_serialization_round_trip_is_exact(self):
        """PLR-01 relabel: serialization is for audit derivability and
        the sidecar only — the executing pipeline never loads it, and
        plateau runs are non-resumable (see
        TestNonResumableFailClosed)."""
        curve = [1.0, 0.9, 0.8, 0.7, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0]
        a = _controller(lr_patience=2, cooldown=1)
        for e, v in enumerate(curve[:5], start=1):
            a.observe(epoch=e, monitor_value=v, apply_fn=_noop_apply)
        snapshot = a.state_dict()
        b = _controller(lr_patience=2, cooldown=1)
        b.load_state_dict(snapshot)
        rec_a, rec_b = [], []
        for e, v in enumerate(curve[5:], start=6):
            rec_a.append(a.observe(epoch=e, monitor_value=v,
                                   apply_fn=_noop_apply))
            rec_b.append(b.observe(epoch=e, monitor_value=v,
                                   apply_fn=_noop_apply))
        assert rec_a == rec_b
        assert a.state_dict() == b.state_dict()

    def test_foreign_contract_state_refused(self):
        c = _controller()
        state = c.state_dict()
        state["contract_id"] = "someone_else.v9"
        with pytest.raises(pl.SacPlateauLrError):
            c.load_state_dict(state)


class TestStructuralTestFactInaccessibility:
    def test_observe_signature_is_closed(self):
        """Acceptance §5: test facts are structurally inaccessible — the
        only inbound facts are one epoch integer and one monitor scalar."""
        sig = inspect.signature(pl.SacPlateauLrController.observe)
        names = [p.name for p in sig.parameters.values()]
        assert names == ["self", "epoch", "monitor_value", "apply_fn"]
        assert all(
            p.kind is inspect.Parameter.KEYWORD_ONLY
            for p in list(sig.parameters.values())[1:]
        )
        c = _controller()
        with pytest.raises(TypeError):
            c.observe(epoch=1, monitor_value=1.0, apply_fn=_noop_apply,
                      test_total_return=99.0)


class _FakeOpt:
    def __init__(self, lr):
        self.param_groups = [{"lr": lr}, {"lr": lr}]


class _FakeSide:
    def __init__(self, opt):
        self.optimizer = opt


class _FakeSac:
    def __init__(self, lr=3e-4, with_ent=True):
        self.actor = _FakeSide(_FakeOpt(lr))
        self.critic = _FakeSide(_FakeOpt(lr))
        self.ent_coef_optimizer = _FakeOpt(lr) if with_ent else None
        self.lr_schedule = lambda progress: lr


class TestApplyToSac:
    def test_updates_every_governed_optimizer_and_schedule(self):
        m = _FakeSac(lr=3e-4, with_ent=True)
        rec = pl.apply_lr_to_sac(m, 1.5e-4)
        assert rec["optimizers_updated"] == ["actor", "critic", "ent_coef"]
        assert rec["optimizers_absent"] == []
        assert rec["old_lrs"] == {"actor": 3e-4, "critic": 3e-4,
                                  "ent_coef": 3e-4}
        for opt in (m.actor.optimizer, m.critic.optimizer,
                    m.ent_coef_optimizer):
            assert all(g["lr"] == 1.5e-4 for g in opt.param_groups)
        # SB3 re-applies lr_schedule inside train(); it MUST now be the
        # constant new LR or the param-group update above is reverted.
        assert isinstance(m.lr_schedule, pl.ConstantLr)
        assert m.lr_schedule(0.37) == 1.5e-4

    def test_fixed_entropy_absence_is_recorded_not_ignored(self):
        m = _FakeSac(with_ent=False)
        rec = pl.apply_lr_to_sac(m, 1e-4)
        assert rec["optimizers_updated"] == ["actor", "critic"]
        assert rec["optimizers_absent"] == ["ent_coef"]
        assert rec["old_lrs"]["ent_coef"] is None

    def test_missing_actor_or_critic_refuses_partial_update(self):
        m = _FakeSac()
        m.critic = _FakeSide(None)
        before = [g["lr"] for g in m.actor.optimizer.param_groups]
        with pytest.raises(pl.SacPlateauLrError):
            pl.apply_lr_to_sac(m, 1e-4)

    @pytest.mark.parametrize("bad", [0.0, -1e-4, float("nan"),
                                     float("inf"), True, "1e-4"])
    def test_bad_new_lr_refused(self, bad):
        with pytest.raises(pl.SacPlateauLrError):
            pl.apply_lr_to_sac(_FakeSac(), bad)

    def test_observed_lrs_reports_absence_as_none(self):
        m = _FakeSac(with_ent=False)
        lrs = pl.observed_sac_lrs(m)
        assert lrs == {"actor": 3e-4, "critic": 3e-4, "ent_coef": None}
        assert pl.observed_sac_lrs(object()) == {
            "actor": None, "critic": None, "ent_coef": None}


class TestBuildFromConfig:
    _GOOD = {"factor": 0.5, "lr_patience": 20, "min_lr": 1e-6,
             "threshold": 0.0, "cooldown": 0}

    def test_absent_config_disables_cleanly(self):
        """Acceptance §5: fixed-LR behavior when disabled — no controller
        exists at all."""
        out = pl.build_controller_from_config(
            {"plateau_lr": None},
            selection_metric="easy_checkpoint_monitor_v1",
            default_start_epoch=40, initial_lr=3e-4)
        assert out is None

    def test_monitor_metric_required(self):
        with pytest.raises(pl.SacPlateauLrError, match="easy_checkpoint"):
            pl.build_controller_from_config(
                {"plateau_lr": dict(self._GOOD)},
                selection_metric="episodic_activity_economic_v1",
                default_start_epoch=40, initial_lr=3e-4)

    @pytest.mark.parametrize("drop", ["factor", "lr_patience", "min_lr",
                                      "threshold", "cooldown"])
    def test_every_contract_number_is_required(self, drop):
        spec = dict(self._GOOD)
        del spec[drop]
        with pytest.raises(pl.SacPlateauLrError, match="missing"):
            pl.build_controller_from_config(
                {"plateau_lr": spec},
                selection_metric="easy_checkpoint_monitor_v1",
                default_start_epoch=40, initial_lr=3e-4)

    def test_unknown_keys_refused(self):
        spec = dict(self._GOOD, verbose=True)
        with pytest.raises(pl.SacPlateauLrError, match="unknown"):
            pl.build_controller_from_config(
                {"plateau_lr": spec},
                selection_metric="easy_checkpoint_monitor_v1",
                default_start_epoch=40, initial_lr=3e-4)

    def test_start_epoch_inherits_declared_default(self):
        c = pl.build_controller_from_config(
            {"plateau_lr": dict(self._GOOD)},
            selection_metric="easy_checkpoint_monitor_v1",
            default_start_epoch=40, initial_lr=3e-4)
        assert c.start_epoch == 40
        c2 = pl.build_controller_from_config(
            {"plateau_lr": dict(self._GOOD, start_epoch=7)},
            selection_metric="easy_checkpoint_monitor_v1",
            default_start_epoch=40, initial_lr=3e-4)
        assert c2.start_epoch == 7


class TestNonResumableFailClosed:
    """AUD-F1-20260821-PLR-01: plateau runs are non-resumable."""

    def test_sidecar_beside_warm_start_refuses(self, tmp_path):
        model = tmp_path / "best_model.zip"
        model.write_bytes(b"x")
        (tmp_path / "best_model.plateau_lr_state.json").write_text("{}")
        with pytest.raises(pl.SacPlateauLrError,
                           match="REFUSED_PLATEAU_RESUME"):
            pl.assert_not_resuming_plateau_run(str(model))

    def test_warm_start_without_sidecar_is_a_new_lifecycle(
            self, tmp_path):
        model = tmp_path / "best_model.zip"
        model.write_bytes(b"x")
        pl.assert_not_resuming_plateau_run(str(model))

    def test_no_warm_start_passes(self):
        pl.assert_not_resuming_plateau_run(None)
        pl.assert_not_resuming_plateau_run("")

    def test_executing_pipeline_calls_the_guard(self):
        """The guard must sit on the executing construction path, not
        only exist as a helper."""
        import inspect as _inspect
        from pipeline_plugins import rl_pipeline_with_validation as rl
        src = _inspect.getsource(rl)
        assert "assert_not_resuming_plateau_run" in src
        idx = src.index("assert_not_resuming_plateau_run")
        assert "warm_start_model" in src[idx:idx + 200]
