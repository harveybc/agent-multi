"""Predeclared bounded-screen aggregator tests (PLR orders 2-3, PLR-06)."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "plateau_screen_aggregate",
        REPO / "tools" / "plateau_screen_aggregate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def tool():
    return _load()


SPEC = {"factor": 0.5, "lr_patience": 20, "min_lr": 1e-6,
        "threshold": 1e-6, "cooldown": 0, "start_epoch": 40}


def _pair_contract(default_seed, **over):
    base = {"data_sha256": "d" * 64,
            "epoch_timesteps": 20000, "max_epochs": 2000,
            "l1_patience": 60, "l1_patience_start_epoch": 40,
            "selection_metric": "easy_checkpoint_monitor_v1",
            "train_days": 120, "val_days": 40, "test_days": 40,
            "device_mask": f"GPU-mask-{default_seed}", "env_origin": "pinned",
            "learning_rate_initial": 3e-4}
    base["seed"] = default_seed
    base.update(over)
    return base


FULL40 = "93880beb" + "0" * 32
OTHER40 = "deadbeef" + "0" * 32


def _report(path, *, seed, policy, best_composite, val_ret=0.01,
            epochs=10, data_sha="d" * 64, reduced_at=(),
            accepted=True, classification=None, commit=FULL40,
            pair_over=None, arm_over=None, salt=""):
    lr = 3e-4
    history = []
    for e in range(1, epochs + 1):
        row = {
            "epoch": e, "selection_metric": "easy_checkpoint_monitor_v1",
            "composite": best_composite if e == epochs // 2 + 1
            else best_composite - 0.05,
            "l1_checkpoint_eligible": True,
            "checkpoint_improved": e in (1, epochs // 2 + 1),
            "val_total_return": val_ret,
            "val_trades": 10, "train_tail_trades": 5,
            "val_max_drawdown_fraction": 0.02,
            "observed_learning_rates": {"actor": lr},
        }
        if policy == "plateau":
            if e in reduced_at:
                row["plateau_lr"] = {"reduced": True, "old_lr": lr,
                                     "new_lr": lr * 0.5}
                lr = lr * 0.5
            else:
                row["plateau_lr"] = {"reduced": False, "old_lr": lr,
                                     "new_lr": lr}
        else:
            row["plateau_lr"] = None
        history.append(row)
    pair = _pair_contract(seed, **(pair_over or {}))
    arm = {"scheduler_policy": policy,
           "plateau_spec": dict(SPEC) if policy == "plateau" else None}
    arm.update(arm_over or {})
    doc = {
        "history": history, "stop_reason": "l1_early_stop",
        "epochs_run": epochs, "elapsed_seconds": 100.0,
        "data_sha256": data_sha, "accepted": accepted,
        "commit": commit,
        "config_sha256": ("f" * 64 if policy == "fixed" else "b" * 64),
        "budgets": {"seed": seed, "epoch_timesteps": 20000,
                    "max_epochs": 2000},
        "stopping_contract": {
            "l1_patience": {"effective": 60},
            "l1_patience_start_epoch": {"effective": 40},
            "classification": classification
            or "BOUNDED_120_40_40_DAY_SCHEDULER_SCREEN"},
        "split_facts": {"traces": {
            r: {"rows": 100, "first_timestamp": "2018-01-01",
                "last_timestamp": "2018-05-01"}
            for r in ("train_epoch", "train_tail_epoch",
                      "validation_epoch")}},
        "pair_contract": pair, "arm_contract": arm,
        "salt": salt,
    }
    path.write_text(json.dumps(doc))


def _screen(tmp_path, deltas, **kw):
    for seed, d in zip((101, 202, 303, 404), deltas):
        _report(tmp_path / f"seed{seed}_fixed_report.json", seed=seed,
                policy="fixed", best_composite=0.10, salt="f", **kw)
        _report(tmp_path / f"seed{seed}_plateau_report.json", seed=seed,
                policy="plateau", best_composite=0.10 + d,
                reduced_at=(5,), salt="p", **kw)
    return tmp_path


class TestPredeclaredRule:
    def test_signal_for(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.02, 0.03, -0.01])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        r = json.loads(out.read_text())
        assert r["outcome"] == "SHORT_SCREEN_SIGNAL_FOR_PLATEAU"
        assert r["dispersion"]["positive_seeds"] == 3

    def test_signal_against(self, tool, tmp_path):
        d = _screen(tmp_path, [-0.01, -0.02, -0.03, 0.01])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        assert json.loads(out.read_text())["outcome"] == (
            "SHORT_SCREEN_SIGNAL_AGAINST")

    def test_inconclusive_split(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.02, -0.01, -0.02])
        out = tmp_path / "agg.json"
        assert tool.main(["--screen-dir", str(d),
                          "--out-json", str(out)]) == 0
        assert json.loads(out.read_text())["outcome"] == "INCONCLUSIVE"

    def test_incomplete_screen_refuses(self, tool, tmp_path, capsys):
        _report(tmp_path / "seed101_fixed_report.json", seed=101,
                policy="fixed", best_composite=0.1)
        assert tool.main(["--screen-dir", str(tmp_path),
                          "--out-json",
                          str(tmp_path / "agg.json")]) == 2
        assert "REFUSED_INCOMPLETE_SCREEN" in capsys.readouterr().out

    def test_wall_clock_is_descriptive_only(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.02, 0.03, 0.04])
        out = tmp_path / "agg.json"
        tool.main(["--screen-dir", str(d), "--out-json", str(out)])
        r = json.loads(out.read_text())
        arm = r["pairs"]["101"]["fixed"]
        assert arm["descriptive_only"][
            "excluded_from_causal_conclusion"] is True


class TestPairIdentity:
    """AUD-F1-20260821-PLR-06 adversarial fixtures."""

    def _one_pair(self, tmp_path, **plateau_kw):
        _screen(tmp_path, [0.01, 0.01, 0.01, 0.01])
        if plateau_kw:
            _report(tmp_path / "seed101_plateau_report.json", seed=101,
                    policy="plateau", best_composite=0.11,
                    reduced_at=(5,), salt="p", **plateau_kw)
        return tmp_path

    def _expect(self, tool, tmp_path, match):
        with pytest.raises(tool.ScreenAggregationError, match=match):
            tool.main(["--screen-dir", str(tmp_path),
                       "--out-json", str(tmp_path / "agg.json")])

    def test_swapped_seed_label_refuses(self, tool, tmp_path):
        d = self._one_pair(tmp_path,
                           pair_over={"seed": 202})
        self._expect(tool, d, "swapped or mislabelled")

    def test_duplicate_report_refuses(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.01, 0.01, 0.01])
        fixed = d / "seed101_fixed_report.json"
        (d / "seed101_plateau_report.json").write_text(
            fixed.read_text())
        # the shared exact-identity check fires first: a duplicated
        # file necessarily carries an identical config_sha256
        self._expect(tool, d, "identical report|identical config_sha256")

    def test_fixed_arm_with_reduction_refuses(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.01, 0.01, 0.01])
        doc = json.loads(
            (d / "seed101_fixed_report.json").read_text())
        doc["history"][4]["plateau_lr"] = {
            "reduced": True, "old_lr": 3e-4, "new_lr": 1.5e-4}
        (d / "seed101_fixed_report.json").write_text(json.dumps(doc))
        self._expect(
            tool, d,
            "not a fixed arm|declares policy|arm identity violated")

    def test_swapped_arm_policy_refuses(self, tool, tmp_path):
        d = self._one_pair(tmp_path,
                           arm_over={"scheduler_policy": "fixed"})
        self._expect(tool, d, "swapped arms")

    def test_extra_factor_difference_refuses(self, tool, tmp_path):
        d = self._one_pair(tmp_path,
                           pair_over={"epoch_timesteps": 10000})
        self._expect(tool, d, "mismatch|swapped")

    def test_nonaccepted_arm_refuses(self, tool, tmp_path):
        d = self._one_pair(tmp_path, accepted=False)
        self._expect(tool, d, "not accepted")

    def test_wrong_plateau_spec_refuses(self, tool, tmp_path):
        d = self._one_pair(
            tmp_path,
            arm_over={"plateau_spec": dict(SPEC, factor=0.9)})
        self._expect(tool, d, "not the\\s+predeclared|predeclared")

    def test_non_halving_reduction_refuses(self, tool, tmp_path):
        d = _screen(tmp_path, [0.01, 0.01, 0.01, 0.01])
        doc = json.loads(
            (d / "seed101_plateau_report.json").read_text())
        doc["history"][4]["plateau_lr"]["new_lr"] = 2e-4
        (d / "seed101_plateau_report.json").write_text(json.dumps(doc))
        self._expect(tool, d, "halving")

    def test_legacy_label_refuses_after_retirement(self, tool,
                                                   tmp_path):
        """§C.6: after the one migrated screen result was committed at
        the working-branch closure, the frozen-tip legacy label and
        derivation path were retired — every legacy-labelled report now
        refuses regardless of commit."""
        d = self._one_pair(tmp_path,
                           classification="long_horizon_contract",
                           commit=FULL40)
        self._expect(tool, d, "legacy\s+exception was retired|retired")

    def test_missing_contracts_refuse_after_retirement(self, tool,
                                                       tmp_path):
        """§C.6: identity derivation from report facts is gone; a
        report without explicit canonical contracts refuses."""
        d = _screen(tmp_path, [0.01, 0.02, 0.03, 0.04])
        for f in d.glob("seed*_report.json"):
            doc = json.loads(f.read_text())
            del doc["pair_contract"], doc["arm_contract"]
            f.write_text(json.dumps(doc))
        self._expect(tool, d, "derivation path was retired")

    def test_no_eligible_checkpoint_refuses(self, tool, tmp_path):
        p = tmp_path / "r.json"
        p.write_text(json.dumps({"history": [
            {"epoch": 1, "composite": 0.0,
             "l1_checkpoint_eligible": False,
             "checkpoint_improved": False}],
            "data_sha256": "d" * 64}))
        with pytest.raises(tool.ScreenAggregationError,
                           match="typed refusal"):
            tool.arm_facts(p)
