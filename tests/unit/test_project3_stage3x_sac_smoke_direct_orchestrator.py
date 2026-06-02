import json

from tools import project3_stage3x_sac_smoke_direct_orchestrator as orch


def test_micro_nsga_task_disables_early_no_trade_abort(tmp_path):
    cfg = tmp_path / "micro.json"
    cfg.write_text(
        json.dumps(
            {
                "micro_nsga_generation": 0,
                "micro_nsga_individual_id": "g00_i01_02",
            }
        ),
        encoding="utf-8",
    )
    task = {
        "config_file": str(cfg),
        "contract_id": "btcusdt_perp__4h__selected__micro_nsga_g00_i01_02",
    }

    assert orch.is_micro_nsga_task(task) is True


def test_non_micro_task_keeps_early_no_trade_abort_enabled(tmp_path):
    cfg = tmp_path / "smoke.json"
    cfg.write_text(json.dumps({"_project3_stage3x_sac_smoke": True}), encoding="utf-8")
    task = {
        "config_file": str(cfg),
        "contract_id": "btcusdt_perp__4h__selected",
    }

    assert orch.is_micro_nsga_task(task) is False


def test_launch_log_name_is_short_for_deep_micro_nsga_ancestry():
    task_id = "stage3x_smoke_03_btcusdt_perp__4h__selected" + (
        "__micro_nsga_g01_i01_00"
        "__micro_nsga_g02_i00_00"
        "__micro_nsga_g03_i00_00"
        "__micro_nsga_g04_i00_00"
        "__micro_nsga_g05_i00_02"
        "__micro_nsga_g06_i00_01"
        "__micro_nsga_g07_i00_00"
        "__s2__base"
    )

    log_name = orch.launch_log_name("omega", task_id)

    assert log_name.endswith(".out")
    assert len(log_name) < 100
    assert log_name == orch.launch_log_name("omega", task_id)
